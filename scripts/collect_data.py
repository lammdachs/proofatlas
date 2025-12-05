#!/usr/bin/env python3
"""
Collect training data from TPTP problems using the config system.

Results are cached per-problem in .data/traces/{config_name}/{problem}.pt
Re-running reuses cached results unless --force is specified.

Usage:
    python scripts/collect_data.py --data-config default
    python scripts/collect_data.py --data-config unit_equality --max-problems 100
    python scripts/collect_data.py --data-config configs/data/custom.json
    python scripts/collect_data.py --data-config default --force  # ignore cache
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from proofatlas.ml.config import DataConfig, ProblemFilters


# Caching
TRACE_CACHE_DIR = ".data/traces"


def get_trace_cache_path(config_name: str, problem_path: str) -> Path:
    """Get cache path for a problem trace."""
    # Use problem path without extension, replace / with _
    problem_name = problem_path.replace("/", "_").replace(".p", "")
    base = Path(__file__).parent.parent
    return base / TRACE_CACHE_DIR / config_name / f"{problem_name}.pt"


def load_cached_trace(config_name: str, problem_path: str) -> Optional[Dict[str, Any]]:
    """Load cached trace if it exists."""
    cache_path = get_trace_cache_path(config_name, problem_path)
    if not cache_path.exists():
        return None
    try:
        return torch.load(cache_path, weights_only=False)
    except Exception:
        return None


def save_trace_cache(config_name: str, problem_path: str, result: Dict[str, Any]):
    """Save trace to cache."""
    cache_path = get_trace_cache_path(config_name, problem_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)


def get_tptp_base() -> Path:
    """Get the TPTP problems base directory."""
    return Path(__file__).parent.parent / ".tptp/TPTP-v9.0.0/Problems"


def get_weights_dir() -> Path:
    """Get the weights directory."""
    return Path(__file__).parent.parent / ".weights"


def load_problem_metadata() -> Dict[str, Any]:
    """Load the problem metadata JSON."""
    metadata_path = Path(__file__).parent.parent / ".data/problem_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Problem metadata not found: {metadata_path}\n"
            "Run: python scripts/extract_problem_metadata.py"
        )
    with open(metadata_path) as f:
        return json.load(f)


def filter_problems(problems: List[Dict], filters: ProblemFilters) -> List[Dict]:
    """Filter problems based on config filters."""
    result = []
    for p in problems:
        if filters.status and p['status'] not in filters.status:
            continue
        if filters.format and p['format'] not in filters.format:
            continue
        if filters.has_equality is not None and p['has_equality'] != filters.has_equality:
            continue
        if filters.is_unit_only is not None and p['is_unit_only'] != filters.is_unit_only:
            continue
        if filters.min_rating is not None and p['rating'] < filters.min_rating:
            continue
        if filters.max_rating is not None and p['rating'] > filters.max_rating:
            continue
        if filters.min_clauses is not None and p['num_clauses'] < filters.min_clauses:
            continue
        if filters.max_clauses is not None and p['num_clauses'] > filters.max_clauses:
            continue
        if filters.domains and p['domain'] not in filters.domains:
            continue
        if filters.exclude_domains and p['domain'] in filters.exclude_domains:
            continue
        result.append(p)
    return result


def collect_from_problem(
    problem_path: Path,
    max_iterations: int,
    timeout_secs: float,
    literal_selection: str = "all",
    selector_name: str = "age_weight",
    weights_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Run prover on a problem and extract training data.

    Args:
        problem_path: Path to TPTP problem file
        max_iterations: Maximum saturation steps
        timeout_secs: Timeout in seconds
        literal_selection: Literal selection strategy
        selector_name: Name of selector (maps to Rust implementation)
        weights_path: Path to weights file for ML selectors (in .weights/)
    """
    from proofatlas import ProofState
    from proofatlas.ml.graph_utils import to_torch_tensors

    try:
        with open(problem_path) as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Read error: {e}"}

    state = ProofState()
    try:
        state.add_clauses_from_tptp(content)
    except Exception as e:
        return {"error": f"Parse error: {e}"}

    # Configure literal selection
    state.set_literal_selection(literal_selection)

    start = time.time()
    try:
        # TODO: Update Rust API to accept selector_name + weights_path
        # For now, still using the old ONNX path interface
        weights_str = str(weights_path) if weights_path else ""
        proof_found = state.run_saturation(max_iterations, timeout_secs, selector_name, weights_str)
    except Exception as e:
        return {"error": f"Saturation error: {e}"}
    elapsed = time.time() - start

    if not proof_found:
        return {"proof_found": False, "time": elapsed}

    try:
        examples = state.extract_training_examples()
        if not examples:
            return {"proof_found": True, "time": elapsed, "examples": 0}

        clause_ids = [e.clause_idx for e in examples]
        graphs = state.clauses_to_graphs(clause_ids)
        graph_tensors = [to_torch_tensors(g) for g in graphs]
        labels = [e.label for e in examples]
        stats = state.get_proof_statistics()

        return {
            "proof_found": True,
            "time": elapsed,
            "examples": len(examples),
            "graphs": graph_tensors,
            "labels": labels,
            "clause_ids": clause_ids,
            "statistics": stats,
        }
    except Exception as e:
        return {"error": f"Extraction error: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Collect training data from TPTP problems")
    parser.add_argument("--data-config", "-d", default="default", help="Data config name or path")
    parser.add_argument("--from-preset", type=str,
                       help="Use traces from bench.py preset (e.g., time_sel21)")
    parser.add_argument("--output", "-o", type=Path, help="Output file (overrides config)")
    parser.add_argument("--max-problems", type=int, help="Maximum problems to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--force", action="store_true", help="Ignore cached results")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.data_config)
    if config_path.exists():
        config = DataConfig.load(config_path)
    else:
        config = DataConfig.load_preset(args.data_config)

    print(f"Data config: {config.name}")
    print(f"Description: {config.description}")

    # Load and filter problems
    print("\nLoading problem metadata...")
    metadata = load_problem_metadata()
    all_problems = metadata["problems"]
    print(f"Total problems: {len(all_problems)}")

    filtered = filter_problems(all_problems, config.problem_filters)
    print(f"After filtering: {len(filtered)}")

    if args.max_problems:
        filtered = filtered[:args.max_problems]
        print(f"Limited to: {len(filtered)}")

    if args.dry_run:
        print("\nDry run - first 20 problems:")
        for p in filtered[:20]:
            print(f"  {p['path']} (rating={p['rating']}, clauses={p['num_clauses']})")
        if len(filtered) > 20:
            print(f"  ... and {len(filtered) - 20} more")
        return

    # Collect data
    tptp_base = get_tptp_base()
    output_path = args.output or Path(config.output.trace_dir) / f"{config.name}_data.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve selector and weights
    selector_name = config.selector.name
    weights_path = None
    if config.selector.weights:
        weights_path = get_weights_dir() / config.selector.weights
        if not weights_path.exists():
            print(f"Warning: Weights file not found: {weights_path}")
            weights_path = None
        else:
            print(f"Using weights: {weights_path}")

    print(f"Using selector: {selector_name}")

    # Determine cache name (use preset if --from-preset, else config name)
    cache_name = args.from_preset if args.from_preset else config.name
    if args.from_preset:
        print(f"Cache: {TRACE_CACHE_DIR}/{cache_name}/ (from bench.py preset)")
    else:
        print(f"Cache: {TRACE_CACHE_DIR}/{cache_name}/ {'(--force: ignoring)' if args.force else '(reusing cached)'}")

    all_graphs, all_labels, all_problem_names, all_clause_ids = [], [], [], []
    successful, failed, no_proof, cached_count = 0, 0, 0, 0
    total_time = 0

    print(f"\nCollecting from {len(filtered)} problems...")
    print(f"Timeout: {config.trace_collection.prover_timeout}s, Max steps: {config.trace_collection.max_steps}")
    print(f"Literal selection: {config.solver.literal_selection}\n")

    for i, problem in enumerate(filtered):
        problem_path = tptp_base / problem["path"]
        pct = 100 * (i + 1) / len(filtered)
        print(f"\r[{pct:5.1f}%] {problem['path']:<40}", end="", flush=True)

        # Check cache first (check preset cache first if specified, then config cache)
        cached = None
        if not args.force:
            cached = load_cached_trace(cache_name, problem["path"])
            # Also check config cache if using preset and not found
            if cached is None and args.from_preset:
                cached = load_cached_trace(config.name, problem["path"])

        if cached is not None:
            result = cached
            cached_count += 1
        else:
            result = collect_from_problem(
                problem_path,
                max_iterations=config.trace_collection.max_steps,
                timeout_secs=config.trace_collection.prover_timeout,
                literal_selection=config.solver.literal_selection,
                selector_name=selector_name,
                weights_path=weights_path,
            )
            # Cache the result (even errors/no-proof, to avoid re-running)
            save_trace_cache(config.name, problem["path"], result)

        if "error" in result:
            failed += 1
            if args.verbose:
                print(f" ERROR: {result['error']}")
        elif not result.get("proof_found"):
            no_proof += 1
            total_time += result.get("time", 0)
        else:
            successful += 1
            total_time += result.get("time", 0)
            if result.get("graphs"):
                all_graphs.extend(result["graphs"])
                all_labels.extend(result["labels"])
                all_problem_names.extend([problem["path"]] * len(result["labels"]))
                all_clause_ids.extend(result["clause_ids"])

    print("\n\n" + "=" * 50)
    print(f"Successful: {successful}, No proof: {no_proof}, Errors: {failed}")
    if cached_count > 0:
        print(f"Cached: {cached_count} (reused from previous runs)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Training examples: {len(all_labels)}")

    if all_labels:
        pos = sum(all_labels)
        print(f"  Positive: {pos} ({100*pos/len(all_labels):.1f}%), Negative: {len(all_labels)-pos}")

        torch.save({
            "config": config.to_dict(),
            "graphs": all_graphs,
            "labels": all_labels,
            "problem_names": all_problem_names,
            "clause_ids": all_clause_ids,
        }, output_path)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
