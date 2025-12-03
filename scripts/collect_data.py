#!/usr/bin/env python3
"""
Collect training data from TPTP problems using the config system.

Usage:
    python scripts/collect_data.py --data-config default
    python scripts/collect_data.py --data-config unit_equality --max-problems 100
    python scripts/collect_data.py --data-config configs/data/custom.json
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


def get_tptp_base() -> Path:
    """Get the TPTP problems base directory."""
    return Path(__file__).parent.parent / ".data/problems/tptp/TPTP-v9.0.0/Problems"


def get_selectors_dir() -> Path:
    """Get the selectors directory."""
    return Path(__file__).parent.parent / ".selectors"


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
    onnx_model_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Run prover on a problem and extract training data."""
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
        onnx_path = str(onnx_model_path) if onnx_model_path else None
        proof_found = state.run_saturation(max_iterations, timeout_secs, onnx_path)
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
    parser.add_argument("--output", "-o", type=Path, help="Output file (overrides config)")
    parser.add_argument("--max-problems", type=int, help="Maximum problems to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

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

    # Resolve ONNX model path
    onnx_model_path = get_selectors_dir() / config.solver.clause_selector
    if not onnx_model_path.exists():
        print(f"Warning: ONNX model not found: {onnx_model_path}")
        onnx_model_path = None
    else:
        print(f"Using clause selector: {onnx_model_path}")

    all_graphs, all_labels, all_problem_names, all_clause_ids = [], [], [], []
    successful, failed, no_proof = 0, 0, 0
    total_time = 0

    print(f"\nCollecting from {len(filtered)} problems...")
    print(f"Timeout: {config.trace_collection.prover_timeout}s, Max steps: {config.trace_collection.max_steps}")
    print(f"Literal selection: {config.solver.literal_selection}\n")

    for i, problem in enumerate(filtered):
        problem_path = tptp_base / problem["path"]
        pct = 100 * (i + 1) / len(filtered)
        print(f"\r[{pct:5.1f}%] {problem['path']:<40}", end="", flush=True)

        result = collect_from_problem(
            problem_path,
            max_iterations=config.trace_collection.max_steps,
            timeout_secs=config.trace_collection.prover_timeout,
            literal_selection=config.solver.literal_selection,
            onnx_model_path=onnx_model_path,
        )

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
