#!/usr/bin/env python3
"""
Prove a single TPTP problem.

USAGE:
    proofatlas problem.p
    proofatlas problem.p --preset gcn_mlp_sel21
    proofatlas problem.p --timeout 30 --literal-selection 21
    proofatlas problem.p --json output.json
    proofatlas --list
"""

import argparse
import json
import sys
import time
from pathlib import Path


def find_project_root() -> Path:
    """Find the proofatlas project root."""
    candidates = [Path.cwd(), Path(__file__).parent.parent.parent.parent]
    for candidate in candidates:
        if (candidate / "configs" / "proofatlas.json").exists():
            return candidate.resolve()

    path = Path.cwd()
    while path != path.parent:
        if (path / "configs" / "proofatlas.json").exists():
            return path.resolve()
        path = path.parent

    return Path.cwd()


def find_tptp_root(base_dir: Path) -> Path:
    """Find the TPTP root directory."""
    tptp_config_path = base_dir / "configs" / "tptp.json"
    if tptp_config_path.exists():
        with open(tptp_config_path) as f:
            tptp_config = json.load(f)
        return base_dir / tptp_config["paths"]["root"]

    for candidate in [base_dir / ".tptp" / "TPTP-v9.0.0", base_dir / ".tptp"]:
        if candidate.exists():
            return candidate

    return base_dir


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def list_presets(base_dir: Path):
    """List available presets."""
    config_path = base_dir / "configs" / "proofatlas.json"
    if not config_path.exists():
        print("Error: configs/proofatlas.json not found", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    presets = config.get("presets", {})

    print("Available presets:")
    for name, preset in sorted(presets.items()):
        desc = preset.get("description", "")
        embedding = preset.get("embedding")
        scorer = preset.get("scorer")

        model_info = ""
        if embedding and scorer:
            model_info = f" [{embedding}+{scorer}]"

        print(f"  {name:<25} {desc}{model_info}")


def main():
    parser = argparse.ArgumentParser(
        description="Prove a single TPTP problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("problem", type=Path, nargs="?", help="Path to TPTP problem file")
    parser.add_argument("--preset", help="Use preset from configs/proofatlas.json")
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument("--timeout", type=int, help="Set timeout in seconds (default: 60)")
    parser.add_argument("--max-clauses", type=int, help="Set max clauses (default: 10000)")
    parser.add_argument(
        "--literal-selection",
        type=int,
        choices=[0, 20, 21, 22],
        help="Literal selection: 0=all, 20=maximal, 21=unique/neg/max, 22=neg/max",
    )
    parser.add_argument("--include", action="append", dest="include_dirs", help="Add include directory")
    parser.add_argument("--json", dest="json_output", help="Export proof attempt to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    base_dir = find_project_root()

    # Handle --list
    if args.list:
        list_presets(base_dir)
        return

    # Require problem file if not --list
    if args.problem is None:
        parser.error("the following arguments are required: problem")

    if not args.problem.exists():
        print(f"Error: File not found: {args.problem}", file=sys.stderr)
        sys.exit(1)

    config_path = base_dir / "configs" / "proofatlas.json"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    presets = config.get("presets", {})

    # Load preset if specified
    preset = {}
    if args.preset:
        if args.preset not in presets:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Use --list to see available presets", file=sys.stderr)
            sys.exit(1)
        preset = presets[args.preset]

    # Get values from preset, then override with command line args
    timeout = args.timeout if args.timeout is not None else preset.get("timeout", 60)
    max_clauses = args.max_clauses if args.max_clauses is not None else preset.get("max_clauses", 0)
    literal_selection = (
        args.literal_selection if args.literal_selection is not None else preset.get("literal_selection", 0)
    )
    age_weight_ratio = preset.get("age_weight_ratio", 0.5)
    max_iterations = preset.get("max_iterations", 0)

    # Check for ML selector in preset
    embedding_type = None
    weights_path = None
    model_name = None

    if preset:
        from proofatlas.ml import is_learned_selector, get_embedding_type, get_model_name, find_weights

        if is_learned_selector(preset):
            embedding_type = get_embedding_type(preset)
            model_name = get_model_name(preset)
            weights_dir = base_dir / ".weights"
            weights_path = find_weights(weights_dir, preset)

            if not weights_path:
                print(f"Error: Model weights not found for {model_name}", file=sys.stderr)
                print(f"Train with: proofatlas-bench --preset {args.preset} --retrain", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f"Using ML selector: {model_name}")
                print(f"  Weights: {weights_path}")

    # Find TPTP root for includes
    tptp_root = find_tptp_root(base_dir)
    include_dirs = args.include_dirs or []
    if tptp_root.exists():
        include_dirs.insert(0, str(tptp_root))

    from proofatlas import ProofState

    try:
        with open(args.problem) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    state = ProofState()

    start = time.time()
    try:
        state.add_clauses_from_tptp(content, str(tptp_root), timeout)
    except Exception as e:
        if "timed out" in str(e).lower():
            elapsed = time.time() - start
            print(f"✗ TIMEOUT in {elapsed:.3f}s")
            print(f"  CNF conversion timed out")
            sys.exit(1)
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    num_clauses = state.num_clauses()
    parse_time = time.time() - start

    print(f"Parsed {num_clauses} clauses from '{args.problem}'")

    if args.verbose:
        if args.preset:
            print(f"Preset: {args.preset}")
        print(f"  timeout: {timeout}s")
        print(f"  literal_selection: {literal_selection}")
        print(f"  max_clauses: {max_clauses or 'unlimited'}")
        print(f"  age_weight_ratio: {age_weight_ratio}")
        print()

    print("Running saturation with:")
    print(f"  Max clauses: {max_clauses or 'unlimited'}")
    print(f"  Timeout: {timeout}s")
    print()

    state.set_literal_selection(str(literal_selection))

    remaining_timeout = max(0.1, timeout - parse_time)

    try:
        proof_found, status = state.run_saturation(
            max_iterations if max_iterations > 0 else max_clauses,
            remaining_timeout,
            age_weight_ratio if not embedding_type else None,
            embedding_type,
            str(weights_path) if weights_path else None,
            model_name,
            None,  # max_clause_memory_mb
        )
    except Exception as e:
        print(f"Error during saturation: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start

    if proof_found:
        print(f"✓ THEOREM PROVED in {elapsed:.3f}s")
    elif status == "saturated":
        print(f"✗ SATURATED in {elapsed:.3f}s")
        print(f"  No proof found - the formula may be satisfiable")
        print(f"  Final clauses: {state.num_clauses()}")
    elif status == "resource_limit":
        print(f"✗ RESOURCE LIMIT in {elapsed:.3f}s")
        print(f"  Exceeded clause or iteration limit")
        print(f"  Final clauses: {state.num_clauses()}")
    else:
        print(f"✗ {status.upper()} in {elapsed:.3f}s")
        print(f"  Final clauses: {state.num_clauses()}")

    # Export to JSON if requested
    if args.json_output:
        result_data = {
            "problem_file": str(args.problem),
            "config": {
                "timeout": timeout,
                "max_clauses": max_clauses,
                "literal_selection": literal_selection,
                "preset": args.preset,
            },
            "result": {
                "status": "proof" if proof_found else status,
                "time_seconds": elapsed,
                "final_clauses": state.num_clauses(),
            },
        }
        try:
            with open(args.json_output, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"\nProof attempt exported to: {args.json_output}")
        except Exception as e:
            print(f"Failed to write JSON: {e}", file=sys.stderr)

    sys.exit(0 if proof_found else 1)


if __name__ == "__main__":
    main()
