#!/usr/bin/env python3
"""
Prove a single TPTP problem.

USAGE:
    proofatlas problem.p
    proofatlas problem.p --config gcn_mlp
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


def list_configs(base_dir: Path):
    """List available configs."""
    config_path = base_dir / "configs" / "proofatlas.json"
    if not config_path.exists():
        print("Error: configs/proofatlas.json not found", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    presets = config.get("presets", {})

    print("Available configs:")
    for name, preset in sorted(presets.items()):
        desc = preset.get("description", "")
        encoder = preset.get("encoder")
        scorer = preset.get("scorer")

        model_info = ""
        if encoder and scorer:
            model_info = f" [{encoder}+{scorer}]"

        print(f"  {name:<25} {desc}{model_info}")


def main():
    parser = argparse.ArgumentParser(
        description="Prove a single TPTP problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("problem", type=Path, nargs="?", help="Path to TPTP problem file")
    parser.add_argument("--config", help="Use config from configs/proofatlas.json")
    parser.add_argument("--list", action="store_true", help="List available configs")
    parser.add_argument("--timeout", type=int, help="Set timeout in seconds (default: 60)")
    parser.add_argument(
        "--literal-selection",
        type=int,
        choices=[0, 20, 21, 22],
        help="Literal selection: 0=all, 20=maximal, 21=unique/neg/max, 22=neg/max",
    )
    parser.add_argument("--memory-limit", type=int, dest="memory_limit_mb", help="Memory limit for clause storage in MB")
    parser.add_argument("--include", action="append", dest="include_dirs", help="Add include directory")
    parser.add_argument("--json", dest="json_output", help="Export proof attempt to JSON file")
    parser.add_argument("--profile", action="store_true", help="Enable profiling (included in --json output)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    base_dir = find_project_root()

    # Handle --list
    if args.list:
        list_configs(base_dir)
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
    if args.config:
        if args.config not in presets:
            print(f"Error: Unknown config '{args.config}'", file=sys.stderr)
            print(f"Use --list to see available configs", file=sys.stderr)
            sys.exit(1)
        preset = presets[args.config]

    # Get values from preset, then override with command line args
    timeout = args.timeout if args.timeout is not None else preset.get("timeout", 60)
    literal_selection = (
        args.literal_selection if args.literal_selection is not None else preset.get("literal_selection", 21)
    )
    age_weight_ratio = preset.get("age_weight_ratio", 0.5)
    max_iterations = preset.get("max_iterations", 0)
    memory_limit_mb = args.memory_limit_mb if args.memory_limit_mb is not None else preset.get("memory_limit_mb")

    # Check for ML selector in preset
    encoder = preset.get("encoder")
    scorer = preset.get("scorer")
    weights_path = None

    if encoder and scorer:
        from proofatlas.ml import find_weights

        model_name = f"{encoder}_{scorer}"
        weights_dir = base_dir / ".weights"
        weights_path = find_weights(weights_dir, preset)

        if not weights_path:
            print(f"Error: Model weights not found for {model_name}", file=sys.stderr)
            print(f"Train with: proofatlas-bench --config {args.config} --retrain", file=sys.stderr)
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
        state.add_clauses_from_tptp(content, str(tptp_root), timeout, memory_limit_mb=memory_limit_mb)
    except Exception as e:
        if "memory limit" in str(e).lower():
            elapsed = time.time() - start
            print(f"✗ RESOURCE LIMIT in {elapsed:.3f}s")
            print(f"  CNF conversion exceeded memory limit")
            sys.exit(1)
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
        if args.config:
            print(f"Preset: {args.config}")
        print(f"  timeout: {timeout}s")
        print(f"  literal_selection: {literal_selection}")
        print(f"  age_weight_ratio: {age_weight_ratio}")
        print()

    print("Running saturation with:")
    print(f"  Timeout: {timeout}s")
    print()

    remaining_timeout = max(0.1, timeout - parse_time)

    try:
        proof_found, status, profile_json, _ = state.run_saturation(
            timeout=remaining_timeout,
            max_iterations=max_iterations if max_iterations > 0 else None,
            literal_selection=literal_selection,
            age_weight_ratio=age_weight_ratio if not encoder else None,
            encoder=encoder,
            scorer=scorer,
            weights_path=str(weights_path) if weights_path else None,
            memory_limit_mb=memory_limit_mb,
            enable_profiling=args.profile,
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
                "literal_selection": literal_selection,
                "preset": args.config,
            },
            "result": {
                "status": "proof" if proof_found else status,
                "time_seconds": elapsed,
                "final_clauses": state.num_clauses(),
            },
        }
        if profile_json is not None:
            result_data["profile"] = json.loads(profile_json)
        try:
            with open(args.json_output, "w") as f:
                json.dump(result_data, f, indent=2)
            print(f"\nProof attempt exported to: {args.json_output}")
        except Exception as e:
            print(f"Failed to write JSON: {e}", file=sys.stderr)

    sys.exit(0 if proof_found else 1)


if __name__ == "__main__":
    main()
