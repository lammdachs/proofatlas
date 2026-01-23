#!/usr/bin/env python3
"""
Prove a single TPTP problem.

USAGE:
    proofatlas-prove problem.p
    proofatlas-prove problem.p --preset time_sel21
    proofatlas-prove problem.p --preset time_sel21 --verbose
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


def main():
    parser = argparse.ArgumentParser(
        description="Prove a single TPTP problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("problem", type=Path, help="Path to TPTP problem file")
    parser.add_argument("--preset", help="Solver preset from proofatlas.json")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if not args.problem.exists():
        print(f"Error: File not found: {args.problem}", file=sys.stderr)
        sys.exit(1)

    base_dir = find_project_root()
    config_path = base_dir / "configs" / "proofatlas.json"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    presets = config.get("presets", {})

    preset_name = args.preset or config.get("defaults", {}).get("preset", "time_sel21")
    if preset_name not in presets:
        print(f"Error: Unknown preset '{preset_name}'", file=sys.stderr)
        print(f"Available: {', '.join(presets.keys())}", file=sys.stderr)
        sys.exit(1)

    preset = presets[preset_name]
    tptp_root = find_tptp_root(base_dir)

    from proofatlas import ProofState

    try:
        with open(args.problem) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    timeout = preset.get("timeout", 60)
    literal_selection = preset.get("literal_selection", 0)
    max_clauses = preset.get("max_clauses", 0)
    age_weight_ratio = preset.get("age_weight_ratio", 0.5)
    selector = preset.get("selector", "age_weight")
    max_iterations = preset.get("max_iterations", 0)

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
        print(f"Preset: {preset_name}")
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
            age_weight_ratio,
            selector,
            None,
        )
    except Exception as e:
        print(f"Error during saturation: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start

    if proof_found:
        print(f"✓ THEOREM PROVED in {elapsed:.3f}s")
        sys.exit(0)
    elif status == "saturated":
        print(f"✗ SATURATED in {elapsed:.3f}s")
        print(f"  No proof found - the formula may be satisfiable")
        print(f"  Final clauses: {state.num_clauses()}")
        sys.exit(1)
    elif status == "resource_limit":
        print(f"✗ TIMEOUT in {elapsed:.3f}s")
        print(f"  Exceeded time limit")
        print(f"  Final clauses: {state.num_clauses()}")
        sys.exit(1)
    else:
        print(f"✗ {status.upper()} in {elapsed:.3f}s")
        print(f"  Final clauses: {state.num_clauses()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
