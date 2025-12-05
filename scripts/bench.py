#!/usr/bin/env python3
"""
Benchmark script for theorem provers.

Usage:
    python scripts/bench.py --prover vampire --preset time_sel0 --problem-set default
    python scripts/bench.py --prover spass --preset time_sel0 --problem-set unit_equality
    python scripts/bench.py --prover proofatlas --problem-set test --max-problems 100
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import csv


@dataclass
class BenchResult:
    problem: str
    status: str  # "proof", "saturated", "timeout", "error"
    time_s: float
    stdout: str = ""
    stderr: str = ""


def load_config(config_path: Path) -> dict:
    """Load a JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def get_problems(base_dir: Path, tptp_config: dict, problem_set_name: str) -> list[Path]:
    """Get list of problem files matching the problem set filters."""
    problem_sets = tptp_config.get("problem_sets", {})
    if problem_set_name not in problem_sets:
        available = list(problem_sets.keys())
        raise ValueError(f"Unknown problem set: {problem_set_name}. Available: {available}")

    filters = problem_sets[problem_set_name]
    problems_dir = base_dir / tptp_config["paths"]["problems"]

    if not problems_dir.exists():
        raise FileNotFoundError(f"TPTP problems directory not found: {problems_dir}\n"
                               "Run: ./scripts/setup_tptp.sh")

    # Load problem metadata if available
    metadata_path = base_dir / ".data" / "problem_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            data = json.load(f)
            # Handle both flat list and {"problems": [...]} formats
            problems_list = data.get("problems", data) if isinstance(data, dict) else data
            metadata = {p["path"]: p for p in problems_list}

    # Collect matching problems
    problems = []
    for domain_dir in sorted(problems_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name

        # Check domain filters
        if "domains" in filters and filters["domains"]:
            if domain not in filters["domains"]:
                continue
        if "exclude_domains" in filters and filters["exclude_domains"]:
            if domain in filters["exclude_domains"]:
                continue

        for problem_file in sorted(domain_dir.glob("*.p")):
            rel_path = str(problem_file.relative_to(problems_dir))
            meta = metadata.get(rel_path, {})

            # Apply filters
            if "status" in filters and filters["status"]:
                if meta.get("status") not in filters["status"]:
                    continue

            if "format" in filters and filters["format"]:
                if meta.get("format") not in filters["format"]:
                    continue

            if "has_equality" in filters and filters["has_equality"] is not None:
                if meta.get("has_equality") != filters["has_equality"]:
                    continue

            if "is_unit_only" in filters and filters["is_unit_only"] is not None:
                if meta.get("is_unit_only") != filters["is_unit_only"]:
                    continue

            if "max_rating" in filters and filters["max_rating"] is not None:
                if meta.get("rating", 1.0) > filters["max_rating"]:
                    continue

            if "min_rating" in filters and filters["min_rating"] is not None:
                if meta.get("rating", 0.0) < filters["min_rating"]:
                    continue

            if "max_clauses" in filters and filters["max_clauses"] is not None:
                if meta.get("num_clauses", 0) > filters["max_clauses"]:
                    continue

            if "min_clauses" in filters and filters["min_clauses"] is not None:
                if meta.get("num_clauses", 0) < filters["min_clauses"]:
                    continue

            problems.append(problem_file)

    return problems


def run_vampire(problem: Path, base_dir: Path, vampire_config: dict, preset_name: str,
                tptp_root: Path) -> BenchResult:
    """Run Vampire on a problem."""
    binary = base_dir / vampire_config["paths"]["binary"]
    if not binary.exists():
        raise FileNotFoundError(f"Vampire not found: {binary}\nRun: ./scripts/setup_vampire.sh")

    presets = vampire_config.get("presets", {})
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    preset = presets[preset_name]

    # Build command
    cmd = [str(binary)]

    # Add options from preset
    if "time_limit" in preset:
        cmd.extend(["--time_limit", str(preset["time_limit"])])
    if "activation_limit" in preset:
        cmd.extend(["--activation_limit", str(preset["activation_limit"])])
    if "selection" in preset:
        cmd.extend(["--selection", str(preset["selection"])])
    if "avatar" in preset:
        cmd.extend(["--avatar", preset["avatar"]])

    # Add include path for axioms
    cmd.extend(["--include", str(tptp_root)])

    # Add problem file
    cmd.append(str(problem))

    # Run
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=preset.get("time_limit", 60) + 5)
        elapsed = time.time() - start

        # Parse result
        if "Refutation found" in result.stdout:
            status = "proof"
        elif "Satisfiable" in result.stdout:
            status = "saturated"
        elif "Time limit" in result.stdout or "Time out" in result.stdout:
            status = "timeout"
        else:
            status = "unknown"

        return BenchResult(
            problem=problem.name,
            status=status,
            time_s=elapsed,
            stdout=result.stdout,
            stderr=result.stderr
        )
    except subprocess.TimeoutExpired:
        return BenchResult(
            problem=problem.name,
            status="timeout",
            time_s=preset.get("time_limit", 60)
        )
    except Exception as e:
        return BenchResult(
            problem=problem.name,
            status="error",
            time_s=time.time() - start,
            stderr=str(e)
        )


def run_spass(problem: Path, base_dir: Path, spass_config: dict, preset_name: str) -> BenchResult:
    """Run SPASS on a problem."""
    binary = base_dir / spass_config["paths"]["binary"]
    if not binary.exists():
        raise FileNotFoundError(f"SPASS not found: {binary}\nRun: ./scripts/setup_spass.sh")

    presets = spass_config.get("presets", {})
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    preset = presets[preset_name]

    # Build command
    cmd = [str(binary), "-TPTP"]  # -TPTP flag for TPTP input format

    # Add options from preset
    if "TimeLimit" in preset:
        cmd.extend(["-TimeLimit=" + str(preset["TimeLimit"])])
    if "Select" in preset:
        cmd.extend(["-Select=" + str(preset["Select"])])

    cmd.append(str(problem))

    # Run
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=preset.get("TimeLimit", 60) + 5)
        elapsed = time.time() - start

        # Parse result
        if "SPASS beagle" in result.stdout or "Proof found" in result.stdout:
            status = "proof"
        elif "Completion found" in result.stdout:
            status = "saturated"
        elif "Ran out of time" in result.stdout:
            status = "timeout"
        else:
            status = "unknown"

        return BenchResult(
            problem=problem.name,
            status=status,
            time_s=elapsed,
            stdout=result.stdout,
            stderr=result.stderr
        )
    except subprocess.TimeoutExpired:
        return BenchResult(
            problem=problem.name,
            status="timeout",
            time_s=preset.get("TimeLimit", 60)
        )
    except Exception as e:
        return BenchResult(
            problem=problem.name,
            status="error",
            time_s=time.time() - start,
            stderr=str(e)
        )


def run_proofatlas(problem: Path, base_dir: Path, proofatlas_config: dict, preset_name: str, tptp_root: Path) -> BenchResult:
    """Run ProofAtlas on a problem."""
    binary = base_dir / proofatlas_config["paths"]["binary"]
    if not binary.exists():
        # Try to build
        print("Building ProofAtlas...")
        subprocess.run(["cargo", "build", "--release"], cwd=base_dir / "rust", check=True)

    if not binary.exists():
        raise FileNotFoundError(f"ProofAtlas not found: {binary}\n"
                               "Run: cd rust && cargo build --release")

    presets = proofatlas_config.get("presets", {})
    defaults = proofatlas_config.get("defaults", {})

    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    preset = {**defaults, **presets[preset_name]}  # Merge defaults with preset

    # Build command
    cmd = [
        str(binary),
        str(problem),
        "--include", str(tptp_root)
    ]

    # Add options from preset
    if "timeout" in preset:
        cmd.extend(["--timeout", str(preset["timeout"])])
    if "max_clauses" in preset:
        cmd.extend(["--max-clauses", str(preset["max_clauses"])])
    if "literal_selection" in preset:
        cmd.extend(["--literal-selection", preset["literal_selection"]])
    if "age_weight_ratio" in preset and preset.get("selector") == "age_weight":
        cmd.extend(["--age-weight", str(preset["age_weight_ratio"])])

    timeout = preset.get("timeout", 60)

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        elapsed = time.time() - start

        if "THEOREM PROVED" in result.stdout or "Proof found" in result.stdout or "SZS status Theorem" in result.stdout:
            status = "proof"
        elif "Saturated" in result.stdout or "SZS status Satisfiable" in result.stdout:
            status = "saturated"
        elif "Timeout" in result.stdout or "SZS status Timeout" in result.stdout:
            status = "timeout"
        else:
            status = "unknown"

        return BenchResult(
            problem=problem.name,
            status=status,
            time_s=elapsed,
            stdout=result.stdout,
            stderr=result.stderr
        )
    except subprocess.TimeoutExpired:
        return BenchResult(
            problem=problem.name,
            status="timeout",
            time_s=timeout
        )
    except Exception as e:
        return BenchResult(
            problem=problem.name,
            status="error",
            time_s=time.time() - start,
            stderr=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark theorem provers")
    parser.add_argument("--prover", required=True, choices=["vampire", "spass", "proofatlas"],
                       help="Which prover to run")
    parser.add_argument("--preset", default="time_sel0",
                       help="Prover preset from config (e.g., time_sel0, activation_sel20)")
    parser.add_argument("--problem-set", default="default",
                       help="Problem set from tptp.json (e.g., default, test, unit_equality)")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Maximum number of problems to run")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed output")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Load configs
    tptp_config = load_config(base_dir / "configs" / "tptp.json")
    tptp_root = base_dir / tptp_config["paths"]["root"]

    # Get problems
    print(f"Loading problem set: {args.problem_set}")
    problems = get_problems(base_dir, tptp_config, args.problem_set)

    if args.max_problems:
        problems = problems[:args.max_problems]

    print(f"Running {args.prover} ({args.preset}) on {len(problems)} problems")
    print()

    # Run benchmark
    results = []
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "unknown": 0}
    total_time = 0

    for i, problem in enumerate(problems, 1):
        if args.prover == "vampire":
            vampire_config = load_config(base_dir / "configs" / "vampire.json")
            result = run_vampire(problem, base_dir, vampire_config, args.preset, tptp_root)
        elif args.prover == "spass":
            spass_config = load_config(base_dir / "configs" / "spass.json")
            result = run_spass(problem, base_dir, spass_config, args.preset)
        else:
            proofatlas_config = load_config(base_dir / "configs" / "proofatlas.json")
            result = run_proofatlas(problem, base_dir, proofatlas_config, args.preset, tptp_root)

        results.append(result)
        stats[result.status] = stats.get(result.status, 0) + 1
        total_time += result.time_s

        # Progress
        status_symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!", "unknown": "?"}
        symbol = status_symbol.get(result.status, "?")

        if args.verbose:
            print(f"[{i}/{len(problems)}] {symbol} {result.problem}: {result.status} ({result.time_s:.2f}s)")
        else:
            print(symbol, end="", flush=True)
            if i % 50 == 0:
                print(f" [{i}/{len(problems)}]")

    if not args.verbose:
        print()

    # Summary
    print()
    print("=" * 50)
    print(f"Results: {len(problems)} problems, {total_time:.1f}s total")
    print(f"  Proofs:    {stats['proof']:4d} ({100*stats['proof']/len(problems):.1f}%)")
    print(f"  Saturated: {stats['saturated']:4d} ({100*stats['saturated']/len(problems):.1f}%)")
    print(f"  Timeout:   {stats['timeout']:4d} ({100*stats['timeout']/len(problems):.1f}%)")
    print(f"  Error:     {stats['error']:4d} ({100*stats['error']/len(problems):.1f}%)")
    print(f"  Unknown:   {stats['unknown']:4d} ({100*stats['unknown']/len(problems):.1f}%)")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["problem", "status", "time_s"])
            for r in results:
                writer.writerow([r.problem, r.status, f"{r.time_s:.3f}"])
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
