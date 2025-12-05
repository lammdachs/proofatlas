#!/usr/bin/env python3
"""
Evaluate all provers with all presets on a problem set.

Usage:
    python scripts/eval.py --problem-set test --max-problems 50
    python scripts/eval.py --problem-set default --output-dir .logs/eval_2024
    python scripts/eval.py --provers proofatlas vampire --presets time_default time_sel20
"""

import argparse
import json
import subprocess
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EvalResult:
    prover: str
    preset: str
    problem: str
    status: str
    time_s: float


def load_config(config_path: Path) -> dict:
    """Load a JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def run_bench(prover: str, preset: str, problem_set: str, max_problems: Optional[int],
              base_dir: Path, verbose: bool = False) -> tuple[list[EvalResult], dict]:
    """Run bench.py for a single prover/preset combination and collect results."""
    cmd = [
        sys.executable, str(base_dir / "scripts" / "bench.py"),
        "--prover", prover,
        "--preset", preset,
        "--problem-set", problem_set,
    ]
    if max_problems:
        cmd.extend(["--max-problems", str(max_problems)])
    if verbose:
        cmd.append("--verbose")

    # Create temp output file
    output_file = base_dir / ".logs" / f"temp_{prover}_{preset}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output", str(output_file)])

    print(f"\n{'='*60}")
    print(f"Running: {prover} / {preset}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=3600)
        elapsed = time.time() - start

        if not verbose and result.returncode != 0:
            print(f"Error: {result.stderr}")
            return [], {}

    except subprocess.TimeoutExpired:
        print("Global timeout expired (1 hour)")
        return [], {}
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return [], {}

    # Parse results from CSV
    results = []
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "unknown": 0}

    if output_file.exists():
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(EvalResult(
                    prover=prover,
                    preset=preset,
                    problem=row["problem"],
                    status=row["status"],
                    time_s=float(row["time_s"])
                ))
                stats[row["status"]] = stats.get(row["status"], 0) + 1

        # Clean up temp file
        output_file.unlink()

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate all provers on a problem set")
    parser.add_argument("--problem-set", default="test",
                       help="Problem set from tptp.json (default: test)")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Maximum number of problems per configuration")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for logs (default: .logs/eval_TIMESTAMP)")
    parser.add_argument("--provers", nargs="+", default=None,
                       help="Specific provers to run (default: all)")
    parser.add_argument("--presets", nargs="+", default=None,
                       help="Specific presets to run (default: all matching)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output from each run")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / ".logs" / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    proofatlas_config = load_config(base_dir / "configs" / "proofatlas.json")
    vampire_config = load_config(base_dir / "configs" / "vampire.json")
    spass_config = load_config(base_dir / "configs" / "spass.json")

    # Determine which provers and presets to run
    prover_configs = {
        "proofatlas": proofatlas_config,
        "vampire": vampire_config,
        "spass": spass_config,
    }

    if args.provers:
        prover_configs = {p: prover_configs[p] for p in args.provers if p in prover_configs}

    # Collect all runs
    runs = []
    for prover, config in prover_configs.items():
        presets = list(config.get("presets", {}).keys())
        if args.presets:
            presets = [p for p in presets if p in args.presets]
        for preset in presets:
            runs.append((prover, preset))

    print(f"Evaluation Configuration")
    print(f"========================")
    print(f"Problem set: {args.problem_set}")
    print(f"Max problems: {args.max_problems or 'all'}")
    print(f"Output dir: {output_dir}")
    print(f"Configurations to run: {len(runs)}")
    for prover, preset in runs:
        print(f"  - {prover} / {preset}")
    print()

    # Run all configurations
    all_results = []
    summary = []

    for i, (prover, preset) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] Running {prover} / {preset}...")

        results, stats = run_bench(
            prover, preset, args.problem_set, args.max_problems,
            base_dir, args.verbose
        )

        all_results.extend(results)

        total = sum(stats.values())
        if total > 0:
            summary.append({
                "prover": prover,
                "preset": preset,
                "total": total,
                "proofs": stats["proof"],
                "proof_rate": f"{100*stats['proof']/total:.1f}%",
                "saturated": stats["saturated"],
                "timeout": stats["timeout"],
                "error": stats["error"],
                "unknown": stats["unknown"],
            })

    # Save detailed results
    results_file = output_dir / "results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prover", "preset", "problem", "status", "time_s"])
        for r in all_results:
            writer.writerow([r.prover, r.preset, r.problem, r.status, f"{r.time_s:.3f}"])

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Prover':<12} {'Preset':<20} {'Total':>6} {'Proofs':>8} {'Rate':>8} {'Sat':>6} {'T/O':>6}")
    print("-" * 80)

    for s in sorted(summary, key=lambda x: (x["prover"], x["preset"])):
        print(f"{s['prover']:<12} {s['preset']:<20} {s['total']:>6} {s['proofs']:>8} {s['proof_rate']:>8} "
              f"{s['saturated']:>6} {s['timeout']:>6}")

    print("-" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Detailed: {results_file}")
    print(f"  - Summary:  {summary_file}")

    # Create a comparison matrix by problem
    if all_results:
        matrix_file = output_dir / "comparison_matrix.csv"

        # Group results by problem
        problems = sorted(set(r.problem for r in all_results))
        configs = sorted(set((r.prover, r.preset) for r in all_results))

        results_dict = {}
        for r in all_results:
            key = (r.problem, r.prover, r.preset)
            results_dict[key] = r.status

        with open(matrix_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            header = ["problem"] + [f"{p}:{pr}" for p, pr in configs]
            writer.writerow(header)
            # Data
            for problem in problems:
                row = [problem]
                for prover, preset in configs:
                    status = results_dict.get((problem, prover, preset), "-")
                    # Abbreviate status
                    abbrev = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!", "unknown": "?"}
                    row.append(abbrev.get(status, status))
                writer.writerow(row)

        print(f"  - Matrix:   {matrix_file}")


if __name__ == "__main__":
    main()
