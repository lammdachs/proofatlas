#!/usr/bin/env python3
"""
Benchmark script for theorem provers.

Usage:
    # Single prover/preset run
    python scripts/bench.py --prover vampire --preset time_sel21 --problem-set test
    python scripts/bench.py --prover proofatlas --problem-set default --max-problems 100

    # Evaluation mode (multiple provers/presets)
    python scripts/bench.py --eval --problem-set test
    python scripts/bench.py --eval --provers proofatlas vampire --presets time_sel21 time_sel22
"""

import argparse
import json
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class BenchResult:
    problem: str
    status: str  # "proof", "saturated", "timeout", "error", "unknown"
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

    cmd = [str(binary)]
    if "time_limit" in preset:
        cmd.extend(["--time_limit", str(preset["time_limit"])])
    if "activation_limit" in preset:
        cmd.extend(["--activation_limit", str(preset["activation_limit"])])
    if "selection" in preset:
        cmd.extend(["--selection", str(preset["selection"])])
    if "avatar" in preset:
        cmd.extend(["--avatar", preset["avatar"]])
    cmd.extend(["--include", str(tptp_root)])
    cmd.append(str(problem))

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=preset.get("time_limit", 60) + 5)
        elapsed = time.time() - start

        if "Refutation found" in result.stdout:
            status = "proof"
        elif "Satisfiable" in result.stdout:
            status = "saturated"
        elif "Time limit" in result.stdout or "Time out" in result.stdout:
            status = "timeout"
        else:
            status = "unknown"

        return BenchResult(problem=problem.name, status=status, time_s=elapsed,
                          stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return BenchResult(problem=problem.name, status="timeout",
                          time_s=preset.get("time_limit", 60))
    except Exception as e:
        return BenchResult(problem=problem.name, status="error",
                          time_s=time.time() - start, stderr=str(e))


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

    cmd = [str(binary), "-TPTP"]
    if "TimeLimit" in preset:
        cmd.append("-TimeLimit=" + str(preset["TimeLimit"]))
    if "Loops" in preset:
        cmd.append("-Loops=" + str(preset["Loops"]))
    if "Select" in preset:
        cmd.append("-Select=" + str(preset["Select"]))
    cmd.append(str(problem))

    timeout = preset.get("TimeLimit", 300) + 5

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start

        if "SPASS beagle" in result.stdout or "Proof found" in result.stdout:
            status = "proof"
        elif "Completion found" in result.stdout:
            status = "saturated"
        elif "Ran out of time" in result.stdout:
            status = "timeout"
        else:
            status = "unknown"

        return BenchResult(problem=problem.name, status=status, time_s=elapsed,
                          stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return BenchResult(problem=problem.name, status="timeout", time_s=timeout - 5)
    except Exception as e:
        return BenchResult(problem=problem.name, status="error",
                          time_s=time.time() - start, stderr=str(e))


def run_proofatlas(problem: Path, base_dir: Path, proofatlas_config: dict,
                   preset_name: str, tptp_root: Path) -> BenchResult:
    """Run ProofAtlas on a problem."""
    binary = base_dir / proofatlas_config["paths"]["binary"]
    if not binary.exists():
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

    preset = {**defaults, **presets[preset_name]}

    cmd = [str(binary), str(problem), "--include", str(tptp_root)]
    if "timeout" in preset:
        cmd.extend(["--timeout", str(preset["timeout"])])
    if "max_clauses" in preset:
        cmd.extend(["--max-clauses", str(preset["max_clauses"])])
    if "literal_selection" in preset:
        cmd.extend(["--literal-selection", str(preset["literal_selection"])])
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

        return BenchResult(problem=problem.name, status=status, time_s=elapsed,
                          stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception as e:
        return BenchResult(problem=problem.name, status="error",
                          time_s=time.time() - start, stderr=str(e))


def run_single(prover: str, preset: str, problems: list[Path], base_dir: Path,
               tptp_root: Path, configs: dict, verbose: bool = False,
               quiet: bool = False) -> tuple[list[BenchResult], dict]:
    """Run a single prover/preset on a list of problems."""
    results = []
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "unknown": 0}

    for i, problem in enumerate(problems, 1):
        if prover == "vampire":
            result = run_vampire(problem, base_dir, configs["vampire"], preset, tptp_root)
        elif prover == "spass":
            result = run_spass(problem, base_dir, configs["spass"], preset)
        else:
            result = run_proofatlas(problem, base_dir, configs["proofatlas"], preset, tptp_root)

        results.append(result)
        stats[result.status] = stats.get(result.status, 0) + 1

        if not quiet:
            symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!", "unknown": "?"}.get(result.status, "?")
            if verbose:
                print(f"[{i}/{len(problems)}] {symbol} {result.problem}: {result.status} ({result.time_s:.2f}s)")
            else:
                print(symbol, end="", flush=True)
                if i % 50 == 0:
                    print(f" [{i}/{len(problems)}]")

    if not quiet and not verbose:
        print()

    return results, stats


def run_eval(provers: list[str], presets: list[str], problems: list[Path], base_dir: Path,
             tptp_root: Path, configs: dict, output_dir: Path, verbose: bool = False):
    """Run evaluation across multiple provers and presets."""
    # Collect all runs
    runs = []
    for prover in provers:
        prover_presets = list(configs[prover].get("presets", {}).keys())
        if presets:
            prover_presets = [p for p in prover_presets if p in presets]
        for preset in prover_presets:
            runs.append((prover, preset))

    print(f"Evaluation Configuration")
    print(f"========================")
    print(f"Problems: {len(problems)}")
    print(f"Output dir: {output_dir}")
    print(f"Configurations: {len(runs)}")
    for prover, preset in runs:
        print(f"  - {prover} / {preset}")
    print()

    all_results = []
    summary = []

    use_progress = not verbose
    runs_iter = tqdm(runs, desc="Configurations", unit="cfg") if use_progress else runs

    for prover, preset in runs_iter:
        if use_progress:
            runs_iter.set_postfix_str(f"{prover}/{preset}")

        results, stats = run_single(prover, preset, problems, base_dir, tptp_root, configs, verbose=False, quiet=True)

        for r in results:
            all_results.append({"prover": prover, "preset": preset, **r.__dict__})

        total = sum(stats.values())
        if total > 0:
            proof_rate = 100 * stats["proof"] / total
            summary.append({
                "prover": prover, "preset": preset, "total": total,
                "proofs": stats["proof"], "proof_rate": f"{proof_rate:.1f}%",
                "saturated": stats["saturated"], "timeout": stats["timeout"],
                "error": stats["error"], "unknown": stats["unknown"],
            })
            if use_progress:
                runs_iter.set_postfix_str(f"{prover}/{preset} [{stats['proof']}/{total}]")

    # Save detailed results
    results_file = output_dir / "results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prover", "preset", "problem", "status", "time_s"])
        for r in all_results:
            writer.writerow([r["prover"], r["preset"], r["problem"], r["status"], f"{r['time_s']:.3f}"])

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
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

    # Create comparison matrix
    if all_results:
        matrix_file = output_dir / "comparison_matrix.csv"
        problems_set = sorted(set(r["problem"] for r in all_results))
        configs_set = sorted(set((r["prover"], r["preset"]) for r in all_results))

        results_dict = {(r["problem"], r["prover"], r["preset"]): r["status"] for r in all_results}

        with open(matrix_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["problem"] + [f"{p}:{pr}" for p, pr in configs_set])
            for problem in problems_set:
                row = [problem]
                for prover, preset in configs_set:
                    status = results_dict.get((problem, prover, preset), "-")
                    abbrev = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!", "unknown": "?"}
                    row.append(abbrev.get(status, status))
                writer.writerow(row)

        print(f"  - Matrix:   {matrix_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark theorem provers")
    parser.add_argument("--eval", action="store_true",
                       help="Evaluation mode: run multiple provers/presets")
    parser.add_argument("--prover", choices=["vampire", "spass", "proofatlas"],
                       help="Prover to run (single mode)")
    parser.add_argument("--provers", nargs="+", choices=["vampire", "spass", "proofatlas"],
                       help="Provers to run (eval mode, default: all)")
    parser.add_argument("--preset", help="Preset to use (single mode)")
    parser.add_argument("--presets", nargs="+", help="Presets to run (eval mode)")
    parser.add_argument("--problem-set", default="test",
                       help="Problem set from tptp.json")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems")
    parser.add_argument("--output", type=str, help="Output CSV file (single mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory (eval mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Load configs
    tptp_config = load_config(base_dir / "configs" / "tptp.json")
    tptp_root = base_dir / tptp_config["paths"]["root"]

    configs = {
        "proofatlas": load_config(base_dir / "configs" / "proofatlas.json"),
        "vampire": load_config(base_dir / "configs" / "vampire.json"),
        "spass": load_config(base_dir / "configs" / "spass.json"),
    }

    # Get problems
    problems = get_problems(base_dir, tptp_config, args.problem_set)
    if args.max_problems:
        problems = problems[:args.max_problems]

    if args.eval or args.provers:
        # Evaluation mode
        provers = args.provers or ["proofatlas", "vampire", "spass"]
        presets = args.presets

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_dir / ".logs" / f"eval_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_eval(provers, presets, problems, base_dir, tptp_root, configs, output_dir, args.verbose)

    elif args.prover:
        # Single mode
        prover = args.prover
        preset = args.preset or configs[prover].get("default_preset", "time_sel21")

        print(f"Running {prover} ({preset}) on {len(problems)} problems\n")

        results, stats = run_single(prover, preset, problems, base_dir, tptp_root, configs, args.verbose)

        total_time = sum(r.time_s for r in results)
        print()
        print("=" * 50)
        print(f"Results: {len(problems)} problems, {total_time:.1f}s total")
        print(f"  Proofs:    {stats['proof']:4d} ({100*stats['proof']/len(problems):.1f}%)")
        print(f"  Saturated: {stats['saturated']:4d} ({100*stats['saturated']/len(problems):.1f}%)")
        print(f"  Timeout:   {stats['timeout']:4d} ({100*stats['timeout']/len(problems):.1f}%)")
        print(f"  Error:     {stats['error']:4d} ({100*stats['error']/len(problems):.1f}%)")
        print(f"  Unknown:   {stats['unknown']:4d} ({100*stats['unknown']/len(problems):.1f}%)")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["problem", "status", "time_s"])
                for r in results:
                    writer.writerow([r.problem, r.status, f"{r.time_s:.3f}"])
            print(f"\nResults saved to: {output_path}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/bench.py --prover vampire --preset time_sel21 --problem-set test")
        print("  python scripts/bench.py --eval --problem-set test")


if __name__ == "__main__":
    main()
