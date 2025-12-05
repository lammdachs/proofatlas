#!/usr/bin/env python3
"""
Benchmark script for theorem provers.

USAGE:
    Runs prover/preset combinations in the background.
    By default runs ALL provers (proofatlas, vampire, spass) with ALL their presets.

    proofatlas-bench                                  # Start job, return immediately
    proofatlas-bench --track                          # Start job and track progress
    proofatlas-bench --prover proofatlas              # One prover, all its presets
    proofatlas-bench --preset time_sel21              # All provers, one preset each
    proofatlas-bench --prover proofatlas vampire --preset activation_sel21

    Filtering:
        --prover    Restrict to specific prover(s): proofatlas, vampire, spass
        --preset    Restrict to specific preset(s): time_sel21, activation_sel20, etc.

    Output: .logs/eval_TIMESTAMP/ containing results.csv, summary.json, comparison_matrix.csv

JOB MANAGEMENT:
    Jobs run in the background and survive disconnection.

    proofatlas-bench --track     # Attach to running job (or start new with tracking)
    proofatlas-bench --status    # One-shot status check
    proofatlas-bench --kill      # Stop running job

COMMON OPTIONS:
    --problem-set NAME    Problem set from configs/tptp.json (default: test)
    --max-problems N      Limit number of problems to run
"""

import argparse
import json
import subprocess
import time
import csv
import os
import signal
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def find_project_root() -> Path:
    """Find the proofatlas project root by looking for marker files."""
    # Try current directory first
    candidates = [
        Path.cwd(),
        Path(__file__).parent.parent,  # scripts/ -> project root
    ]

    for candidate in candidates:
        if (candidate / "configs" / "tptp.json").exists():
            return candidate.resolve()

    # Walk up from current directory
    path = Path.cwd()
    while path != path.parent:
        if (path / "configs" / "tptp.json").exists():
            return path.resolve()
        path = path.parent

    raise FileNotFoundError(
        "Could not find proofatlas project root. "
        "Run from the project directory or set PROOFATLAS_ROOT."
    )


# Job management
JOB_FILE = ".logs/bench_job.json"


def get_job_file(base_dir: Path) -> Path:
    return base_dir / JOB_FILE


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_job_status(base_dir: Path) -> Optional[dict]:
    """Get the current job status, or None if no job is running."""
    job_file = get_job_file(base_dir)
    if not job_file.exists():
        return None

    try:
        with open(job_file) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # Check if process is still running
    if not is_process_running(job["pid"]):
        return None

    # Read progress from log file
    log_file = Path(job.get("log_file", ""))
    if log_file.exists():
        try:
            with open(log_file) as f:
                lines = f.readlines()
            # Find last progress line
            for line in reversed(lines):
                if line.startswith("PROGRESS:"):
                    parts = line.strip().split(":")
                    if len(parts) >= 5:
                        job["current_config"] = int(parts[1])
                        job["total_configs"] = int(parts[2])
                        job["current_problem"] = int(parts[3])
                        job["total_problems"] = int(parts[4])
                    if len(parts) >= 8:
                        job["proofs"] = int(parts[5])
                        job["saturated"] = int(parts[6])
                        job["timeout"] = int(parts[7])
                    break
        except IOError:
            pass

    return job


def save_job_status(base_dir: Path, pid: int, args: list, output_dir: Path, log_file: Path):
    """Save the current job status."""
    job_file = get_job_file(base_dir)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    job = {
        "pid": pid,
        "args": args,
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
    }

    with open(job_file, "w") as f:
        json.dump(job, f, indent=2)


def clear_job_status(base_dir: Path):
    """Remove the job status file."""
    job_file = get_job_file(base_dir)
    if job_file.exists():
        job_file.unlink()


def kill_job(base_dir: Path) -> bool:
    """Kill the running job. Returns True if a job was killed."""
    job = get_job_status(base_dir)
    if not job:
        return False

    try:
        os.kill(job["pid"], signal.SIGTERM)
        clear_job_status(base_dir)
        return True
    except (OSError, ProcessLookupError):
        clear_job_status(base_dir)
        return False


def format_job_status(job: dict) -> str:
    """Format job status as a single line."""
    if not job:
        return "No job running"

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    parts = [f"[{hours}h{minutes:02d}m]"]

    if "current_config" in job:
        parts.append(f"cfg {job['current_config']}/{job['total_configs']}")
        parts.append(f"prob {job['current_problem']}/{job['total_problems']}")

    if "proofs" in job:
        total = job["proofs"] + job["saturated"] + job["timeout"]
        if total > 0:
            rate = 100 * job["proofs"] / total
            parts.append(f"+{job['proofs']} ~{job['saturated']} T{job['timeout']} ({rate:.1f}%)")

    return " | ".join(parts)


def print_job_status(base_dir: Path):
    """Print the current job status."""
    job = get_job_status(base_dir)
    if not job:
        print("No job currently running.")
        return

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    print(f"Job running (PID: {job['pid']})")
    print(f"  Started:  {start.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m ago)")
    print(f"  Output:   {job['output_dir']}")

    if "current_config" in job:
        print(f"  Progress: config {job['current_config']}/{job['total_configs']}, "
              f"problem {job['current_problem']}/{job['total_problems']}")

    if "proofs" in job:
        total = job["proofs"] + job["saturated"] + job["timeout"]
        if total > 0:
            rate = 100 * job["proofs"] / total
            print(f"  Results:  +{job['proofs']} ~{job['saturated']} T{job['timeout']} ({rate:.1f}% proofs)")

    print(f"\nTo stop: proofatlas-bench --kill")


def track_job(base_dir: Path, poll_interval: float = 1.0):
    """Continuously display job status until it completes."""
    import sys

    last_status = ""
    while True:
        job = get_job_status(base_dir)
        if not job:
            # Clear line and print final message
            sys.stdout.write("\r" + " " * len(last_status) + "\r")
            sys.stdout.flush()
            print("Job completed.")
            break

        status = format_job_status(job)
        # Clear previous line and print new status
        sys.stdout.write("\r" + " " * len(last_status) + "\r")
        sys.stdout.write(status)
        sys.stdout.flush()
        last_status = status

        time.sleep(poll_interval)


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
               quiet: bool = False, use_progress: bool = False,
               progress_desc: str = None, log_file = None,
               config_idx: int = 0, total_configs: int = 0) -> tuple[list[BenchResult], dict]:
    """Run a single prover/preset on a list of problems."""
    results = []
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "unknown": 0}

    desc = progress_desc or f"{prover}/{preset}"
    problems_iter = tqdm(problems, desc=desc, unit="prob", leave=False) if use_progress else problems

    for i, problem in enumerate(problems_iter if use_progress else problems, 1):
        if prover == "vampire":
            result = run_vampire(problem, base_dir, configs["vampire"], preset, tptp_root)
        elif prover == "spass":
            result = run_spass(problem, base_dir, configs["spass"], preset)
        else:
            result = run_proofatlas(problem, base_dir, configs["proofatlas"], preset, tptp_root)

        results.append(result)
        stats[result.status] = stats.get(result.status, 0) + 1

        if log_file:
            # Write progress for background job monitoring
            log_file.write(f"PROGRESS:{config_idx}:{total_configs}:{i}:{len(problems)}:"
                          f"{stats['proof']}:{stats['saturated']}:{stats['timeout']}\n")
            log_file.flush()

        if use_progress:
            problems_iter.set_postfix_str(f"+{stats['proof']} ~{stats['saturated']} T{stats['timeout']}")
        elif not quiet:
            symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!", "unknown": "?"}.get(result.status, "?")
            if verbose:
                print(f"[{i}/{len(problems)}] {symbol} {result.problem}: {result.status} ({result.time_s:.2f}s)")
            else:
                print(symbol, end="", flush=True)
                if i % 50 == 0:
                    print(f" [{i}/{len(problems)}]")

    if not quiet and not verbose and not use_progress:
        print()

    return results, stats


def run_eval(provers: list[str], presets: list[str], problems: list[Path], base_dir: Path,
             tptp_root: Path, configs: dict, output_dir: Path, verbose: bool = False,
             log_file = None):
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

    for idx, (prover, preset) in enumerate(runs, 1):
        desc = f"[{idx}/{len(runs)}] {prover}/{preset}"
        results, stats = run_single(prover, preset, problems, base_dir, tptp_root, configs,
                                    verbose=False, quiet=True, use_progress=use_progress, progress_desc=desc,
                                    log_file=log_file, config_idx=idx, total_configs=len(runs))

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
    parser.add_argument("--prover", nargs="+", choices=["vampire", "spass", "proofatlas"],
                       help="Prover(s) to run (default: all)")
    parser.add_argument("--preset", nargs="+",
                       help="Preset(s) to run (default: all)")
    parser.add_argument("--problem-set", default="test",
                       help="Problem set from tptp.json")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Job management
    parser.add_argument("--track", action="store_true",
                       help="Track progress (attach to running job or start with tracking)")
    parser.add_argument("--status", action="store_true",
                       help="Check status of running job (one-shot)")
    parser.add_argument("--kill", action="store_true",
                       help="Stop running job")

    args = parser.parse_args()

    base_dir = find_project_root()

    # Handle job management commands first
    if args.status:
        print_job_status(base_dir)
        return

    if args.track:
        job = get_job_status(base_dir)
        if job:
            track_job(base_dir)
            return
        # No job running - fall through to start a new one with tracking

    if args.kill:
        if kill_job(base_dir):
            print("Job killed.")
        else:
            print("No job currently running.")
        return

    # Check if a job is already running (for any job-starting command)
    existing = get_job_status(base_dir)
    if existing:
        print(f"Error: A job is already running (PID: {existing['pid']})")
        print(f"Use --status to check progress, --track to monitor, or --kill to stop it.")
        sys.exit(1)

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

    # Determine provers and presets
    provers = args.prover or ["proofatlas", "vampire", "spass"]
    presets = args.preset  # None means all presets

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / ".logs" / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fork to background
    log_file_path = output_dir / "bench.log"

    pid = os.fork()
    if pid > 0:
        # Parent process
        save_job_status(base_dir, pid, sys.argv, output_dir, log_file_path)
        print(f"Started background job (PID: {pid})")
        print(f"Output: {output_dir}")
        print(f"Use --track to monitor, --status to check, --kill to stop.")
        if args.track:
            time.sleep(0.5)  # Give child time to start
            print()
            track_job(base_dir)
    else:
        # Child process - detach and run
        os.setsid()
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

        try:
            run_eval(provers, presets, problems, base_dir, tptp_root, configs,
                     output_dir, verbose=True, log_file=sys.stdout)
        finally:
            clear_job_status(base_dir)
            sys.stdout.close()
        os._exit(0)


if __name__ == "__main__":
    main()
