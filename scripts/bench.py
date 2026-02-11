#!/usr/bin/env python3
"""
Benchmark proofatlas theorem prover.

USAGE:
    proofatlas-bench                           # Run all configs
    proofatlas-bench --config gcn_mlp    # Run specific config
    proofatlas-bench --list                    # List available configs

    proofatlas-bench --status                  # Check job status
    proofatlas-bench --kill                    # Stop running job

CACHING:
    Results cached in .data/runs/proofatlas/<preset>/<problem>.json
    Use --rerun to force re-evaluation of cached results.

ML MODELS:
    Train models separately with scripts/train.py, then benchmark:
        python scripts/train.py --config gcn_mlp
        proofatlas-bench --config gcn_mlp

    For ML presets, a scoring server is auto-launched during evaluation.

OUTPUT:
    .data/runs/proofatlas/<preset>/     - Per-problem results
"""

import os

# Limit torch/MKL threads before any imports that load libtorch.
# Without this, parallel workers (--cpu-workers>1) cause massive thread contention
# as each subprocess inherits multithreaded libtorch from the parent fork.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Prevent CUDA initialization in the daemon and its forked worker subprocesses.
# Without this, importing proofatlas loads libtorch_cuda.so which initializes a
# CUDA context. Each multiprocessing.Process(fork) child inherits this context,
# and after ~340 forks the accumulated GPU memory corruption crashes the scoring
# server. Only the scoring server subprocess (started via Popen with its own env
# setting CUDA_VISIBLE_DEVICES=<gpu_id>) should touch the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from bench_jobs import (
    clear_job_status, clear_pids, daemonize, get_job_file, get_job_status,
    get_log_file, kill_job, log, print_job_status, save_job_status,
)
from bench_provers import BenchResult, run_single_problem

# Lazy imports for ML functionality (avoid loading PyTorch for --list)
_ml_module = None


def _get_ml():
    """Lazily import proofatlas.ml to avoid slow startup for simple commands."""
    global _ml_module
    if _ml_module is None:
        from proofatlas import ml as _ml
        _ml_module = _ml
    return _ml_module


def find_project_root() -> Path:
    """Find the proofatlas project root."""
    candidates = [Path.cwd(), Path(__file__).parent.parent]
    for candidate in candidates:
        if (candidate / "configs" / "proofatlas.json").exists():
            return candidate.resolve()

    path = Path.cwd()
    while path != path.parent:
        if (path / "configs" / "proofatlas.json").exists():
            return path.resolve()
        path = path.parent

    raise FileNotFoundError("Could not find proofatlas project root.")


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def get_all_prover_configs(base_dir: Path) -> dict:
    """Get configs from all provers, with installation status.

    Returns dict mapping prover name to:
        - config: the loaded config dict
        - installed: whether the prover binary exists
        - binary: path to binary (or None for proofatlas)
    """
    provers = {}

    # proofatlas (always available via Python bindings)
    proofatlas_config_path = base_dir / "configs" / "proofatlas.json"
    if proofatlas_config_path.exists():
        provers["proofatlas"] = {
            "config": load_config(proofatlas_config_path),
            "installed": True,
            "binary": None,
        }

    # vampire
    vampire_config_path = base_dir / "configs" / "vampire.json"
    if vampire_config_path.exists():
        vampire_config = load_config(vampire_config_path)
        vampire_binary = base_dir / vampire_config["paths"]["binary"]
        provers["vampire"] = {
            "config": vampire_config,
            "installed": vampire_binary.exists(),
            "binary": vampire_binary if vampire_binary.exists() else None,
        }

    # spass
    spass_config_path = base_dir / "configs" / "spass.json"
    if spass_config_path.exists():
        spass_config = load_config(spass_config_path)
        spass_binary = base_dir / spass_config["paths"]["binary"]
        provers["spass"] = {
            "config": spass_config,
            "installed": spass_binary.exists(),
            "binary": spass_binary if spass_binary.exists() else None,
        }

    return provers


def find_config(config_name: str, all_provers: dict) -> tuple:
    """Find which prover a config belongs to.

    Returns (prover_name, preset_dict) or (None, None) if not found.
    """
    for prover_name, prover_info in all_provers.items():
        presets = prover_info["config"].get("presets", {})
        if config_name in presets:
            return prover_name, presets[config_name]
    return None, None


def list_configs(base_dir: Path):
    """List available configs from all provers."""
    all_provers = get_all_prover_configs(base_dir)

    if not all_provers:
        print("Error: No prover configs found")
        return

    for prover_name, prover_info in all_provers.items():
        presets = prover_info["config"].get("presets", {})
        installed = prover_info["installed"]

        status = "" if installed else " (not installed)"
        print(f"\n{prover_name}{status}:")

        for name, preset in sorted(presets.items()):
            desc = preset.get("description", "")
            encoder = preset.get("encoder")
            scorer = preset.get("scorer")

            model_info = ""
            if encoder and scorer:
                model_info = f" [{encoder}+{scorer}]"

            print(f"  {name:<25} {desc}{model_info}")


def get_problems(base_dir: Path, tptp_config: dict, problem_set_name: str) -> list[Path]:
    """Get list of problem files matching the problem set filters."""
    problem_sets = tptp_config.get("problem_sets", {})
    if problem_set_name not in problem_sets:
        available = list(problem_sets.keys())
        raise ValueError(f"Unknown problem set: {problem_set_name}. Available: {available}")

    filters = problem_sets[problem_set_name]
    problems_dir = base_dir / tptp_config["paths"]["problems"]

    if not problems_dir.exists():
        raise FileNotFoundError(f"TPTP problems not found: {problems_dir}")

    # Load metadata
    metadata_path = base_dir / ".data" / "problem_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            data = json.load(f)
            problems_list = data.get("problems", data) if isinstance(data, dict) else data
            metadata = {p["path"]: p for p in problems_list}

    # Check for explicit problem list (inline or from file)
    explicit_problems = filters.get("problems")
    if "problems_file" in filters:
        problems_file = base_dir / filters["problems_file"]
        if problems_file.exists():
            with open(problems_file) as f:
                explicit_problems = [line.strip() for line in f if line.strip()]
    if explicit_problems:
        explicit_set = set(explicit_problems)

    problems = []
    for domain_dir in sorted(problems_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        if "domains" in filters and filters["domains"] and domain not in filters["domains"]:
            continue
        if "exclude_domains" in filters and domain in filters.get("exclude_domains", []):
            continue

        for problem_file in sorted(domain_dir.glob("*.p")):
            rel_path = str(problem_file.relative_to(problems_dir))
            meta = metadata.get(rel_path, {})

            # Filter by explicit problem names (without .p extension)
            if explicit_problems:
                problem_name = problem_file.stem
                if problem_name not in explicit_set:
                    continue

            if "status" in filters and filters["status"]:
                if meta.get("status") not in filters["status"]:
                    continue
            if "format" in filters and filters["format"]:
                if meta.get("format") not in filters["format"]:
                    continue
            if "max_rating" in filters and filters["max_rating"] is not None:
                if meta.get("rating", 1.0) > filters["max_rating"]:
                    continue
            if "max_clauses" in filters and filters["max_clauses"] is not None:
                if meta.get("num_clauses", 0) > filters["max_clauses"]:
                    continue
            if "max_term_depth" in filters and filters["max_term_depth"] is not None:
                if meta.get("max_term_depth", 0) > filters["max_term_depth"]:
                    continue
            if "max_clause_size" in filters and filters["max_clause_size"] is not None:
                if meta.get("max_clause_size", 0) > filters["max_clause_size"]:
                    continue
            if "has_equality" in filters and filters["has_equality"] is not None:
                if meta.get("has_equality") != filters["has_equality"]:
                    continue
            if "is_unit_only" in filters and filters["is_unit_only"] is not None:
                if meta.get("is_unit_only") != filters["is_unit_only"]:
                    continue

            problems.append(problem_file)

    return problems


# Result persistence

def get_run_result_path(base_dir: Path, prover: str, preset_name: str, problem: Path) -> Path:
    """Get path to result file for a problem."""
    return base_dir / ".data" / "runs" / prover / preset_name / f"{problem.stem}.json"


def load_run_result(base_dir: Path, prover: str, preset_name: str, problem: Path) -> Optional[BenchResult]:
    """Load existing result if available."""
    result_file = get_run_result_path(base_dir, prover, preset_name, problem)
    if not result_file.exists():
        return None
    try:
        with open(result_file) as f:
            data = json.load(f)
        return BenchResult(
            problem=data["problem"],
            status=data["status"],
            time_s=data["time_s"],
        )
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def save_run_result(base_dir: Path, prover: str, preset_name: str, result: BenchResult):
    """Save individual result to .data/runs/<prover>/<preset>/<problem>.json"""
    runs_dir = base_dir / ".data" / "runs" / prover / preset_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    problem_name = Path(result.problem).stem
    result_file = runs_dir / f"{problem_name}.json"

    data = {
        "problem": result.problem,
        "status": result.status,
        "time_s": result.time_s,
        "prover": prover,
        "preset": preset_name,
        "timestamp": datetime.now().isoformat(),
    }

    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# Web export

def export_benchmark_progress(base_dir: Path, prover: str, preset_name: str,
                               stats: dict, completed: int, total: int):
    """Export current benchmark progress to web/data/benchmarks/<prover>_<preset>.json"""
    output_dir = base_dir / "web" / "data" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prover}_{preset_name}.json"

    data = {
        "generated": datetime.now().isoformat(),
        "prover": prover,
        "preset": preset_name,
        "completed": completed,
        "total": total,
        "progress_pct": 100 * completed / total if total else 0,
        "stats": {
            "proof": stats.get("proof", 0),
            "saturated": stats.get("saturated", 0),
            "resource_limit": stats.get("resource_limit", 0),
            "error": stats.get("error", 0),
        },
        "proof_rate": 100 * stats.get("proof", 0) / completed if completed else 0,
    }

    try:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        # Update index file
        _update_benchmark_index(output_dir)
    except Exception:
        pass  # Don't fail benchmark if export fails


def _update_benchmark_index(output_dir: Path):
    """Update index.json with list of all benchmark runs."""
    index_file = output_dir / "index.json"
    runs = []

    for f in sorted(output_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        runs.append(f.stem)

    index = {
        "generated": datetime.now().isoformat(),
        "runs": runs,
    }

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


# Scoring server management

def _start_scoring_server_subprocess(encoder, scorer, weights_path, socket_path, use_cuda, gpu_id=None):
    """Start a scoring server in a subprocess (blocks in that subprocess).

    Returns (Popen, stderr_path). stderr is written to a temp file so the pipe
    buffer can't fill up and block the server (which would freeze socket handlers
    and cause 'connection reset' errors from workers).
    """
    import subprocess as sp
    import tempfile

    script = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path("{Path(__file__).parent.parent / "python"}").resolve()))
from proofatlas.proofatlas import start_scoring_server
start_scoring_server(
    encoder="{encoder}",
    scorer="{scorer}",
    weights_path="{weights_path}",
    socket_path="{socket_path}",
    use_cuda={use_cuda},
)
'''
    env = {**os.environ}
    if use_cuda:
        # Restore GPU visibility for the server subprocess (the parent hides GPUs
        # via CUDA_VISIBLE_DEVICES="" to prevent fork-inherited CUDA contexts).
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    stderr_file = tempfile.NamedTemporaryFile(prefix="proofatlas-server-", suffix=".log", delete=False)

    proc = sp.Popen(
        [sys.executable, '-c', script],
        stdout=sp.DEVNULL,
        stderr=stderr_file,
        cwd=str(Path(__file__).parent.parent),
        env=env,
    )
    stderr_file.close()  # Process inherited the fd, we can close our handle
    return proc, stderr_file.name


def _wait_for_socket(socket_path, timeout=10):
    """Wait for a Unix socket file to appear."""
    start = time.time()
    while time.time() - start < timeout:
        if Path(socket_path).exists():
            return True
        time.sleep(0.1)
    return False


# Evaluation

STATUS_SYMBOLS = {"proof": "+", "saturated": "~", "resource_limit": "R", "error": "!"}


def _log_result(result, stats, index, total, log_file, base_dir, prover, preset_name, cached=False):
    """Update stats, print progress, log for --status parsing, export to web."""
    stats[result.status] = stats.get(result.status, 0) + 1
    symbol = STATUS_SYMBOLS[result.status]
    if cached:
        stats["skip"] += 1
        print(f"[{index}/{total}] S{symbol} {result.problem} (cached)")
    else:
        print(f"[{index}/{total}] {symbol} {result.problem} ({result.time_s:.2f}s)")
    log_file.write(f"PROGRESS:{index}:{total}:{stats['proof']}:{stats['resource_limit']}\n")
    log_file.flush()
    sys.stdout.flush()
    export_benchmark_progress(base_dir, prover, preset_name, stats, index, total)


def _restart_server(i, server_procs, socket_paths, stderr_logs, preset, weights_path, use_cuda, reason):
    """Restart a single scoring server. Returns True on success."""
    old_proc = server_procs[i]
    if old_proc.poll() is None:
        old_proc.terminate()
        try:
            old_proc.wait(timeout=5)
        except Exception:
            old_proc.kill()
            old_proc.wait()
    print(f"Scoring server {i}: {reason}, restarting...")
    sys.stdout.flush()
    gpu_id = i if use_cuda else None
    new_proc, new_log = _start_scoring_server_subprocess(
        encoder=preset["encoder"],
        scorer=preset["scorer"],
        weights_path=weights_path,
        socket_path=socket_paths[i],
        use_cuda=use_cuda,
        gpu_id=gpu_id,
    )
    server_procs[i] = new_proc
    stderr_logs[i] = new_log
    if _wait_for_socket(socket_paths[i]):
        print(f"Scoring server {i} restarted at {socket_paths[i]}")
        sys.stdout.flush()
        return True
    else:
        print(f"ERROR: Scoring server {i} failed to restart")
        sys.stdout.flush()
        return False


def _check_servers(server_procs, socket_paths, stderr_logs, preset, weights_path, use_cuda,
                   completed_count=0):
    """Check server health and restart any dead servers. Returns number restarted."""
    if not server_procs:
        return 0
    restarted = 0
    for i, proc in enumerate(server_procs):
        if proc.poll() is not None:
            rc = proc.returncode
            if rc < 0:
                desc = f"died (signal {-rc})"
            elif rc == 0:
                desc = "died (clean exit)"
            else:
                desc = f"died (exit code {rc})"
            if _restart_server(i, server_procs, socket_paths, stderr_logs,
                               preset, weights_path, use_cuda, desc):
                restarted += 1
    return restarted


def run_evaluation(base_dir: Path, problems: list[Path], tptp_root: Path,
                   prover: str, preset: dict, log_file,
                   preset_name: str = None, weights_path: str = None,
                   binary: Path = None, trace_preset: str = None,
                   rerun: bool = False, n_jobs: int = 1,
                   use_cuda: bool = False, gpu_workers: int = 0,
                   collect_traces: bool = False):
    """Run evaluation on problems with the specified prover."""
    stats = {"proof": 0, "saturated": 0, "resource_limit": 0, "error": 0, "skip": 0}

    # Start scoring server(s) for ML selectors
    server_procs = []
    socket_paths = []
    stderr_logs = []
    ml = _get_ml()

    if prover == "proofatlas" and ml.is_learned_selector(preset) and weights_path:
        model_label = f"{preset['encoder']}_{preset['scorer']}"

        if gpu_workers > 0:
            # Multi-GPU: one server per GPU
            print(f"\nStarting {gpu_workers} scoring server(s) for {model_label}...")
            for i in range(gpu_workers):
                sp = f"/tmp/proofatlas-scoring-{os.getpid()}-{i}.sock"
                proc, stderr_path = _start_scoring_server_subprocess(
                    encoder=preset["encoder"],
                    scorer=preset["scorer"],
                    weights_path=weights_path,
                    socket_path=sp,
                    use_cuda=True,
                    gpu_id=i,
                )
                server_procs.append(proc)
                socket_paths.append(sp)
                stderr_logs.append(stderr_path)
        else:
            # CPU: single server
            print(f"\nStarting scoring server for {model_label}...")
            sp = f"/tmp/proofatlas-scoring-{os.getpid()}.sock"
            proc, stderr_path = _start_scoring_server_subprocess(
                encoder=preset["encoder"],
                scorer=preset["scorer"],
                weights_path=weights_path,
                socket_path=sp,
                use_cuda=False,
            )
            server_procs.append(proc)
            socket_paths.append(sp)
            stderr_logs.append(stderr_path)

        # Wait for all sockets
        for sp in socket_paths:
            if not _wait_for_socket(sp):
                idx = socket_paths.index(sp)
                server_procs[idx].terminate()
                server_procs[idx].wait(timeout=5)
                print(f"ERROR: Scoring server failed to start at {sp}")
                # Read stderr from log file
                try:
                    stderr_content = Path(stderr_logs[idx]).read_text(errors="replace")
                    for line in stderr_content.splitlines()[-10:]:
                        print(f"  Server: {line}")
                except Exception:
                    pass
                # Terminate any servers that did start
                for proc in server_procs:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                for s in socket_paths:
                    Path(s).unlink(missing_ok=True)
                for log_path in stderr_logs:
                    Path(log_path).unlink(missing_ok=True)
                return stats

        if len(socket_paths) == 1:
            print(f"Scoring server ready at {socket_paths[0]}")
        else:
            print(f"All {len(socket_paths)} scoring servers ready")

    try:
        if prover == "proofatlas":
            model_label = f"{preset['encoder']}_{preset['scorer']}" if ml.is_learned_selector(preset) else "age_weight"
            print(f"\nEvaluating {len(problems)} problems with {model_label}" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))
            if weights_path:
                print(f"Weights: {weights_path}")
        else:
            print(f"\nEvaluating {len(problems)} problems" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))

        collect_trace = collect_traces and (prover == "proofatlas")

        # Prepare work items
        work_items = [
            (problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun, use_cuda)
            for problem in problems
        ]

        if n_jobs > 1:
            # Parallel execution — each thread spawns a subprocess via run_proofatlas()
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(run_single_problem, item, socket_path=socket_paths[i % len(socket_paths)] if socket_paths else None): i for i, item in enumerate(work_items)}
                completed = 0

                for future in as_completed(futures):
                    completed += 1
                    _check_servers(server_procs, socket_paths, stderr_logs, preset, weights_path, use_cuda,
                                   completed_count=completed)
                    try:
                        status, result = future.result()
                        _log_result(result, stats, completed, len(problems), log_file, base_dir, prover, preset_name, cached=(status == "skip"))
                    except Exception as e:
                        print(f"ERROR: {e}")
                        stats["error"] += 1
        else:
            # Sequential execution
            for i, item in enumerate(work_items, 1):
                _check_servers(server_procs, socket_paths, stderr_logs, preset, weights_path, use_cuda,
                               completed_count=i - 1)
                # Periodic garbage collection to prevent OOM
                if i % 100 == 0:
                    import gc
                    gc.collect()

                try:
                    sp = socket_paths[(i - 1) % len(socket_paths)] if socket_paths else None
                    status, result = run_single_problem(item, socket_path=sp)
                    _log_result(result, stats, i, len(problems), log_file, base_dir, prover, preset_name, cached=(status == "skip"))
                except Exception as e:
                    print(f"ERROR processing {item[0].name}: {e}")
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()

        # Print summary (individual results saved to .data/runs/)
        # Note: skip count is separate (skipped problems are also counted in their status)
        total = len(problems)
        proof_rate = 100 * stats["proof"] / total if total else 0

        print(f"\n{'='*60}")
        skip_str = f" S{stats['skip']}" if stats["skip"] else ""
        print(f"Results: +{stats['proof']} ~{stats['saturated']} R{stats['resource_limit']}{skip_str} ({proof_rate:.1f}% proofs)")

    finally:
        # Terminate scoring server subprocesses
        for proc in server_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        for sp in socket_paths:
            Path(sp).unlink(missing_ok=True)
        for log_path in stderr_logs:
            Path(log_path).unlink(missing_ok=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark and train theorem provers")
    parser.add_argument("--config", nargs="*",
                       help="Config(s) to run (default: all)")
    parser.add_argument("--problem-set",
                       help="Problem set from tptp.json (default: from config)")
    parser.add_argument("--rerun", action="store_true",
                       help="Re-evaluate problems even if cached results exist")
    parser.add_argument("--cpu-workers", type=int, default=1,
                       help="Number of parallel CPU workers (default: 1)")
    parser.add_argument("--gpu-workers", type=int, default=0,
                       help="Number of GPUs for ML inference (0=CPU, N>0=distribute across N GPUs)")

    parser.add_argument("--trace", action="store_true",
                       help="Collect training traces (.npz) for successful proofs")

    # Job management
    parser.add_argument("--status", action="store_true",
                       help="Check job status")
    parser.add_argument("--kill", action="store_true",
                       help="Stop running job")
    parser.add_argument("--list", action="store_true",
                       help="List available configs and exit")

    args = parser.parse_args()
    base_dir = find_project_root()

    # List configs
    if args.list:
        list_configs(base_dir)
        return

    # Job management
    if args.status:
        print_job_status(base_dir)
        return

    if args.kill:
        had_job = kill_job(base_dir)
        if had_job:
            print("Job killed.")
        else:
            print("No job file found. Killing any orphaned processes...")
        return

    existing = get_job_status(base_dir)
    if existing:
        print(f"Error: Job already running (PID: {existing['pid']})")
        print("Use --status or --kill")
        sys.exit(1)

    # Load configs
    tptp_config = load_config(base_dir / "configs" / "tptp.json")
    tptp_root = base_dir / tptp_config["paths"]["root"]

    # Determine problem set (use default from config if not specified)
    problem_set = args.problem_set
    if problem_set is None:
        problem_set = tptp_config.get("defaults", {}).get("problem_set")
        if problem_set is None:
            print("Error: No --problem-set specified and no default in tptp.json")
            sys.exit(1)

    # Get all prover configs (installed or not)
    all_provers = get_all_prover_configs(base_dir)
    if not all_provers:
        print("Error: No prover configs found")
        sys.exit(1)

    # Build list of runs
    runs = []

    if args.config:
        # Run specified configs
        for config_name in args.config:
            prover_name, preset = find_config(config_name, all_provers)

            if prover_name is None:
                print(f"Error: Unknown config '{config_name}'")
                print(f"Use --list to see available configs")
                sys.exit(1)

            prover_info = all_provers[prover_name]
            if not prover_info["installed"]:
                print(f"Error: Config '{config_name}' requires {prover_name}, which is not installed")
                print(f"Run 'python scripts/setup_{prover_name}.py' to install it")
                sys.exit(1)

            runs.append({
                "prover": prover_name,
                "preset_name": config_name,
                "preset": preset,
                "binary": prover_info["binary"],
                "config": prover_info["config"],
            })
    else:
        # Default: run all proofatlas presets
        if "proofatlas" not in all_provers:
            print("Error: proofatlas config not found")
            sys.exit(1)

        prover_info = all_provers["proofatlas"]
        presets = prover_info["config"].get("presets", {})
        for preset_name, preset in presets.items():
            runs.append({
                "prover": "proofatlas",
                "preset_name": preset_name,
                "preset": preset,
                "binary": prover_info["binary"],
                "config": prover_info["config"],
            })

    if not runs:
        print("Error: No matching prover/preset combinations")
        sys.exit(1)

    # Get problems
    problems = get_problems(base_dir, tptp_config, problem_set)

    # Validate worker configuration
    gpu_workers = args.gpu_workers
    cpu_workers = args.cpu_workers

    if gpu_workers < 0:
        print("Error: --gpu-workers cannot be negative")
        sys.exit(1)

    # Check if any runs use ML selectors
    ml = _get_ml()
    has_ml_runs = any(
        r["prover"] == "proofatlas" and ml.is_learned_selector(r["preset"])
        for r in runs
    )

    if gpu_workers > 0:
        if not has_ml_runs:
            print("Warning: --gpu-workers has no effect without ML selector configs, ignoring")
            gpu_workers = 0
        else:
            # Check GPU count via subprocess because CUDA_VISIBLE_DEVICES=""
            # is set in this process to prevent fork-inherited CUDA contexts.
            # Remove the override so the child sees all GPUs.
            import subprocess
            check_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            try:
                result = subprocess.run(
                    [sys.executable, "-c",
                     "import torch; print(torch.cuda.device_count())"],
                    capture_output=True, text=True, timeout=30,
                    env=check_env,
                )
                available_gpus = int(result.stdout.strip())
            except Exception:
                print("Error: --gpu-workers requires PyTorch with CUDA support")
                sys.exit(1)

            if available_gpus == 0:
                print("Error: --gpu-workers > 0 but no CUDA GPUs detected")
                print("Use --gpu-workers 0 for CPU inference")
                sys.exit(1)

            if gpu_workers > available_gpus:
                print(f"Error: --gpu-workers {gpu_workers} exceeds available GPUs ({available_gpus})")
                sys.exit(1)

    if gpu_workers > 0 and gpu_workers > cpu_workers:
        print(f"Note: Clamping --gpu-workers from {gpu_workers} to {cpu_workers} (no more servers than workers)")
        gpu_workers = cpu_workers

    use_cuda_eval = gpu_workers > 0

    log_file_path = get_log_file(base_dir)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Daemonize
    is_daemon, grandchild_pid = daemonize(log_file_path)
    if not is_daemon:
        time.sleep(0.1)
        prover_names = sorted(set(r["prover"] for r in runs))
        print(f"Started job (PID: {grandchild_pid})")
        print(f"Provers: {', '.join(prover_names)}, Configs: {len(runs)}, Problems: {len(problems)}")
        print("Use --status to check, --kill to stop")
        return

    # --- Daemon process from here ---

    # Log startup info
    print(f"Benchmark daemon started (PID: {os.getpid()})")
    print(f"Working directory: {base_dir}")
    print(f"Configs: {len(runs)}, Problems: {len(problems)}")
    print(f"Backend: {'cuda (' + str(gpu_workers) + ' GPUs)' if use_cuda_eval else 'cpu'}")
    sys.stdout.flush()

    # Clear any stale PID tracking and save job status
    clear_pids(base_dir)
    try:
        save_job_status(base_dir, os.getpid(), sys.argv, len(runs))
        print(f"Job file saved: {get_job_file(base_dir)}")
    except Exception as e:
        print(f"WARNING: Failed to save job status: {e}")
        print("Use 'ps aux | grep proofatlas' to find this process")
    sys.stdout.flush()

    # Set up signal handlers to log unexpected termination
    def signal_handler(signum, frame):
        sig_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT",
                     signal.SIGQUIT: "SIGQUIT", signal.SIGABRT: "SIGABRT"}
        sig_name = sig_names.get(signum, f"signal {signum}")
        print(f"\nRECEIVED {sig_name} - exiting")
        sys.stdout.flush()
        clear_pids(base_dir)
        clear_job_status(base_dir)
        sys.stdout.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    num_runs = len(runs)

    try:
        for run_idx, run in enumerate(runs, 1):
            prover = run["prover"]
            preset_name = run["preset_name"]
            preset = run["preset"]
            binary = run["binary"]
            prover_config = run["config"]

            config_label = f"{prover}/{preset_name}"
            trace_preset = preset.get("traces") or preset_name

            # Log config progress for --status parsing
            print(f"CONFIG:{config_label}:{run_idx}:{num_runs}")
            sys.stdout.flush()

            print(f"\n{'='*60}")
            print(f"Running: {config_label} ({run_idx}/{num_runs})")
            print(f"{'='*60}\n")

            weights_path = None
            current_preset = preset

            # Check weights for ML selectors (training now handled by scripts/train.py)
            ml = _get_ml()
            if prover == "proofatlas" and ml.is_learned_selector(preset):
                model_name = f"{preset['encoder']}_{preset['scorer']}"
                encoder_type = ml.get_encoder_type(preset)
                log(f"[{preset_name}] Learned selector: {model_name} (encoder: {encoder_type})")
                sys.stdout.flush()

                weights_dir = base_dir / ".weights"
                existing_weights = ml.find_weights(weights_dir, preset)

                if existing_weights:
                    log(f"[{preset_name}] Found weights: {existing_weights}")
                    weights_path = existing_weights
                else:
                    log(f"[{preset_name}] ERROR: No weights found in {weights_dir}")
                    log(f"[{preset_name}] Train first with: python scripts/train.py --config {preset_name}")
                    sys.stdout.flush()
                    continue
                sys.stdout.flush()
            elif prover == "proofatlas":
                log(f"[{preset_name}] Heuristic selector (no training needed)")
                sys.stdout.flush()
            else:
                log(f"[{preset_name}] External prover: {prover}")
                sys.stdout.flush()

            # Run evaluation
            log(f"[{preset_name}] Starting evaluation on {len(problems)} problems...")
            sys.stdout.flush()
            # Rust expects weights directory (find_weights already returns the directory)
            weights_dir_str = str(weights_path) if weights_path else None
            run_evaluation(
                base_dir, problems, tptp_root,
                prover=prover, preset=current_preset,
                log_file=sys.stdout,
                preset_name=preset_name, weights_path=weights_dir_str,
                binary=binary, trace_preset=trace_preset,
                rerun=args.rerun, n_jobs=args.cpu_workers,
                use_cuda=use_cuda_eval, gpu_workers=gpu_workers,
                collect_traces=args.trace,
            )
            log(f"[{preset_name}] Evaluation complete")
            sys.stdout.flush()

        print(f"\n{'='*60}")
        print("All benchmarks complete")
        print(f"{'='*60}")
        sys.stdout.flush()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_pids(base_dir)
        clear_job_status(base_dir)
        sys.stdout.close()
    os._exit(0)


if __name__ == "__main__":
    main()
