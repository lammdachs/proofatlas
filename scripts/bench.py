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

    CPU workers use an in-process pipeline (no external server needed).

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
import subprocess
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

    # Load metadata from index.json alongside Problems/
    index_path = problems_dir.parent / "index.json"
    metadata = {}
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
            metadata = {p["path"]: p for p in data["problems"]}

    problems = []
    for domain_dir in sorted(problems_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        if "domains" in filters and domain not in filters["domains"]:
            continue

        for problem_file in sorted(domain_dir.glob("*.p")):
            rel_path = str(problem_file.relative_to(problems_dir))
            meta = metadata.get(rel_path, {})

            if "status" in filters:
                if meta.get("status") not in filters["status"]:
                    continue
            if "format" in filters:
                if meta.get("format") not in filters["format"]:
                    continue
            if "max_total_size_bytes" in filters:
                if meta.get("total_size", 0) > filters["max_total_size_bytes"]:
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




def run_evaluation(base_dir: Path, problems: list[Path], tptp_root: Path,
                   prover: str, preset: dict, log_file,
                   preset_name: str = None, weights_path: str = None,
                   binary: Path = None, trace_preset: str = None,
                   rerun: bool = False, n_jobs: int = 1,
                   use_cuda: bool = False,
                   collect_traces: bool = False):
    """Run evaluation on problems with the specified prover."""
    stats = {"proof": 0, "saturated": 0, "resource_limit": 0, "error": 0, "skip": 0}
    ml = _get_ml()

    if prover == "proofatlas":
        model_label = f"{preset['encoder']}_{preset['scorer']}" if ml.is_learned_selector(preset) else "age_weight"
        print(f"\nEvaluating {len(problems)} problems with {model_label}" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))
        if weights_path:
            print(f"Weights: {weights_path}")
    else:
        print(f"\nEvaluating {len(problems)} problems" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))

    collect_trace = collect_traces and (prover == "proofatlas")

    if prover == "proofatlas":
        # Persistent worker pool — reuses ProofAtlas (and ML model) across problems
        from bench_provers import ProofAtlasPool

        pool = ProofAtlasPool(
            n_workers=n_jobs, preset=preset, base_dir=base_dir,
            tptp_root=tptp_root, weights_path=weights_path,
            use_cuda=use_cuda,
            collect_trace=collect_trace, trace_preset=trace_preset,
        )
        pool.start()

        try:
            if n_jobs > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading

                problem_iter = iter(enumerate(problems, 1))
                iter_lock = threading.Lock()
                log_lock = threading.Lock()
                completed_count = [0]

                def _pool_worker_fn(worker_idx):
                    worker = pool.workers[worker_idx]
                    while True:
                        with iter_lock:
                            try:
                                i, problem = next(problem_iter)
                            except StopIteration:
                                return

                        existing = load_run_result(base_dir, prover, preset_name, problem)
                        if existing and not rerun:
                            with log_lock:
                                completed_count[0] += 1
                                _log_result(existing, stats, completed_count[0], len(problems), log_file, base_dir, prover, preset_name, cached=True)
                            continue

                        try:
                            result = worker.submit(problem, pool.process_timeout)
                            if result.status == "timeout":
                                result.status = "resource_limit"
                            save_run_result(base_dir, prover, preset_name, result)
                        except Exception as e:
                            result = BenchResult(problem=problem.name, status="error", time_s=0)
                            print(f"ERROR: {e}")

                        with log_lock:
                            completed_count[0] += 1
                            _log_result(result, stats, completed_count[0], len(problems), log_file, base_dir, prover, preset_name, cached=False)

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(_pool_worker_fn, i) for i in range(n_jobs)]
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception as e:
                            print(f"ERROR in pool worker: {e}")
                            stats["error"] += 1
            else:
                # Sequential execution with persistent worker
                for i, problem in enumerate(problems, 1):
                    existing = load_run_result(base_dir, prover, preset_name, problem)
                    if existing and not rerun:
                        _log_result(existing, stats, i, len(problems), log_file, base_dir, prover, preset_name, cached=True)
                        continue

                    try:
                        result = pool.workers[0].submit(problem, pool.process_timeout)
                        if result.status == "timeout":
                            result.status = "resource_limit"
                        save_run_result(base_dir, prover, preset_name, result)
                        _log_result(result, stats, i, len(problems), log_file, base_dir, prover, preset_name, cached=False)
                    except Exception as e:
                        print(f"ERROR processing {problem.name}: {e}")
                        sys.stdout.flush()
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
        finally:
            pool.shutdown()

    else:
        # External provers (Vampire, SPASS): subprocess per problem
        work_items = [
            (problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun, use_cuda)
            for problem in problems
        ]

        if n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(run_single_problem, item): i for i, item in enumerate(work_items)}
                completed = 0

                for future in as_completed(futures):
                    completed += 1
                    try:
                        status, result = future.result()
                        _log_result(result, stats, completed, len(problems), log_file, base_dir, prover, preset_name, cached=(status == "skip"))
                    except Exception as e:
                        print(f"ERROR: {e}")
                        stats["error"] += 1
        else:
            for i, item in enumerate(work_items, 1):
                if i % 100 == 0:
                    import gc
                    gc.collect()

                try:
                    status, result = run_single_problem(item)
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
    parser.add_argument("--cuda", action="store_true",
                       help="Use CUDA for ML inference")

    parser.add_argument("--timeout", type=float, default=None,
                       help="Override per-problem timeout (seconds)")
    parser.add_argument("--trace", action="store_true",
                       help="Collect training traces (.npz) for successful proofs")

    # Job management
    parser.add_argument("--status", action="store_true",
                       help="Check job status")
    parser.add_argument("--kill", action="store_true",
                       help="Stop running job")
    parser.add_argument("--list", action="store_true",
                       help="List available configs and exit")
    parser.add_argument("--foreground", action="store_true",
                       help="Run in foreground (skip daemonization, for pipeline/debugging)")

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

    if not args.foreground:
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

    # Override timeout if specified (both timeout and max_iterations apply; whichever fires first)
    if args.timeout is not None:
        for run in runs:
            run["preset"] = {**run["preset"], "timeout": args.timeout}

    # Get problems
    problems = get_problems(base_dir, tptp_config, problem_set)

    # Validate worker configuration
    use_cuda_eval = args.cuda

    # Ensure base MiniLM weights exist for trace embedding.
    # Must run before daemonize — torch.jit.trace can't run after fork.
    if args.trace:
        weights_dir = base_dir / ".weights"
        minilm_model = weights_dir / "base_minilm.pt"
        minilm_tokenizer = weights_dir / "base_minilm_tokenizer" / "tokenizer.json"
        if not (minilm_model.exists() and minilm_tokenizer.exists()):
            print("Downloading and exporting base MiniLM for trace embedding...")
            sys.stdout.flush()
            # Run in subprocess to isolate torch.jit.trace from our process.
            # Remove CUDA_VISIBLE_DEVICES="" (set at top of bench.py) — it causes
            # double-free crashes in torch.jit.trace with transformers 5.x SDPA.
            export_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            rc = subprocess.run(
                [sys.executable, str(base_dir / "scripts" / "setup_minilm.py")],
                cwd=str(base_dir),
                env=export_env,
            ).returncode
            if rc == 0:
                print(f"Base MiniLM exported to {weights_dir}")
            else:
                print("Warning: Failed to export base MiniLM.")
                print("Traces will lack pre-computed embeddings.")
            sys.stdout.flush()

    log_file_path = get_log_file(base_dir)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Daemonize (unless --foreground)
    if not args.foreground:
        is_daemon, grandchild_pid = daemonize(log_file_path)
        if not is_daemon:
            time.sleep(0.1)
            prover_names = sorted(set(r["prover"] for r in runs))
            print(f"Started job (PID: {grandchild_pid})")
            print(f"Provers: {', '.join(prover_names)}, Configs: {len(runs)}, Problems: {len(problems)}")
            print(f"Log: tail -f {log_file_path}")
            print("Use --status to check, --kill to stop")
            return

        # --- Daemon process from here ---

        # Log startup info
        print(f"Benchmark daemon started (PID: {os.getpid()})")
        print(f"Working directory: {base_dir}")
        print(f"Configs: {len(runs)}, Problems: {len(problems)}")
        print(f"Backend: {'cuda' if use_cuda_eval else 'cpu'}")
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
                    log(f"[{preset_name}] Train first with: proofatlas-train --config {preset_name}")
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
                use_cuda=use_cuda_eval,
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
        if not args.foreground:
            clear_pids(base_dir)
            clear_job_status(base_dir)
            sys.stdout.close()

    if not args.foreground:
        os._exit(0)


if __name__ == "__main__":
    main()
