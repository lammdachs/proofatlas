#!/usr/bin/env python3
"""
Benchmark and train proofatlas theorem prover.

USAGE:
    proofatlas-bench                           # Run all configs
    proofatlas-bench --config gcn_mlp_sel21    # Run specific config
    proofatlas-bench --config gcn_mlp_sel21 --retrain  # Retrain model
    proofatlas-bench --list                    # List available configs

    proofatlas-bench --status                  # Check job status
    proofatlas-bench --kill                    # Stop running job

CACHING:
    Results cached in .data/runs/proofatlas/<preset>/<problem>.json
    Use --rerun to force re-evaluation of cached results.
    Use --retrain to force retraining of ML models.

ML MODELS:
    Presets with embedding+scorer (e.g., gcn_mlp_sel21) automatically:
    1. Collect traces using age_weight baseline (if none exist)
    2. Train the model
    3. Export to .weights/{embedding}_{scorer}.pt

OUTPUT:
    .weights/{embedding}_{scorer}.pt    - TorchScript model
    .data/traces/<preset>/              - Training traces
    .data/runs/proofatlas/<preset>/     - Per-problem results
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

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


# Job management
JOB_FILE = ".data/bench_job.json"
LOG_FILE = ".data/bench.log"
PID_FILE = ".data/bench_pids.txt"


def get_job_file(base_dir: Path) -> Path:
    return base_dir / JOB_FILE


def get_log_file(base_dir: Path) -> Path:
    return base_dir / LOG_FILE


def get_pid_file(base_dir: Path) -> Path:
    return base_dir / PID_FILE


def register_pid(base_dir: Path, pid: int):
    """Register a spawned process PID for cleanup (Unix only)."""
    if sys.platform == "win32":
        return
    pid_file = get_pid_file(base_dir)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "a") as f:
        f.write(f"{pid}\n")


def clear_pids(base_dir: Path):
    """Clear the PID tracking file."""
    pid_file = get_pid_file(base_dir)
    if pid_file.exists():
        pid_file.unlink()


def kill_tracked_pids(base_dir: Path) -> int:
    """Kill all tracked PIDs and clear the file (Unix only)."""
    if sys.platform == "win32":
        return 0
    pid_file = get_pid_file(base_dir)
    if not pid_file.exists():
        return 0

    killed = 0
    try:
        pids = pid_file.read_text().splitlines()
        for line in pids:
            if not line:
                continue
            try:
                os.kill(int(line), signal.SIGKILL)
                killed += 1
            except (ValueError, OSError, ProcessLookupError):
                pass
        pid_file.unlink(missing_ok=True)
    except (IOError, OSError):
        pass

    return killed


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_job_status(base_dir: Path) -> Optional[dict]:
    job_file = get_job_file(base_dir)
    if not job_file.exists():
        return None

    try:
        with open(job_file) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if not is_process_running(job["pid"]):
        return None

    log_file = Path(job.get("log_file", ""))
    if log_file.exists():
        try:
            with open(log_file) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("PROGRESS:"):
                    parts = line.strip().split(":")
                    if len(parts) >= 5:
                        job["current"] = int(parts[1])
                        job["total"] = int(parts[2])
                        job["proofs"] = int(parts[3])
                        job["timeout"] = int(parts[4])
                    break
                elif line.startswith("TRAIN:"):
                    parts = line.strip().split(":")
                    if len(parts) >= 4:
                        job["train_epoch"] = int(parts[1])
                        job["train_total"] = int(parts[2])
                        job["train_loss"] = float(parts[3])
                    break
        except IOError:
            pass

    return job


def save_job_status(base_dir: Path, pid: int, args: list, num_configs: int = 1):
    job_file = get_job_file(base_dir)
    job_file.parent.mkdir(parents=True, exist_ok=True)

    job = {
        "pid": pid,
        "args": args,
        "log_file": str(get_log_file(base_dir)),
        "start_time": datetime.now().isoformat(),
        "num_configs": num_configs,
    }

    with open(job_file, "w") as f:
        json.dump(job, f, indent=2)


def clear_job_status(base_dir: Path):
    job_file = get_job_file(base_dir)
    if job_file.exists():
        job_file.unlink()


def kill_job(base_dir: Path) -> bool:
    import subprocess

    job = get_job_status(base_dir)

    # Step 1: Clear job status to stop spawning new processes
    clear_job_status(base_dir)

    # Step 2: Kill the main daemon process
    if job:
        try:
            os.kill(job['pid'], signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    # Step 3: Kill tracked PIDs and worker processes (Unix only)
    if sys.platform != "win32":
        # Kill any proofatlas-bench worker processes
        subprocess.run(["pkill", "-9", "-f", "proofatlas-bench.*--config"], capture_output=True)

        # Kill tracked prover PIDs
        max_iterations = 10
        for _ in range(max_iterations):
            killed = kill_tracked_pids(base_dir)
            if killed == 0:
                break
            time.sleep(0.2)

        # Kill any remaining prover processes from this project
        subprocess.run(["pkill", "-9", "-f", str(base_dir / ".vampire")], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", str(base_dir / ".spass")], capture_output=True)

    return job is not None


def print_job_status(base_dir: Path):
    job = get_job_status(base_dir)
    if not job:
        print("No job currently running.")
        return

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    print(f"Job running (PID: {job['pid']})")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m ago)")

    # Parse log file for progress
    log_file = Path(job.get("log_file", ""))
    if log_file.exists():
        last_progress = None
        last_train = None
        last_config = None  # CONFIG:name:index:total
        phase = None

        with open(log_file) as f:
            for line in f:
                if line.startswith("PROGRESS:"):
                    last_progress = line.strip()
                    phase = "eval"
                elif line.startswith("TRAIN:"):
                    last_train = line.strip()
                    phase = "train"
                elif line.startswith("CONFIG:"):
                    last_config = line.strip()
                elif "Collecting traces" in line:
                    phase = "collect"
                elif "Training" in line and "problems" in line:
                    phase = "train"

        # Check if we're in trace collection
        is_collecting = False
        with open(log_file) as f:
            content = f.read()
            if "Collecting traces" in content:
                eval_count = content.count("Evaluating")
                is_collecting = eval_count == 1

        # Show current config with progress
        if last_config:
            parts = last_config.split(":")
            if len(parts) >= 4:
                config_name, config_idx, config_total = parts[1], parts[2], parts[3]
                # config_name is "prover/preset" format
                if "/" in config_name:
                    prover, preset = config_name.split("/", 1)
                    print(f"  Prover:  {prover}")
                    print(f"  Config:  {preset} ({config_idx}/{config_total})")
                else:
                    print(f"  Config:  {config_name} ({config_idx}/{config_total})")

        if phase == "train" and last_train:
            parts = last_train.split(":")
            if len(parts) >= 4:
                epoch, max_epochs, loss = parts[1], parts[2], parts[3]
                print(f"  Training: epoch {epoch}/{max_epochs}, loss={float(loss):.4f}")
        elif is_collecting and last_progress:
            parts = last_progress.split(":")
            if len(parts) >= 5:
                current, total, proofs, timeout = parts[1:5]
                print(f"  Collecting traces: {current}/{total}, +{proofs} proofs")
        elif phase == "eval" and last_progress:
            parts = last_progress.split(":")
            if len(parts) >= 5:
                current, total, proofs, timeout = parts[1:5]
                print(f"  Evaluating: {current}/{total}, +{proofs} proofs, T{timeout} timeout")
        else:
            print("  Starting...")
    else:
        print("  Starting...")

    print(f"\nTo stop: proofatlas-bench --kill")


@dataclass
class BenchResult:
    problem: str
    status: str  # "proof", "saturated", "timeout", "error"
    time_s: float


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def list_configs(base_dir: Path):
    """List available configs."""
    config_path = base_dir / "configs" / "proofatlas.json"
    if not config_path.exists():
        print("Error: configs/proofatlas.json not found")
        return

    config = load_config(config_path)
    presets = config.get("presets", {})

    print("Available configs:")
    for name, preset in sorted(presets.items()):
        desc = preset.get("description", "")
        embedding = preset.get("embedding")
        scorer = preset.get("scorer")

        model_info = ""
        if embedding and scorer:
            model_info = f" [{embedding}+{scorer}]"

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


# Prover execution

def _run_proofatlas_inner(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                          weights_path: str = None, collect_trace: bool = False,
                          trace_preset: str = None) -> BenchResult:
    """Inner function that actually runs ProofAtlas (called in subprocess)."""
    from proofatlas import ProofState

    timeout = preset.get("timeout", 10)

    try:
        with open(problem) as f:
            content = f.read()
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=0)

    # Start timer before parsing (CNF conversion counts against timeout)
    start = time.time()

    state = ProofState()
    try:
        # Pass timeout to parsing to prevent CNF conversion hangs
        state.add_clauses_from_tptp(content, str(tptp_root), timeout)
    except Exception as e:
        elapsed = time.time() - start
        # Check if this was a timeout during CNF conversion
        if "timed out" in str(e).lower():
            return BenchResult(problem=problem.name, status="timeout", time_s=elapsed)
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)

    literal_selection = str(preset.get("literal_selection", 21))
    state.set_literal_selection(literal_selection)

    max_iterations = preset.get("max_iterations", 0)  # 0 means no limit
    max_clause_memory_mb = preset.get("max_clause_memory_mb")  # None means no limit
    ml = _get_ml()
    is_learned = ml.is_learned_selector(preset)
    age_weight_ratio = preset.get("age_weight_ratio", 0.167)
    # Derive embedding type from embedding field: "graph", "string", or None
    embedding_type = ml.get_embedding_type(preset) if is_learned else None
    # Model name for .pt file: {embedding}_{scorer} for modular design
    model_name = ml.get_model_name(preset) if is_learned else None

    # Initialize CUDA if using string embedding (required for tch to detect CUDA)
    if embedding_type == "string":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
                _ = torch.tensor([1.0]).cuda()  # Force CUDA context creation
        except Exception:
            pass  # If torch not available or CUDA init fails, fall back to CPU

    # Remaining time after parsing
    elapsed_parsing = time.time() - start
    remaining_timeout = max(0.1, timeout - elapsed_parsing)

    try:
        proof_found, status = state.run_saturation(
            max_iterations,
            float(remaining_timeout),
            float(age_weight_ratio) if not is_learned else None,
            embedding_type,  # "graph", "string", or None
            weights_path,
            model_name,
            max_clause_memory_mb,
        )
    except Exception as e:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Map status to benchmark format (resource_limit -> timeout for compatibility)
    if status == "resource_limit":
        status = "timeout"

    # Collect trace for training
    if collect_trace and proof_found and trace_preset:
        try:
            trace_json = state.extract_structured_trace(elapsed)
            _get_ml().save_trace(base_dir / ".data" / "traces", trace_preset, problem.name, trace_json)
        except Exception:
            pass

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def get_num_gpus() -> int:
    """Get the number of available CUDA GPUs.

    Uses nvidia-smi instead of torch to avoid initializing CUDA in the parent
    process, which would interfere with tch-rs in forked subprocesses.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Count lines that start with "GPU"
            return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _worker_process(problem_str, base_dir_str, preset, tptp_root_str, weights_path,
                    collect_trace, trace_preset, result_queue, gpu_id=None):
    """Worker function that runs in subprocess and sends result via queue."""
    # Reset signal handlers inherited from parent daemon process.
    # Without this, if the worker is killed (e.g., timeout), it would run
    # the parent's signal handler which deletes the job file!
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    # Set CUDA device for this worker
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = _run_proofatlas_inner(
            Path(problem_str), Path(base_dir_str), preset, Path(tptp_root_str),
            weights_path, collect_trace, trace_preset
        )
        result_queue.put((result.status, result.time_s))
    except Exception as e:
        result_queue.put(("error", 0))


def run_proofatlas(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                   weights_path: str = None, collect_trace: bool = False,
                   trace_preset: str = None, gpu_id: int = None) -> BenchResult:
    """Run ProofAtlas on a problem in a subprocess.

    Uses multiprocessing to isolate crashes (e.g., stack overflow on deeply
    nested terms) so they don't take down the entire benchmark process.
    """
    import multiprocessing

    timeout = preset.get("timeout", 10)
    process_timeout = timeout + 10  # Extra time for overhead

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker_process,
        args=(str(problem), str(base_dir), preset, str(tptp_root),
              weights_path, collect_trace, trace_preset, result_queue, gpu_id)
    )

    start = time.time()
    proc.start()
    proc.join(timeout=process_timeout)
    elapsed = time.time() - start

    if proc.is_alive():
        # Process hung - kill it
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return BenchResult(problem=problem.name, status="timeout", time_s=elapsed)

    if proc.exitcode != 0:
        # Process crashed (e.g., stack overflow gives exit code 134)
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)

    try:
        status, elapsed_inner = result_queue.get_nowait()
        return BenchResult(problem=problem.name, status=status, time_s=elapsed_inner)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=elapsed)


def run_vampire(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run Vampire on a problem."""
    import subprocess

    timeout = preset.get("time_limit", 10)
    selection = preset.get("selection", 21)
    avatar = preset.get("avatar", "off")
    memory_limit = preset.get("memory_limit")
    activation_limit = preset.get("activation_limit")

    cmd = [
        str(binary),
        "--include", str(tptp_root),
        "--time_limit", str(timeout),
        "--selection", str(selection),
        "--avatar", avatar,
    ]

    if memory_limit is not None:
        cmd.extend(["--memory_limit", str(memory_limit)])

    if activation_limit is not None:
        cmd.extend(["--activation_limit", str(activation_limit)])

    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Parse Vampire output
    if "Refutation found" in output or "Termination reason: Refutation" in output:
        status = "proof"
    elif "Termination reason: Satisfiable" in output:
        status = "saturated"
    elif "Termination reason: Time limit" in output or elapsed >= timeout:
        status = "timeout"
    elif "Termination reason: Memory limit" in output:
        status = "timeout"  # Memory limit treated as resource limit
    elif "Termination reason: Activation limit" in output:
        status = "timeout"  # Activation limit treated as resource limit
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_spass(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run SPASS on a problem."""
    import subprocess

    timeout = preset.get("TimeLimit", 10)
    selection = preset.get("Select", 1)
    memory = preset.get("Memory")
    loops = preset.get("Loops")

    # SPASS requires TPTP format with -TPTP flag
    cmd = [
        str(binary),
        "-TPTP",
        f"-TimeLimit={timeout}",
        f"-Select={selection}",
    ]

    if memory is not None:
        cmd.append(f"-Memory={memory}")

    if loops is not None:
        cmd.append(f"-Loops={loops}")

    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TPTP": str(tptp_root)},
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    # Parse SPASS output
    # SPASS says "SPASS beiseite: Proof found." for proofs
    if "Proof found" in output:
        status = "proof"
    elif "Completion found" in output:
        status = "saturated"
    elif "Maximal number of loops exceeded" in output:
        status = "timeout"  # Loop limit treated as resource limit
    elif elapsed >= timeout or "SPASS broke down" in output:
        status = "timeout"
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


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
            "timeout": stats.get("timeout", 0),
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


# Global variable for worker GPU assignment (set by worker_init)
_worker_gpu_id = None


def _run_single_problem_with_worker_gpu(args):
    """Wrapper that uses worker's assigned GPU."""
    global _worker_gpu_id
    return _run_single_problem(args, gpu_id=_worker_gpu_id)


def _run_single_problem(args, gpu_id=None):
    """Worker function for execution."""
    problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun = args

    try:
        # Check if already evaluated (skip unless --rerun)
        existing = load_run_result(base_dir, prover, preset_name, problem)
        if existing and not rerun:
            return ("skip", existing)

        if prover == "proofatlas":
            result = run_proofatlas(
                problem, base_dir, preset, tptp_root,
                weights_path=weights_path, collect_trace=collect_trace,
                trace_preset=trace_preset, gpu_id=gpu_id,
            )
        elif prover == "vampire":
            result = run_vampire(problem, base_dir, preset, binary, tptp_root)
        elif prover == "spass":
            result = run_spass(problem, base_dir, preset, binary, tptp_root)
        else:
            result = BenchResult(problem=problem.name, status="error", time_s=0)

        # Save individual result to .data/runs/
        save_run_result(base_dir, prover, preset_name, result)
        return ("run", result)
    except Exception as e:
        return ("error", BenchResult(problem=problem.name, status="error", time_s=0))


def run_evaluation(base_dir: Path, problems: list[Path], tptp_root: Path,
                   prover: str, preset: dict, log_file,
                   preset_name: str = None, weights_path: str = None,
                   binary: Path = None, trace_preset: str = None,
                   rerun: bool = False, n_jobs: int = 1):
    """Run evaluation on problems with the specified prover."""
    stats = {"proof": 0, "saturated": 0, "timeout": 0, "error": 0, "skip": 0}

    if prover == "proofatlas":
        ml = _get_ml()
        model_label = ml.get_model_name(preset) if ml.is_learned_selector(preset) else "age_weight"
        print(f"\nEvaluating {len(problems)} problems with {model_label}" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))
        if weights_path:
            print(f"Weights: {weights_path}")
    else:
        print(f"\nEvaluating {len(problems)} problems" + (f" ({n_jobs} jobs)" if n_jobs > 1 else ""))

    # Always collect traces for proofatlas
    collect_trace = (prover == "proofatlas")

    # Detect GPUs for worker distribution
    num_gpus = get_num_gpus()
    if num_gpus > 0 and prover == "proofatlas":
        print(f"Distributing workers across {num_gpus} GPU(s)")

    # Prepare work items (GPU assigned per-worker, not per-problem)
    work_items = [
        (problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun)
        for problem in problems
    ]

    if n_jobs > 1:
        # Parallel execution with GPU assignment per worker
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        # Shared counter for assigning GPUs to workers
        gpu_counter = multiprocessing.Value('i', 0)

        def worker_init(counter, n_gpus):
            """Initialize worker with a GPU assignment."""
            global _worker_gpu_id
            if n_gpus > 0:
                with counter.get_lock():
                    _worker_gpu_id = counter.value % n_gpus
                    counter.value += 1
            else:
                _worker_gpu_id = None

        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=worker_init,
                                 initargs=(gpu_counter, num_gpus)) as executor:
            futures = {executor.submit(_run_single_problem_with_worker_gpu, item): i for i, item in enumerate(work_items)}
            completed = 0

            for future in as_completed(futures):
                completed += 1
                try:
                    status, result = future.result()
                    if status == "skip":
                        stats[result.status] = stats.get(result.status, 0) + 1
                        stats["skip"] += 1
                        symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                        print(f"[{completed}/{len(problems)}] S{symbol} {result.problem} (cached)")
                    else:
                        stats[result.status] = stats.get(result.status, 0) + 1
                        symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                        print(f"[{completed}/{len(problems)}] {symbol} {result.problem} ({result.time_s:.2f}s)")

                    log_file.write(f"PROGRESS:{completed}:{len(problems)}:{stats['proof']}:{stats['timeout']}\n")
                    log_file.flush()
                    sys.stdout.flush()
                    export_benchmark_progress(base_dir, prover, preset_name, stats, completed, len(problems))
                except Exception as e:
                    print(f"ERROR: {e}")
                    stats["error"] += 1
    else:
        # Sequential execution
        for i, item in enumerate(work_items, 1):
            # Periodic garbage collection to prevent OOM
            if i % 100 == 0:
                import gc
                gc.collect()

            try:
                status, result = _run_single_problem(item)
                if status == "skip":
                    stats[result.status] = stats.get(result.status, 0) + 1
                    stats["skip"] += 1
                    symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                    print(f"[{i}/{len(problems)}] S{symbol} {result.problem} (cached)")
                else:
                    stats[result.status] = stats.get(result.status, 0) + 1
                    symbol = {"proof": "+", "saturated": "~", "timeout": "T", "error": "!"}[result.status]
                    print(f"[{i}/{len(problems)}] {symbol} {result.problem} ({result.time_s:.2f}s)")

                log_file.write(f"PROGRESS:{i}:{len(problems)}:{stats['proof']}:{stats['timeout']}\n")
                log_file.flush()
                sys.stdout.flush()
                export_benchmark_progress(base_dir, prover, preset_name, stats, i, len(problems))
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
    print(f"Results: +{stats['proof']} ~{stats['saturated']} T{stats['timeout']}{skip_str} ({proof_rate:.1f}% proofs)")

    return stats


def get_available_provers(base_dir: Path) -> dict:
    """Get available provers and their configs."""
    provers = {}

    # proofatlas is always available (Python bindings)
    proofatlas_config_path = base_dir / "configs" / "proofatlas.json"
    if proofatlas_config_path.exists():
        provers["proofatlas"] = {
            "config": load_config(proofatlas_config_path),
            "binary": None,  # Uses Python bindings
        }

    # Check for vampire
    vampire_config_path = base_dir / "configs" / "vampire.json"
    if vampire_config_path.exists():
        vampire_config = load_config(vampire_config_path)
        vampire_binary = base_dir / vampire_config["paths"]["binary"]
        if vampire_binary.exists():
            provers["vampire"] = {
                "config": vampire_config,
                "binary": vampire_binary,
            }

    # Check for spass
    spass_config_path = base_dir / "configs" / "spass.json"
    if spass_config_path.exists():
        spass_config = load_config(spass_config_path)
        spass_binary = base_dir / spass_config["paths"]["binary"]
        if spass_binary.exists():
            provers["spass"] = {
                "config": spass_config,
                "binary": spass_binary,
            }

    return provers


def main():
    parser = argparse.ArgumentParser(description="Benchmark and train theorem provers")
    parser.add_argument("--config", nargs="*",
                       help="Config(s) to run (default: all)")
    parser.add_argument("--problem-set",
                       help="Problem set from tptp.json (default: from config)")
    parser.add_argument("--retrain", action="store_true",
                       help="Retrain models even if cached weights exist")
    parser.add_argument("--rerun", action="store_true",
                       help="Re-evaluate problems even if cached results exist")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1)")

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

    # Get available provers
    available_provers = get_available_provers(base_dir)
    if not available_provers:
        print("Error: No provers available")
        sys.exit(1)

    # Only run proofatlas
    if "proofatlas" not in available_provers:
        print("Error: proofatlas not available")
        sys.exit(1)

    prover_info = available_provers["proofatlas"]
    presets = prover_info["config"].get("presets", {})

    # Build list of runs
    runs = []

    if args.config:
        # Run specified configs
        for preset_name in args.config:
            if preset_name not in presets:
                print(f"Error: Unknown config '{preset_name}'")
                print(f"Use --list to see available configs")
                sys.exit(1)
            runs.append({
                "prover": "proofatlas",
                "preset_name": preset_name,
                "preset": presets[preset_name],
                "binary": prover_info["binary"],
                "config": prover_info["config"],
            })
    else:
        # Default: run all presets
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

    log_file_path = get_log_file(base_dir)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Double fork to fully daemonize (survives terminal close, SSH disconnect)
    # Use a pipe to communicate grandchild PID back to parent
    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # First parent: wait for intermediate child and read grandchild PID
        os.close(write_fd)
        os.waitpid(pid, 0)
        grandchild_pid = int(os.read(read_fd, 32).decode().strip())
        os.close(read_fd)

        # Job status is saved by grandchild - just wait briefly for it
        time.sleep(0.1)
        prover_names = sorted(set(r["prover"] for r in runs))
        print(f"Started job (PID: {grandchild_pid})")
        print(f"Provers: {', '.join(prover_names)}, Configs: {len(runs)}, Problems: {len(problems)}")
        print("Use --status to check, --kill to stop")
        return

    # First child: become session leader and fork again
    os.close(read_fd)
    os.setsid()
    import signal
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    pid2 = os.fork()
    if pid2 > 0:
        # Intermediate child: send grandchild PID to parent and exit
        os.write(write_fd, f"{pid2}\n".encode())
        os.close(write_fd)
        os._exit(0)

    # Second child (grandchild): the actual daemon
    os.close(write_fd)

    # Close stdin and redirect stdout/stderr to log file early
    # so any errors during startup are captured
    sys.stdin.close()
    os.close(0)
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    # Log startup info
    print(f"Benchmark daemon started (PID: {os.getpid()})")
    print(f"Working directory: {base_dir}")
    print(f"Configs: {len(runs)}, Problems: {len(problems)}")
    sys.stdout.flush()

    # Clear any stale PID tracking and save job status
    clear_pids(base_dir)
    job_file_error = None
    try:
        save_job_status(base_dir, os.getpid(), sys.argv, len(runs))
        print(f"Job file saved: {get_job_file(base_dir)}")
    except Exception as e:
        job_file_error = str(e)
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

    # Extract problem names for filtering traces
    problem_names = {p.stem for p in problems}
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

            # Training only supported for proofatlas
            ml = _get_ml()
            if prover == "proofatlas" and ml.is_learned_selector(preset):
                model_name = ml.get_model_name(preset)
                embedding_type = ml.get_embedding_type(preset)
                print(f"[{preset_name}] Learned selector: {model_name} (embedding: {embedding_type})")
                sys.stdout.flush()

                weights_dir = base_dir / ".weights"
                existing_weights = ml.find_weights(weights_dir, preset)

                if existing_weights and not args.retrain:
                    print(f"[{preset_name}] Found cached weights: {existing_weights}")
                    weights_path = existing_weights
                else:
                    if existing_weights:
                        print(f"[{preset_name}] --retrain specified, will retrain (existing: {existing_weights})")
                    else:
                        print(f"[{preset_name}] No cached weights found in {weights_dir}")
                    sys.stdout.flush()

                    # First collect traces with age_weight if none exist
                    traces_dir = base_dir / ".data" / "traces"
                    trace_preset_dir = traces_dir / trace_preset
                    proofatlas_presets = prover_config.get("presets", {})

                    existing_traces = list(trace_preset_dir.glob("*.json")) if trace_preset_dir.exists() else []
                    if not existing_traces:
                        print(f"[{preset_name}] No traces found in {trace_preset_dir}")
                        print(f"[{preset_name}] Collecting traces using age_weight baseline...")
                        sys.stdout.flush()
                        trace_source_preset = proofatlas_presets.get(trace_preset, preset)
                        run_evaluation(
                            base_dir, problems, tptp_root,
                            prover="proofatlas", preset=trace_source_preset,
                            log_file=sys.stdout,
                            preset_name=trace_preset, trace_preset=trace_preset,
                            rerun=True,  # Always run for trace collection
                        )
                        existing_traces = list(trace_preset_dir.glob("*.json")) if trace_preset_dir.exists() else []
                        print(f"[{preset_name}] Trace collection complete: {len(existing_traces)} traces")
                    else:
                        print(f"[{preset_name}] Found {len(existing_traces)} existing traces in {trace_preset_dir}")
                    sys.stdout.flush()

                    # Train with lazy loading (traces loaded one at a time)
                    print(f"[{preset_name}] Starting training (lazy loading)...")
                    sys.stdout.flush()
                    try:
                        weights_path = ml.run_training(
                            preset=preset,
                            trace_dir=trace_preset_dir,
                            weights_dir=weights_dir,
                            configs_dir=base_dir / "configs",
                            problem_names=problem_names,
                            web_data_dir=base_dir / "web" / "data",
                            log_file=sys.stdout,
                        )
                        print(f"[{preset_name}] Training complete, weights saved to: {weights_path}")
                    except ValueError as e:
                        print(f"[{preset_name}] WARNING: {e}, falling back to age_weight")
                        current_preset = proofatlas_presets.get(trace_preset, preset)
                        weights_path = None
                    sys.stdout.flush()
            elif prover == "proofatlas":
                print(f"[{preset_name}] Heuristic selector (no training needed)")
                sys.stdout.flush()
            else:
                print(f"[{preset_name}] External prover: {prover}")
                sys.stdout.flush()

            # Run evaluation
            print(f"[{preset_name}] Starting evaluation on {len(problems)} problems...")
            sys.stdout.flush()
            run_evaluation(
                base_dir, problems, tptp_root,
                prover=prover, preset=current_preset,
                log_file=sys.stdout,
                preset_name=preset_name, weights_path=str(weights_path) if weights_path else None,
                binary=binary, trace_preset=trace_preset,
                rerun=args.rerun, n_jobs=args.n_jobs,
            )
            print(f"[{preset_name}] Evaluation complete")
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
