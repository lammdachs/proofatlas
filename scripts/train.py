#!/usr/bin/env python3
"""
Train ML models for ProofAtlas clause selection.

USAGE:
    python scripts/train.py --config gcn_mlp              # Train model
    python scripts/train.py --config gcn_mlp --use-cuda    # Train on GPU
    python scripts/train.py --config gcn_mlp --daemon       # Train in background
    python scripts/train.py --status                        # Check job status
    python scripts/train.py --kill                          # Stop training job

This script:
1. Collects proof traces using age_weight baseline (if none exist)
2. Trains the model in a subprocess (CUDA isolation)
3. Saves weights to .weights/

After training, run evaluation with:
    proofatlas-bench --config gcn_mlp
"""

import os

# Limit torch/MKL threads before any imports that load libtorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

TRAIN_PREFIX = "train"


def log(msg: str):
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def find_project_root() -> Path:
    """Find the proofatlas project root."""
    path = Path(__file__).resolve().parent.parent
    if (path / "crates" / "proofatlas").exists():
        return path
    raise RuntimeError(f"Cannot find project root (tried {path})")


def load_config(path: Path) -> dict:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def run_training_subprocess(preset, trace_dir, weights_dir, configs_dir, problem_names,
                             web_data_dir, use_cuda=False):
    """Run training in a subprocess to avoid CUDA context issues.

    CUDA contexts don't survive fork. Running training in a subprocess
    keeps the main process CUDA-free.
    """
    # Serialize arguments to a temp file (problem_names can be large)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        args_file = f.name
        pickle.dump({
            'preset': preset,
            'trace_dir': str(trace_dir),
            'weights_dir': str(weights_dir),
            'configs_dir': str(configs_dir),
            'problem_names': list(problem_names) if problem_names else None,
            'web_data_dir': str(web_data_dir) if web_data_dir else None,
        }, f)

    # Run training script
    script = f'''
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path("{Path(__file__).parent.parent / "python"}").resolve()))

with open("{args_file}", "rb") as f:
    args = pickle.load(f)

from proofatlas import ml
result = ml.run_training(
    preset=args["preset"],
    trace_dir=Path(args["trace_dir"]),
    weights_dir=Path(args["weights_dir"]),
    configs_dir=Path(args["configs_dir"]),
    problem_names=set(args["problem_names"]) if args["problem_names"] else None,
    web_data_dir=Path(args["web_data_dir"]) if args["web_data_dir"] else None,
    log_file=sys.stdout,
)
print("TRAINING_RESULT:" + str(result))
'''

    try:
        env = None
        if not use_cuda:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )

        # Forward output
        if proc.stdout:
            for line in proc.stdout.splitlines():
                if line.startswith("TRAINING_RESULT:"):
                    result_path = Path(line.split(":", 1)[1])
                    return result_path
                else:
                    print(line)

        if proc.returncode != 0:
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    print(f"ERROR: {line}", file=sys.stderr)
            raise ValueError(f"Training failed with exit code {proc.returncode}")

        raise ValueError("Training completed but no result path found")
    finally:
        # Clean up temp file
        Path(args_file).unlink(missing_ok=True)


def collect_traces(base_dir, problems, tptp_root, preset, trace_preset, use_cuda):
    """Collect proof traces using age_weight baseline."""
    from proofatlas import ProofState

    traces_dir = base_dir / ".data" / "traces" / trace_preset
    traces_dir.mkdir(parents=True, exist_ok=True)

    existing = list(traces_dir.glob("*.json"))
    if existing:
        log(f"Found {len(existing)} existing traces in {traces_dir}")
        return

    log(f"No traces found in {traces_dir}")
    log("Collecting traces using age_weight baseline...")

    timeout = preset.get("timeout", 10)
    literal_selection = preset.get("literal_selection", 21)
    age_weight_ratio = preset.get("age_weight_ratio", 0.167)
    memory_limit = preset.get("memory_limit")
    from proofatlas import ml
    collected = 0

    for i, problem in enumerate(problems, 1):
        try:
            with open(problem) as f:
                content = f.read()

            state = ProofState()
            state.add_clauses_from_tptp(content, str(tptp_root), timeout, memory_limit=memory_limit)

            proof_found, status, _, _ = state.run_saturation(
                timeout=float(timeout),
                literal_selection=literal_selection,
                age_weight_ratio=float(age_weight_ratio),
                memory_limit=memory_limit,
            )

            if proof_found:
                trace_json = state.extract_structured_trace(timeout)
                ml.save_trace(base_dir / ".data" / "traces", trace_preset, problem.name, trace_json)
                collected += 1

            if i % 100 == 0:
                log(f"  [{i}/{len(problems)}] {collected} traces collected")

        except Exception:
            continue

    log(f"Trace collection complete: {collected} traces")


def get_problems(base_dir, tptp_config, problem_set):
    """Get list of TPTP problem files matching the problem set criteria."""
    tptp_root = base_dir / tptp_config["paths"]["root"]
    problems_dir = base_dir / tptp_config["paths"]["problems"]

    if not problems_dir.exists():
        print(f"Error: TPTP problems directory not found at {problems_dir}")
        sys.exit(1)

    ps = tptp_config.get("problem_sets", {}).get(problem_set)
    if not ps:
        print(f"Error: Unknown problem set '{problem_set}'")
        sys.exit(1)

    # Simple filter: get all .p files from specified domains
    problems = []
    domains = ps.get("domains")
    exclude_domains = ps.get("exclude_domains", [])

    for p_file in sorted(problems_dir.rglob("*.p")):
        domain = p_file.parent.name
        if domains and domain not in domains:
            continue
        if domain in exclude_domains:
            continue
        problems.append(p_file)

    return problems


def handle_status(base_dir):
    """Print current training job status."""
    from bench_jobs import get_job_status, get_log_file

    job_file_path = base_dir / f".data/{TRAIN_PREFIX}_job.json"
    if not job_file_path.exists():
        print("No training job currently running.")
        return

    try:
        with open(job_file_path) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("No training job currently running.")
        return

    from bench_jobs import is_process_running
    if not is_process_running(job["pid"]):
        print("No training job currently running (stale job file).")
        job_file_path.unlink(missing_ok=True)
        return

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    print(f"Training job running (PID: {job['pid']})")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m ago)")
    print(f"  Config:  {job.get('config', 'unknown')}")

    # Parse log file for progress
    log_file = get_log_file(base_dir, prefix=TRAIN_PREFIX)
    if log_file.exists():
        last_train = None
        with open(log_file) as f:
            for line in f:
                if line.startswith("TRAIN:"):
                    last_train = line.strip()
                elif "Epoch" in line and "train=" in line:
                    last_train = line.strip()

        if last_train:
            print(f"  Progress: {last_train}")
        else:
            print("  Starting...")
    else:
        print("  Starting...")

    print(f"\nTo stop: python scripts/train.py --kill")


def handle_kill(base_dir):
    """Kill the running training job."""
    from bench_jobs import (
        is_process_running, kill_tracked_pids,
        get_pid_file, get_log_file,
    )
    import signal

    job_file_path = base_dir / f".data/{TRAIN_PREFIX}_job.json"
    if not job_file_path.exists():
        print("No training job to kill.")
        return

    try:
        with open(job_file_path) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("No training job to kill (invalid job file).")
        job_file_path.unlink(missing_ok=True)
        return

    # Clear job file first to prevent respawns
    job_file_path.unlink(missing_ok=True)

    # Kill main process
    pid = job.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"Killed training process (PID: {pid})")
        except (OSError, ProcessLookupError):
            log(f"Training process (PID: {pid}) already stopped")

    # Kill tracked sub-processes
    killed = kill_tracked_pids(base_dir)
    if killed:
        log(f"Killed {killed} sub-processes")

    # Clean up PID file
    pid_file = get_pid_file(base_dir, prefix=TRAIN_PREFIX)
    pid_file.unlink(missing_ok=True)


def run_training_job(args, base_dir, preset, problems, tptp_config):
    """Run the actual training (called directly or as daemon)."""
    from proofatlas import ml
    from bench_jobs import register_pid, get_pid_file

    model_name = f"{preset['encoder']}_{preset['scorer']}"
    tptp_root = base_dir / tptp_config["paths"]["root"]
    weights_dir = Path(args.weights_dir) if args.weights_dir else base_dir / ".weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Collect traces if needed
    trace_preset = preset.get("traces") or args.config
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        trace_dir = base_dir / ".data" / "traces" / trace_preset
        existing_traces = list(trace_dir.glob("*.json")) if trace_dir.exists() else []
        if not existing_traces:
            collect_traces(base_dir, problems, tptp_root, preset, trace_preset, args.use_cuda)

    # Train model
    problem_names = {p.stem for p in problems}
    log("Starting training...")

    try:
        weights_path = run_training_subprocess(
            preset=preset,
            trace_dir=trace_dir,
            weights_dir=weights_dir,
            configs_dir=base_dir / "configs",
            problem_names=problem_names,
            web_data_dir=base_dir / "web" / "data",
            use_cuda=args.use_cuda,
        )
        log(f"Training complete! Weights saved to: {weights_path}")
        log(f"Run evaluation with: proofatlas-bench --config {args.config}")
    except ValueError as e:
        log(f"Training failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train ML models for ProofAtlas")
    parser.add_argument("--config",
                       help="Config name (e.g., gcn_mlp)")
    parser.add_argument("--problem-set",
                       help="Problem set from tptp.json (default: from config)")
    parser.add_argument("--use-cuda", action="store_true",
                       help="Use CUDA for training")
    parser.add_argument("--trace-dir",
                       help="Directory containing training traces (overrides auto-collection)")
    parser.add_argument("--weights-dir",
                       help="Where to save trained weights (default: .weights/)")
    parser.add_argument("--daemon", action="store_true",
                       help="Run training in background (survives SSH disconnect)")
    parser.add_argument("--status", action="store_true",
                       help="Show status of running training job")
    parser.add_argument("--kill", action="store_true",
                       help="Kill running training job")

    args = parser.parse_args()
    base_dir = find_project_root()

    # Handle --status and --kill without requiring --config
    if args.status:
        handle_status(base_dir)
        return

    if args.kill:
        handle_kill(base_dir)
        return

    if not args.config:
        parser.error("--config is required (unless using --status or --kill)")

    # Load configs
    prover_config = load_config(base_dir / "configs" / "proofatlas.json")
    tptp_config = load_config(base_dir / "configs" / "tptp.json")

    presets = prover_config.get("presets", {})
    if args.config not in presets:
        print(f"Error: Unknown config '{args.config}'")
        print(f"Available: {', '.join(sorted(presets.keys()))}")
        sys.exit(1)

    preset = presets[args.config]
    from proofatlas import ml
    if not ml.is_learned_selector(preset):
        print(f"Error: Config '{args.config}' is not an ML selector (no encoder/scorer)")
        sys.exit(1)

    model_name = f"{preset['encoder']}_{preset['scorer']}"
    log(f"Training model: {model_name}")

    # Determine problem set
    problem_set = args.problem_set
    if problem_set is None:
        problem_set = tptp_config.get("defaults", {}).get("problem_set")
        if problem_set is None:
            print("Error: No --problem-set specified and no default in tptp.json")
            sys.exit(1)

    problems = get_problems(base_dir, tptp_config, problem_set)
    log(f"Problem set '{problem_set}': {len(problems)} problems")

    # Check for existing weights
    weights_dir = Path(args.weights_dir) if args.weights_dir else base_dir / ".weights"
    existing_weights = ml.find_weights(weights_dir, preset)
    if existing_weights:
        log(f"Found existing weights: {existing_weights}")
        log("Will overwrite with new training run")

    if args.daemon:
        from bench_jobs import daemonize, get_log_file, save_job_status

        log_file_path = get_log_file(base_dir, prefix=TRAIN_PREFIX)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        is_daemon, grandchild_pid = daemonize(log_file_path)

        if not is_daemon:
            # Parent process: save job status and exit
            job_file = base_dir / f".data/{TRAIN_PREFIX}_job.json"
            job_file.parent.mkdir(parents=True, exist_ok=True)
            job = {
                "pid": grandchild_pid,
                "config": args.config,
                "start_time": datetime.now().isoformat(),
                "log_file": str(log_file_path),
            }
            with open(job_file, "w") as f:
                json.dump(job, f, indent=2)

            log(f"Training started in background (PID: {grandchild_pid})")
            log(f"  Log: {log_file_path}")
            log(f"  Status: python scripts/train.py --status")
            log(f"  Stop:   python scripts/train.py --kill")
            return

        # Daemon process: run training
        run_training_job(args, base_dir, preset, problems, tptp_config)

        # Clean up job file on completion
        job_file = base_dir / f".data/{TRAIN_PREFIX}_job.json"
        job_file.unlink(missing_ok=True)
    else:
        # Foreground mode
        run_training_job(args, base_dir, preset, problems, tptp_config)


if __name__ == "__main__":
    main()
