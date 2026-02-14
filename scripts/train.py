#!/usr/bin/env python3
"""
Train ML models for ProofAtlas clause selection.

USAGE:
    proofatlas-train --config gcn_mlp              # Train model (daemonizes)
    proofatlas-train --config gcn_mlp --use-cuda   # Train on GPU
    proofatlas-train --config gcn_mlp --foreground  # Run in foreground (for pipeline/debugging)
    proofatlas-train --status                       # Check job status
    proofatlas-train --kill                         # Stop training job

    # Worker pipeline flags:
    proofatlas-train --config gcn_mlp --cpu-workers 4       # Parallel batch prep
    proofatlas-train --config gcn_mlp --gpu-workers 2       # Multi-GPU DDP
    proofatlas-train --config gcn_mlp --batch-size 16M      # Max tensor bytes per batch
    proofatlas-train --config gcn_mlp --accumulate-batches 4  # Gradient accumulation

This script:
1. Trains the model (requires pre-collected .npz traces)
2. Saves weights to .weights/

Traces must be collected first via:
    proofatlas-bench --config age_weight --trace

After training, run evaluation with:
    proofatlas-bench --config gcn_mlp
"""

import os

# Limit torch/MKL threads before any imports that load libtorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Suppress noisy tokenizer parallelism warning in forked workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

TRAIN_PREFIX = "train"


def parse_byte_size(s: str) -> int:
    """Parse a human-readable byte size string (e.g. '16M', '512K', '1G') to bytes.

    Supports suffixes: K/KB (KiB), M/MB (MiB), G/GB (GiB). Plain integers as bytes.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty byte size string")

    suffixes = {
        'K': 1024, 'KB': 1024,
        'M': 1024 ** 2, 'MB': 1024 ** 2,
        'G': 1024 ** 3, 'GB': 1024 ** 3,
    }

    upper = s.upper()
    for suffix, multiplier in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if upper.endswith(suffix):
            num_str = s[:len(s) - len(suffix)]
            try:
                return int(float(num_str) * multiplier)
            except ValueError:
                raise ValueError(f"invalid byte size: {s!r}")

    try:
        return int(s)
    except ValueError:
        raise ValueError(f"invalid byte size: {s!r}")


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


def check_cuda_device_count() -> int:
    """Check available CUDA GPU count via subprocess (avoids CUDA context in main)."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import torch; print(torch.cuda.device_count())'],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def validate_args(args, config):
    """Validate CLI arguments and return resolved values.

    Returns (batch_size, accumulate_batches) after validation.
    """
    errors = []

    if args.cpu_workers < 0:
        errors.append(f"--cpu-workers must be >= 0 (got {args.cpu_workers})")
    if args.gpu_workers < 1:
        errors.append(f"--gpu-workers must be >= 1 (got {args.gpu_workers})")

    # Batch size (bytes)
    batch_size = None
    if args.batch_size is not None:
        try:
            batch_size = parse_byte_size(args.batch_size)
        except ValueError as e:
            errors.append(f"--batch-size: {e}")
        else:
            if batch_size <= 0:
                errors.append(f"--batch-size must be > 0 (got {args.batch_size})")

    # Resolve accumulate_batches
    accumulate_batches = args.accumulate_batches
    if accumulate_batches is not None and accumulate_batches <= 0:
        errors.append(f"--accumulate-batches must be > 0 (got {accumulate_batches})")

    # GPU checks
    if args.gpu_workers > 1:
        available = check_cuda_device_count()
        if available == 0:
            errors.append("--gpu-workers > 1 requires CUDA GPUs, but none detected")
        elif args.gpu_workers > available:
            errors.append(
                f"--gpu-workers {args.gpu_workers} exceeds available GPUs ({available})"
            )

    # Accumulation must cover all GPU workers
    if accumulate_batches is not None and accumulate_batches < args.gpu_workers:
        errors.append(
            f"--accumulate-batches ({accumulate_batches}) must be >= "
            f"--gpu-workers ({args.gpu_workers})"
        )

    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if batch_size is not None:
        if batch_size >= 1024 * 1024:
            log(f"Batch size: {batch_size // (1024*1024)}M bytes")
        else:
            log(f"Batch size: {batch_size // 1024}K bytes")

    return batch_size, accumulate_batches


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

    # Kill main process and its process group (DDP/DataLoader workers)
    pid = job.get("pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGKILL)
            log(f"Killed training process group (PGID: {pid})")
        except (OSError, ProcessLookupError):
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


def _ddp_train_worker(rank, preset, trace_dir, weights_dir, configs_dir, problem_names,
                      web_data_dir, cpu_workers, world_size, batch_size,
                      accumulate_batches, max_epochs):
    """DDP worker function at module level so it can be pickled by mp.spawn."""
    from proofatlas.ml.training import run_training

    run_training(
        preset=preset,
        trace_dir=trace_dir,
        weights_dir=weights_dir,
        configs_dir=configs_dir,
        problem_names=problem_names,
        web_data_dir=web_data_dir,
        log_file=sys.stdout,
        cpu_workers=cpu_workers,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        accumulate_batches=accumulate_batches,
        max_epochs=max_epochs,
    )


def run_training_job(args, base_dir, preset, problems, tptp_config):
    """Run the actual training (called directly or as daemon)."""
    from proofatlas.ml.training import run_training
    from proofatlas.ml.weights import get_model_name

    model_name = get_model_name(preset)
    weights_dir = Path(args.weights_dir) if args.weights_dir else base_dir / ".weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Validate CLI args
    batch_size, accumulate_batches = validate_args(args, preset)

    # Resolve trace directory
    trace_preset = preset.get("traces") or args.config
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        trace_dir = base_dir / ".data" / "traces" / trace_preset

    # Check that traces exist
    if not trace_dir.exists() or not any(trace_dir.glob("**/*.npz")):
        log(f"Error: No .npz trace files found in {trace_dir}")
        log(f"Collect traces first with: proofatlas-bench --config age_weight --trace")
        sys.exit(1)

    # Train model
    problem_names = {p.stem for p in problems}
    log("Starting training...")

    # Multi-GPU DDP
    if args.gpu_workers > 1:
        import torch.multiprocessing as mp

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        mp.spawn(
            _ddp_train_worker,
            args=(preset, trace_dir, weights_dir, base_dir / "configs",
                  problem_names, base_dir / "web" / "data", args.cpu_workers,
                  args.gpu_workers, batch_size, accumulate_batches,
                  args.max_epochs),
            nprocs=args.gpu_workers,
            join=True,
        )
    else:
        weights_path = run_training(
            preset=preset,
            trace_dir=trace_dir,
            weights_dir=weights_dir,
            configs_dir=base_dir / "configs",
            problem_names=problem_names,
            web_data_dir=base_dir / "web" / "data",
            log_file=sys.stdout,
            cpu_workers=args.cpu_workers,
            batch_size=batch_size,
            accumulate_batches=accumulate_batches,
            force_cpu=not args.use_cuda,
            max_epochs=args.max_epochs,
        )

    log(f"Training complete! Weights saved to: {weights_dir}")
    log(f"Run evaluation with: proofatlas-bench --config {args.config}")


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
    parser.add_argument("--foreground", action="store_true",
                       help="Run in foreground (skip daemonization, for pipeline/debugging)")
    parser.add_argument("--status", action="store_true",
                       help="Show status of running training job")
    parser.add_argument("--kill", action="store_true",
                       help="Kill running training job")

    # Worker pipeline flags
    parser.add_argument("--cpu-workers", type=int, default=2,
                       help="DataLoader workers for parallel batch preparation (default: 2)")
    parser.add_argument("--gpu-workers", type=int, default=1,
                       help="DDP processes across GPUs (default: 1, no DDP)")
    parser.add_argument("--batch-size", type=str, default=None,
                       help="Max tensor bytes per micro-batch, e.g. 64K (default: 64K)")
    parser.add_argument("--accumulate-batches", type=int, default=None,
                       help="Gradient accumulation steps (default: from config)")
    parser.add_argument("--max-epochs", type=int, default=None,
                       help="Override max training epochs (default: from config)")

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
    from proofatlas.ml.weights import is_learned_selector
    if not is_learned_selector(preset):
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
    from proofatlas.ml.weights import find_weights
    existing_weights = find_weights(weights_dir, preset)
    if existing_weights:
        log(f"Found existing weights: {existing_weights}")
        log("Will overwrite with new training run")

    if args.foreground:
        # Foreground mode: run directly (for pipeline use and debugging)
        run_training_job(args, base_dir, preset, problems, tptp_config)
        return

    # Daemon mode (default): daemonize like bench.py
    from bench_jobs import daemonize, get_log_file

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
        log(f"  Log: tail -f {log_file_path}")
        log(f"  Status: proofatlas-train --status")
        log(f"  Stop:   proofatlas-train --kill")
        return

    # --- Daemon process from here ---

    # Set up signal handlers to log unexpected termination
    from bench_jobs import clear_pids

    def signal_handler(signum, frame):
        sig_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT",
                     signal.SIGQUIT: "SIGQUIT", signal.SIGABRT: "SIGABRT"}
        sig_name = sig_names.get(signum, f"signal {signum}")
        print(f"\nRECEIVED {sig_name} - exiting")
        sys.stdout.flush()
        job_file = base_dir / f".data/{TRAIN_PREFIX}_job.json"
        job_file.unlink(missing_ok=True)
        clear_pids(base_dir)
        sys.stdout.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    try:
        run_training_job(args, base_dir, preset, problems, tptp_config)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up job file on completion
        job_file = base_dir / f".data/{TRAIN_PREFIX}_job.json"
        job_file.unlink(missing_ok=True)
        sys.stdout.close()
    os._exit(0)


if __name__ == "__main__":
    main()
