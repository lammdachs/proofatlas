#!/usr/bin/env python3
"""
Train ML models for ProofAtlas clause selection.

USAGE:
    python scripts/train.py --config gcn_mlp              # Train model
    python scripts/train.py --config gcn_mlp --use-cuda    # Train on GPU
    python scripts/train.py --config gcn_mlp --daemon       # Train in background
    python scripts/train.py --status                        # Check job status
    python scripts/train.py --kill                          # Stop training job

    # Worker pipeline flags:
    python scripts/train.py --config gcn_mlp --cpu-workers 4       # Parallel batch prep
    python scripts/train.py --config gcn_mlp --gpu-workers 2       # Multi-GPU DDP
    python scripts/train.py --config gcn_mlp --batch-size 16M      # Max batch tensor bytes
    python scripts/train.py --config gcn_mlp --accumulate-batches 4  # Gradient accumulation

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

import argparse
import json
import subprocess
import sys
import time
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


def parse_byte_size(s: str) -> int:
    """Parse a human-readable byte size string.

    Supports suffixes: K/KB, M/MB, G/GB (case-insensitive).
    Plain integers are treated as bytes.

    Examples:
        "16M" -> 16777216
        "512K" -> 524288
        "2G" -> 2147483648
        "1048576" -> 1048576
    """
    s = s.strip().upper()
    multipliers = {
        'K': 1024,
        'KB': 1024,
        'M': 1024 ** 2,
        'MB': 1024 ** 2,
        'G': 1024 ** 3,
        'GB': 1024 ** 3,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            num = s[:-len(suffix)].strip()
            try:
                return int(float(num) * mult)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid byte size: '{s}'")

    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid byte size: '{s}'. Use suffixes like 16M, 512K, 2G"
        )


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.1f}G"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.1f}M"
    elif n >= 1024:
        return f"{n / 1024:.1f}K"
    else:
        return f"{n}B"


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

    Returns (max_batch_bytes, accumulate_batches) after validation.
    """
    errors = []

    if args.cpu_workers < 0:
        errors.append(f"--cpu-workers must be >= 0 (got {args.cpu_workers})")
    if args.gpu_workers < 1:
        errors.append(f"--gpu-workers must be >= 1 (got {args.gpu_workers})")

    # Parse batch size
    max_batch_bytes = None
    if args.batch_size is not None:
        max_batch_bytes = parse_byte_size(args.batch_size)
        if max_batch_bytes <= 0:
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

    # Log effective batch info
    if max_batch_bytes is not None:
        eff_accum = accumulate_batches or 1
        effective = max_batch_bytes * eff_accum * args.gpu_workers
        log(f"Effective batch: {_fmt_bytes(max_batch_bytes)} x {eff_accum} accum "
            f"x {args.gpu_workers} GPUs = ~{_fmt_bytes(effective)}")

    return max_batch_bytes, accumulate_batches


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
    from proofatlas.ml.training import run_training
    from proofatlas.ml.weights import get_model_name

    model_name = get_model_name(preset)
    weights_dir = Path(args.weights_dir) if args.weights_dir else base_dir / ".weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Validate CLI args
    max_batch_bytes, accumulate_batches = validate_args(args, preset)

    # Resolve trace directory
    trace_preset = preset.get("traces") or args.config
    if args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        trace_dir = base_dir / ".data" / "traces" / trace_preset

    # Check that traces exist
    if not trace_dir.exists() or not any(trace_dir.glob("*.npz")):
        log(f"Error: No .npz trace files found in {trace_dir}")
        log(f"Collect traces first with: proofatlas-bench --config age_weight --trace")
        sys.exit(1)

    # Train model
    problem_names = {p.stem for p in problems}
    log("Starting training...")

    # Multi-GPU DDP
    if args.gpu_workers > 1:
        import torch.multiprocessing as mp

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        def train_worker(rank):
            run_training(
                preset=preset,
                trace_dir=trace_dir,
                weights_dir=weights_dir,
                configs_dir=base_dir / "configs",
                problem_names=problem_names,
                web_data_dir=base_dir / "web" / "data",
                log_file=sys.stdout,
                cpu_workers=args.cpu_workers,
                rank=rank,
                world_size=args.gpu_workers,
                max_batch_bytes=max_batch_bytes,
                accumulate_batches=accumulate_batches,
            )

        mp.spawn(train_worker, nprocs=args.gpu_workers, join=True)
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
            max_batch_bytes=max_batch_bytes,
            accumulate_batches=accumulate_batches,
            force_cpu=not args.use_cuda,
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
    parser.add_argument("--daemon", action="store_true",
                       help="Run training in background (survives SSH disconnect)")
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
                       help="Max collated tensor bytes per micro-batch (default: 16M). "
                            "Supports suffixes: K, M, G (e.g., 16M, 512K, 2G)")
    parser.add_argument("--accumulate-batches", type=int, default=None,
                       help="Gradient accumulation steps (default: from config)")

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
