#!/usr/bin/env python3
"""
Train and evaluate all ML configs in sequence, then push results.

USAGE:
    proofatlas-pipeline                                    # Train+bench all ML configs
    proofatlas-pipeline --configs gcn_mlp gcn_attention     # Specific configs
    proofatlas-pipeline --status                            # Check pipeline status
    proofatlas-pipeline --kill                              # Stop pipeline

    # Pass-through flags:
    proofatlas-pipeline --use-cuda --batch-size 512K --cpu-workers 4 --gpu-workers 2
    proofatlas-pipeline --rerun                             # Re-run existing bench results
    proofatlas-pipeline --max-epochs 4                      # Override training epochs

For each config:
1. Runs proofatlas-train --foreground (training)
2. Runs proofatlas-bench --foreground (evaluation)
3. After all configs: git commit + push results
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

PIPELINE_PREFIX = "pipeline"


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


def get_ml_configs(base_dir: Path) -> list[str]:
    """Get all ML preset names (those with encoder + scorer keys)."""
    config = load_config(base_dir / "configs" / "proofatlas.json")
    presets = config.get("presets", {})
    return sorted(
        name for name, preset in presets.items()
        if preset.get("encoder") and preset.get("scorer")
    )


def handle_status(base_dir: Path):
    """Print current pipeline job status."""
    from bench_jobs import is_process_running, get_log_file

    job_file = base_dir / f".data/{PIPELINE_PREFIX}_job.json"
    if not job_file.exists():
        print("No pipeline job currently running.")
        return

    try:
        with open(job_file) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("No pipeline job currently running.")
        return

    if not is_process_running(job["pid"]):
        print("No pipeline job currently running (stale job file).")
        job_file.unlink(missing_ok=True)
        return

    start = datetime.fromisoformat(job["start_time"])
    elapsed = datetime.now() - start
    hours = elapsed.seconds // 3600
    minutes = (elapsed.seconds % 3600) // 60

    print(f"Pipeline job running (PID: {job['pid']})")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m ago)")
    print(f"  Configs: {', '.join(job.get('configs', []))}")

    # Parse log file for progress
    log_file = get_log_file(base_dir, prefix=PIPELINE_PREFIX)
    if log_file.exists():
        last_marker = None
        with open(log_file) as f:
            for line in f:
                if line.startswith("PIPELINE:"):
                    last_marker = line.strip()

        if last_marker:
            parts = last_marker.split(":")
            if len(parts) >= 5:
                step, total, config_name, phase = parts[1], parts[2], parts[3], parts[4]
                print(f"  Progress: {step}/{total} - {config_name} ({phase})")
        else:
            print("  Starting...")
    else:
        print("  Starting...")

    print(f"\nTo stop: proofatlas-pipeline --kill")


def handle_kill(base_dir: Path):
    """Kill the running pipeline job and any child processes."""
    from bench_jobs import is_process_running, kill_tracked_pids, get_pid_file

    job_file = base_dir / f".data/{PIPELINE_PREFIX}_job.json"
    if not job_file.exists():
        print("No pipeline job to kill.")
        return

    try:
        with open(job_file) as f:
            job = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("No pipeline job to kill (invalid job file).")
        job_file.unlink(missing_ok=True)
        return

    # Clear job file first
    job_file.unlink(missing_ok=True)

    # Kill child process if tracked
    child_pid = job.get("child_pid")
    if child_pid:
        try:
            os.kill(child_pid, signal.SIGKILL)
            log(f"Killed child process (PID: {child_pid})")
        except (OSError, ProcessLookupError):
            pass

    # Kill main daemon process
    pid = job.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"Killed pipeline process (PID: {pid})")
        except (OSError, ProcessLookupError):
            log(f"Pipeline process (PID: {pid}) already stopped")

    # Kill any tracked sub-processes
    killed = kill_tracked_pids(base_dir)
    if killed:
        log(f"Killed {killed} sub-processes")

    pid_file = get_pid_file(base_dir, prefix=PIPELINE_PREFIX)
    pid_file.unlink(missing_ok=True)


def update_child_pid(job_file: Path, child_pid: int):
    """Update the job file with the current child process PID."""
    try:
        with open(job_file) as f:
            job = json.load(f)
        job["child_pid"] = child_pid
        with open(job_file, "w") as f:
            json.dump(job, f, indent=2)
    except (json.JSONDecodeError, IOError):
        pass


def run_pipeline(configs: list[str], args, base_dir: Path):
    """Run train→bench for each config, then push results."""
    total_steps = len(configs)
    completed = []
    failed = []
    job_file = base_dir / f".data/{PIPELINE_PREFIX}_job.json"

    for step, config_name in enumerate(configs, 1):
        # --- Training phase ---
        print(f"\n{'='*60}")
        print(f"PIPELINE:  {config_name}  ({step}/{total_steps})")
        print(f"{'='*60}")

        # Emit marker for --status parsing
        print(f"PIPELINE:{step}:{total_steps}:{config_name}:training")
        sys.stdout.flush()

        log(f"[{config_name}] Starting training...")

        train_cmd = [
            sys.executable, "-m", "proofatlas.cli.train",
            "--config", config_name,
            "--foreground",
        ]
        if args.use_cuda:
            train_cmd.append("--use-cuda")
        if args.batch_size:
            train_cmd.extend(["--batch-size", args.batch_size])
        if args.cpu_workers is not None:
            train_cmd.extend(["--cpu-workers", str(args.cpu_workers)])
        if args.gpu_workers is not None:
            train_cmd.extend(["--gpu-workers", str(args.gpu_workers)])
        if args.accumulate_batches is not None:
            train_cmd.extend(["--accumulate-batches", str(args.accumulate_batches)])
        if args.max_epochs is not None:
            train_cmd.extend(["--max-epochs", str(args.max_epochs)])

        proc = subprocess.Popen(
            train_cmd,
            stdout=sys.stdout, stderr=sys.stderr,
            cwd=str(base_dir),
        )
        update_child_pid(job_file, proc.pid)
        rc = proc.wait()

        if rc != 0:
            log(f"[{config_name}] Training failed (exit code {rc})")
            failed.append(config_name)
            sys.stdout.flush()
            continue

        log(f"[{config_name}] Training complete")
        sys.stdout.flush()

        # --- Evaluation phase ---
        print(f"PIPELINE:{step}:{total_steps}:{config_name}:evaluating")
        sys.stdout.flush()

        log(f"[{config_name}] Starting evaluation...")

        bench_cmd = [
            sys.executable, "-m", "proofatlas.cli.bench",
            "--config", config_name,
            "--foreground",
        ]
        if args.cpu_workers is not None:
            bench_cmd.extend(["--cpu-workers", str(args.cpu_workers)])
        if args.gpu_workers is not None:
            bench_cmd.extend(["--gpu-workers", str(args.gpu_workers)])
        if args.rerun:
            bench_cmd.append("--rerun")

        proc = subprocess.Popen(
            bench_cmd,
            stdout=sys.stdout, stderr=sys.stderr,
            cwd=str(base_dir),
        )
        update_child_pid(job_file, proc.pid)
        rc = proc.wait()

        if rc != 0:
            log(f"[{config_name}] Evaluation failed (exit code {rc})")
            failed.append(config_name)
            sys.stdout.flush()
            continue

        log(f"[{config_name}] Evaluation complete")
        completed.append(config_name)
        sys.stdout.flush()

    # --- Push results ---
    if completed:
        print(f"\nPIPELINE:{total_steps}:{total_steps}:push:pushing")
        sys.stdout.flush()

        log(f"Pushing results for: {', '.join(completed)}")

        subprocess.run(["git", "add", "web/data/"], cwd=str(base_dir))
        commit_msg = f"[skip ci] Update results: {', '.join(completed)}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(base_dir), capture_output=True, text=True,
        )
        if result.returncode == 0:
            push_result = subprocess.run(
                ["git", "push"], cwd=str(base_dir),
                capture_output=True, text=True,
            )
            if push_result.returncode == 0:
                log("Results pushed successfully")
            else:
                log(f"Git push failed: {push_result.stderr.strip()}")
        else:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                log("No new results to commit")
            else:
                log(f"Git commit failed: {result.stderr.strip()}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Pipeline complete")
    if completed:
        print(f"  Completed: {', '.join(completed)}")
    if failed:
        print(f"  Failed:    {', '.join(failed)}")
    print(f"{'='*60}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ML configs in sequence")
    parser.add_argument("--configs", nargs="*",
                       help="Config names to run (default: all ML configs)")

    # Job management
    parser.add_argument("--status", action="store_true",
                       help="Check pipeline status")
    parser.add_argument("--kill", action="store_true",
                       help="Stop pipeline")

    # Pass-through training flags
    parser.add_argument("--use-cuda", action="store_true",
                       help="Use CUDA for training")
    parser.add_argument("--batch-size", type=str, default=None,
                       help="Max batch tensor bytes (e.g., 16M, 512K)")
    parser.add_argument("--cpu-workers", type=int, default=None,
                       help="Number of CPU workers")
    parser.add_argument("--gpu-workers", type=int, default=None,
                       help="Number of GPU workers")
    parser.add_argument("--accumulate-batches", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--max-epochs", type=int, default=None,
                       help="Override max training epochs")

    # Pass-through bench flags
    parser.add_argument("--rerun", action="store_true",
                       help="Re-evaluate cached bench results")

    args = parser.parse_args()
    base_dir = find_project_root()

    # Handle --status and --kill
    if args.status:
        handle_status(base_dir)
        return

    if args.kill:
        handle_kill(base_dir)
        return

    # Resolve configs
    if args.configs:
        configs = args.configs
        # Validate config names
        all_ml = get_ml_configs(base_dir)
        prover_config = load_config(base_dir / "configs" / "proofatlas.json")
        all_presets = prover_config.get("presets", {})
        for c in configs:
            if c not in all_presets:
                print(f"Error: Unknown config '{c}'")
                print(f"Available ML configs: {', '.join(all_ml)}")
                sys.exit(1)
            preset = all_presets[c]
            if not (preset.get("encoder") and preset.get("scorer")):
                print(f"Error: Config '{c}' is not an ML selector (no encoder/scorer)")
                sys.exit(1)
    else:
        configs = get_ml_configs(base_dir)
        if not configs:
            print("Error: No ML configs found in proofatlas.json")
            sys.exit(1)

    log(f"Pipeline configs: {', '.join(configs)}")

    # Check for existing pipeline job
    from bench_jobs import is_process_running
    job_file = base_dir / f".data/{PIPELINE_PREFIX}_job.json"
    if job_file.exists():
        try:
            with open(job_file) as f:
                existing = json.load(f)
            if is_process_running(existing["pid"]):
                print(f"Error: Pipeline already running (PID: {existing['pid']})")
                print("Use --status or --kill")
                sys.exit(1)
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # Daemonize
    from bench_jobs import daemonize, get_log_file, clear_pids

    log_file_path = get_log_file(base_dir, prefix=PIPELINE_PREFIX)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    is_daemon, grandchild_pid = daemonize(log_file_path)

    if not is_daemon:
        # Parent process: save job status and exit
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "pid": grandchild_pid,
            "configs": configs,
            "start_time": datetime.now().isoformat(),
            "log_file": str(log_file_path),
        }
        with open(job_file, "w") as f:
            json.dump(job, f, indent=2)

        log(f"Pipeline started in background (PID: {grandchild_pid})")
        log(f"  Configs: {', '.join(configs)}")
        log(f"  Log: tail -f {log_file_path}")
        log(f"  Status: proofatlas-pipeline --status")
        log(f"  Stop:   proofatlas-pipeline --kill")
        return

    # --- Daemon process from here ---

    # Set up signal handlers
    def signal_handler(signum, frame):
        sig_names = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT",
                     signal.SIGQUIT: "SIGQUIT", signal.SIGABRT: "SIGABRT"}
        sig_name = sig_names.get(signum, f"signal {signum}")
        print(f"\nRECEIVED {sig_name} - exiting")
        sys.stdout.flush()
        job_file.unlink(missing_ok=True)
        clear_pids(base_dir)
        sys.stdout.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)

    try:
        run_pipeline(configs, args, base_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        job_file.unlink(missing_ok=True)
        clear_pids(base_dir)
        sys.stdout.close()
    os._exit(0)


if __name__ == "__main__":
    main()
