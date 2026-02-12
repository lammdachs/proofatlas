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

DEFAULT_PREFIX = "pipeline"
PIPELINE_PREFIX = DEFAULT_PREFIX  # May be overridden by --job-prefix


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

    # Kill child process group (includes DataLoader/DDP workers)
    child_pid = job.get("child_pid")
    if child_pid:
        try:
            os.killpg(child_pid, signal.SIGKILL)
            log(f"Killed child process group (PGID: {child_pid})")
        except (OSError, ProcessLookupError):
            # Fall back to killing individual process
            try:
                os.kill(child_pid, signal.SIGKILL)
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


def sync_all_run_results(base_dir: Path):
    """Combine per-problem JSONs from .data/runs/ into single files in web/data/runs/."""
    runs_src = base_dir / ".data" / "runs"
    runs_dst = base_dir / "web" / "data" / "runs"
    runs_dst.mkdir(parents=True, exist_ok=True)

    if not runs_src.exists():
        log("No .data/runs/ directory found — skipping run sync")
        return

    index_entries = []

    for prover_dir in sorted(runs_src.iterdir()):
        if not prover_dir.is_dir():
            continue
        prover = prover_dir.name

        for config_dir in sorted(prover_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            preset = config_dir.name
            run_key = f"{prover}_{preset}"

            results = []
            for problem_file in sorted(config_dir.glob("*.json")):
                try:
                    with open(problem_file) as f:
                        data = json.load(f)
                    results.append({
                        "problem": data.get("problem", problem_file.stem + ".p"),
                        "status": data.get("status", "unknown"),
                        "time_s": data.get("time_s", 0),
                        "timestamp": data.get("timestamp", ""),
                    })
                except (json.JSONDecodeError, IOError):
                    continue

            if not results:
                continue

            combined = {
                "prover": prover,
                "preset": preset,
                "results": results,
            }

            out_path = runs_dst / f"{run_key}.json"
            with open(out_path, "w") as f:
                json.dump(combined, f, separators=(",", ":"))

            index_entries.append(run_key)
            log(f"[{run_key}] Wrote {len(results)} results to web/data/runs/")

    # Write index
    with open(runs_dst / "index.json", "w") as f:
        json.dump({"runs": sorted(index_entries)}, f, indent=2)

    log(f"Run index: {len(index_entries)} configs")


def push_results(base_dir: Path, configs: list[str]) -> bool:
    """Git add, pull, commit, push web/data/. Returns True if pushed successfully.

    Uses file locking to prevent race conditions on shared storage.
    """
    import fcntl

    sync_all_run_results(base_dir)

    lock_file = base_dir / ".data" / "git.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(lock_file, "w") as lock_fd:
            # Acquire exclusive lock (blocking)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            subprocess.run(["git", "add", "web/data/"], cwd=str(base_dir))

            commit_msg = f"[skip ci] Update results: {', '.join(configs)}"
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(base_dir), capture_output=True, text=True,
            )
            if result.returncode != 0:
                if "nothing to commit" in result.stdout + result.stderr:
                    log("No new results to commit")
                    return True
                log(f"Git commit failed: {result.stderr.strip()}")
                return False

            # Pull with rebase to incorporate any remote changes
            pull = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=str(base_dir), capture_output=True, text=True,
            )
            if pull.returncode != 0:
                log(f"Git pull --rebase failed: {pull.stderr.strip()}")
                log("Commit is local — will retry on next push interval")
                return False

            push = subprocess.run(
                ["git", "push"],
                cwd=str(base_dir), capture_output=True, text=True,
            )
            if push.returncode != 0:
                log(f"Git push failed: {push.stderr.strip()}")
                return False

            log(f"Pushed results for: {', '.join(configs)}")
            return True
    except OSError as e:
        log(f"Failed to acquire git lock: {e}")
        return False


def maybe_push(base_dir: Path, unpushed: list[str], last_push_time: float,
               push_interval: float, force: bool = False) -> tuple[list[str], float]:
    """Push if enough time has elapsed or force=True. Returns (remaining_unpushed, new_last_push_time)."""
    if not unpushed:
        return unpushed, last_push_time
    if not force and (time.time() - last_push_time) < push_interval:
        return unpushed, last_push_time

    print(f"PIPELINE:0:0:push:pushing")
    sys.stdout.flush()

    if push_results(base_dir, unpushed):
        return [], time.time()
    else:
        # Keep unpushed list so next push attempt includes them
        return unpushed, last_push_time


def upload_weights(base_dir: Path, configs: list[str]):
    """Upload trained weights for completed configs to a rolling GitHub release.

    Uses file locking for release creation to prevent race conditions on shared storage.
    """
    import fcntl

    weights_dir = base_dir / ".weights"
    if not weights_dir.exists():
        log("No .weights/ directory — skipping weight upload")
        return

    tag = "weights"
    release_name = "Model Weights"
    release_body = (
        f"Rolling backup of trained model weights.\n\n"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Configs: {', '.join(configs)}"
    )

    lock_file = base_dir / ".data" / "gh_release.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Lock only for release creation/check to avoid race condition
    try:
        with open(lock_file, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Ensure the tag and release exist
            check = subprocess.run(
                ["gh", "release", "view", tag],
                cwd=str(base_dir), capture_output=True, text=True,
            )
            if check.returncode != 0:
                log("Creating weights release...")
                result = subprocess.run(
                    ["gh", "release", "create", tag,
                     "--title", release_name,
                     "--notes", release_body,
                     "--latest=false"],
                    cwd=str(base_dir), capture_output=True, text=True,
                )
                if result.returncode != 0:
                    log(f"Failed to create release: {result.stderr.strip()}")
                    return
            else:
                # Update release notes
                subprocess.run(
                    ["gh", "release", "edit", tag,
                     "--notes", release_body],
                    cwd=str(base_dir), capture_output=True, text=True,
                )
    except OSError as e:
        log(f"Failed to acquire release lock: {e}")
        return

    # Collect weight files for completed configs
    assets = []
    for config_name in configs:
        pt_file = weights_dir / f"{config_name}.pt"
        if pt_file.exists():
            assets.append(str(pt_file))
        # Also upload tokenizer dirs (sentence models)
        tok_dir = weights_dir / f"{config_name}_tokenizer"
        if tok_dir.is_dir():
            # Tar the tokenizer dir for upload
            tar_path = weights_dir / f"{config_name}_tokenizer.tar.gz"
            subprocess.run(
                ["tar", "czf", str(tar_path), "-C", str(weights_dir), f"{config_name}_tokenizer"],
                capture_output=True,
            )
            if tar_path.exists():
                assets.append(str(tar_path))

    if not assets:
        log("No weight files found for completed configs")
        return

    # Upload with --clobber to overwrite existing assets
    cmd = ["gh", "release", "upload", tag, "--clobber"] + assets
    result = subprocess.run(
        cmd, cwd=str(base_dir), capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"Weight upload failed: {result.stderr.strip()}")
    else:
        log(f"Uploaded weights: {', '.join(Path(a).name for a in assets)}")

    # Clean up any temp tarballs
    for config_name in configs:
        tar_path = weights_dir / f"{config_name}_tokenizer.tar.gz"
        tar_path.unlink(missing_ok=True)


def run_pipeline(configs: list[str], args, base_dir: Path):
    """Run train→bench for each config, pushing results incrementally."""
    total_steps = len(configs)
    completed = []
    failed = []
    unpushed = []
    last_push_time = time.time()
    push_interval = args.push_interval
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
            start_new_session=True,
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
            start_new_session=True,
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
        unpushed.append(config_name)
        sys.stdout.flush()

        # Push if enough time has elapsed
        unpushed, last_push_time = maybe_push(
            base_dir, unpushed, last_push_time, push_interval,
        )

    # Final push for any remaining results
    unpushed, last_push_time = maybe_push(
        base_dir, unpushed, last_push_time, push_interval, force=True,
    )

    # Upload weights for completed configs
    if completed:
        print(f"PIPELINE:0:0:weights:uploading")
        sys.stdout.flush()
        upload_weights(base_dir, completed)

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
    parser.add_argument("--job-prefix", type=str, default=DEFAULT_PREFIX,
                       help="Prefix for job/log files (for multi-node setups)")

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

    # Push control
    parser.add_argument("--push-interval", type=int, default=300,
                       help="Seconds between result pushes (default: 300 = 5 min)")

    args = parser.parse_args()
    base_dir = find_project_root()

    # Set job prefix for multi-node setups
    global PIPELINE_PREFIX
    PIPELINE_PREFIX = args.job_prefix

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
        # Kill child process group (train/bench and their workers)
        try:
            with open(job_file) as f:
                child_pid = json.load(f).get("child_pid")
            if child_pid:
                os.killpg(child_pid, signal.SIGKILL)
        except Exception:
            pass
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
