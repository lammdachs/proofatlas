"""Job and daemon management for proofatlas-bench.

Handles PID tracking, job status persistence, daemonization,
job status display, and result syncing/pushing. Only depends on stdlib.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Job file paths (relative to project root)
JOB_FILE = ".data/bench_job.json"
LOG_FILE = ".data/bench.log"
PID_FILE = ".data/bench_pids.txt"


def log(msg: str):
    """Print a log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_job_file(base_dir: Path, prefix: str = "bench") -> Path:
    return base_dir / f".data/{prefix}_job.json"


def get_log_file(base_dir: Path, prefix: str = "bench") -> Path:
    return base_dir / f".data/{prefix}.log"


def get_pid_file(base_dir: Path, prefix: str = "bench") -> Path:
    return base_dir / f".data/{prefix}_pids.txt"


def register_pid(base_dir: Path, pid: int):
    """Register a spawned process PID for cleanup."""
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
    """Kill all tracked PIDs and clear the file."""
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

    # Step 3: Kill tracked PIDs and worker processes
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


def daemonize(log_file_path: Path):
    """Double-fork to fully daemonize (survives terminal close, SSH disconnect).

    Returns (is_daemon, grandchild_pid):
        - In the parent: (False, grandchild_pid)
        - In the daemon (grandchild): (True, 0)
    """
    read_fd, write_fd = os.pipe()

    pid = os.fork()
    if pid > 0:
        # First parent: wait for intermediate child and read grandchild PID
        os.close(write_fd)
        os.waitpid(pid, 0)
        grandchild_pid = int(os.read(read_fd, 32).decode().strip())
        os.close(read_fd)
        return (False, grandchild_pid)

    # First child: become session leader and fork again
    os.close(read_fd)
    os.setsid()
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
    # so any errors during startup are captured.
    # Use os.dup2 to redirect the OS-level file descriptors (fd 1/2),
    # not just Python's sys.stdout/stderr — C extensions (libtorch,
    # tokenizers) write directly to fd 1/2, bypassing Python.
    sys.stdin.close()
    os.close(0)
    log_file = open(log_file_path, "w")
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = log_file
    sys.stderr = sys.stdout

    return (True, 0)


# =============================================================================
# Result syncing and pushing
# =============================================================================


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
    import subprocess

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


def upload_weights(base_dir: Path, configs: list[str]):
    """Upload trained weights for completed configs to a rolling GitHub release.

    Uses file locking for release creation to prevent race conditions on shared storage.
    """
    import fcntl
    import subprocess

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
