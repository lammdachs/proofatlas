"""Job and daemon management for proofatlas-bench.

Handles PID tracking, job status persistence, daemonization,
and job status display. Only depends on stdlib.
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
    # so any errors during startup are captured
    sys.stdin.close()
    os.close(0)
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    return (True, 0)
