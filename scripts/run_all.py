#!/usr/bin/env python3
"""
Full experiment script: traces, training, evaluation, and push.

Runs in the foreground by default. Use nohup for headless operation:
    nohup python scripts/run_all.py --use-cuda --cpu-workers 56 --gpu-workers 4 &

USAGE:
    python scripts/run_all.py --use-cuda --cpu-workers 8 --gpu-workers 1

    python scripts/run_all.py --configs gcn_mlp sentence_mlp  # Specific configs
    python scripts/run_all.py --problem-set puz              # Evaluate on PUZ only
    python scripts/run_all.py --max-epochs 4            # Short training run
    python scripts/run_all.py --skip-traces             # Skip phase 1
    python scripts/run_all.py --skip-training           # Skip phase 2
    python scripts/run_all.py --skip-push               # Don't git push
    python scripts/run_all.py --rerun                   # Force re-run everything
    python scripts/run_all.py --kill                    # Kill a running experiment

Phases:
  1. Trace collection + age_weight baseline evaluation
  2. Training (step-limited ML configs on GPU via --use-cuda)
  3. Step-limited evaluation — per-config device selection:
       *_mlp (non-sentence)    → CPU  (lightweight encoder + MLP scorer)
       *_attention, *_transformer, sentence_* → GPU  (--gpu-workers)
  4. Wall-time evaluation (time_* configs, reuse trained weights)
  5. Push results + upload weights
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

PID_FILE_NAME = ".data/run_all.pid"


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def elapsed_str(start: float) -> str:
    secs = int(time.time() - start)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def find_project_root() -> Path:
    path = Path(__file__).resolve().parent.parent
    if (path / "crates" / "proofatlas").exists():
        return path
    raise RuntimeError(f"Cannot find project root (tried {path})")


def load_presets(base_dir: Path) -> dict:
    with open(base_dir / "configs" / "proofatlas.json") as f:
        return json.load(f).get("presets", {})


def get_step_configs(presets: dict) -> list[str]:
    """Step-limited ML configs (for training + step-limited evaluation)."""
    return sorted(
        name for name, preset in presets.items()
        if preset.get("encoder") and preset.get("scorer")
        and not name.startswith("time_")
    )


def get_time_configs(presets: dict) -> list[str]:
    """Wall-time ML configs (for time-limited evaluation only, reuse weights)."""
    return sorted(
        name for name in presets
        if name.startswith("time_")
    )


# GPU eval when: sentence encoder (MiniLM, 33M params per clause) or
# attention/transformer scorer (cross-attention is expensive per step).
# CPU eval only for: GCN/features encoder + MLP scorer (all lightweight ops).
GPU_EVAL_ENCODERS = {"sentence"}
GPU_EVAL_SCORERS = {"attention", "transformer"}


# =============================================================================
# PID file management
# =============================================================================

def get_pid_file(base_dir: Path) -> Path:
    return base_dir / PID_FILE_NAME


def write_pid_file(base_dir: Path, child_pid: int | None = None):
    """Write PID file with main PID and optional child PID."""
    pid_file = get_pid_file(base_dir)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    data = {"pid": os.getpid()}
    if child_pid is not None:
        data["child_pgid"] = child_pid
    with open(pid_file, "w") as f:
        json.dump(data, f)


def update_child_pid(base_dir: Path, child_pid: int):
    """Update the PID file with the current child process group ID."""
    pid_file = get_pid_file(base_dir)
    try:
        with open(pid_file) as f:
            data = json.load(f)
        data["child_pgid"] = child_pid
        with open(pid_file, "w") as f:
            json.dump(data, f)
    except (json.JSONDecodeError, IOError):
        pass


def remove_pid_file(base_dir: Path):
    get_pid_file(base_dir).unlink(missing_ok=True)


def handle_kill(base_dir: Path):
    """Kill a running experiment and all its child processes."""
    pid_file = get_pid_file(base_dir)
    if not pid_file.exists():
        print("No running experiment found.")
        return

    try:
        with open(pid_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("Corrupt PID file, removing.")
        pid_file.unlink(missing_ok=True)
        return

    main_pid = data.get("pid")
    child_pgid = data.get("child_pgid")

    # Kill child process group first (train/bench and their workers)
    if child_pgid:
        try:
            os.killpg(child_pgid, signal.SIGTERM)
            log(f"Sent SIGTERM to child process group (PGID {child_pgid})")
            time.sleep(1)
            os.killpg(child_pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    # Kill main process
    if main_pid:
        try:
            os.kill(main_pid, signal.SIGTERM)
            log(f"Sent SIGTERM to run_all.py (PID {main_pid})")
            time.sleep(0.5)
            os.kill(main_pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    pid_file.unlink(missing_ok=True)
    log("Experiment killed.")


# =============================================================================
# Subprocess execution with PID tracking
# =============================================================================

def run_subprocess(cmd: list[str], base_dir: Path) -> int:
    """Run a subprocess, forwarding signals and tracking its PGID."""
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=str(base_dir),
        start_new_session=True,
    )

    # Track child PGID so --kill can find it
    update_child_pid(base_dir, proc.pid)

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def forward_signal(signum, frame):
        try:
            os.killpg(proc.pid, signum)
        except (OSError, ProcessLookupError):
            pass

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    try:
        rc = proc.wait()
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

    return rc


def phase_header(phase: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Phase {phase}: {title}")
    print(f"{'='*60}\n")
    sys.stdout.flush()


def run_experiment(args, base_dir: Path):
    """Run the full experiment pipeline."""
    overall_start = time.time()

    # Load presets for config validation and per-config device decisions
    presets = load_presets(base_dir)

    # Resolve step-limited ML configs (for training + step eval)
    all_step = get_step_configs(presets)
    all_time = get_time_configs(presets)

    if args.configs:
        # User-specified configs: split into step and time
        configs = [c for c in args.configs if not c.startswith("time_")]
        time_configs = [c for c in args.configs if c.startswith("time_")]
        for c in configs:
            if c not in all_step:
                print(f"Error: '{c}' is not a valid step-limited ML config")
                print(f"Available: {', '.join(all_step)}")
                sys.exit(1)
        for c in time_configs:
            if c not in all_time:
                print(f"Error: '{c}' is not a valid wall-time config")
                print(f"Available: {', '.join(all_time)}")
                sys.exit(1)
    else:
        configs = all_step
        time_configs = all_time
        if not configs:
            print("Error: No ML configs found in proofatlas.json")
            sys.exit(1)

    # Partition configs by eval device
    all_eval = configs + time_configs
    gpu_eval = [c for c in all_eval
                if presets[c].get("encoder") in GPU_EVAL_ENCODERS
                or presets[c].get("scorer") in GPU_EVAL_SCORERS]
    cpu_eval = [c for c in all_eval if c not in gpu_eval]

    print(f"{'='*60}")
    print(f"  ProofAtlas Full Experiment")
    print(f"  Step configs: {', '.join(configs)}")
    if time_configs:
        print(f"  Time configs: {', '.join(time_configs)}")
    if args.problem_set:
        print(f"  Problem set: {args.problem_set}")
    print(f"  Train:  {'GPU' if args.use_cuda else 'CPU'}")
    if gpu_eval and args.gpu_workers:
        print(f"  Eval:   CPU ({', '.join(cpu_eval)})")
        print(f"          GPU ({', '.join(gpu_eval)})")
    else:
        print(f"  Eval:   CPU (all)")
    print(f"  Phases: {'traces' if not args.skip_traces else '-'}"
          f" | {'training' if not args.skip_training else '-'}"
          f" | step-eval"
          f" | {'time-eval' if time_configs else '-'}"
          f" | {'push' if not args.skip_push else '-'}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # ── Phase 1: Trace Collection + Baseline ──────────────────────

    if not args.skip_traces:
        phase_start = time.time()
        phase_header(1, "Trace Collection + Baseline Evaluation")

        trace_dir = base_dir / ".data" / "traces" / "age_weight"
        npz_count = len(list(trace_dir.glob("*.npz"))) if trace_dir.exists() else 0
        results_dir = base_dir / ".data" / "runs" / "proofatlas" / "age_weight"
        results_exist = results_dir.exists() and any(results_dir.glob("*.json"))

        if npz_count >= 10 and not args.rerun:
            log(f"Traces already collected ({npz_count} NPZ files), skipping")
        else:
            if npz_count < 10 and results_exist and not args.rerun:
                log(f"Results exist but only {npz_count} traces — adding --rerun to collect traces")
                needs_rerun = True
            else:
                needs_rerun = args.rerun

            cmd = [
                sys.executable, "-m", "proofatlas.cli.bench",
                "--config", "age_weight",
                "--trace", "--foreground",
            ]
            if args.cpu_workers is not None:
                cmd.extend(["--cpu-workers", str(args.cpu_workers)])
            if args.problem_set is not None:
                cmd.extend(["--problem-set", args.problem_set])
            if needs_rerun:
                cmd.append("--rerun")

            log(f"Running: {' '.join(cmd)}")
            rc = run_subprocess(cmd, base_dir)
            if rc != 0:
                log(f"WARNING: Trace collection exited with code {rc}")

        log(f"Phase 1 done ({elapsed_str(phase_start)})")
    else:
        log("Skipping phase 1 (traces)")

    # ── Phase 2: Training ─────────────────────────────────────────

    train_failed = []

    if not args.skip_training:
        phase_start = time.time()
        phase_header(2, "Training")

        for i, config in enumerate(configs, 1):
            weights_file = base_dir / ".weights" / f"{config}.pt"
            if weights_file.exists() and not args.rerun:
                log(f"[{i}/{len(configs)}] {config}: weights exist, skipping")
                continue

            log(f"[{i}/{len(configs)}] {config}: starting training...")
            cmd = [
                sys.executable, "-m", "proofatlas.cli.train",
                "--config", config, "--foreground",
            ]
            if args.use_cuda:
                cmd.append("--use-cuda")
            if args.cpu_workers is not None:
                cmd.extend(["--cpu-workers", str(args.cpu_workers)])
            if args.gpu_workers is not None:
                cmd.extend(["--gpu-workers", str(args.gpu_workers)])
            if args.max_epochs is not None:
                cmd.extend(["--max-epochs", str(args.max_epochs)])

            rc = run_subprocess(cmd, base_dir)
            if rc != 0:
                log(f"[{i}/{len(configs)}] {config}: training FAILED (exit {rc})")
                train_failed.append(config)
            else:
                log(f"[{i}/{len(configs)}] {config}: training complete")

        log(f"Phase 2 done ({elapsed_str(phase_start)})")
    else:
        log("Skipping phase 2 (training)")

    # ── Phase 3: Step-Limited Evaluation ─────────────────────────

    phase_start = time.time()
    phase_header(3, "Step-Limited Evaluation")

    eval_completed = []
    eval_failed = []

    for i, config in enumerate(configs, 1):
        use_gpu = args.gpu_workers is not None and config in gpu_eval
        device_tag = "GPU" if use_gpu else "CPU"
        log(f"[{i}/{len(configs)}] {config}: starting evaluation ({device_tag})...")
        cmd = [
            sys.executable, "-m", "proofatlas.cli.bench",
            "--config", config, "--foreground",
        ]
        if args.cpu_workers is not None:
            cmd.extend(["--cpu-workers", str(args.cpu_workers)])
        if use_gpu:
            cmd.extend(["--gpu-workers", str(args.gpu_workers)])
        if args.problem_set is not None:
            cmd.extend(["--problem-set", args.problem_set])
        if args.rerun:
            cmd.append("--rerun")

        rc = run_subprocess(cmd, base_dir)
        if rc != 0:
            log(f"[{i}/{len(configs)}] {config}: evaluation FAILED (exit {rc})")
            eval_failed.append(config)
        else:
            log(f"[{i}/{len(configs)}] {config}: evaluation complete")
            eval_completed.append(config)

    log(f"Phase 3 done ({elapsed_str(phase_start)})")

    # ── Phase 4: Wall-Time Evaluation ─────────────────────────────

    time_completed = []
    time_failed = []

    if time_configs:
        phase_start = time.time()
        phase_header(4, "Wall-Time Evaluation")

        for i, config in enumerate(time_configs, 1):
            use_gpu = args.gpu_workers is not None and config in gpu_eval
            device_tag = "GPU" if use_gpu else "CPU"
            log(f"[{i}/{len(time_configs)}] {config}: starting wall-time evaluation ({device_tag})...")
            cmd = [
                sys.executable, "-m", "proofatlas.cli.bench",
                "--config", config, "--foreground",
            ]
            if args.cpu_workers is not None:
                cmd.extend(["--cpu-workers", str(args.cpu_workers)])
            if use_gpu:
                cmd.extend(["--gpu-workers", str(args.gpu_workers)])
            if args.problem_set is not None:
                cmd.extend(["--problem-set", args.problem_set])
            if args.rerun:
                cmd.append("--rerun")

            rc = run_subprocess(cmd, base_dir)
            if rc != 0:
                log(f"[{i}/{len(time_configs)}] {config}: wall-time evaluation FAILED (exit {rc})")
                time_failed.append(config)
            else:
                log(f"[{i}/{len(time_configs)}] {config}: wall-time evaluation complete")
                time_completed.append(config)

        log(f"Phase 4 done ({elapsed_str(phase_start)})")

    # ── Phase 5: Push Results ─────────────────────────────────────

    if not args.skip_push:
        phase_header(5, "Push Results")
        from bench_jobs import push_results, upload_weights

        all_completed = eval_completed + time_completed
        if not args.skip_traces:
            all_completed = ["age_weight"] + all_completed
        if all_completed:
            push_results(base_dir, all_completed)
            upload_weights(base_dir, eval_completed)
        else:
            log("No completed configs to push")
    else:
        log("Skipping phase 5 (push)")

    # ── Summary ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Experiment Complete ({elapsed_str(overall_start)})")
    print(f"{'='*60}")
    if eval_completed:
        print(f"  Step-limited: {', '.join(eval_completed)}")
    if time_completed:
        print(f"  Wall-time:    {', '.join(time_completed)}")
    if train_failed:
        print(f"  Train failures: {', '.join(train_failed)}")
    if eval_failed or time_failed:
        print(f"  Eval failures:  {', '.join(eval_failed + time_failed)}")
    if not eval_completed and not time_completed and not train_failed and not eval_failed:
        print("  All phases skipped or no configs to process")
    print(f"{'='*60}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Full experiment: traces → training → evaluation → push",
    )
    parser.add_argument("--configs", nargs="*",
                        help="ML config names to train+evaluate (default: all)")

    # Job management
    parser.add_argument("--kill", action="store_true",
                        help="Kill a running experiment")

    # Phase skips
    parser.add_argument("--skip-traces", action="store_true",
                        help="Skip phase 1 (trace collection + baseline)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip phase 2 (training)")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip phase 4 (git push + weight upload)")

    # Pass-through flags
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA for training")
    parser.add_argument("--cpu-workers", type=int, default=None,
                        help="Number of CPU workers for bench/trace")
    parser.add_argument("--gpu-workers", type=int, default=None,
                        help="GPU workers for training DDP and eval (attention/transformer/sentence)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run everything (ignore caches)")
    parser.add_argument("--problem-set", default=None,
                        help="Problem set to evaluate on (default: from tptp.json)")

    args = parser.parse_args()
    base_dir = find_project_root()

    if args.kill:
        handle_kill(base_dir)
        return

    # Write PID file so --kill can find us
    write_pid_file(base_dir)

    # Clean up PID file on exit (normal or signal)
    def cleanup_and_exit(signum, frame):
        remove_pid_file(base_dir)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, cleanup_and_exit)

    try:
        run_experiment(args, base_dir)
    finally:
        remove_pid_file(base_dir)


if __name__ == "__main__":
    main()
