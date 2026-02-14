#!/usr/bin/env python3
"""
Quick smoke test for the full experiment pipeline.

Runs everything briefly on the 'test' problem set (small PUZ+SYN problems):
  1. Trace collection with age_weight baseline
  2. Training for 1 epoch (all ML configs)
  3. Step-limited evaluation (all ML configs)
  4. Wall-time evaluation (all time_* configs)
  5. External provers: vampire + spass (if installed)

USAGE:
    python scripts/run_test.py                  # Full smoke test
    python scripts/run_test.py --cpu-workers 4  # Parallel workers
    python scripts/run_test.py --use-cuda       # GPU training
    python scripts/run_test.py --use-cuda --cpu-workers 8 --gpu-workers 1
    python scripts/run_test.py --rerun           # Force re-run everything
    python scripts/run_test.py --kill            # Kill running smoke test
"""

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

PROBLEM_SET = "test"
MAX_EPOCHS = 1
PID_FILE = ".data/run_test.pid"

# Configs that benefit from GPU evaluation (same as run_all.py)
GPU_EVAL_ENCODERS = {"sentence"}
GPU_EVAL_SCORERS = {"attention", "transformer"}


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def elapsed_str(start: float) -> str:
    secs = int(time.time() - start)
    m, s = divmod(secs, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def find_project_root() -> Path:
    path = Path(__file__).resolve().parent.parent
    if (path / "crates" / "proofatlas").exists():
        return path
    raise RuntimeError(f"Cannot find project root (tried {path})")


def write_pid(base_dir: Path):
    pid_path = base_dir / PID_FILE
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid(base_dir: Path):
    pid_path = base_dir / PID_FILE
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def kill_running(base_dir: Path) -> int:
    pid_path = base_dir / PID_FILE
    if not pid_path.exists():
        print("No running smoke test found.")
        return 1
    try:
        pid = int(pid_path.read_text().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"Killed smoke test (pid {pid}).")
        pid_path.unlink(missing_ok=True)
        return 0
    except (ProcessLookupError, PermissionError, ValueError) as e:
        print(f"Could not kill: {e}")
        pid_path.unlink(missing_ok=True)
        return 1


_interrupted = False


def run_cmd(cmd: list[str], base_dir: Path) -> int:
    """Run a subprocess in foreground. Ctrl+C kills child and aborts the script."""
    global _interrupted
    if _interrupted:
        return 130

    log(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr,
        cwd=str(base_dir), start_new_session=True,
    )

    def on_interrupt(signum, frame):
        global _interrupted
        _interrupted = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, on_interrupt)
    signal.signal(signal.SIGTERM, on_interrupt)

    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def load_presets(base_dir: Path) -> dict:
    with open(base_dir / "configs" / "proofatlas.json") as f:
        return json.load(f).get("presets", {})


def get_step_configs(presets: dict) -> list[str]:
    return sorted(
        name for name, p in presets.items()
        if p.get("encoder") and p.get("scorer") and not name.startswith("time_")
    )


def get_time_configs(presets: dict) -> list[str]:
    return sorted(name for name in presets if name.startswith("time_"))


def get_external_provers(base_dir: Path) -> list[dict]:
    """Return installed external prover configs."""
    provers = []
    for name, config_file in [("vampire", "vampire.json"), ("spass", "spass.json")]:
        path = base_dir / "configs" / config_file
        if not path.exists():
            continue
        with open(path) as f:
            config = json.load(f)
        binary = base_dir / config["paths"]["binary"]
        if binary.exists():
            default_preset = config.get("default_preset", "time")
            provers.append({"name": name, "preset": default_preset})
    return provers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick smoke test for the full pipeline")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for training")
    parser.add_argument("--cpu-workers", type=int, default=1, help="CPU workers (default: 1)")
    parser.add_argument("--gpu-workers", type=int, default=None, help="GPU workers for evaluation")
    parser.add_argument("--timeout", type=float, default=None, help="Override per-problem timeout (seconds)")
    parser.add_argument("--rerun", action="store_true", help="Force re-run everything")
    parser.add_argument("--kill", action="store_true", help="Kill running smoke test")
    args = parser.parse_args()

    base_dir = find_project_root()

    if args.kill:
        sys.exit(kill_running(base_dir))

    # Start a new process group so --kill can terminate everything
    os.setpgrp()
    write_pid(base_dir)
    presets = load_presets(base_dir)
    step_configs = get_step_configs(presets)
    time_configs = get_time_configs(presets)
    external = get_external_provers(base_dir)

    # Partition configs by eval device
    all_eval = step_configs + time_configs
    gpu_eval = set(c for c in all_eval
                   if presets[c].get("encoder") in GPU_EVAL_ENCODERS
                   or presets[c].get("scorer") in GPU_EVAL_SCORERS) if args.gpu_workers else set()

    overall_start = time.time()

    print(f"{'='*60}")
    print(f"  ProofAtlas Smoke Test")
    print(f"  Problem set: {PROBLEM_SET}")
    print(f"  Step configs: {len(step_configs)}")
    print(f"  Time configs: {len(time_configs)}")
    print(f"  External: {', '.join(p['name'] for p in external) or 'none'}")
    print(f"  Training: {MAX_EPOCHS} epoch, {'GPU' if args.use_cuda else 'CPU'}")
    if args.timeout is not None:
        print(f"  Timeout: {args.timeout}s (override)")
    if gpu_eval:
        cpu_names = [c for c in all_eval if c not in gpu_eval]
        gpu_names = [c for c in all_eval if c in gpu_eval]
        print(f"  Eval:   CPU ({', '.join(cpu_names)})")
        print(f"          GPU ({', '.join(gpu_names)})")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    results = {"ok": [], "fail": []}

    def record(phase, name, rc):
        tag = f"{phase}/{name}"
        if rc == 0:
            results["ok"].append(tag)
        else:
            results["fail"].append(tag)
            log(f"FAILED: {tag} (exit {rc})")

    # ── Phase 1: Traces + baseline ─────────────────────────────────

    phase_start = time.time()
    log("Phase 1: Trace collection + age_weight baseline")

    cmd = [
        sys.executable, "-m", "proofatlas.cli.bench",
        "--config", "age_weight",
        "--trace", "--foreground",
        "--problem-set", PROBLEM_SET,
    ]
    if args.cpu_workers > 1:
        cmd.extend(["--cpu-workers", str(args.cpu_workers)])
    if args.rerun:
        cmd.append("--rerun")
    if args.timeout is not None:
        cmd.extend(["--timeout", str(args.timeout)])

    rc = run_cmd(cmd, base_dir)
    record("traces", "age_weight", rc)
    log(f"Phase 1 done ({elapsed_str(phase_start)})\n")

    if _interrupted:
        log("Interrupted."); remove_pid(base_dir); sys.exit(130)

    # ── Phase 2: Training (1 epoch) ────────────────────────────────

    phase_start = time.time()
    log(f"Phase 2: Training ({MAX_EPOCHS} epoch, {len(step_configs)} configs)")

    for i, config in enumerate(step_configs, 1):
        weights_file = base_dir / ".weights" / f"{config}.pt"
        if weights_file.exists() and not args.rerun:
            log(f"  [{i}/{len(step_configs)}] {config}: weights exist, skipping")
            results["ok"].append(f"train/{config}")
            continue

        log(f"  [{i}/{len(step_configs)}] {config}: training...")
        cmd = [
            sys.executable, "-m", "proofatlas.cli.train",
            "--config", config, "--foreground",
            "--max-epochs", str(MAX_EPOCHS),
        ]
        if args.use_cuda:
            cmd.append("--use-cuda")
        if args.cpu_workers > 1:
            cmd.extend(["--cpu-workers", str(args.cpu_workers)])

        rc = run_cmd(cmd, base_dir)
        record("train", config, rc)

    log(f"Phase 2 done ({elapsed_str(phase_start)})\n")

    if _interrupted:
        log("Interrupted."); remove_pid(base_dir); sys.exit(130)

    # ── Phase 3: Step-limited evaluation ───────────────────────────

    phase_start = time.time()
    log(f"Phase 3: Step-limited evaluation ({len(step_configs)} configs)")

    for i, config in enumerate(step_configs, 1):
        use_gpu = config in gpu_eval
        device_tag = "GPU" if use_gpu else "CPU"
        log(f"  [{i}/{len(step_configs)}] {config} ({device_tag})")
        cmd = [
            sys.executable, "-m", "proofatlas.cli.bench",
            "--config", config, "--foreground",
            "--problem-set", PROBLEM_SET,
        ]
        if args.cpu_workers > 1:
            cmd.extend(["--cpu-workers", str(args.cpu_workers)])
        if use_gpu:
            cmd.extend(["--gpu-workers", str(args.gpu_workers)])
        if args.rerun:
            cmd.append("--rerun")
        if args.timeout is not None:
            cmd.extend(["--timeout", str(args.timeout)])

        rc = run_cmd(cmd, base_dir)
        record("step-eval", config, rc)

    log(f"Phase 3 done ({elapsed_str(phase_start)})\n")

    if _interrupted:
        log("Interrupted."); remove_pid(base_dir); sys.exit(130)

    # ── Phase 4: Wall-time evaluation ──────────────────────────────

    if time_configs:
        phase_start = time.time()
        log(f"Phase 4: Wall-time evaluation ({len(time_configs)} configs)")

        for i, config in enumerate(time_configs, 1):
            use_gpu = config in gpu_eval
            device_tag = "GPU" if use_gpu else "CPU"
            log(f"  [{i}/{len(time_configs)}] {config} ({device_tag})")
            cmd = [
                sys.executable, "-m", "proofatlas.cli.bench",
                "--config", config, "--foreground",
                "--problem-set", PROBLEM_SET,
            ]
            if args.cpu_workers > 1:
                cmd.extend(["--cpu-workers", str(args.cpu_workers)])
            if use_gpu:
                cmd.extend(["--gpu-workers", str(args.gpu_workers)])
            if args.rerun:
                cmd.append("--rerun")
            if args.timeout is not None:
                cmd.extend(["--timeout", str(args.timeout)])

            rc = run_cmd(cmd, base_dir)
            record("time-eval", config, rc)

        log(f"Phase 4 done ({elapsed_str(phase_start)})\n")

    if _interrupted:
        log("Interrupted."); remove_pid(base_dir); sys.exit(130)

    # ── Phase 5: External provers ──────────────────────────────────

    if external:
        phase_start = time.time()
        log(f"Phase 5: External provers ({', '.join(p['name'] for p in external)})")

        for prover in external:
            name = prover["name"]
            preset = prover["preset"]
            log(f"  {name}/{preset}")
            cmd = [
                sys.executable, "-m", "proofatlas.cli.bench",
                "--config", preset, "--foreground",
                "--problem-set", PROBLEM_SET,
            ]
            if args.cpu_workers > 1:
                cmd.extend(["--cpu-workers", str(args.cpu_workers)])
            if args.rerun:
                cmd.append("--rerun")
            if args.timeout is not None:
                cmd.extend(["--timeout", str(args.timeout)])

            rc = run_cmd(cmd, base_dir)
            record("external", f"{name}/{preset}", rc)

        log(f"Phase 5 done ({elapsed_str(phase_start)})\n")

    # ── Summary ────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Smoke Test Complete ({elapsed_str(overall_start)})")
    print(f"{'='*60}")
    print(f"  Passed: {len(results['ok'])}")
    if results["fail"]:
        print(f"  FAILED: {len(results['fail'])}")
        for f in results["fail"]:
            print(f"    - {f}")
    else:
        print(f"  All steps passed!")
    print(f"{'='*60}")
    sys.stdout.flush()

    remove_pid(base_dir)
    sys.exit(1 if results["fail"] else 0)


if __name__ == "__main__":
    main()
