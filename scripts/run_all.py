#!/usr/bin/env python3
"""
Full experiment script: traces, training, evaluation, and push.

Designed for interactive SSH sessions (tmux/screen). Runs in foreground only.

USAGE:
    python scripts/run_all.py --use-cuda --cpu-workers 8 --gpu-workers 1

    python scripts/run_all.py --configs gcn_mlp sentence_mlp  # Specific configs
    python scripts/run_all.py --max-epochs 4            # Short training run
    python scripts/run_all.py --skip-traces             # Skip phase 1
    python scripts/run_all.py --skip-training           # Skip phase 2
    python scripts/run_all.py --skip-push               # Don't git push
    python scripts/run_all.py --rerun                   # Force re-run everything

Phases:
  1. Trace collection + age_weight baseline evaluation
  2. Training (all ML configs on GPU via --use-cuda)
  3. Evaluation — per-config device selection:
       *_mlp (non-sentence)    → CPU  (lightweight encoder + MLP scorer)
       *_attention, *_transformer, sentence_* → GPU  (--gpu-workers)
  4. Push results + upload weights
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

# For importing pipeline.py helpers
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


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


def get_ml_configs(presets: dict) -> list[str]:
    return sorted(
        name for name, preset in presets.items()
        if preset.get("encoder") and preset.get("scorer")
    )


# GPU eval when: sentence encoder (MiniLM, 33M params per clause) or
# attention/transformer scorer (cross-attention is expensive per step).
# CPU eval only for: GCN/features encoder + MLP scorer (all lightweight ops).
GPU_EVAL_ENCODERS = {"sentence"}
GPU_EVAL_SCORERS = {"attention", "transformer"}


def run_subprocess(cmd: list[str], base_dir: Path) -> int:
    """Run a subprocess, forwarding SIGINT/SIGTERM to it."""
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=str(base_dir),
        start_new_session=True,
    )

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


def main():
    parser = argparse.ArgumentParser(
        description="Full experiment: traces → training → evaluation → push",
    )
    parser.add_argument("--configs", nargs="*",
                        help="ML config names to train+evaluate (default: all)")

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

    args = parser.parse_args()
    base_dir = find_project_root()
    overall_start = time.time()

    # Load presets for config validation and per-config device decisions
    presets = load_presets(base_dir)

    # Resolve ML configs
    if args.configs:
        configs = args.configs
        all_ml = get_ml_configs(presets)
        for c in configs:
            if c not in all_ml:
                print(f"Error: '{c}' is not a valid ML config")
                print(f"Available: {', '.join(all_ml)}")
                sys.exit(1)
    else:
        configs = get_ml_configs(presets)
        if not configs:
            print("Error: No ML configs found in proofatlas.json")
            sys.exit(1)

    # Partition configs by eval device
    gpu_eval = [c for c in configs
                if presets[c].get("encoder") in GPU_EVAL_ENCODERS
                or presets[c].get("scorer") in GPU_EVAL_SCORERS]
    cpu_eval = [c for c in configs if c not in gpu_eval]

    print(f"{'='*60}")
    print(f"  ProofAtlas Full Experiment")
    print(f"  ML configs: {', '.join(configs)}")
    print(f"  Train:  {'GPU' if args.use_cuda else 'CPU'}")
    if gpu_eval and args.gpu_workers:
        print(f"  Eval:   CPU ({', '.join(cpu_eval)})")
        print(f"          GPU ({', '.join(gpu_eval)})")
    else:
        print(f"  Eval:   CPU (all)")
    print(f"  Phases: {'traces' if not args.skip_traces else '-'}"
          f" | {'training' if not args.skip_training else '-'}"
          f" | eval"
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

    # ── Phase 3: Evaluation ───────────────────────────────────────

    phase_start = time.time()
    phase_header(3, "Evaluation")

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

    # ── Phase 4: Push Results ─────────────────────────────────────

    if not args.skip_push:
        phase_header(4, "Push Results")
        from pipeline import sync_all_run_results, push_results, upload_weights

        all_completed = ["age_weight"] + eval_completed if not args.skip_traces else eval_completed
        if all_completed:
            push_results(base_dir, all_completed)
            upload_weights(base_dir, eval_completed)
        else:
            log("No completed configs to push")
    else:
        log("Skipping phase 4 (push)")

    # ── Summary ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Experiment Complete ({elapsed_str(overall_start)})")
    print(f"{'='*60}")
    if eval_completed:
        print(f"  Evaluated: {', '.join(eval_completed)}")
    if train_failed:
        print(f"  Train failures: {', '.join(train_failed)}")
    if eval_failed:
        print(f"  Eval failures:  {', '.join(eval_failed)}")
    if not eval_completed and not train_failed and not eval_failed:
        print("  All phases skipped or no configs to process")
    print(f"{'='*60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
