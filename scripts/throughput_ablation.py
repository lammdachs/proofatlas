#!/usr/bin/env python3
"""Cache × batching ablation on the embed_score (transformer) configuration.

Grid:
  - 16 iter-cap problems (throughput_sat16)
  - Cells:
      (cache=T, mode=batched)  -- async with embed_batch_size=large
      (cache=T, mode=eager)    -- async with embed_batch_size=1
      (cache=F, mode=batched)
      (cache=F, mode=eager)
      (cache=T, mode=sequential)  -- strawman baseline (drain-after-each-submit)
  - 3 reps per cell

Output: .data/throughput_ablation/<cell>__<problem>__rep<N>.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_TPTP = PROJECT_ROOT / "configs" / "tptp.json"
CONFIGS_PROVER = PROJECT_ROOT / "configs" / "proofatlas.json"
TPTP_ROOT = PROJECT_ROOT / ".tptp" / "TPTP-v9.0.0"
WEIGHTS_DIR = PROJECT_ROOT / ".weights"

CONFIG = "gcn_struct_transformer_xcl_r6"
BIG_BATCH = 1_000_000

CELLS = [
    ("cache_batched",    dict(cache_embeddings=True,  inference_mode="async",      embed_batch_size=BIG_BATCH)),
    ("cache_eager",      dict(cache_embeddings=True,  inference_mode="async",      embed_batch_size=1)),
    ("nocache_batched",  dict(cache_embeddings=False, inference_mode="async",      embed_batch_size=BIG_BATCH)),
    ("nocache_eager",    dict(cache_embeddings=False, inference_mode="async",      embed_batch_size=1)),
    ("cache_sequential", dict(cache_embeddings=True,  inference_mode="sequential")),
]


def load_problems():
    cfg = json.loads(CONFIGS_TPTP.read_text())
    return cfg["problem_sets"]["throughput_sat16"]["problems"]


def load_preset():
    return json.loads(CONFIGS_PROVER.read_text())["presets"][CONFIG]


def build_atlas(preset, cell_kwargs):
    from proofatlas import ProofAtlas
    return ProofAtlas(
        encoder=preset["encoder"], scorer=preset["scorer"],
        model_name=CONFIG, weights_path=str(WEIGHTS_DIR), use_cuda=True,
        timeout=float(preset.get("timeout", 60)),
        max_iterations=preset.get("max_iterations", 256),
        literal_selection=preset.get("literal_selection", 21),
        memory_limit=preset.get("memory_limit", 64),
        include_dir=str(TPTP_ROOT),
        enable_profiling=True,
        temperature=float(preset.get("temperature", 1.0)),
        **cell_kwargs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output-dir", default=".data/throughput_ablation")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems()
    preset = load_preset()

    total = len(CELLS) * len(problems) * args.reps
    done = skipped = failed = 0
    t_start = time.perf_counter()
    print(f"Cells: {[c[0] for c in CELLS]}")
    print(f"Problems: {len(problems)}, reps: {args.reps}, total runs: {total}")

    for cell_name, cell_kwargs in CELLS:
        atlas = None
        for prob in problems:
            for rep in range(args.reps):
                tag = f"{cell_name}__{prob.replace('/','_')}__rep{rep}.json"
                out_file = out_dir / tag
                if out_file.exists() and not args.rerun:
                    skipped += 1; done += 1; continue
                if atlas is None:
                    atlas = build_atlas(preset, cell_kwargs)
                try:
                    t0 = time.perf_counter()
                    prv = atlas.prove(str(TPTP_ROOT / "Problems" / prob))
                    dt = time.perf_counter() - t0
                    s = prv.statistics()
                    pj = prv.profile_json()
                    profile = json.loads(pj) if pj else None
                    r = dict(
                        cell=cell_name, problem=prob, rep=rep,
                        config=CONFIG,
                        wall_s=dt, status=prv.status, proof_found=prv.proof_found,
                        iterations=s.get("iterations", 0),
                        clause_count=s.get("clause_count", 0),
                        profile=profile,
                        timestamp=datetime.now().isoformat(),
                    )
                except Exception as e:
                    failed += 1
                    r = dict(cell=cell_name, problem=prob, rep=rep,
                             error=str(e), timestamp=datetime.now().isoformat())
                out_file.write_text(json.dumps(r))
                done += 1
                elapsed = time.perf_counter() - t_start
                eta = elapsed / max(done - skipped, 1) * (total - done)
                summary = "ok"
                if "error" in r: summary = f"ERR: {r['error'][:30]}"
                else: summary = f"{'P' if r['proof_found'] else r['status'][:5]} wall={r['wall_s']:5.1f}s iter={r['iterations']}"
                print(f"[{done:4d}/{total}] {100*done/total:5.1f}% ETA {eta:5.0f}s  "
                      f"{cell_name:18s} {prob:25s} rep{rep}  {summary}")

    print(f"\nDone. skipped={skipped} failed={failed} elapsed={time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
