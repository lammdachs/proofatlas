#!/usr/bin/env python3
"""Async vs sequential inference throughput experiment.

Iterates (problem x config x mode x rep) over the throughput16 problem set,
recording per-iteration telemetry for each run. Skips cells that already have
a result file unless --rerun is given.

Output: .data/throughput/<config>__<mode>__<problem>__rep<N>.json
Each file has: wall_s, status, proof_found, iterations, profile (per-iter trace).
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

DEFAULT_CONFIGS = [
    "gcn_struct_transformer_xcl_r6",
    "gcn_struct_mlp_xcl_r6",
    "age_weight",
]
DEFAULT_MODES = ["async", "sequential"]


def load_preset(name):
    presets = json.loads(CONFIGS_PROVER.read_text())["presets"]
    if name not in presets:
        raise ValueError(f"preset {name!r} not in configs/proofatlas.json")
    return presets[name]


def load_problems(problem_set):
    cfg = json.loads(CONFIGS_TPTP.read_text())
    if problem_set not in cfg["problem_sets"]:
        raise ValueError(f"problem set {problem_set!r} not in configs/tptp.json")
    return cfg["problem_sets"][problem_set]["problems"]


def build_atlas(preset_name, preset, inference_mode):
    from proofatlas import ProofAtlas
    kwargs = dict(
        timeout=float(preset.get("timeout", 60)),
        max_iterations=preset.get("max_iterations", 256),
        literal_selection=preset.get("literal_selection", 21),
        memory_limit=preset.get("memory_limit", 64),
        include_dir=str(TPTP_ROOT),
        enable_profiling=True,
        temperature=float(preset.get("temperature", 1.0)),
    )
    if "encoder" in preset:
        kwargs.update(
            encoder=preset["encoder"],
            scorer=preset["scorer"],
            weights_path=str(WEIGHTS_DIR),
            model_name=preset_name,
            inference_mode=inference_mode,
        )
    else:
        kwargs["age_weight_ratio"] = float(preset.get("age_weight_ratio", 0.5))
    return ProofAtlas(**kwargs)


def run_one(atlas, problem_relpath):
    full = str(TPTP_ROOT / "Problems" / problem_relpath)
    t0 = time.perf_counter()
    prv = atlas.prove(full)
    wall = time.perf_counter() - t0
    pj = prv.profile_json()
    profile = json.loads(pj) if pj else None
    stats = prv.statistics()
    return {
        "wall_s": wall,
        "status": prv.status,
        "proof_found": prv.proof_found,
        "iterations": stats.get("iterations", 0),
        "clause_count": stats.get("clause_count", 0),
        "profile": profile,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--problem-set", default="throughput16")
    ap.add_argument("--problems", nargs="+", default=None,
                    help="Override: explicit problem relpaths (skips --problem-set)")
    ap.add_argument("--output-dir", default=".data/throughput")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = args.problems or load_problems(args.problem_set)

    # Build (config, mode) cells; for non-ML configs (no encoder) use "noop" mode.
    cells = []
    for cfg in args.configs:
        is_ml = "encoder" in load_preset(cfg)
        if is_ml:
            for m in args.modes:
                cells.append((cfg, m))
        else:
            cells.append((cfg, "noop"))

    print(f"Problems: {len(problems)}")
    print(f"Configs:  {args.configs}")
    print(f"Modes:    {args.modes}")
    print(f"Reps:     {args.reps}")
    print(f"Cells:    {len(cells)}")
    print(f"Out dir:  {out_dir}")
    print(f"Total runs: {len(cells) * len(problems) * args.reps}")
    print()

    done = skipped = failed = 0
    total = len(cells) * len(problems) * args.reps
    t_start = time.perf_counter()

    for cfg, mode in cells:
        preset = load_preset(cfg)
        atlas = None  # build lazily; share within (cfg, mode)
        for problem in problems:
            for rep in range(args.reps):
                tag = f"{cfg}__{mode}__{problem.replace('/', '_')}__rep{rep}.json"
                out_file = out_dir / tag
                if out_file.exists() and not args.rerun:
                    skipped += 1
                    done += 1
                    continue
                if atlas is None:
                    eff_mode = mode if mode != "noop" else "async"
                    atlas = build_atlas(cfg, preset, eff_mode)
                try:
                    r = run_one(atlas, problem)
                    r.update(config=cfg, mode=mode, problem=problem, rep=rep,
                             timestamp=datetime.now().isoformat())
                except Exception as e:
                    failed += 1
                    r = dict(config=cfg, mode=mode, problem=problem, rep=rep,
                             error=str(e), timestamp=datetime.now().isoformat())
                out_file.write_text(json.dumps(r))
                done += 1
                elapsed = time.perf_counter() - t_start
                eta_s = elapsed / max(done - skipped, 1) * (total - done)
                summary = "ok"
                if "error" in r:
                    summary = f"ERR: {r['error'][:30]}"
                elif "proof_found" in r:
                    summary = (f"{'P' if r['proof_found'] else r['status'][:5]} "
                               f"wall={r['wall_s']:5.1f}s iter={r['iterations']}")
                print(f"[{done:4d}/{total}] {100*done/total:5.1f}%  ETA {eta_s:5.0f}s  "
                      f"{cfg[:32]:32s} {mode:10s} {problem:25s} rep{rep}  {summary}")

    print(f"\nDone. skipped={skipped} failed={failed} total={total}  "
          f"elapsed={time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
