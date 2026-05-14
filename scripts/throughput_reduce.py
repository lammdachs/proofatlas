#!/usr/bin/env python3
"""Reduce raw throughput-bench JSON files into a summary table.

For each (problem, config, mode) cell, computes:
- wall_s_median across reps
- iter/s = iterations / wall_s
- per-iter t/i median (ms)
- per-stage medians (ms)

Reads .data/throughput/*.json and prints a markdown-style table.

Usage:
    python scripts/throughput_reduce.py
    python scripts/throughput_reduce.py --csv summary.csv
    python scripts/throughput_reduce.py --skip-cold N   (drop first N iters)
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / ".data" / "throughput"


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Write summary CSV here")
    ap.add_argument("--skip-cold", type=int, default=1,
                    help="Drop first N iters per run (model-load cold start)")
    args = ap.parse_args()

    # cells[(problem, config, mode)] = list of (wall_s, iterations, status,
    #                                            proof, iter_trace) across reps
    cells = defaultdict(list)
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        prof = d.get("profile") or {}
        trace = prof.get("iter_trace", [])
        cells[(d["problem"], d["config"], d["mode"])].append({
            "wall_s": d["wall_s"],
            "iterations": d["iterations"],
            "status": d["status"],
            "proof": d["proof_found"],
            "trace": trace,
        })

    rows = []
    for (problem, config, mode), reps in sorted(cells.items()):
        # Combine all reps' iter traces (after cold-start skip)
        all_iters = []
        for r in reps:
            all_iters.extend(r["trace"][args.skip_cold:])
        if not all_iters:
            continue
        wall_s_med = median([r["wall_s"] for r in reps])
        iters_med = median([r["iterations"] for r in reps])
        iter_per_s = iters_med / wall_s_med if wall_s_med > 0 else float("nan")
        t_total_med_ms = median([x["t_total_ns"] / 1e6 for x in all_iters])
        t_select_med_ms = median([x["t_select_ns"] / 1e6 for x in all_iters])
        t_proc_new_med_ms = median([x["t_process_new_ns"] / 1e6 for x in all_iters])
        t_fwd_med_ms = median([x["t_forward_simplify_ns"] / 1e6 for x in all_iters])
        t_bwd_med_ms = median([x["t_backward_simplify_ns"] / 1e6 for x in all_iters])
        t_gen_med_ms = median([x["t_generate_ns"] / 1e6 for x in all_iters])
        proof = reps[0]["proof"]  # ought to be consistent across reps
        status = reps[0]["status"]
        rows.append({
            "problem": problem,
            "config": config,
            "mode": mode,
            "reps": len(reps),
            "wall_s_med": wall_s_med,
            "iters_med": iters_med,
            "iter_per_s": iter_per_s,
            "t_total_med_ms": t_total_med_ms,
            "t_select_med_ms": t_select_med_ms,
            "t_proc_new_med_ms": t_proc_new_med_ms,
            "t_fwd_med_ms": t_fwd_med_ms,
            "t_bwd_med_ms": t_bwd_med_ms,
            "t_gen_med_ms": t_gen_med_ms,
            "proof": proof,
            "status": status,
        })

    # Print summary table
    header = (
        f"{'problem':25s}  {'config':32s}  {'mode':10s}  "
        f"{'reps':>4s}  {'wall':>6s}  {'iters':>5s}  {'iter/s':>7s}  "
        f"{'t/i':>6s}  {'sel':>6s}  {'proc':>6s}  {'gen':>5s}  status"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['problem']:25s}  {r['config'][:32]:32s}  {r['mode']:10s}  "
            f"{r['reps']:4d}  {r['wall_s_med']:6.2f}  {int(r['iters_med']):5d}  "
            f"{r['iter_per_s']:7.2f}  "
            f"{r['t_total_med_ms']:6.2f}  {r['t_select_med_ms']:6.2f}  "
            f"{r['t_proc_new_med_ms']:6.2f}  {r['t_gen_med_ms']:5.2f}  "
            f"{'P' if r['proof'] else r['status'][:5]}"
        )

    # Pivot: async vs sequential side-by-side for each (problem, ML config)
    print()
    print("=== Async vs Sequential gap (ML configs only) ===")
    pivot = defaultdict(dict)
    for r in rows:
        if r["mode"] in ("async", "sequential"):
            pivot[(r["problem"], r["config"])][r["mode"]] = r
    print(f"{'problem':25s}  {'config':32s}  "
          f"{'async t/i':>9s}  {'seq t/i':>9s}  {'Δms':>6s}  "
          f"{'async ips':>10s}  {'seq ips':>10s}  {'speedup':>7s}")
    for (problem, config), modes in sorted(pivot.items()):
        if "async" in modes and "sequential" in modes:
            a, s = modes["async"], modes["sequential"]
            delta = s["t_total_med_ms"] - a["t_total_med_ms"]
            speedup = a["iter_per_s"] / s["iter_per_s"] if s["iter_per_s"] > 0 else float("nan")
            print(
                f"{problem:25s}  {config[:32]:32s}  "
                f"{a['t_total_med_ms']:9.2f}  {s['t_total_med_ms']:9.2f}  {delta:+6.2f}  "
                f"{a['iter_per_s']:10.2f}  {s['iter_per_s']:10.2f}  {speedup:7.2f}x"
            )

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
