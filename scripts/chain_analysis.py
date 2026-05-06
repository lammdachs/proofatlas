#!/usr/bin/env python3
"""Chain analysis: does LAT 'narrow' onto a contracting subset?

Computes per-round solved sets, intra-chain symmetric differences,
carry-over rates, and overlap with baselines for LAT and XCL chains.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/apluska/proofatlas/.data/runs/proofatlas")


def solved_set(run_dir: Path) -> set[str]:
    out = set()
    for p in run_dir.iterdir():
        if not p.name.endswith(".json"):
            continue
        with p.open() as f:
            d = json.load(f)
        if d.get("status") == "proof":
            out.add(d["problem"])
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def report_chain(label: str, rounds: list[str], baseline_aw: set, baseline_r0: set):
    print(f"\n=== {label} ===")
    print(f"  baseline (age_weight): {len(baseline_aw)} solved")
    print(f"  baseline r0 (gcn_struct_*): {len(baseline_r0)} solved")
    sets = []
    for r, run in enumerate(rounds):
        path = ROOT / run
        if not path.exists():
            print(f"  r{r+1} {run}: MISSING")
            continue
        s = solved_set(path)
        sets.append((run, s))
        print(f"  r{r+1} {run}: {len(s)} solved")
    print()
    print("  round-to-round transitions (kept = solved by both / lost = prev only / gained = curr only):")
    prev_label, prev = "r0", baseline_r0
    for label2, s in sets:
        kept = len(prev & s)
        lost = len(prev - s)
        gained = len(s - prev)
        carry = kept / len(prev) if prev else 0.0
        nov = gained / len(s) if s else 0.0
        print(f"    {prev_label:30s} -> {label2:30s} | kept {kept:5d} | lost {lost:5d} | gained {gained:5d} | "
              f"carryover {carry:.3f} | novelty {nov:.3f} | jaccard {jaccard(prev, s):.3f}")
        prev_label, prev = label2, s
    print()
    if sets:
        last_label, last = sets[-1]
        kept_aw = len(baseline_aw & last)
        lost_aw = len(baseline_aw - last)
        gained_aw = len(last - baseline_aw)
        print(f"  vs age_weight baseline ({last_label}):")
        print(f"    kept {kept_aw} | lost {lost_aw} | gained {gained_aw} | jaccard {jaccard(baseline_aw, last):.3f}")
        kept_r0 = len(baseline_r0 & last)
        lost_r0 = len(baseline_r0 - last)
        gained_r0 = len(last - baseline_r0)
        print(f"  vs round-0 baseline (last):")
        print(f"    kept {kept_r0} | lost {lost_r0} | gained {gained_r0} | jaccard {jaccard(baseline_r0, last):.3f}")
    return sets


def main():
    aw = solved_set(ROOT / "age_weight")
    mlp_r0 = solved_set(ROOT / "gcn_struct_mlp")
    tr_r0 = solved_set(ROOT / "gcn_struct_transformer")

    mlp_lat = [f"gcn_struct_mlp_lat_r{i}" for i in range(1, 8)]
    mlp_xcl = [f"gcn_struct_mlp_xcl_r{i}" for i in range(1, 8)]
    tr_lat = [f"gcn_struct_transformer_lat_r{i}" for i in range(1, 8)]
    tr_xcl = [f"gcn_struct_transformer_xcl_r{i}" for i in range(1, 8)]

    print(f"baseline age_weight: {len(aw)}")
    print(f"baseline gcn_struct_mlp r0: {len(mlp_r0)}")
    print(f"baseline gcn_struct_transformer r0: {len(tr_r0)}")
    print(f"aw vs mlp_r0 jaccard: {jaccard(aw, mlp_r0):.3f}")
    print(f"aw vs tr_r0 jaccard: {jaccard(aw, tr_r0):.3f}")

    mlp_lat_sets = report_chain("MLP LAT", mlp_lat, aw, mlp_r0)
    mlp_xcl_sets = report_chain("MLP XCL", mlp_xcl, aw, mlp_r0)
    tr_lat_sets = report_chain("Transformer LAT", tr_lat, aw, tr_r0)
    tr_xcl_sets = report_chain("Transformer XCL", tr_xcl, aw, tr_r0)

    # Cross-chain check: at the last round, what fraction of LAT-solved is
    # also solved by XCL? And how do their unique-problems counts compare?
    def cross(name, lat, xcl, baseline_r0):
        if not lat or not xcl:
            return
        _, lat_last = lat[-1]
        _, xcl_last = xcl[-1]
        print(f"\n=== Cross {name} ===")
        print(f"  LAT last ∩ XCL last = {len(lat_last & xcl_last)}")
        print(f"  LAT last ∖ XCL last = {len(lat_last - xcl_last)} (LAT-unique)")
        print(f"  XCL last ∖ LAT last = {len(xcl_last - lat_last)} (XCL-unique)")
        print(f"  baseline r0 ∖ LAT last = {len(baseline_r0 - lat_last)} (lost by LAT)")
        print(f"  baseline r0 ∖ XCL last = {len(baseline_r0 - xcl_last)} (lost by XCL)")
        print(f"  XCL last ∖ baseline r0 = {len(xcl_last - baseline_r0)} (XCL gain over r0)")
        print(f"  LAT last ∖ baseline r0 = {len(lat_last - baseline_r0)} (LAT gain over r0)")

    cross("MLP", mlp_lat_sets, mlp_xcl_sets, mlp_r0)
    cross("Transformer", tr_lat_sets, tr_xcl_sets, tr_r0)


if __name__ == "__main__":
    main()
