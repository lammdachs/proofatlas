#!/usr/bin/env python3
"""Cross-labeling overlap statistics.

For a (runner, source) pair, compute on every problem P where the
runner failed and source succeeded:

    overlap(P) = |source_proof_clauses(P) ∩ runner_generated_clauses(P)|
                 -------------------------------------------------------
                                |source_proof_clauses(P)|

Reports the distribution (count, min, p10, p25, median, p75, p90, max, mean)
across problems for several (runner, source) pairs used in the paper.
"""
import json
import numpy as np
import os
from pathlib import Path
from statistics import median

ROOT = Path("/home/apluska/proofatlas/.data/traces")


def load_clause_strings(config: str, status: str, problem: str):
    p = ROOT / config / status / f"{problem}.clause_strings.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_labels(config: str, problem: str):
    p = ROOT / config / "solved" / f"{problem}.graph.npz"
    if not p.exists():
        return None
    g = np.load(p, allow_pickle=True)
    return g["labels"]


def proof_clauses(config: str, problem: str) -> set[str] | None:
    strings = load_clause_strings(config, "solved", problem)
    labels = load_labels(config, problem)
    if strings is None or labels is None:
        return None
    if len(strings) != len(labels):
        return None
    return {s for s, l in zip(strings, labels) if l == 1}


def runner_generated(runner: str, problem: str) -> set[str] | None:
    strings = load_clause_strings(runner, "unsolved", problem)
    if strings is None:
        return None
    return set(strings)


def list_status(config: str, status: str) -> set[str]:
    d = ROOT / config / status
    if not d.exists():
        return set()
    out = set()
    for f in os.listdir(d):
        if f.endswith(".clause_strings.json"):
            out.add(f[: -len(".clause_strings.json")])
    return out


def overlap_stats(runner: str, source: str):
    """Compute overlap distribution for a (runner, source) pair."""
    runner_failed = list_status(runner, "unsolved")
    source_solved = list_status(source, "solved")
    pairable = sorted(runner_failed & source_solved)

    overlaps = []
    proof_sizes = []
    skipped = 0
    for P in pairable:
        proof = proof_clauses(source, P)
        gen = runner_generated(runner, P)
        if proof is None or gen is None:
            skipped += 1
            continue
        if not proof:
            skipped += 1
            continue
        overlap = len(proof & gen) / len(proof)
        overlaps.append(overlap)
        proof_sizes.append(len(proof))

    return overlaps, proof_sizes, len(pairable), skipped


def summarize(overlaps, proof_sizes):
    if not overlaps:
        return "no problems"
    a = np.array(overlaps)
    return {
        "n_problems": len(a),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median_proof_size": int(np.median(proof_sizes)),
        "frac_full_overlap": float((a == 1.0).mean()),
        "frac_zero_overlap": float((a == 0.0).mean()),
    }


def main():
    # Canonical pairs: the cross-labeling sources actually used in our XCL chains.
    pairs = [
        # First-round XCL: runner is the round-0 model itself running the
        # benchmark; sources are age_weight (then becomes round-0 in priority).
        ("gcn_struct_mlp", "age_weight"),
        ("gcn_struct_transformer", "age_weight"),
        # Round-2 XCL: runner is xcl_r1; source is xcl_r1's predecessor (round 0)
        # but the priority-ordered list puts xcl_r1 first when it solves a problem;
        # for problems xcl_r1 also failed, age_weight is a fallback. Most common
        # mapping is xcl_r{n} runner + xcl_r{n-1} source, fallback age_weight.
        ("gcn_struct_mlp_xcl_r1", "gcn_struct_mlp"),
        ("gcn_struct_mlp_xcl_r1", "age_weight"),
        ("gcn_struct_mlp_xcl_r2", "gcn_struct_mlp_xcl_r1"),
        ("gcn_struct_mlp_xcl_r3", "gcn_struct_mlp_xcl_r2"),
        ("gcn_struct_transformer_xcl_r1", "gcn_struct_transformer"),
        ("gcn_struct_transformer_xcl_r1", "age_weight"),
        ("gcn_struct_transformer_xcl_r2", "gcn_struct_transformer_xcl_r1"),
        ("gcn_struct_transformer_xcl_r3", "gcn_struct_transformer_xcl_r2"),
    ]

    print(f"{'runner':40s} {'source':40s} {'n':>5s} {'med':>6s} {'mean':>6s} "
          f"{'p25':>6s} {'p75':>6s} {'min':>6s} {'max':>6s} {'=1':>5s} {'=0':>5s} "
          f"{'med|P|':>7s}")
    print("-" * 145)
    for runner, source in pairs:
        overlaps, sizes, n_pairable, skipped = overlap_stats(runner, source)
        s = summarize(overlaps, sizes)
        if isinstance(s, str):
            print(f"{runner:40s} {source:40s} -- {s}")
            continue
        print(f"{runner:40s} {source:40s} "
              f"{s['n_problems']:5d} "
              f"{s['median']:6.2f} "
              f"{s['mean']:6.2f} "
              f"{s['p25']:6.2f} "
              f"{s['p75']:6.2f} "
              f"{s['min']:6.2f} "
              f"{s['max']:6.2f} "
              f"{s['frac_full_overlap']:5.2f} "
              f"{s['frac_zero_overlap']:5.2f} "
              f"{s['median_proof_size']:7d}")


if __name__ == "__main__":
    main()
