#!/usr/bin/env python3
"""Determinism check: for each (config, problem), verify that the async and
sequential runs produced the same iter count, proof outcome, and final |U|/|P|.

If iter count or proof outcome differs, the sequential drain has changed prover
semantics — a bug. Reads .data/throughput/*.json produced by throughput_bench.py.

Usage:
    python scripts/throughput_determinism.py
    python scripts/throughput_determinism.py --rep 0    # check rep 0 only
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / ".data" / "throughput"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=0,
                    help="Which rep to compare (modes must agree for at least this rep)")
    args = ap.parse_args()

    # Group: cells[(config, problem)][mode] = result dict
    cells = defaultdict(dict)
    for f in OUT.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("rep") != args.rep:
            continue
        if "error" in d:
            continue
        cells[(d["config"], d["problem"])][d["mode"]] = d

    diffs = []
    same = 0
    for (cfg, prob), by_mode in sorted(cells.items()):
        modes = sorted(by_mode)
        if "async" not in modes or "sequential" not in modes:
            continue  # not an ML cell (age_weight has only "noop")
        a = by_mode["async"]
        s = by_mode["sequential"]
        diff_fields = []
        if a["iterations"] != s["iterations"]:
            diff_fields.append(f"iter {a['iterations']} vs {s['iterations']}")
        if a["proof_found"] != s["proof_found"]:
            diff_fields.append(f"proof {a['proof_found']} vs {s['proof_found']}")
        if a["status"] != s["status"]:
            diff_fields.append(f"status {a['status']} vs {s['status']}")
        if a.get("clause_count") != s.get("clause_count"):
            diff_fields.append(f"clauses {a.get('clause_count')} vs {s.get('clause_count')}")
        if diff_fields:
            diffs.append((cfg, prob, diff_fields))
        else:
            same += 1

    print(f"Compared {same + len(diffs)} (config, problem) pairs across async vs sequential.")
    print(f"Matched (deterministic): {same}")
    print(f"Diverged:                {len(diffs)}")
    if diffs:
        print()
        print("Divergences:")
        for cfg, prob, fields in diffs:
            print(f"  {cfg}  {prob}: {'; '.join(fields)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
