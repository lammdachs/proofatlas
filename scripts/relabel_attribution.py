#!/usr/bin/env python3
"""For each XCL training round, attribute cross-labels to source generation.

The priority-ordered source list at round n is
  (xcl_{n-1}, ..., xcl_1, r0, age_weight).
For each runner-failed problem that some source has a proof for, the
relabel is taken from the FIRST matching source. This script counts,
per training round, how many relabels come from each source.
"""
import json
import os
from pathlib import Path

RUNS = Path("/home/apluska/proofatlas/.data/runs/proofatlas")
_CACHE: dict[str, frozenset] = {}


def solved_set(config: str) -> frozenset:
    if config in _CACHE:
        return _CACHE[config]
    d = RUNS / config
    if not d.exists():
        _CACHE[config] = frozenset()
        return _CACHE[config]
    out: set = set()
    for f in d.iterdir():
        if not f.name.endswith(".json"):
            continue
        with f.open() as fh:
            r = json.load(fh)
        if r["status"] == "proof":
            out.add(r["problem"])
    _CACHE[config] = frozenset(out)
    return _CACHE[config]


def failed_set(config: str) -> frozenset:
    """All problems where the config did NOT produce a proof."""
    if config not in _CACHE or _CACHE.get(config + "@failed") is None:
        d = RUNS / config
        if not d.exists():
            return frozenset()
        out: set = set()
        for f in d.iterdir():
            if not f.name.endswith(".json"):
                continue
            with f.open() as fh:
                r = json.load(fh)
            if r["status"] != "proof":
                out.add(r["problem"])
        _CACHE[config + "@failed"] = frozenset(out)
    return _CACHE[config + "@failed"]


def attribute(runner: str, sources: list[str]) -> dict[str, int]:
    """For each runner-failed problem, attribute the relabel to first matching source."""
    fails = failed_set(runner)
    counts = {s: 0 for s in sources}
    for P in fails:
        for s in sources:
            if P in solved_set(s):
                counts[s] += 1
                break
    return counts


def main():
    chains = {
        "MLP": ("gcn_struct_mlp", [f"gcn_struct_mlp_xcl_r{i}" for i in range(1, 8)]),
        "Tr": ("gcn_struct_transformer", [f"gcn_struct_transformer_xcl_r{i}" for i in range(1, 8)]),
    }
    for label, (r0, rounds) in chains.items():
        print(f"\n=== {label} XCL chain: relabel attribution per round ===")
        # build header — sources can be up to len(rounds) prior XCL + r0 + aw
        all_sources = list(reversed(rounds)) + [r0, "age_weight"]
        # Print header with shortened source names
        short = lambda s: ("xcl_r" + s.split("_r")[-1]) if "_xcl_r" in s else (s.replace("gcn_struct_", "") if "gcn_struct" in s else s)
        header = " ".join(f"{short(s):>9s}" for s in all_sources)
        print(f"{'round':5s} {'total':>6s}  {header}")
        # round 1: runner = r0; sources = [r0, age_weight]
        runner = r0
        sources = [r0, "age_weight"]
        c = attribute(runner, sources)
        # pad with 0s for positions corresponding to xcl rounds (none yet at round 1)
        cells = []
        for s in all_sources:
            cells.append(f"{c.get(s, 0):>9d}")
        total = sum(c.values())
        print(f"{'r1':5s} {total:>6d}  {' '.join(cells)}")
        # rounds 2..len(rounds)
        for n in range(2, len(rounds) + 1):
            runner = rounds[n - 2]
            prior = list(reversed(rounds[: n - 2]))  # xcl_{n-2}, ..., xcl_1
            sources = prior + [r0, "age_weight"]
            c = attribute(runner, sources)
            cells = []
            for s in all_sources:
                cells.append(f"{c.get(s, 0):>9d}")
            total = sum(c.values())
            print(f"{'r' + str(n):5s} {total:>6d}  {' '.join(cells)}")


if __name__ == "__main__":
    main()
