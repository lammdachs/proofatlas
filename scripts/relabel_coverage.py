#!/usr/bin/env python3
"""How many failed attempts get cross-labeled (bench-authoritative).

Reads the per-problem run JSONs (.data/runs/proofatlas/<config>/*.json)
once per config, caches the solved/failed sets, and reports for each
XCL training round:
  - runner solved/failed
  - relabel coverage = fraction of runner-failed for which some
    source in the priority list has a proof
  - total positive-label-bearing training problems = solved + relabeled
"""
import json
from pathlib import Path

RUNS = Path("/home/apluska/proofatlas/.data/runs/proofatlas")
_CACHE: dict[str, tuple[frozenset, frozenset]] = {}


def status_sets(config: str):
    if config in _CACHE:
        return _CACHE[config]
    d = RUNS / config
    if not d.exists():
        _CACHE[config] = (frozenset(), frozenset())
        return _CACHE[config]
    solved, failed = set(), set()
    for f in d.iterdir():
        if not f.name.endswith(".json"):
            continue
        with f.open() as fh:
            r = json.load(fh)
        problem = r["problem"]
        if r["status"] == "proof":
            solved.add(problem)
        else:
            failed.add(problem)
    _CACHE[config] = (frozenset(solved), frozenset(failed))
    return _CACHE[config]


def coverage(runner: str, sources: list[str]):
    runner_solved, runner_failed = status_sets(runner)
    union_source_solved: set = set()
    for s in sources:
        ss, _ = status_sets(s)
        union_source_solved |= ss
    relabeled = runner_failed & union_source_solved
    new_solved = runner_solved - union_source_solved
    return {
        "runner_solved": len(runner_solved),
        "runner_failed": len(runner_failed),
        "relabeled": len(relabeled),
        "new": len(new_solved),
    }


def main():
    mlp_lat = [f"gcn_struct_mlp_lat_r{i}" for i in range(1, 8)]
    mlp_xcl = [f"gcn_struct_mlp_xcl_r{i}" for i in range(1, 8)]
    tr_lat = [f"gcn_struct_transformer_lat_r{i}" for i in range(1, 8)]
    tr_xcl = [f"gcn_struct_transformer_xcl_r{i}" for i in range(1, 8)]

    def chain(prefix: str, rounds: list[str], r0: str):
        print(f"\n=== {prefix} XCL chain ===")
        print(f"{'round':6s} {'runner':36s} "
              f"{'solved':>7s} {'failed':>7s} {'relab':>7s} {'new':>7s}")
        # round 1 training: runner = r0; sources for new-counting = [age_weight] only
        # (r0 is the runner itself; previous configs are baselines outside the chain)
        runner = r0
        sources_relab = [r0, "age_weight"]
        sources_new = ["age_weight"]
        s = coverage(runner, sources_relab)
        # recompute new with sources_new
        rs, _ = status_sets(runner)
        union_new_src = set()
        for src in sources_new:
            ss, _ = status_sets(src)
            union_new_src |= ss
        s["new"] = len(rs - union_new_src)
        print(f"{'r1':6s} {runner:36s} "
              f"{s['runner_solved']:7d} {s['runner_failed']:7d} "
              f"{s['relabeled']:7d} {s['new']:7d}")
        # round n training (n>=2): runner = xcl_{n-1};
        # sources = (xcl_{n-2}, ..., xcl_1, r0, age_weight).
        # Only emit rows for training rounds whose produced model exists
        # (i.e., n in {2..len(rounds)}).
        for n in range(2, len(rounds) + 1):
            runner = rounds[n - 2]
            prior = list(reversed(rounds[: n - 2]))
            sources = prior + [r0, "age_weight"]
            s = coverage(runner, sources)
            print(f"{'r' + str(n):6s} {runner:36s} "
                  f"{s['runner_solved']:7d} {s['runner_failed']:7d} "
                  f"{s['relabeled']:7d} {s['new']:7d}")

    chain("MLP LAT", mlp_lat, "gcn_struct_mlp")
    chain("MLP XCL", mlp_xcl, "gcn_struct_mlp")
    chain("Tr LAT", tr_lat, "gcn_struct_transformer")
    chain("Tr XCL", tr_xcl, "gcn_struct_transformer")


if __name__ == "__main__":
    main()
