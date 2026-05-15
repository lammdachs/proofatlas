#!/usr/bin/env python3
"""Generate paper figures from .data/throughput_ablation/.

Produces three figures in ML4SP/figures/:
  - cache_bars.pdf         headline 2x2 wall-time bars per cell
  - latency_cdf.pdf        per-iter CDF, eager vs batched (cache=T only)
  - speedup_vs_ci.pdf      per-problem batched/eager speedup vs |Δ|/iter

Also prints a summary table for the paper's headline.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / ".data" / "throughput_ablation"
OUT = ROOT / "ML4SP" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Cell ordering and palette
CELLS = ["cache_batched", "cache_eager", "nocache_batched", "nocache_eager", "cache_sequential"]
CELL_LABELS = {
    "cache_batched":     "cache, batched",
    "cache_eager":       "cache, eager",
    "nocache_batched":   "no cache, batched",
    "nocache_eager":     "no cache, eager",
    "cache_sequential":  "strawman (seq)",
}
CELL_COLORS = {
    "cache_batched":     "#456878",
    "cache_eager":       "#9E2D39",
    "nocache_batched":   "#B58C18",
    "nocache_eager":     "#7A5C00",
    "cache_sequential":  "#4A6444",
}


def load_runs():
    """Return runs[cell][problem] = list of per-rep dicts."""
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        runs[d["cell"]][d["problem"]].append(d)
    return runs


def per_problem_median_wall(runs, cell):
    """Return list of (problem, median_wall_s) for this cell."""
    out = []
    for prob, reps in runs.get(cell, {}).items():
        ts = [r["wall_s"] for r in reps]
        if ts:
            out.append((prob, statistics.median(ts)))
    return out


def fig_cache_bars(runs):
    """Headline figure: median per-problem wall time per cell."""
    rows = []
    for cell in CELLS:
        per_prob = per_problem_median_wall(runs, cell)
        if not per_prob:
            continue
        walls = [w for _, w in per_prob]
        rows.append((cell, statistics.median(walls),
                     statistics.median(walls) - np.percentile(walls, 25),
                     np.percentile(walls, 75) - statistics.median(walls),
                     len(walls)))
    if not rows:
        return None
    cells, medians, lo, hi, ns = zip(*rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(cells))
    bars = ax.bar(x, medians, yerr=[lo, hi], color=[CELL_COLORS[c] for c in cells],
                  edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in cells], rotation=0, fontsize=9)
    ax.set_ylabel("Median per-problem wall time (s)")
    ax.set_title(f"Wall time by cell (median over {ns[0]} problems, error bars: IQR)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, n) in enumerate(zip(medians, ns)):
        ax.text(i, m + max(hi) * 0.1, f"{m:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = OUT / "cache_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_latency_cdf(runs):
    """Per-iter latency CDF: cache_batched vs cache_eager."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for cell in ("cache_batched", "cache_eager"):
        ts = []
        for prob, reps in runs.get(cell, {}).items():
            for r in reps:
                prof = r.get("profile") or {}
                for x in (prof.get("iter_trace") or [])[1:]:  # skip cold start
                    ts.append(x["t_total_ns"] / 1e6)
        if not ts:
            continue
        ts.sort()
        y = np.arange(1, len(ts) + 1) / len(ts)
        ax.plot(ts, y, label=CELL_LABELS[cell], color=CELL_COLORS[cell], linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Per-iteration wall time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-iteration latency CDF (cache=T)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "latency_cdf.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_speedup_vs_ci(runs):
    """Per-problem speedup (batched_walls / eager_walls) vs |Δ|/iter (cache=T)."""
    rb = runs.get("cache_batched", {})
    re_ = runs.get("cache_eager", {})
    xs, ys = [], []
    for prob in set(rb) & set(re_):
        bw = statistics.median([r["wall_s"] for r in rb[prob]])
        ew = statistics.median([r["wall_s"] for r in re_[prob]])
        if bw <= 0:
            continue
        speedup = ew / bw   # >1 means batched is faster
        # Estimate |Δ|/iter from the trace
        cg = []
        for r in rb[prob]:
            prof = r.get("profile") or {}
            for x in (prof.get("iter_trace") or [])[1:]:
                cg.append(x.get("clauses_generated", 0))
        if not cg:
            continue
        xs.append(statistics.mean(cg))
        ys.append(speedup)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(xs, ys, color="#456878", alpha=0.75, s=40, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"Mean $|\Delta|$/iteration")
    ax.set_ylabel("Speedup: eager / batched")
    ax.set_title("Per-problem batched-vs-eager speedup (cache=T)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "speedup_vs_ci.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def print_table(runs):
    print("\n=== headline numbers ===")
    print(f"{'cell':22s}  n_problems  median_wall_s   IQR_lo  IQR_hi")
    for cell in CELLS:
        per_prob = per_problem_median_wall(runs, cell)
        if not per_prob:
            print(f"{cell:22s}  (no data)")
            continue
        walls = sorted([w for _, w in per_prob])
        med = statistics.median(walls)
        q25 = np.percentile(walls, 25)
        q75 = np.percentile(walls, 75)
        print(f"{cell:22s}  {len(walls):10d}  {med:12.2f}   {q25:6.2f}  {q75:6.2f}")


def main():
    runs = load_runs()
    f1 = fig_cache_bars(runs)
    f2 = fig_latency_cdf(runs)
    f3 = fig_speedup_vs_ci(runs)
    for f in (f1, f2, f3):
        if f: print(f"wrote {f.relative_to(ROOT)}")
    print_table(runs)


if __name__ == "__main__":
    main()
