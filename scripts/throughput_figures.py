#!/usr/bin/env python3
"""Generate paper figures from raw throughput-bench data.

Outputs to ML4SP/figures/:
  - latency_cdf.pdf        per-iter latency CDF, one curve per (config, mode)
  - stage_breakdown.pdf    per-stage median bars, async vs sequential side-by-side
  - speedup_vs_ci.pdf      c/i × (seq→async speedup), one marker per (problem, config)

Reads .data/throughput/*.json.
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
DATA = ROOT / ".data" / "throughput"
OUT = ROOT / "ML4SP" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# lammdachs palette (matches web/training)
COLORS = {
    "gcn_struct_transformer_xcl_r6": {"async": "#456878", "sequential": "#9E2D39", "noop": "#456878"},
    "gcn_struct_mlp_xcl_r6":         {"async": "#B58C18", "sequential": "#7A5C00", "noop": "#B58C18"},
    "age_weight":                    {"async": "#4A6444", "sequential": "#4A6444", "noop": "#4A6444"},
}
SHORT = {
    "gcn_struct_transformer_xcl_r6": "transformer",
    "gcn_struct_mlp_xcl_r6":         "mlp",
    "age_weight":                    "age-weight",
}
STAGES = [
    ("t_forward_simplify_ns",  "forward simp",   "#7A5C00"),
    ("t_backward_simplify_ns", "backward simp",  "#B58C18"),
    ("t_select_ns",            "select",         "#9E2D39"),
    ("t_generate_ns",          "generate",       "#456878"),
    ("t_add_inferences_ns",    "add",            "#4A6444"),
]


def load_cells(skip_cold=1):
    """Return cells[(config, mode)] = list of (problem, [iter_trace_entries])."""
    cells = defaultdict(list)
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        prof = d.get("profile") or {}
        trace = prof.get("iter_trace", [])[skip_cold:]
        if not trace:
            continue
        cells[(d["config"], d["mode"])].append({
            "problem": d["problem"],
            "rep": d["rep"],
            "wall_s": d["wall_s"],
            "iterations": d["iterations"],
            "trace": trace,
        })
    return cells


def fig_latency_cdf(cells):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (cfg, mode), reps in sorted(cells.items()):
        if mode == "noop":
            continue
        ts = []
        for r in reps:
            ts.extend(x["t_total_ns"] / 1e6 for x in r["trace"])
        if not ts:
            continue
        ts = sorted(ts)
        y = np.arange(1, len(ts) + 1) / len(ts)
        ax.plot(ts, y, label=f"{SHORT[cfg]} ({mode})", color=COLORS[cfg][mode],
                linestyle="-" if mode == "async" else "--", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Per-iteration wall time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-iteration latency CDF (pooled across problems and reps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = OUT / "latency_cdf.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_stage_breakdown(cells):
    """Per (config, mode), median stage time in ms, stacked bar."""
    # Order configs / modes
    ordering = []
    for cfg in ["gcn_struct_transformer_xcl_r6", "gcn_struct_mlp_xcl_r6", "age_weight"]:
        for mode in ["async", "sequential", "noop"]:
            if (cfg, mode) in cells:
                ordering.append((cfg, mode))
    if not ordering:
        return None
    labels = [f"{SHORT[c]}\n{m}" for c, m in ordering]
    stage_medians = {key: [] for key, _, _ in STAGES}
    for c, m in ordering:
        # Pool all iters across all reps and problems
        traces = []
        for r in cells[(c, m)]:
            traces.extend(r["trace"])
        for key, _, _ in STAGES:
            vals = [x[key] / 1e6 for x in traces]
            stage_medians[key].append(statistics.median(vals) if vals else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ordering))
    bottom = np.zeros(len(ordering))
    for key, label, color in STAGES:
        vals = np.array(stage_medians[key])
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median per-iter time (ms)")
    ax.set_title("Per-stage breakdown of one given-clause iteration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "stage_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_speedup_vs_ci(cells):
    """Per (problem, ML config): x=max c/i, y=async iter/s ÷ sequential iter/s.
    >1 means async is faster."""
    # First compute iters/s per (problem, config, mode) using median wall_s
    by_pcm = defaultdict(list)
    for (cfg, mode), reps in cells.items():
        for r in reps:
            by_pcm[(r["problem"], cfg, mode)].append((r["wall_s"], r["iterations"], r["trace"]))
    # Compute median c/i per problem (across all configs)
    by_problem_ci = defaultdict(list)
    for (prob, cfg, mode), runs in by_pcm.items():
        for w, it, tr in runs:
            if it > 0:
                # avg clauses_generated per iter
                cg = [x["clauses_generated"] for x in tr]
                if cg:
                    by_problem_ci[prob].append(sum(cg) / len(cg))
    prob_ci = {p: max(v) for p, v in by_problem_ci.items() if v}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg in ["gcn_struct_transformer_xcl_r6", "gcn_struct_mlp_xcl_r6"]:
        xs, ys = [], []
        for prob in prob_ci:
            a = by_pcm.get((prob, cfg, "async"))
            s = by_pcm.get((prob, cfg, "sequential"))
            if not a or not s:
                continue
            a_w = statistics.median([x[0] for x in a])
            s_w = statistics.median([x[0] for x in s])
            a_i = statistics.median([x[1] for x in a])
            s_i = statistics.median([x[1] for x in s])
            if a_w <= 0 or s_w <= 0:
                continue
            a_ips = a_i / a_w
            s_ips = s_i / s_w
            speedup = a_ips / s_ips if s_ips > 0 else float("nan")
            xs.append(prob_ci[prob])
            ys.append(speedup)
        ax.scatter(xs, ys, label=SHORT[cfg],
                   color=COLORS[cfg]["async"], alpha=0.7, s=40, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"Per-iter |$\Delta$| (mean clauses generated per iter)")
    ax.set_ylabel("Speedup: async iter/s ÷ sequential iter/s")
    ax.set_xscale("log")
    ax.set_title(r"Async speedup vs problem |$\Delta$|/iter")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "speedup_vs_ci.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_iter_gantt():
    """Schematic Gantt-style diagram of one given-clause iteration, sync vs async.

    Shows where the prover thread waits and how that wait moves off-thread under
    the async architecture. Schematic (not derived from a single run): aims to
    illustrate the overlap mechanism, not particular wall-clock numbers.
    """
    import matplotlib.patches as patches

    # Timings (ms). |Delta|=4 new clauses; per-embed backend cost t_e=3 ms;
    # per-clause forward/backward simp f=b=1 ms; score_context t_s=4 ms;
    # activate+generate+add = 1 ms; channel/submit overhead negligible (0.1 ms).
    f, b, t_e, t_s, taga = 1.0, 1.0, 3.0, 4.0, 1.0
    delta = 4

    # ---- sync schedule ----
    sync_prover = []
    sync_backend = []
    t = 0.0
    for i in range(delta):
        sync_prover.append(("forward", t, f, "#7A5C00")); t += f
        sync_prover.append(("submit", t, 0.2, "#B58C18"))
        sync_backend.append(("embed", t + 0.2, t_e, "#456878"))
        sync_prover.append(("wait", t + 0.2, t_e, "#cccccc")); t += 0.2 + t_e
        sync_prover.append(("backward", t, b, "#9E2D39")); t += b
    sync_prover.append(("select", t, 0.2, "#B58C18"))
    sync_backend.append(("score", t + 0.2, t_s, "#456878"))
    sync_prover.append(("wait", t + 0.2, t_s, "#cccccc")); t += 0.2 + t_s
    sync_prover.append(("act/gen/add", t, taga, "#4A6444")); t += taga
    sync_total = t

    # ---- async schedule ----
    async_prover = []
    async_backend = []
    t = 0.0
    submit_times = []  # backend can start whenever a request arrives
    for i in range(delta):
        async_prover.append(("forward", t, f, "#7A5C00")); t += f
        async_prover.append(("submit", t, 0.2, "#B58C18"))
        submit_times.append(t + 0.2)
        t += 0.2
        async_prover.append(("backward", t, b, "#9E2D39")); t += b
    # Backend grabs the first request when free, batches subsequent ones that arrive
    # during the forward pass; in our schematic the backend can keep up so we
    # show ONE batched forward pass over the queued requests.
    batch_start = submit_times[0]
    batch_end = batch_start + t_e
    # Drain wait: prover at this point is at time t; if batch_end > t, prover waits
    async_backend.append(("embed batch", batch_start, t_e, "#456878"))
    async_prover.append(("select", t, 0.2, "#B58C18"))
    t += 0.2
    if batch_end > t:
        async_prover.append(("drain", t, batch_end - t, "#cccccc"))
        t = batch_end
    async_backend.append(("score", t, t_s, "#456878"))
    async_prover.append(("wait", t, t_s, "#cccccc")); t += t_s
    async_prover.append(("act/gen/add", t, taga, "#4A6444")); t += taga
    async_total = t

    # ---- draw ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.15})
    xmax = max(sync_total, async_total) + 0.5

    def draw_lane(ax, entries, title):
        for label, start, dur, color in entries:
            rect = patches.Rectangle((start, 0.1), dur, 0.8, facecolor=color,
                                     edgecolor="black", linewidth=0.4)
            ax.add_patch(rect)
            if dur >= 0.8:
                ax.text(start + dur / 2, 0.5, label, ha="center", va="center",
                        fontsize=7, color="white" if color in ("#9E2D39", "#456878", "#7A5C00", "#4A6444") else "black")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([title], fontsize=9)
        ax.set_xticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_visible(True)

    draw_lane(axes[0], sync_prover,    "sync, prover")
    draw_lane(axes[1], sync_backend,   "sync, backend")
    draw_lane(axes[2], async_prover,   "async, prover")
    draw_lane(axes[3], async_backend,  "async, backend")

    axes[-1].set_xticks(list(range(0, int(xmax) + 1, 2)))
    axes[-1].set_xlabel("time (ms, schematic)")

    # Vertical reference lines at end of each iteration
    for ax in axes:
        ax.axvline(sync_total, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.axvline(async_total, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    axes[0].set_title(
        r"One given-clause iteration with $|\Delta|=4$: how the embed wait moves off the prover thread under async",
        fontsize=9.5, pad=4,
    )
    fig.tight_layout()
    out = OUT / "iteration_gantt.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_worker_scaling():
    """Aggregate iter/s vs number of prover workers; one curve per (config, mode)."""
    src = ROOT / ".data" / "throughput_workers"
    if not src.exists():
        return None
    rows = [json.loads(f.read_text()) for f in src.glob("*.json")]
    if not rows:
        return None
    fig, (ax_agg, ax_pw) = plt.subplots(1, 2, figsize=(11, 4.3))
    for cfg in ["gcn_struct_transformer_xcl_r6", "gcn_struct_mlp_xcl_r6"]:
        for mode in ["async", "sequential"]:
            sub = sorted([r for r in rows if r["config"] == cfg and r["mode"] == mode],
                         key=lambda r: r["n_workers"])
            if not sub:
                continue
            xs = [r["n_workers"] for r in sub]
            ys_agg = [r["aggregate_iter_per_s"] for r in sub]
            ys_pw  = [r["per_worker_iter_per_s"] for r in sub]
            ax_agg.plot(xs, ys_agg, marker="o", label=f"{SHORT[cfg]} ({mode})",
                        color=COLORS[cfg][mode],
                        linestyle="-" if mode == "async" else "--")
            ax_pw.plot(xs, ys_pw, marker="o", label=f"{SHORT[cfg]} ({mode})",
                       color=COLORS[cfg][mode],
                       linestyle="-" if mode == "async" else "--")
    ax_agg.set_xlabel("Prover workers")
    ax_agg.set_ylabel("Aggregate iter/s")
    ax_agg.set_title("Aggregate throughput (8-problem subset)")
    ax_agg.legend(fontsize=8, loc="best")
    ax_agg.grid(True, alpha=0.3)
    ax_agg.set_xticks([1, 2, 4, 8])
    ax_pw.set_xlabel("Prover workers")
    ax_pw.set_ylabel("Per-worker iter/s")
    ax_pw.set_title("Per-worker throughput")
    ax_pw.legend(fontsize=8, loc="best")
    ax_pw.grid(True, alpha=0.3)
    ax_pw.set_xticks([1, 2, 4, 8])
    fig.tight_layout()
    out = OUT / "worker_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    cells = load_cells()
    print(f"Loaded {sum(len(v) for v in cells.values())} run records across {len(cells)} (config, mode) cells")
    f1 = fig_latency_cdf(cells)
    f2 = fig_iter_gantt()
    f3 = fig_speedup_vs_ci(cells)
    f4 = fig_worker_scaling()
    for f in (f1, f2, f3, f4):
        if f: print(f"wrote {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
