#!/usr/bin/env python3
"""
Export benchmark results from .data/runs to wasm/data/benchmarks.json

Creates a JSON file with:
- Summary statistics per prover/preset
- Individual problem results (for searchable table)
"""

import json
from collections import defaultdict
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_all_results(runs_dir: Path) -> list:
    """Load all cached results from .data/runs/"""
    results = []

    for prover_dir in sorted(runs_dir.iterdir()):
        if not prover_dir.is_dir():
            continue
        prover = prover_dir.name

        for preset_dir in sorted(prover_dir.iterdir()):
            if not preset_dir.is_dir():
                continue
            preset = preset_dir.name

            for result_file in sorted(preset_dir.glob("*.json")):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    results.append({
                        "prover": prover,
                        "preset": preset,
                        "problem": data["problem"],
                        "status": data["status"],
                        "time_s": data["time_s"],
                    })
                except Exception:
                    continue

    return results


def compute_summary(results: list) -> dict:
    """Compute summary statistics per prover/preset."""
    stats = defaultdict(lambda: {
        "total": 0,
        "proof": 0,
        "saturated": 0,
        "timeout": 0,
        "error": 0,
        "total_time": 0,
        "proof_times": [],
    })

    for r in results:
        key = f"{r['prover']}/{r['preset']}"
        stats[key]["total"] += 1
        stats[key][r["status"]] = stats[key].get(r["status"], 0) + 1
        stats[key]["total_time"] += r["time_s"]
        if r["status"] == "proof":
            stats[key]["proof_times"].append(r["time_s"])

    # Compute derived stats
    summary = []
    for key, s in sorted(stats.items()):
        prover, preset = key.split("/", 1)
        proof_rate = 100 * s["proof"] / s["total"] if s["total"] > 0 else 0
        avg_proof_time = sum(s["proof_times"]) / len(s["proof_times"]) if s["proof_times"] else 0

        summary.append({
            "prover": prover,
            "preset": preset,
            "total": s["total"],
            "proof": s["proof"],
            "saturated": s["saturated"],
            "timeout": s["timeout"],
            "error": s.get("error", 0),
            "proof_rate": round(proof_rate, 1),
            "avg_proof_time": round(avg_proof_time, 3),
            "total_time": round(s["total_time"], 1),
        })

    return summary


def compute_problem_comparison(results: list) -> dict:
    """Create a comparison matrix: problem -> {prover/preset -> status}"""
    problems = defaultdict(dict)

    for r in results:
        key = f"{r['prover']}/{r['preset']}"
        problems[r["problem"]][key] = {
            "status": r["status"],
            "time_s": r["time_s"],
        }

    return dict(problems)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export benchmark results for web display")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output file (default: wasm/data/benchmarks.json)")
    args = parser.parse_args()

    root = get_project_root()
    runs_dir = root / ".data" / "runs"
    output_path = args.output or root / "wasm" / "data" / "benchmarks.json"

    if not runs_dir.exists():
        print(f"Error: No benchmark results found at {runs_dir}")
        print("Run benchmarks first: proofatlas-bench")
        return

    print(f"Loading results from {runs_dir}...")
    results = load_all_results(runs_dir)
    print(f"  Loaded {len(results)} results")

    if not results:
        print("No results to export")
        return

    print("Computing summary statistics...")
    summary = compute_summary(results)

    print("Building problem comparison matrix...")
    problems = compute_problem_comparison(results)

    # Get list of all configs
    configs = sorted(set(f"{r['prover']}/{r['preset']}" for r in results))

    # Build output
    output = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "total_results": len(results),
        "configs": configs,
        "summary": summary,
        "problems": problems,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {output_path}")
    print(f"  Configurations: {len(configs)}")
    print(f"  Problems: {len(problems)}")

    # Print summary table
    print("\nSummary:")
    print(f"{'Prover':<12} {'Preset':<20} {'Total':>6} {'Proofs':>7} {'Rate':>7}")
    print("-" * 60)
    for s in summary:
        print(f"{s['prover']:<12} {s['preset']:<20} {s['total']:>6} {s['proof']:>7} {s['proof_rate']:>6.1f}%")


if __name__ == "__main__":
    main()
