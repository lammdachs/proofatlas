#!/usr/bin/env python3
"""
Export benchmark and training results for web display.

USAGE:
    python scripts/export.py                              # Export all results
    python scripts/export.py --benchmarks                 # Export benchmarks only
    python scripts/export.py --training                   # Export training only
    python scripts/export.py --problem-set test           # Limit to 'test' problem set
    python scripts/export.py --configs proofatlas vampire # Only include these provers
    python scripts/export.py --base-only                  # Skip learned selectors

OUTPUT:
    web/data/benchmarks.json  - Benchmark results per prover/preset
    web/data/training.json    - Training runs and available weights
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_problem_set(root: Path, problem_set_name: str) -> set[str]:
    """Load problem names from a problem set definition."""
    tptp_config_path = root / "configs" / "tptp.json"
    if not tptp_config_path.exists():
        raise FileNotFoundError(f"TPTP config not found: {tptp_config_path}")

    with open(tptp_config_path) as f:
        tptp_config = json.load(f)

    problem_sets = tptp_config.get("problem_sets", {})
    if problem_set_name not in problem_sets:
        available = list(problem_sets.keys())
        raise ValueError(f"Unknown problem set: {problem_set_name}. Available: {available}")

    filters = problem_sets[problem_set_name]
    problems_dir = root / tptp_config["paths"]["problems"]

    if not problems_dir.exists():
        raise FileNotFoundError(f"TPTP problems not found: {problems_dir}")

    # Load metadata
    metadata_path = root / ".data" / "problem_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Problem metadata not found: {metadata_path}\n"
            "Run: python scripts/setup_tptp.py --scan"
        )

    with open(metadata_path) as f:
        data = json.load(f)
        problems_list = data.get("problems", data) if isinstance(data, dict) else data
        metadata = {p["path"]: p for p in problems_list}

    # Filter problems
    matching = set()
    for path, meta in metadata.items():
        # Status filter
        if "status" in filters:
            if meta.get("status", "").lower() not in [s.lower() for s in filters["status"]]:
                continue

        # Format filter
        if "format" in filters:
            if meta.get("format", "").lower() not in [f.lower() for f in filters["format"]]:
                continue

        # Domain filter (include)
        if "domains" in filters:
            domain = path.split("/")[0] if "/" in path else path[:3]
            if domain not in filters["domains"]:
                continue

        # Domain filter (exclude)
        if "exclude_domains" in filters:
            domain = path.split("/")[0] if "/" in path else path[:3]
            if domain in filters["exclude_domains"]:
                continue

        # Rating filter
        if "max_rating" in filters:
            if meta.get("rating", 1.0) > filters["max_rating"]:
                continue

        # Clause count filter
        if "max_clauses" in filters:
            if meta.get("num_clauses", float("inf")) > filters["max_clauses"]:
                continue

        # Equality filter
        if "has_equality" in filters:
            if meta.get("has_equality", False) != filters["has_equality"]:
                continue

        # Unit-only filter
        if "is_unit_only" in filters:
            if meta.get("is_unit_only", False) != filters["is_unit_only"]:
                continue

        # Extract problem name from path (e.g., "PUZ/PUZ001-1.p" -> "PUZ001-1.p")
        problem_name = path.split("/")[-1] if "/" in path else path
        matching.add(problem_name)

    return matching


# Benchmark export

def load_benchmark_results(runs_dir: Path) -> list:
    """Load all cached results from .data/runs/"""
    results = []

    if not runs_dir.exists():
        return results

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


def filter_results(
    results: list,
    problem_set: set[str] | None = None,
    configs: list[str] | None = None,
    base_only: bool = False,
) -> list:
    """Filter results by problem set and/or configs."""
    filtered = results

    if problem_set is not None:
        filtered = [r for r in filtered if r["problem"] in problem_set]

    if configs is not None:
        # configs can be "prover" or "prover/preset"
        def matches_config(r):
            for cfg in configs:
                if "/" in cfg:
                    prover, preset = cfg.split("/", 1)
                    if r["prover"] == prover and r["preset"] == preset:
                        return True
                else:
                    if r["prover"] == cfg:
                        return True
            return False
        filtered = [r for r in filtered if matches_config(r)]

    if base_only:
        # Skip learned selectors (presets with "model" in the name)
        filtered = [r for r in filtered if "model" not in r["preset"]]

    return filtered


def compute_benchmark_summary(results: list) -> list:
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


def export_benchmarks(
    root: Path,
    output_path: Path,
    problem_set_name: str | None = None,
    configs: list[str] | None = None,
    base_only: bool = False,
):
    """Export benchmark results to JSON."""
    runs_dir = root / ".data" / "runs"

    print(f"Loading benchmark results from {runs_dir}...")
    results = load_benchmark_results(runs_dir)
    print(f"  Loaded {len(results)} results")

    if not results:
        print("No benchmark results to export")
        return False

    # Load problem set if specified
    problem_set = None
    if problem_set_name:
        print(f"Loading problem set '{problem_set_name}'...")
        problem_set = load_problem_set(root, problem_set_name)
        print(f"  Problem set contains {len(problem_set)} problems")

    # Filter results
    if problem_set or configs or base_only:
        results = filter_results(results, problem_set, configs, base_only)
        print(f"  After filtering: {len(results)} results")

    if not results:
        print("No results match the filters")
        return False

    summary = compute_benchmark_summary(results)
    problems = compute_problem_comparison(results)
    found_configs = sorted(set(f"{r['prover']}/{r['preset']}" for r in results))

    output = {
        "generated": datetime.now().isoformat(),
        "problem_set": problem_set_name,
        "configs_filter": configs,
        "total_results": len(results),
        "configs": found_configs,
        "summary": summary,
        "problems": problems,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported benchmarks to {output_path}")
    print(f"  Configurations: {len(found_configs)}")
    print(f"  Problems: {len(problems)}")

    # Print summary table
    print("\nBenchmark Summary:")
    print(f"{'Prover':<12} {'Preset':<20} {'Total':>6} {'Proofs':>7} {'Rate':>7}")
    print("-" * 60)
    for s in summary:
        print(f"{s['prover']:<12} {s['preset']:<20} {s['total']:>6} {s['proof']:>7} {s['proof_rate']:>6.1f}%")

    return True


# Training export

def load_training_runs(logs_dir: Path) -> list:
    """Load all training results from .logs/bench_* directories."""
    runs = []

    if not logs_dir.exists():
        return runs

    # Look for bench_* directories (unified bench.py output)
    for run_dir in sorted(logs_dir.glob("bench_*")):
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "summary.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file) as f:
                results = json.load(f)

            # Parse training progress from log file
            log_file = run_dir / "bench.log"
            loss_history = []
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        if line.startswith("TRAIN:"):
                            parts = line.strip().split(":")
                            if len(parts) >= 4:
                                loss_history.append({
                                    "epoch": int(parts[1]),
                                    "max_epochs": int(parts[2]),
                                    "train_loss": float(parts[3]),
                                })

            runs.append({
                "name": run_dir.name,
                "preset": results.get("preset", "unknown"),
                "proofs": results.get("proofs", 0),
                "total": results.get("total", 0),
                "proof_rate": results.get("proof_rate", "0%"),
                "weights": results.get("weights"),
                "loss_history": loss_history,
            })
        except Exception as e:
            print(f"  Warning: Failed to load {run_dir.name}: {e}")
            continue

    return runs


def load_available_weights(weights_dir: Path) -> list:
    """Load list of available trained weights."""
    weights = []

    if not weights_dir.exists():
        return weights

    for weights_file in sorted(weights_dir.glob("*.safetensors")):
        try:
            from safetensors import safe_open
            with safe_open(weights_file, framework="pt") as f:
                metadata = f.metadata() or {}

            weights.append({
                "name": weights_file.stem,
                "file": weights_file.name,
                "model_type": metadata.get("model_type", "unknown"),
                "hidden_dim": int(metadata.get("hidden_dim", 0)),
                "num_layers": int(metadata.get("num_layers", 0)),
                "size_bytes": weights_file.stat().st_size,
            })
        except ImportError:
            weights.append({
                "name": weights_file.stem,
                "file": weights_file.name,
                "size_bytes": weights_file.stat().st_size,
            })
        except Exception:
            continue

    return weights


def load_architectures(models_config_path: Path) -> dict:
    """Load available model architectures from config."""
    if not models_config_path.exists():
        return {}

    with open(models_config_path) as f:
        config = json.load(f)

    return config.get("architectures", {})


def export_training(root: Path, output_path: Path):
    """Export training results to JSON."""
    logs_dir = root / ".logs"
    weights_dir = root / ".weights"
    models_config = root / "configs" / "models.json"

    print(f"Loading training runs from {logs_dir}...")
    runs = load_training_runs(logs_dir)
    print(f"  Found {len(runs)} training runs")

    print(f"Loading available weights from {weights_dir}...")
    weights = load_available_weights(weights_dir)
    print(f"  Found {len(weights)} weight files")

    print(f"Loading model architectures...")
    architectures = load_architectures(models_config)
    print(f"  Found {len(architectures)} architectures")

    # Find best run by proof rate
    best_run = None
    if runs:
        for run in runs:
            rate_str = run.get("proof_rate", "0%").rstrip("%")
            try:
                rate = float(rate_str)
                if best_run is None or rate > float(best_run.get("proof_rate", "0%").rstrip("%")):
                    best_run = run
            except ValueError:
                pass

    output = {
        "generated": datetime.now().isoformat(),
        "total_runs": len(runs),
        "total_weights": len(weights),
        "summary": {
            "best_preset": best_run["preset"] if best_run else None,
            "best_proof_rate": best_run["proof_rate"] if best_run else None,
        },
        "architectures": architectures,
        "weights": weights,
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported training to {output_path}")

    if weights:
        print("\nAvailable Weights:")
        for w in weights:
            size_kb = w.get("size_bytes", 0) / 1024
            print(f"  {w['name']}: {w.get('model_type', 'unknown')} ({size_kb:.1f} KB)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export results for web display")
    parser.add_argument("--benchmarks", action="store_true",
                       help="Export benchmarks only")
    parser.add_argument("--training", action="store_true",
                       help="Export training only")
    parser.add_argument("--problem-set", type=str, metavar="NAME",
                       help="Limit to problems in this problem set (e.g., test, default)")
    parser.add_argument("--configs", type=str, nargs="+", metavar="CFG",
                       help="Only include these provers/configs (e.g., proofatlas vampire/time_sel0)")
    parser.add_argument("--base-only", action="store_true",
                       help="Only run base configs (skip learned selectors)")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory (default: web/data/)")
    args = parser.parse_args()

    root = get_project_root()
    output_dir = args.output_dir or root / "web" / "data"

    # If neither flag specified, export both
    export_both = not args.benchmarks and not args.training

    if args.benchmarks or export_both:
        export_benchmarks(root, output_dir / "benchmarks.json",
                         problem_set_name=args.problem_set,
                         configs=args.configs,
                         base_only=args.base_only)
        print()

    if args.training or export_both:
        export_training(root, output_dir / "training.json")


if __name__ == "__main__":
    main()
