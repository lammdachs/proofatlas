#!/usr/bin/env python3
"""
Export benchmark and training results for web display.

USAGE:
    python scripts/export.py                    # Export all results (uses default problem set)
    python scripts/export.py --benchmarks       # Export benchmarks only
    python scripts/export.py --training         # Export training only
    python scripts/export.py --problem-set test # Limit to 'test' problem set
    python scripts/export.py --prover proofatlas # Only include this prover
    python scripts/export.py --preset time_sel0 # Only include this preset
    python scripts/export.py --base-only        # Skip learned selectors

The default problem set is read from configs/tptp.json (defaults.problem_set).

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


def get_default_problem_set(root: Path) -> str | None:
    """Get the default problem set from tptp.json."""
    tptp_config_path = root / "configs" / "tptp.json"
    if not tptp_config_path.exists():
        return None

    with open(tptp_config_path) as f:
        tptp_config = json.load(f)

    return tptp_config.get("defaults", {}).get("problem_set")


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
    total = len(metadata)
    for i, (path, meta) in enumerate(metadata.items(), 1):
        if i % 5000 == 0 or i == total:
            print(f"\r  Filtering problems: {i}/{total}", end="", flush=True)

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

        # Term depth filter
        if "max_term_depth" in filters:
            if meta.get("max_term_depth", float("inf")) > filters["max_term_depth"]:
                continue

        # Clause size filter
        if "max_clause_size" in filters:
            if meta.get("max_clause_size", float("inf")) > filters["max_clause_size"]:
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

    print()  # newline after progress
    return matching


# Benchmark export

def load_benchmark_results(runs_dir: Path) -> list:
    """Load all cached results from .data/runs/"""
    results = []

    if not runs_dir.exists():
        return results

    # First count total files for progress
    all_files = []
    for prover_dir in sorted(runs_dir.iterdir()):
        if not prover_dir.is_dir():
            continue
        for preset_dir in sorted(prover_dir.iterdir()):
            if not preset_dir.is_dir():
                continue
            all_files.extend(preset_dir.glob("*.json"))

    total = len(all_files)
    if total == 0:
        return results

    for i, result_file in enumerate(all_files, 1):
        if i % 1000 == 0 or i == total:
            print(f"\r  Loading results: {i}/{total}", end="", flush=True)

        try:
            # Extract prover/preset from path
            preset = result_file.parent.name
            prover = result_file.parent.parent.name

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

    print()  # newline after progress
    return results


def filter_results(
    results: list,
    problem_set: set[str] | None = None,
    prover: str | None = None,
    preset: str | None = None,
    base_only: bool = False,
) -> list:
    """Filter results by problem set, prover, preset, and base_only."""
    filtered = results

    if problem_set is not None:
        filtered = [r for r in filtered if r["problem"] in problem_set]

    if prover is not None:
        filtered = [r for r in filtered if r["prover"] == prover]

    if preset is not None:
        filtered = [r for r in filtered if r["preset"] == preset]

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
    prover: str | None = None,
    preset: str | None = None,
    base_only: bool = False,
):
    """Export benchmark results to JSON."""
    runs_dir = root / ".data" / "runs"

    print(f"Loading benchmark results...")
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
        print(f"  Found {len(problem_set)} matching problems")

    # Filter results
    if problem_set or prover or preset or base_only:
        print("Filtering results...")
        before = len(results)
        results = filter_results(results, problem_set, prover, preset, base_only)
        print(f"  {before} -> {len(results)} results")

    if not results:
        print("No results match the filters")
        return False

    print("Computing statistics...")
    summary = compute_benchmark_summary(results)
    problems = compute_problem_comparison(results)
    found_configs = sorted(set(f"{r['prover']}/{r['preset']}" for r in results))

    output = {
        "generated": datetime.now().isoformat(),
        "problem_set": problem_set_name,
        "prover_filter": prover,
        "preset_filter": preset,
        "base_only": base_only,
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
    """Load all training results from .logs/*/metrics.json."""
    runs = []

    if not logs_dir.exists():
        return runs

    # Look for directories with metrics.json (new format from proofatlas-train)
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            config = metrics.get("config", {})
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            epochs = metrics.get("epochs", [])

            # Build epoch history
            epoch_history = []
            for e in epochs:
                epoch_history.append({
                    "epoch": e.get("epoch", 0),
                    "train_loss": e.get("train_loss"),
                    "val_loss": e.get("val_loss"),
                    "val_acc": e.get("val_acc"),
                    "val_mrr": e.get("val_mrr"),
                    "learning_rate": e.get("learning_rate"),
                })

            runs.append({
                "name": metrics.get("run_name", run_dir.name),
                "start_time": metrics.get("start_time"),
                "end_time": metrics.get("end_time"),
                "total_time_seconds": metrics.get("total_time_seconds"),
                "termination_reason": metrics.get("termination_reason"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_loss": metrics.get("best_val_loss"),
                "model": {
                    "type": model_config.get("type", "unknown"),
                    "hidden_dim": model_config.get("hidden_dim"),
                    "num_layers": model_config.get("num_layers"),
                    "input_dim": model_config.get("input_dim"),
                    "scorer_type": model_config.get("scorer_type"),
                },
                "training": {
                    "batch_size": training_config.get("batch_size"),
                    "learning_rate": training_config.get("learning_rate"),
                    "max_epochs": training_config.get("max_epochs"),
                    "patience": training_config.get("patience"),
                    "loss_type": training_config.get("loss_type"),
                },
                "epochs": epoch_history,
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


def load_architectures(config_path: Path) -> dict:
    """Load available architectures from config."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = json.load(f)

    return config.get("architectures", {})


def export_training(root: Path, output_path: Path):
    """Export training results to JSON."""
    logs_dir = root / ".logs"
    weights_dir = root / ".weights"
    embeddings_config = root / "configs" / "embeddings.json"
    scorers_config = root / "configs" / "scorers.json"

    print(f"Loading training runs from {logs_dir}...")
    runs = load_training_runs(logs_dir)
    print(f"  Found {len(runs)} training runs")

    print(f"Loading available weights from {weights_dir}...")
    weights = load_available_weights(weights_dir)
    print(f"  Found {len(weights)} weight files")

    print(f"Loading architectures...")
    embeddings = load_architectures(embeddings_config)
    scorers = load_architectures(scorers_config)
    print(f"  Found {len(embeddings)} embeddings, {len(scorers)} scorers")

    # Find best run by val_loss
    best_run = None
    if runs:
        for run in runs:
            val_loss = run.get("best_val_loss")
            if val_loss is not None:
                if best_run is None or val_loss < best_run.get("best_val_loss", float("inf")):
                    best_run = run

    output = {
        "generated": datetime.now().isoformat(),
        "total_runs": len(runs),
        "total_weights": len(weights),
        "summary": {
            "best_run": best_run["name"] if best_run else None,
            "best_val_loss": best_run["best_val_loss"] if best_run else None,
        },
        "embeddings": embeddings,
        "scorers": scorers,
        "weights": weights,
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported training to {output_path}")

    if runs:
        print("\nTraining Runs:")
        print(f"{'Name':<35} {'Model':<8} {'Epochs':>7} {'Val Loss':>10} {'Val Acc':>8}")
        print("-" * 75)
        for r in runs:
            model_type = r.get("model", {}).get("type", "?")
            num_epochs = len(r.get("epochs", []))
            val_loss = r.get("best_val_loss")
            val_loss_str = f"{val_loss:.4f}" if val_loss else "N/A"
            # Get final val_acc from last epoch
            epochs = r.get("epochs", [])
            val_acc = epochs[-1].get("val_acc") if epochs else None
            val_acc_str = f"{val_acc*100:.1f}%" if val_acc else "N/A"
            print(f"{r['name']:<35} {model_type:<8} {num_epochs:>7} {val_loss_str:>10} {val_acc_str:>8}")

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
    parser.add_argument("--prover", type=str,
                       help="Prover to include (default: all)")
    parser.add_argument("--preset", type=str,
                       help="Preset to include (default: all)")
    parser.add_argument("--base-only", action="store_true",
                       help="Only include base configs (skip learned selectors)")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory (default: web/data/)")
    parser.add_argument("--commit", action="store_true",
                       help="Commit exported files to git")
    args = parser.parse_args()

    root = get_project_root()
    output_dir = args.output_dir or root / "web" / "data"

    # Use default problem set if not specified
    problem_set_name = args.problem_set
    if problem_set_name is None:
        problem_set_name = get_default_problem_set(root)

    # If neither flag specified, export both
    export_both = not args.benchmarks and not args.training

    if args.benchmarks or export_both:
        export_benchmarks(root, output_dir / "benchmarks.json",
                         problem_set_name=problem_set_name,
                         prover=args.prover,
                         preset=args.preset,
                         base_only=args.base_only)
        print()

    if args.training or export_both:
        export_training(root, output_dir / "training.json")

    if args.commit:
        import subprocess

        files_to_commit = []
        if args.benchmarks or export_both:
            files_to_commit.append(str(output_dir / "benchmarks.json"))
        if args.training or export_both:
            files_to_commit.append(str(output_dir / "training.json"))

        # Stage files
        subprocess.run(["git", "add"] + files_to_commit, cwd=root, check=True)

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=root,
            capture_output=True
        )
        if result.returncode != 0:
            # There are staged changes, commit them
            subprocess.run(
                ["git", "commit", "-m", "Update exported web data"],
                cwd=root,
                check=True
            )
            print("\nCommitted exported files to git")
        else:
            print("\nNo changes to commit")


if __name__ == "__main__":
    main()
