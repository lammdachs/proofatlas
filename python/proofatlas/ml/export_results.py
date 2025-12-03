"""
Export training results to website-friendly format.

Converts training logs to JSON files that can be hosted on GitHub Pages
and visualized with the interactive charts.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional


def export_run(
    log_dir: Path,
    run_name: str,
    output_dir: Path,
):
    """
    Export a single training run to website format.

    Args:
        log_dir: Directory containing training logs
        run_name: Name of the run to export
        output_dir: Output directory for website data
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)

    metrics_file = log_dir / run_name / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Create output directory
    run_output = output_dir / run_name
    run_output.mkdir(parents=True, exist_ok=True)

    # Copy metrics file
    shutil.copy(metrics_file, run_output / "metrics.json")

    print(f"Exported {run_name} to {run_output}")


def export_all_runs(
    log_dir: Path,
    output_dir: Path,
    run_names: Optional[List[str]] = None,
):
    """
    Export all training runs to website format.

    Args:
        log_dir: Directory containing training logs
        output_dir: Output directory for website data
        run_names: Optional list of specific runs to export (default: all)
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)

    # Find all runs
    if run_names is None:
        run_names = []
        for d in log_dir.iterdir():
            if d.is_dir() and (d / "metrics.json").exists():
                run_names.append(d.name)

    if not run_names:
        print("No runs found to export")
        return

    # Export each run
    for run_name in run_names:
        try:
            export_run(log_dir, run_name, output_dir)
        except Exception as e:
            print(f"Failed to export {run_name}: {e}")

    # Create runs index
    runs_index = {
        "runs": sorted(run_names, reverse=True),
    }

    with open(output_dir / "runs.json", "w") as f:
        json.dump(runs_index, f, indent=2)

    print(f"\nExported {len(run_names)} runs to {output_dir}")
    print(f"Runs index: {output_dir / 'runs.json'}")


def create_sample_data(output_dir: Path):
    """
    Create sample training data for testing the visualization.

    Args:
        output_dir: Output directory for sample data
    """
    import random
    from datetime import datetime, timedelta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample runs
    model_types = ["gcn", "gat", "transformer"]
    runs = []

    for model_type in model_types:
        run_name = f"{model_type}_sample_{datetime.now().strftime('%Y%m%d')}"
        runs.append(run_name)

        # Generate sample metrics
        epochs = []
        base_train_loss = random.uniform(1.5, 2.5)
        base_val_loss = random.uniform(1.8, 2.8)
        lr = 1e-3

        for epoch in range(50):
            decay = 0.95 ** epoch
            noise = random.uniform(0.9, 1.1)

            train_loss = base_train_loss * decay * noise
            val_loss = base_val_loss * decay * noise * 1.1
            val_acc = min(0.95, 0.5 + 0.4 * (1 - decay) + random.uniform(-0.05, 0.05))
            lr *= 0.98

            epochs.append({
                "epoch": epoch,
                "timestamp": (datetime.now() - timedelta(hours=50-epoch)).isoformat(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": lr,
            })

        # Generate sample evaluations
        evaluations = []
        for eval_epoch in [9, 19, 29, 39, 49]:
            success_rate = 0.2 + 0.5 * (eval_epoch / 50) + random.uniform(-0.1, 0.1)
            evaluations.append({
                "epoch": eval_epoch,
                "timestamp": epochs[eval_epoch]["timestamp"],
                "success_rate": min(1.0, max(0.0, success_rate)),
                "num_success": int(success_rate * 20),
                "num_problems": 20,
                "avg_time": random.uniform(2.0, 8.0),
            })

        metrics = {
            "run_name": run_name,
            "start_time": (datetime.now() - timedelta(hours=50)).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config": {
                "model_type": model_type,
                "hidden_dim": 64,
                "num_layers": 3,
                "num_heads": 4,
                "dropout": 0.1,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "max_epochs": 50,
            },
            "epochs": epochs,
            "evaluations": evaluations,
            "best_epoch": min(range(len(epochs)), key=lambda i: epochs[i]["val_loss"]),
            "best_val_loss": min(e["val_loss"] for e in epochs),
            "total_time_seconds": random.uniform(1800, 7200),
        }

        # Save metrics
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Created sample run: {run_name}")

    # Create runs index
    runs_index = {"runs": runs}
    with open(output_dir / "runs.json", "w") as f:
        json.dump(runs_index, f, indent=2)

    print(f"\nCreated {len(runs)} sample runs in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export training results")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing training logs")
    parser.add_argument("--output-dir", type=Path, default=Path("wasm/data"),
                        help="Output directory for website data")
    parser.add_argument("--sample", action="store_true",
                        help="Create sample data for testing")

    args = parser.parse_args()

    if args.sample:
        create_sample_data(args.output_dir)
    else:
        export_all_runs(args.log_dir, args.output_dir)
