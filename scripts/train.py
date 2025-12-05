#!/usr/bin/env python3
"""
Train clause selection models and export weights for Burn.

This script trains a model using collected data and exports weights to safetensors
format for use by the Rust/Burn theorem prover.

Usage:
    # Train from bench.py traces (aggregates automatically)
    python scripts/train.py --preset time_sel21 --training gcn

    # Train from pre-aggregated data file
    python scripts/train.py --data data.pt --training gcn

    # With custom options
    python scripts/train.py --preset time_sel21 --training gcn --run-name my_experiment
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from proofatlas.ml.config import TrainingConfig
from proofatlas.ml.training import (
    ClauseDataset,
    collate_clause_batch,
    train,
    save_model,
)


def get_traces_dir() -> Path:
    """Get the traces directory."""
    return Path(__file__).parent.parent / ".data" / "traces"


def load_traces_from_preset(preset: str):
    """Load and aggregate traces from a bench.py preset.

    Traces are stored as individual .pt files in .data/traces/{preset}/
    """
    traces_dir = get_traces_dir() / preset
    if not traces_dir.exists():
        raise FileNotFoundError(
            f"No traces found for preset '{preset}' at {traces_dir}\n"
            f"Run: proofatlas-bench --prover proofatlas --preset {preset} --trace"
        )

    trace_files = list(traces_dir.glob("*.pt"))
    if not trace_files:
        raise FileNotFoundError(f"No trace files in {traces_dir}")

    print(f"Loading traces from {traces_dir}...")
    print(f"  Found {len(trace_files)} trace files")

    all_graphs = []
    all_labels = []
    all_problem_names = []
    successful = 0
    skipped = 0

    for trace_file in sorted(trace_files):
        try:
            trace = torch.load(trace_file, weights_only=False)
        except Exception:
            skipped += 1
            continue

        # Skip non-proof traces
        if not trace.get("proof_found") or not trace.get("graphs"):
            skipped += 1
            continue

        problem_name = trace_file.stem
        all_graphs.extend(trace["graphs"])
        all_labels.extend(trace["labels"])
        all_problem_names.extend([problem_name] * len(trace["labels"]))
        successful += 1

    print(f"  Loaded: {successful} problems, Skipped: {skipped}")
    print(f"  Examples: {len(all_labels)}")

    if all_labels:
        pos = sum(all_labels)
        print(f"  Positive: {pos} ({100*pos/len(all_labels):.1f}%)")

    return {
        "graphs": all_graphs,
        "labels": all_labels,
        "problem_names": all_problem_names,
    }


def load_data(data_path: Path):
    """Load pre-aggregated training data from a .pt file."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"  Examples: {len(data['labels'])}")
    print(f"  Problems: {len(set(data['problem_names']))}")

    if data['labels']:
        pos = sum(data['labels'])
        print(f"  Positive: {pos} ({100*pos/len(data['labels']):.1f}%)")

    return data


def split_data(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train/val/test sets."""
    import random
    random.seed(seed)

    n = len(data['labels'])
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    def make_dataset(idxs):
        return ClauseDataset(
            node_features=[data['graphs'][i]['node_features'] for i in idxs],
            edge_indices=[data['graphs'][i]['edge_index'] for i in idxs],
            labels=[data['labels'][i] for i in idxs],
        )

    train_ds = make_dataset(indices[:train_end])
    val_ds = make_dataset(indices[train_end:val_end])
    test_ds = make_dataset(indices[val_end:])

    return train_ds, val_ds, test_ds


def get_weights_dir() -> Path:
    """Get the weights directory."""
    return Path(__file__).parent.parent / ".weights"


def export_to_safetensors(model: torch.nn.Module, path: Path, config: TrainingConfig):
    """Export model weights to safetensors format for Burn."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("WARNING: safetensors not installed. Install with: pip install safetensors")
        print("Skipping safetensors export.")
        return

    # Get state dict and convert to safetensors format
    state_dict = model.state_dict()

    # Save metadata about the model architecture
    metadata = {
        "model_type": config.model.type,
        "hidden_dim": str(config.model.hidden_dim),
        "num_layers": str(config.model.num_layers),
        "num_heads": str(config.model.num_heads),
        "input_dim": str(config.model.input_dim),
    }

    save_file(state_dict, path, metadata=metadata)
    print(f"Weights exported to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train clause selection model and export weights")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--preset", "-p", help="Load traces from bench.py preset (e.g., time_sel21)")
    data_group.add_argument("--data", "-d", type=Path, help="Pre-aggregated training data file (.pt)")
    parser.add_argument("--training", "-t", default="gcn", help="Training config name or path")
    parser.add_argument("--run-name", help="Run name (default: auto-generated)")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--output", "-o", type=Path, help="Output weights path (default: .weights/<name>.safetensors)")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory (default: logs/<run_name>)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.training)
    if config_path.exists():
        config = TrainingConfig.load(config_path)
    else:
        config = TrainingConfig.load_preset(args.training)

    # Override with CLI args
    if args.run_name:
        config.name = args.run_name
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    # Generate run name if not set
    if not config.name:
        config.name = f"{config.model.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print(f"Training: {config.name}")
    print("=" * 60)
    print(f"Model: {config.model.type}, hidden={config.model.hidden_dim}, layers={config.model.num_layers}")
    print(f"Training: epochs={config.training.max_epochs}, batch={config.training.batch_size}, lr={config.training.learning_rate}")
    print()

    # Load data
    if args.preset:
        data = load_traces_from_preset(args.preset)
    else:
        data = load_data(args.data)

    if not data['labels']:
        print("ERROR: No training examples in data file")
        sys.exit(1)

    # Split data
    print("\nSplitting data...")
    train_ds, val_ds, test_ds = split_data(data, seed=42)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Train
    print("\nStarting training...")
    model, metrics = train(train_ds, val_ds, config)

    # Save PyTorch checkpoint
    checkpoint_dir = args.checkpoint_dir or Path(".logs") / config.name
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, checkpoint_path, config)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Export to safetensors for Burn
    weights_path = args.output or get_weights_dir() / f"{config.name}.safetensors"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_safetensors(model, weights_path, config)

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best epoch: {metrics.get('best_epoch', 'N/A')}")
    best_val_loss = metrics.get('best_val_loss')
    if best_val_loss is not None:
        print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {metrics.get('total_time_seconds', 0):.1f}s")
    print(f"\nTo use this selector in data config:")
    print(f'  "selector": {{"name": "{config.model.type}", "weights": "{weights_path.name}"}}')


if __name__ == "__main__":
    main()
