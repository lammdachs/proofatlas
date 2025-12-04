#!/usr/bin/env python3
"""
Train clause selection models and export to ONNX.

This script trains a model using collected data and exports it to ONNX format
for use by the theorem prover.

Usage:
    python scripts/train.py --data data.pt --selector gcn
    python scripts/train.py --data data.pt --selector configs/selectors/gat.json
    python scripts/train.py --data data.pt --selector gcn --run-name my_experiment
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import torch
from proofatlas.ml.config import SelectorConfig
from proofatlas.ml.training import (
    ClauseDataset,
    collate_clause_batch,
    train,
    save_model,
)
from proofatlas.selectors import export_to_onnx


def load_data(data_path: Path):
    """Load collected training data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path)

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


def get_selectors_dir() -> Path:
    """Get the selectors directory."""
    return Path(__file__).parent.parent / ".selectors"


def main():
    parser = argparse.ArgumentParser(description="Train clause selection model and export to ONNX")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Training data file (.pt)")
    parser.add_argument("--selector", "-s", default="gcn", help="Selector config name or path")
    parser.add_argument("--run-name", help="Run name (default: auto-generated)")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--output", "-o", type=Path, help="Output ONNX model path (default: .selectors/<name>.onnx)")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory (default: logs/<run_name>)")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.selector)
    if config_path.exists():
        config = SelectorConfig.load(config_path)
    else:
        config = SelectorConfig.load_preset(args.selector)

    # Check that this selector requires training
    if not config.requires_training:
        print(f"ERROR: Selector '{config.name}' does not require training (no training section in config)")
        print("This selector is a heuristic that can be exported directly without training.")
        sys.exit(1)

    # Override with CLI args
    if args.run_name:
        config.name = args.run_name
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.gpus is not None:
        config.distributed.num_gpus = args.gpus

    # Generate run name if not set
    if not config.name:
        config.name = f"{config.model.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print(f"Training: {config.name}")
    print("=" * 60)
    print(f"Model: {config.model.type}, hidden={config.model.hidden_dim}, layers={config.model.num_layers}")
    print(f"Training: epochs={config.training.max_epochs}, batch={config.training.batch_size}, lr={config.training.learning_rate}")
    print(f"Distributed: gpus={config.distributed.num_gpus}, strategy={config.distributed.strategy}")
    print()

    # Load data
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
    checkpoint_dir = args.checkpoint_dir or Path(config.logging.log_dir) / config.name
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, checkpoint_path, config)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Export to ONNX
    onnx_path = args.output or get_selectors_dir() / f"{config.name}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if model needs adjacency matrix
    needs_adj = config.model.type in ["gcn", "gat", "graphsage", "gnn_transformer"]

    export_to_onnx(model, str(onnx_path), include_adj=needs_adj)
    print(f"ONNX model exported to: {onnx_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best epoch: {metrics.get('best_epoch', 'N/A')}")
    best_val_loss = metrics.get('best_val_loss')
    if best_val_loss is not None:
        print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {metrics.get('total_time_seconds', 0):.1f}s")
    print(f"\nTo use this selector, reference: {onnx_path.name}")


if __name__ == "__main__":
    main()
