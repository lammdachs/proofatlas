#!/usr/bin/env python3
"""
Train clause selection models using the config system.

Usage:
    python scripts/train.py --data data.pt --training-config default
    python scripts/train.py --data data.pt --training-config configs/training/gat_large.json
    python scripts/train.py --data data.pt --training-config default --run-name my_experiment
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


def main():
    parser = argparse.ArgumentParser(description="Train clause selection model")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Training data file (.pt)")
    parser.add_argument("--training-config", "-t", default="default", help="Training config name or path")
    parser.add_argument("--run-name", help="Run name (default: auto-generated)")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--gpus", type=int, help="Override number of GPUs")
    parser.add_argument("--output", "-o", type=Path, help="Output model path")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.training_config)
    if config_path.exists():
        config = TrainingConfig.load(config_path)
    else:
        config = TrainingConfig.load_preset(args.training_config)

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

    # Save model
    output_path = args.output or Path(config.logging.log_dir) / config.name / "model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output_path, config)
    print(f"\nModel saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best epoch: {metrics.get('best_epoch', 'N/A')}")
    print(f"  Best val loss: {metrics.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Total time: {metrics.get('total_time_seconds', 0):.1f}s")


if __name__ == "__main__":
    main()
