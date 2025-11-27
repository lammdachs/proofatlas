"""Training script for clause selection GNN"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

from .model import create_model
from .data_collection import load_training_dataset, TrainingDataset


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model
    model_type: str = "gcn"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    pooling: str = "mean"

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10  # Early stopping patience

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is the remainder

    # Class imbalance
    use_class_weights: bool = True


def create_pyg_dataset(dataset: TrainingDataset) -> List[Data]:
    """Convert TrainingDataset to list of PyTorch Geometric Data objects"""
    pyg_data = []

    for i, (graph, label) in enumerate(zip(dataset.graphs, dataset.labels)):
        data = Data(
            x=graph["x"],
            edge_index=graph["edge_index"],
            y=torch.tensor([label.item()], dtype=torch.float32),
        )
        pyg_data.append(data)

    return pyg_data


def split_dataset(
    dataset: List[Data],
    train_split: float = 0.8,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Split dataset into train/val/test"""
    if shuffle:
        import random
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = [dataset[i] for i in indices]

    n = len(dataset)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    return dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute class weights for imbalanced data"""
    pos = labels.sum().item()
    neg = len(labels) - pos

    if pos == 0 or neg == 0:
        return torch.tensor([1.0])

    # Weight inversely proportional to class frequency
    pos_weight = neg / pos
    return torch.tensor([pos_weight])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss"""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(predictions, batch.y.squeeze())

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate model, return (loss, accuracy, precision, recall)"""
    model.eval()
    total_loss = 0
    correct = 0
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            predictions = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(predictions, batch.y.squeeze())

            total_loss += loss.item() * batch.num_graphs

            # Binary predictions
            pred_binary = (torch.sigmoid(predictions) > 0.5).float()
            labels = batch.y.squeeze()

            correct += (pred_binary == labels).sum().item()
            true_positives += ((pred_binary == 1) & (labels == 1)).sum().item()
            predicted_positives += (pred_binary == 1).sum().item()
            actual_positives += (labels == 1).sum().item()

    n = len(loader.dataset)
    accuracy = correct / n
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    return total_loss / n, accuracy, precision, recall


def train(
    train_data: List[Data],
    val_data: List[Data],
    config: TrainingConfig,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a clause scoring model.

    Args:
        train_data: Training data
        val_data: Validation data
        config: Training configuration
        device: Device to train on (default: auto-detect)
        verbose: Print progress

    Returns:
        (trained_model, training_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training on {device}")
        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)

    # Create model
    model = create_model(
        model_type=config.model_type,
        node_feature_dim=20,  # Fixed for our clause graphs
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pooling=config.pooling,
    ).to(device)

    if verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

    # Loss function with class weights
    if config.use_class_weights:
        labels = torch.tensor([d.y.item() for d in train_data])
        pos_weight = compute_class_weights(labels)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop with early stopping
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, val_prec, val_rec = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                  f"prec={val_prec:.4f}, rec={val_rec:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def save_model(model: nn.Module, path: Path, config: Optional[TrainingConfig] = None):
    """Save model and optionally config"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__ if config else None,
    }, path)


def load_model(path: Path, device: Optional[torch.device] = None) -> nn.Module:
    """Load a saved model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", {})
    model = create_model(
        model_type=config.get("model_type", "gcn"),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        pooling=config.get("pooling", "mean"),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train clause selection GNN")
    parser.add_argument("data_path", type=Path, help="Path to training data (.pt file)")
    parser.add_argument("--output", "-o", type=Path, default=Path("models/clause_gnn.pt"),
                        help="Output model path")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    dataset = load_training_dataset(args.data_path)

    # Convert to PyG format
    pyg_data = create_pyg_dataset(dataset)
    print(f"Total examples: {len(pyg_data)}")

    # Split
    train_data, val_data, test_data = split_dataset(pyg_data)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Train
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    model, history = train(train_data, val_data, config)

    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc, test_prec, test_rec = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.4f}, prec={test_prec:.4f}, rec={test_rec:.4f}")

    # Save model
    save_model(model, args.output, config)
    print(f"Model saved to {args.output}")
