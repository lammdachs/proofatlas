"""
Training infrastructure for clause selection models.

Uses PyTorch Lightning for multi-GPU training with JSON logging
for web-based visualization.
"""

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import Logger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from ..selectors import create_model, normalize_adjacency, edge_index_to_adjacency
from .config import SelectorConfig


# =============================================================================
# Dataset
# =============================================================================


class ClauseDataset(torch.utils.data.Dataset):
    """Dataset of clause selection examples."""

    def __init__(
        self,
        node_features: List[torch.Tensor],
        edge_indices: List[torch.Tensor],
        labels: List[int],
        pool_matrices: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            node_features: List of [num_nodes, 13] tensors
            edge_indices: List of [2, num_edges] tensors
            labels: List of selected clause indices
            pool_matrices: Optional pre-computed pool matrices
        """
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.labels = labels
        self.pool_matrices = pool_matrices

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        item = {
            "node_features": self.node_features[idx],
            "edge_index": self.edge_indices[idx],
            "label": self.labels[idx],
        }
        if self.pool_matrices is not None:
            item["pool_matrix"] = self.pool_matrices[idx]
        return item


def collate_clause_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for clause batches."""
    # Concatenate all node features
    node_features = torch.cat([b["node_features"] for b in batch], dim=0)

    # Offset edge indices and concatenate
    edge_indices = []
    offset = 0
    for b in batch:
        edge_indices.append(b["edge_index"] + offset)
        offset += b["node_features"].size(0)
    edge_index = torch.cat(edge_indices, dim=1)

    # Build adjacency matrix
    num_nodes = node_features.size(0)
    adj = edge_index_to_adjacency(edge_index, num_nodes, add_self_loops=True)
    adj_norm = normalize_adjacency(adj, add_self_loops=False)  # Already added

    # Build pool matrix if not provided
    if "pool_matrix" in batch[0]:
        # Offset and concatenate pool matrices
        pool_matrices = []
        col_offset = 0
        for b in batch:
            pm = b["pool_matrix"]
            # Create expanded matrix with correct column offset
            expanded = torch.zeros(pm.size(0), num_nodes)
            expanded[:, col_offset:col_offset + pm.size(1)] = pm
            pool_matrices.append(expanded)
            col_offset += pm.size(1)
        pool_matrix = torch.cat(pool_matrices, dim=0)
    else:
        # Create simple pool matrix (one clause per example)
        pool_matrix = torch.zeros(len(batch), num_nodes)
        offset = 0
        for i, b in enumerate(batch):
            n = b["node_features"].size(0)
            pool_matrix[i, offset:offset + n] = 1.0 / n
            offset += n

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {
        "node_features": node_features,
        "adj": adj_norm,
        "pool_matrix": pool_matrix,
        "labels": labels,
    }


# =============================================================================
# Lightning Module
# =============================================================================


if LIGHTNING_AVAILABLE:

    class ClauseSelectionModule(L.LightningModule):
        """Lightning module for clause selection training."""

        def __init__(self, config: SelectorConfig):
            super().__init__()
            self.config = config
            self.save_hyperparameters(config.to_dict())

            # Create model
            self.model = create_model(
                model_type=config.model.type,
                hidden_dim=config.model.hidden_dim,
                num_layers=config.model.num_layers,
                num_heads=config.model.num_heads,
                dropout=config.model.dropout,
            )

            # Track whether model needs adjacency matrix
            self.needs_adj = config.model.type in ["gcn", "gat", "graphsage", "gnn_transformer"]

        def forward(self, node_features, adj, pool_matrix):
            if self.needs_adj:
                return self.model(node_features, adj, pool_matrix)
            else:
                return self.model(node_features, pool_matrix)

        def training_step(self, batch, batch_idx):
            scores = self(batch["node_features"], batch["adj"], batch["pool_matrix"])
            loss = F.cross_entropy(scores, batch["labels"])

            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            scores = self(batch["node_features"], batch["adj"], batch["pool_matrix"])
            loss = F.cross_entropy(scores, batch["labels"])

            # Accuracy
            preds = scores.argmax(dim=-1)
            acc = (preds == batch["labels"]).float().mean()

            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate * self.config.scheduler.min_lr_ratio,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }


# =============================================================================
# JSON Logger for Web Visualization
# =============================================================================


class JSONLogger:
    """Logger that writes metrics to JSON for web visualization."""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file = self.log_dir / run_name / "metrics.json"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "run_name": run_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "epochs": [],
            "evaluations": [],
        }

    def log_config(self, config: SelectorConfig):
        self.metrics["config"] = config.to_dict()
        self._save()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float, lr: float):
        self.metrics["epochs"].append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": lr,
        })
        self._save()

    def log_evaluation(self, epoch: int, results: Dict[str, Any]):
        self.metrics["evaluations"].append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **results,
        })
        self._save()

    def log_final(self, best_epoch: int, best_val_loss: float, total_time: float):
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["best_epoch"] = best_epoch
        self.metrics["best_val_loss"] = best_val_loss
        self.metrics["total_time_seconds"] = total_time
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


if LIGHTNING_AVAILABLE:

    class JSONLoggingCallback(Callback):
        """Lightning callback for JSON logging."""

        def __init__(self, logger: JSONLogger):
            self.logger = logger
            self.epoch_metrics = {}

        def on_train_epoch_end(self, trainer, pl_module):
            # Collect metrics
            metrics = {k: v.item() if torch.is_tensor(v) else v
                       for k, v in trainer.callback_metrics.items()}

            train_loss = metrics.get("train_loss", 0)
            val_loss = metrics.get("val_loss", 0)
            val_acc = metrics.get("val_acc", 0)
            lr = trainer.optimizers[0].param_groups[0]["lr"]

            self.logger.log_epoch(
                epoch=trainer.current_epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr,
            )


# =============================================================================
# Proof Evaluation Callback
# =============================================================================


if LIGHTNING_AVAILABLE:

    class ProofEvaluationCallback(Callback):
        """Evaluate model on actual theorem proving problems."""

        def __init__(
            self,
            problems: List[str],
            logger: JSONLogger,
            eval_every_n_epochs: int = 10,
            timeout: float = 10.0,
        ):
            self.problems = problems
            self.logger = logger
            self.eval_every_n_epochs = eval_every_n_epochs
            self.timeout = timeout

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch

            # Only evaluate every N epochs
            if (epoch + 1) % self.eval_every_n_epochs != 0:
                return

            # Only run on rank 0
            if trainer.global_rank != 0:
                return

            if not self.problems:
                return

            results = self._evaluate_problems(pl_module)
            self.logger.log_evaluation(epoch, results)

            # Log to trainer
            trainer.logger.log_metrics({
                "proof_success_rate": results["success_rate"],
                "proof_avg_time": results["avg_time"],
            }, step=trainer.global_step)

        def _evaluate_problems(self, pl_module) -> Dict[str, Any]:
            """Run prover on evaluation problems."""
            try:
                from proofatlas import parse_tptp_file, SaturationState, SaturationConfig
            except ImportError:
                return {
                    "success_rate": 0.0,
                    "avg_time": 0.0,
                    "num_problems": 0,
                    "error": "proofatlas not available",
                }

            pl_module.eval()
            device = pl_module.device

            results = []
            for problem_path in self.problems:
                try:
                    start_time = time.time()

                    # Parse problem
                    formula = parse_tptp_file(problem_path, [])

                    # Create saturation state
                    config = SaturationConfig()
                    config.timeout = self.timeout
                    state = SaturationState(formula.clauses, config)

                    # TODO: Set learned clause selector
                    # This requires exporting model to ONNX and loading in Rust

                    # Run saturation
                    result = state.saturate()
                    elapsed = time.time() - start_time

                    results.append({
                        "problem": problem_path,
                        "success": result.is_proof(),
                        "time": elapsed,
                    })
                except Exception as e:
                    results.append({
                        "problem": problem_path,
                        "success": False,
                        "time": self.timeout,
                        "error": str(e),
                    })

            # Aggregate
            num_success = sum(1 for r in results if r["success"])
            total_time = sum(r["time"] for r in results)

            return {
                "success_rate": num_success / len(results) if results else 0.0,
                "num_success": num_success,
                "num_problems": len(results),
                "avg_time": total_time / len(results) if results else 0.0,
                "problems": results,
            }


# =============================================================================
# Training Functions
# =============================================================================


def train(
    train_dataset: ClauseDataset,
    val_dataset: ClauseDataset,
    config: SelectorConfig,
) -> Tuple[nn.Module, Dict]:
    """
    Train a clause selection model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration

    Returns:
        (trained_model, metrics_dict)
    """
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning required. Install with: pip install lightning")

    # Generate run name if not provided
    if not config.name:
        config.name = f"{config.model.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_clause_batch,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_clause_batch,
        num_workers=4,
        pin_memory=True,
    )

    # Create module
    module = ClauseSelectionModule(config)

    # Create JSON logger
    json_logger = JSONLogger(config.logging.log_dir, config.name)
    json_logger.log_config(config)

    # Callbacks
    callbacks = [
        JSONLoggingCallback(json_logger),
        ModelCheckpoint(
            dirpath=Path(config.logging.log_dir) / config.name / "checkpoints",
            filename="{epoch}-{val_loss:.4f}",
            monitor=config.checkpointing.monitor,
            mode=config.checkpointing.mode,
            save_top_k=config.checkpointing.save_top_k,
        ),
        EarlyStopping(
            monitor=config.checkpointing.monitor,
            patience=config.training.patience,
            mode=config.checkpointing.mode,
        ),
    ]

    # Add proof evaluation callback if eval is configured
    if config.evaluation.num_eval_problems > 0:
        callbacks.append(ProofEvaluationCallback(
            problems=[],  # TODO: Load eval problems from data config
            logger=json_logger,
            eval_every_n_epochs=config.evaluation.eval_every_n_epochs,
            timeout=config.evaluation.eval_timeout,
        ))

    # Determine devices
    if torch.cuda.is_available():
        devices = config.distributed.num_gpus if config.distributed.num_gpus > 0 else "auto"
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=config.distributed.strategy if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        enable_progress_bar=config.logging.enable_progress_bar,
        log_every_n_steps=config.logging.log_every_n_steps,
    )

    # Train
    start_time = time.time()
    trainer.fit(module, train_loader, val_loader)
    total_time = time.time() - start_time

    # Log final metrics
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    json_logger.log_final(
        best_epoch=trainer.current_epoch,
        best_val_loss=float(best_val_loss) if best_val_loss else 0.0,
        total_time=total_time,
    )

    # Load best model
    if best_checkpoint:
        module = ClauseSelectionModule.load_from_checkpoint(best_checkpoint, config=config)

    return module.model, json_logger.metrics


def save_model(model: nn.Module, path: Path, config: Optional[SelectorConfig] = None):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict() if config else None,
    }, path)


def load_model(path: Path, device: Optional[torch.device] = None) -> nn.Module:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    config_dict = checkpoint.get("config", {})

    # Handle both old flat config and new nested config
    model_config = config_dict.get("model", config_dict)

    model = create_model(
        model_type=model_config.get("type", model_config.get("model_type", "gcn")),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 3),
        num_heads=model_config.get("num_heads", 4),
        dropout=model_config.get("dropout", 0.1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# =============================================================================
# Legacy compatibility
# =============================================================================


def create_pyg_dataset(dataset):
    """Legacy function for PyG compatibility."""
    raise NotImplementedError(
        "create_pyg_dataset is deprecated. Use ClauseDataset directly."
    )


def split_dataset(
    node_features: List[torch.Tensor],
    edge_indices: List[torch.Tensor],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[ClauseDataset, ClauseDataset, ClauseDataset]:
    """Split data into train/val/test datasets."""
    import random
    random.seed(seed)

    n = len(node_features)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    def make_dataset(idxs):
        return ClauseDataset(
            node_features=[node_features[i] for i in idxs],
            edge_indices=[edge_indices[i] for i in idxs],
            labels=[labels[i] for i in idxs],
        )

    return (
        make_dataset(indices[:train_end]),
        make_dataset(indices[train_end:val_end]),
        make_dataset(indices[val_end:]),
    )


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    import argparse
    from .config import ModelConfig, TrainingParams, DistributedConfig, LoggingConfig

    parser = argparse.ArgumentParser(description="Train clause selection model")
    parser.add_argument("--config", type=Path, help="Config JSON file or preset name")
    parser.add_argument("--model-type", default="gcn", choices=["gcn", "gat", "graphsage", "transformer", "gnn_transformer", "mlp"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-name", default="")

    args = parser.parse_args()

    if args.config:
        # Try loading as file path first, then as preset name
        config_path = Path(args.config)
        if config_path.exists():
            config = SelectorConfig.load(config_path)
        else:
            config = SelectorConfig.load_preset(str(args.config))
    else:
        # Build config from CLI args
        config = SelectorConfig(
            name=args.run_name,
            model=ModelConfig(
                type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
            ),
            training=TrainingParams(
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                learning_rate=args.lr,
            ),
            distributed=DistributedConfig(
                num_gpus=args.gpus,
            ),
            logging=LoggingConfig(
                log_dir=args.log_dir,
            ),
        )

    print(f"Config: {config}")
    print("\nTo train, create datasets and call train(train_dataset, val_dataset, config)")
