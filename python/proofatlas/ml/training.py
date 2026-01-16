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
# Contrastive Loss
# =============================================================================


def info_nce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for clause selection.

    For each positive (proof clause), compute loss against all negatives.

    Args:
        scores: [batch_size] clause scores from model
        labels: [batch_size] binary labels (1=proof clause, 0=not)
        temperature: softmax temperature (lower = sharper)

    Returns:
        Scalar loss
    """
    scores = scores / temperature

    pos_mask = labels.bool()
    neg_mask = ~pos_mask

    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()

    if num_pos == 0 or num_neg == 0:
        # Fallback to BCE if no positive/negative examples
        return F.binary_cross_entropy_with_logits(scores, labels.float())

    pos_scores = scores[pos_mask]  # [num_pos]
    neg_scores = scores[neg_mask]  # [num_neg]

    # For each positive, compute log-softmax over (positive, all negatives)
    # This is equivalent to: -log(exp(pos) / (exp(pos) + sum(exp(neg))))

    # Numerically stable: log_softmax = pos - logsumexp([pos, neg1, neg2, ...])
    neg_logsumexp = torch.logsumexp(neg_scores, dim=0)  # scalar

    # For each positive: loss = -pos + log(exp(pos) + exp(neg_logsumexp))
    #                         = -pos + logsumexp([pos, neg_logsumexp])
    losses = -pos_scores + torch.logsumexp(
        torch.stack([pos_scores, neg_logsumexp.expand_as(pos_scores)], dim=0),
        dim=0
    )

    return losses.mean()


def margin_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    num_pairs: int = 16,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss for clause selection.

    Sample (positive, negative) pairs and train positive to score higher.

    Args:
        scores: [batch_size] clause scores from model
        labels: [batch_size] binary labels (1=proof clause, 0=not)
        margin: margin between positive and negative scores
        num_pairs: number of pairs to sample per positive

    Returns:
        Scalar loss
    """
    pos_mask = labels.bool()
    neg_mask = ~pos_mask

    num_pos = pos_mask.sum()
    num_neg = neg_mask.sum()

    if num_pos == 0 or num_neg == 0:
        return F.binary_cross_entropy_with_logits(scores, labels.float())

    pos_scores = scores[pos_mask]  # [num_pos]
    neg_scores = scores[neg_mask]  # [num_neg]

    # Sample negative indices for each positive
    neg_indices = torch.randint(0, num_neg, (num_pos, num_pairs), device=scores.device)
    sampled_neg = neg_scores[neg_indices]  # [num_pos, num_pairs]

    # Expand positive scores for pairwise comparison
    pos_expanded = pos_scores.unsqueeze(1).expand(-1, num_pairs)  # [num_pos, num_pairs]

    # Margin ranking loss: max(0, margin - (pos - neg))
    losses = F.relu(margin - (pos_expanded - sampled_neg))

    return losses.mean()


# =============================================================================
# Per-Proof Loss Functions
# =============================================================================


def info_nce_loss_per_proof(
    scores: torch.Tensor,
    labels: torch.Tensor,
    proof_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE loss computed separately for each proof, then averaged.

    This is the correct formulation: positives and negatives should come
    from the same proof search, not mixed across different problems.

    Args:
        scores: [total_clauses] clause scores from model
        labels: [total_clauses] binary labels (1=proof clause, 0=not)
        proof_ids: [total_clauses] which proof each clause belongs to
        temperature: softmax temperature (lower = sharper)

    Returns:
        Scalar loss (mean over proofs)
    """
    unique_proofs = proof_ids.unique()
    losses = []

    for proof_id in unique_proofs:
        mask = proof_ids == proof_id
        proof_scores = scores[mask]
        proof_labels = labels[mask]

        loss = info_nce_loss(proof_scores, proof_labels, temperature)
        losses.append(loss)

    return torch.stack(losses).mean()


# =============================================================================
# Dataset
# =============================================================================


class ProofDataset(torch.utils.data.Dataset):
    """
    Dataset that loads structured JSON traces and converts to graphs/strings.

    Each item is a random prefix of a proof search, simulating the
    partial information available during actual inference. This ensures
    the training distribution matches the inference distribution.

    The structured format preserves all symbol names and clause structure,
    enabling both graph-based and sentence-based training from the same data.
    """

    def __init__(
        self,
        trace_dir: Path,
        output_type: str = "graph",  # "graph" or "string"
        min_prefix_clauses: int = 10,
        sample_prefix: bool = True,
        max_clauses: Optional[int] = None,
    ):
        """
        Args:
            trace_dir: Directory containing .json trace files
            output_type: "graph" for GNN models, "string" for sentence models
            min_prefix_clauses: Minimum prefix length to sample
            sample_prefix: If True, sample random prefix; if False, use full proof
            max_clauses: Optional limit on clauses per proof (for memory)
        """
        self.trace_dir = Path(trace_dir)
        self.trace_files = sorted(self.trace_dir.glob("*.json"))
        self.output_type = output_type
        self.min_prefix_clauses = min_prefix_clauses
        self.sample_prefix = sample_prefix
        self.max_clauses = max_clauses

        if not self.trace_files:
            raise ValueError(f"No JSON trace files found in {trace_dir}")

        # Lazy-load converters
        self._clause_to_graph = None
        self._clause_to_string = None

    def __len__(self):
        return len(self.trace_files)

    def __getitem__(self, idx):
        import json
        import random

        with open(self.trace_files[idx]) as f:
            trace = json.load(f)

        clauses = trace["clauses"]
        n_clauses = len(clauses)

        # Sample a random prefix to simulate partial proof search
        if self.sample_prefix and n_clauses > self.min_prefix_clauses:
            prefix_len = random.randint(self.min_prefix_clauses, n_clauses)
            clauses = clauses[:prefix_len]

        # Apply max_clauses limit if needed
        if self.max_clauses and len(clauses) > self.max_clauses:
            # Keep all positives, sample negatives
            pos_indices = [i for i, c in enumerate(clauses) if c.get("label", 0) == 1]
            neg_indices = [i for i, c in enumerate(clauses) if c.get("label", 0) == 0]

            max_neg = self.max_clauses - len(pos_indices)
            if max_neg > 0 and len(neg_indices) > max_neg:
                neg_indices = random.sample(neg_indices, max_neg)

            indices = sorted(pos_indices + neg_indices)
            clauses = [clauses[i] for i in indices]

        # Extract labels
        labels = [c.get("label", 0) for c in clauses]

        if self.output_type == "graph":
            # Convert to graph tensors
            if self._clause_to_graph is None:
                from .structured import clause_to_graph
                self._clause_to_graph = clause_to_graph

            max_age = len(clauses)
            graphs = [self._clause_to_graph(c, max_age) for c in clauses]

            return {
                "graphs": graphs,
                "labels": labels,
                "problem": self.trace_files[idx].stem,
            }
        else:
            # Convert to strings
            if self._clause_to_string is None:
                from .structured import clause_to_string
                self._clause_to_string = clause_to_string

            strings = [self._clause_to_string(c) for c in clauses]

            return {
                "strings": strings,
                "labels": labels,
                "problem": self.trace_files[idx].stem,
            }


def collate_sentence_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for sentence-based batches.

    Returns clause strings and labels with proof_ids for per-proof loss.
    """
    all_strings = []
    all_labels = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        strings = item["strings"]
        labels = item["labels"]

        all_strings.extend(strings)
        all_labels.extend(labels)
        all_proof_ids.extend([proof_idx] * len(strings))

    return {
        "strings": all_strings,
        "labels": torch.tensor(all_labels, dtype=torch.float32),
        "proof_ids": torch.tensor(all_proof_ids, dtype=torch.long),
    }


def collate_proof_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for proof batches.

    Combines all clauses from all proofs into single tensors,
    with proof_ids to track which proof each clause belongs to.

    New architecture (IJCAR26 plan):
    - Node features (3d): type, arity, arg_pos (for GCN encoder)
    - Clause features (3d): age, role, size (for scorer, sinusoidal encoded)
    """
    all_node_features = []
    all_edge_indices = []
    all_clause_features = []
    all_labels = []
    all_proof_ids = []

    node_offset = 0

    for proof_idx, item in enumerate(batch):
        graphs = item["graphs"]
        labels = item["labels"]

        for graph, label in zip(graphs, labels):
            # Offset edge indices
            edge_index = graph["edge_index"] + node_offset
            all_edge_indices.append(edge_index)

            all_node_features.append(graph["x"])
            all_labels.append(label)
            all_proof_ids.append(proof_idx)

            # Collect clause features if present
            if "clause_features" in graph:
                all_clause_features.append(graph["clause_features"])

            node_offset += graph["num_nodes"]

    # Concatenate all
    node_features = torch.cat(all_node_features, dim=0)
    edge_index = torch.cat(all_edge_indices, dim=1)
    labels = torch.tensor(all_labels, dtype=torch.float32)
    proof_ids = torch.tensor(all_proof_ids, dtype=torch.long)

    # Build adjacency matrix
    num_nodes = node_features.size(0)
    adj = edge_index_to_adjacency(edge_index, num_nodes, add_self_loops=True)
    adj_norm = normalize_adjacency(adj, add_self_loops=False)

    # Build pool matrix (one row per clause)
    num_clauses = len(all_labels)
    pool_matrix = torch.zeros(num_clauses, num_nodes)

    clause_idx = 0
    node_offset = 0
    for item in batch:
        for graph in item["graphs"]:
            n = graph["num_nodes"]
            pool_matrix[clause_idx, node_offset:node_offset + n] = 1.0 / n
            clause_idx += 1
            node_offset += n

    result = {
        "node_features": node_features,
        "adj": adj_norm,
        "pool_matrix": pool_matrix,
        "labels": labels,
        "proof_ids": proof_ids,
    }

    # Stack clause features if present
    if all_clause_features:
        result["clause_features"] = torch.stack(all_clause_features, dim=0)

    return result


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
            node_features: List of [num_nodes, 8] tensors (raw features)
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

    # Labels can be binary (0/1) or indices - handle both
    raw_labels = [b["label"] for b in batch]
    # Use long for indices (legacy), but the loss functions handle conversion
    labels = torch.tensor(raw_labels, dtype=torch.long)

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

            # Loss configuration
            self.loss_type = getattr(config.training, 'loss_type', 'info_nce')
            self.temperature = getattr(config.training, 'temperature', 1.0)

        def forward(self, node_features, adj, pool_matrix, clause_features=None):
            if self.needs_adj:
                return self.model(node_features, adj, pool_matrix, clause_features)
            else:
                return self.model(node_features, pool_matrix, clause_features)

        def _compute_loss(self, scores, labels, proof_ids=None):
            """Compute loss based on configuration."""
            if self.loss_type == 'info_nce':
                if proof_ids is not None:
                    # Per-proof InfoNCE (correct formulation)
                    return info_nce_loss_per_proof(scores, labels, proof_ids, temperature=self.temperature)
                else:
                    # Legacy: mixed clauses from different proofs
                    return info_nce_loss(scores, labels, temperature=self.temperature)
            elif self.loss_type == 'margin':
                return margin_ranking_loss(scores, labels)
            elif self.loss_type == 'bce':
                return F.binary_cross_entropy_with_logits(scores, labels.float())
            else:
                # Legacy: cross-entropy for multi-class selection
                return F.cross_entropy(scores, labels)

        def _compute_metrics(self, scores, labels):
            """Compute ranking metrics."""
            pos_mask = labels.bool()
            neg_mask = ~pos_mask

            num_pos = pos_mask.sum()
            num_neg = neg_mask.sum()

            if num_pos == 0 or num_neg == 0:
                return {"acc": torch.tensor(0.0, device=scores.device), "mrr": torch.tensor(0.0, device=scores.device)}

            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]

            # Binary accuracy: predict positive if score > 0
            preds = (scores > 0).float()
            acc = (preds == labels.float()).float().mean()

            # Mean Reciprocal Rank: for each positive, what's its rank among all?
            # Higher score = better rank (rank 1 is best)
            all_scores = scores.unsqueeze(0)  # [1, batch]
            pos_ranks = (scores.unsqueeze(1) < all_scores).sum(dim=1).float() + 1  # [batch]
            pos_ranks = pos_ranks[pos_mask]  # ranks of positive examples
            mrr = (1.0 / pos_ranks).mean()

            return {"acc": acc, "mrr": mrr}

        def training_step(self, batch, batch_idx):
            clause_features = batch.get("clause_features")
            scores = self(batch["node_features"], batch["adj"], batch["pool_matrix"], clause_features)
            proof_ids = batch.get("proof_ids")
            loss = self._compute_loss(scores, batch["labels"], proof_ids)

            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            clause_features = batch.get("clause_features")
            scores = self(batch["node_features"], batch["adj"], batch["pool_matrix"], clause_features)
            proof_ids = batch.get("proof_ids")
            loss = self._compute_loss(scores, batch["labels"], proof_ids)

            metrics = self._compute_metrics(scores, batch["labels"])

            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            self.log("val_acc", metrics["acc"], prog_bar=True, sync_dist=True)
            self.log("val_mrr", metrics["mrr"], prog_bar=True, sync_dist=True)
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

    def log_final(self, best_epoch: int, best_val_loss: float, total_time: float, termination_reason: str = "unknown"):
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["best_epoch"] = best_epoch
        self.metrics["best_val_loss"] = best_val_loss
        self.metrics["total_time_seconds"] = total_time
        self.metrics["termination_reason"] = termination_reason
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
# Export Callback for Web Visualization
# =============================================================================


if LIGHTNING_AVAILABLE:

    class ExportCallback(Callback):
        """Export current training run to web/data/training/<run_name>.json after each epoch."""

        def __init__(self, run_name: str, metrics_file: Path, output_dir: Path):
            self.run_name = run_name
            self.metrics_file = metrics_file
            self.output_dir = output_dir

        def on_train_epoch_end(self, trainer, pl_module):
            # Only run on rank 0
            if trainer.global_rank != 0:
                return

            try:
                self._export_current_run()
            except Exception:
                # Don't fail training if export fails
                pass

        def _export_current_run(self):
            """Export current run to individual JSON file."""
            import json as json_module
            from datetime import datetime as dt

            if not self.metrics_file.exists():
                return

            # Load current run's metrics
            with open(self.metrics_file) as f:
                metrics = json_module.load(f)

            config = metrics.get("config", {})
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            epochs = metrics.get("epochs", [])

            epoch_history = [{
                "epoch": e.get("epoch", 0),
                "train_loss": e.get("train_loss"),
                "val_loss": e.get("val_loss"),
                "val_acc": e.get("val_acc"),
                "val_mrr": e.get("val_mrr"),
                "learning_rate": e.get("learning_rate"),
            } for e in epochs]

            output = {
                "generated": dt.now().isoformat(),
                "name": metrics.get("run_name", self.run_name),
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
            }

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.output_dir / f"{self.run_name}.json"
            with open(output_file, "w") as f:
                json_module.dump(output, f, indent=2)

            # Update index file with list of all runs
            self._update_index()

        def _update_index(self):
            """Update index.json with list of all available runs."""
            import json as json_module
            from datetime import datetime as dt

            index_file = self.output_dir / "index.json"
            runs = []

            # Scan directory for run files
            for f in sorted(self.output_dir.glob("*.json")):
                if f.name == "index.json":
                    continue
                runs.append(f.stem)

            index = {
                "generated": dt.now().isoformat(),
                "runs": runs,
            }

            with open(index_file, "w") as f:
                json_module.dump(index, f, indent=2)


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


def train_from_traces(
    train_trace_dir: Path,
    val_trace_dir: Path,
    config: SelectorConfig,
    max_clauses_per_proof: Optional[int] = 1000,
    min_prefix_clauses: int = 10,
) -> Tuple[nn.Module, Dict]:
    """
    Train a clause selection model from trace directories.

    This is the recommended training function. Each trace file contains
    one complete proof search in structured JSON format. During training,
    random prefixes are sampled to match the partial-information setting
    at inference time.

    Args:
        train_trace_dir: Directory with training trace files (.json)
        val_trace_dir: Directory with validation trace files (.json)
        config: Training configuration
        max_clauses_per_proof: Limit clauses per proof (for memory)
        min_prefix_clauses: Minimum prefix length to sample

    Returns:
        (trained_model, metrics_dict)
    """
    train_dataset = ProofDataset(
        train_trace_dir,
        output_type="graph",
        max_clauses=max_clauses_per_proof,
        min_prefix_clauses=min_prefix_clauses,
        sample_prefix=True,  # Training: sample random prefixes
    )
    val_dataset = ProofDataset(
        val_trace_dir,
        output_type="graph",
        max_clauses=max_clauses_per_proof,
        min_prefix_clauses=min_prefix_clauses,
        sample_prefix=False,  # Validation: use full proofs for consistent metrics
    )

    return train(train_dataset, val_dataset, config)


def train(
    train_dataset,
    val_dataset,
    config: SelectorConfig,
) -> Tuple[nn.Module, Dict]:
    """
    Train a clause selection model.

    Args:
        train_dataset: Training dataset (ProofDataset or ClauseDataset)
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

    # Select collate function based on dataset type
    # Check if dataset returns proof-style batches (with 'graphs' key)
    is_proof_dataset = isinstance(train_dataset, ProofDataset) or hasattr(train_dataset, '_clause_to_graph')
    if is_proof_dataset:
        collate_fn = collate_proof_batch
    else:
        collate_fn = collate_clause_batch

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Create module
    module = ClauseSelectionModule(config)

    # Create JSON logger
    json_logger = JSONLogger(config.logging.log_dir, config.name)
    json_logger.log_config(config)

    # Callbacks
    log_dir_path = Path(config.logging.log_dir)
    project_root = log_dir_path.parent if log_dir_path.name == ".logs" else log_dir_path.parent.parent
    metrics_file = log_dir_path / config.name / "metrics.json"

    callbacks = [
        JSONLoggingCallback(json_logger),
        ExportCallback(
            run_name=config.name,
            metrics_file=metrics_file,
            output_dir=project_root / "web" / "data" / "training",
        ),
        ModelCheckpoint(
            dirpath=log_dir_path / config.name / "checkpoints",
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
        logger=False,  # We use our own JSONLogger
        enable_progress_bar=config.logging.enable_progress_bar,
        log_every_n_steps=config.logging.log_every_n_steps,
    )

    # Train
    start_time = time.time()
    trainer.fit(module, train_loader, val_loader)
    total_time = time.time() - start_time

    # Determine termination reason
    early_stop_callback = next(
        (cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)), None
    )
    if early_stop_callback and early_stop_callback.stopped_epoch > 0:
        termination_reason = f"early_stopping (patience={config.training.patience})"
    elif trainer.current_epoch >= config.training.max_epochs - 1:
        termination_reason = f"max_epochs ({config.training.max_epochs})"
    else:
        termination_reason = "interrupted"

    # Log final metrics
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    json_logger.log_final(
        best_epoch=trainer.current_epoch,
        best_val_loss=float(best_val_loss) if best_val_loss else 0.0,
        total_time=total_time,
        termination_reason=termination_reason,
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
