"""
Training infrastructure for clause selection models.

Provides datasets, collate functions, and loss functions for training
clause selection models. The training loop is in bench.py (run_training).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..selectors import create_model
from .config import TrainingConfig

# Use orjson for faster JSON loading if available
try:
    import orjson
    _ORJSON_AVAILABLE = True
except ImportError:
    _ORJSON_AVAILABLE = False


def _load_json(path: Path) -> dict:
    """Load JSON file using orjson if available, else standard json."""
    if _ORJSON_AVAILABLE:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    else:
        with open(path) as f:
            return json.load(f)


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

    All data is loaded and converted upfront in __init__ for simplicity.
    """

    def __init__(
        self,
        trace_dir: Path,
        output_type: str = "graph",  # "graph" or "string"
        max_clauses: Optional[int] = None,
        problem_names: Optional[set] = None,
        trace_files: Optional[List[Path]] = None,
    ):
        """
        Args:
            trace_dir: Directory containing .json trace files
            output_type: "graph" for GNN models, "string" for sentence models
            max_clauses: Optional limit on clauses per proof (for memory)
            problem_names: Optional set of problem names to include
            trace_files: Optional explicit list of trace files (overrides trace_dir)
        """
        self.output_type = output_type
        self.max_clauses = max_clauses

        # Get trace files
        if trace_files is not None:
            files = trace_files
        elif trace_dir:
            trace_dir = Path(trace_dir)
            files = sorted(trace_dir.glob("*.json"))
            if problem_names is not None:
                files = [f for f in files if f.stem in problem_names]
        else:
            files = []

        if not files:
            raise ValueError("No JSON trace files found")

        # Load all data upfront
        self.items = []
        for trace_file in files:
            item = self._load_trace(trace_file)
            if item is not None:
                self.items.append(item)

        if not self.items:
            raise ValueError("No valid traces found (all filtered or empty)")

    def _load_trace(self, trace_file: Path) -> Optional[Dict]:
        """Load and convert a single trace file."""
        import random

        try:
            trace = _load_json(trace_file)
        except Exception:
            return None

        if not trace.get("proof_found") or not trace.get("clauses"):
            return None

        clauses = trace["clauses"]

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

        labels = [c.get("label", 0) for c in clauses]

        if self.output_type == "graph":
            from .structured import clause_to_graph

            max_age = len(clauses)
            graphs = [clause_to_graph(c, max_age) for c in clauses]

            return {
                "graphs": graphs,
                "labels": labels,
                "problem": trace_file.stem,
            }
        else:
            from .structured import clause_to_string

            strings = [clause_to_string(c) for c in clauses]

            return {
                "strings": strings,
                "labels": labels,
                "problem": trace_file.stem,
            }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


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

    Combines all clause graphs from all proofs into batched tensors,
    with proof_ids to track which proof each clause belongs to.

    Uses batch_graphs which outputs sparse adjacency and pool matrices
    for memory-efficient processing of large batches.
    """
    from .structured import batch_graphs

    # Collect all graphs and labels, tracking proof membership
    all_graphs = []
    all_labels = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        graphs = item["graphs"]
        labels = item["labels"]

        all_graphs.extend(graphs)
        all_labels.extend(labels)
        all_proof_ids.extend([proof_idx] * len(graphs))

    # Build batched tensors with sparse adjacency and pool matrices
    batched = batch_graphs(all_graphs, labels=all_labels)

    return {
        "node_features": batched["x"],
        "adj": batched["adj"],
        "pool_matrix": batched["pool_matrix"],
        "labels": batched["y"],
        "proof_ids": torch.tensor(all_proof_ids, dtype=torch.long),
        "clause_features": batched.get("clause_features"),
    }


# =============================================================================
# JSON Logger for Web Visualization
# =============================================================================


class JSONLogger:
    """Logger that writes metrics to JSON for web visualization.

    Writes to web/data/training/{run_name}.json and updates index.json.
    The web interface reads these files to display training progress.
    """

    def __init__(self, web_data_dir: Path, run_name: str):
        """Initialize logger.

        Args:
            web_data_dir: Path to web/data directory
            run_name: Name for this training run (used as filename)
        """
        self.web_data_dir = Path(web_data_dir)
        self.training_dir = self.web_data_dir / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name
        self.log_file = self.training_dir / f"{run_name}.json"
        self.index_file = self.training_dir / "index.json"

        self.metrics = {
            "generated": datetime.now().isoformat(),
            "name": run_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_time_seconds": None,
            "termination_reason": None,
            "best_epoch": None,
            "best_val_loss": None,
            "model": {},
            "training": {},
            "epochs": [],
            "evaluations": [],
        }
        self._save()
        self._update_index()

    def log_config(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """Log model and training configuration."""
        self.metrics["model"] = model_config
        self.metrics["training"] = training_config
        self._save()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  val_acc: Optional[float] = None, lr: Optional[float] = None):
        """Log metrics for one epoch."""
        self.metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_mrr": None,
            "learning_rate": lr,
        })
        self._save()

    def log_evaluation(self, epoch: int, results: Dict[str, Any]):
        """Log evaluation results."""
        self.metrics["evaluations"].append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **results,
        })
        self._save()

    def log_final(self, best_epoch: Optional[int], best_val_loss: Optional[float],
                  total_time: float, termination_reason: str = "completed"):
        """Log final training results."""
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["best_epoch"] = best_epoch
        self.metrics["best_val_loss"] = best_val_loss
        self.metrics["total_time_seconds"] = total_time
        self.metrics["termination_reason"] = termination_reason
        self._save()

    def _save(self):
        """Save metrics to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _update_index(self):
        """Update index.json with this run."""
        # Load existing index
        if self.index_file.exists():
            with open(self.index_file) as f:
                index = json.load(f)
        else:
            index = {"runs": []}

        # Add this run if not already present
        if self.run_name not in index["runs"]:
            index["runs"].append(self.run_name)
            index["runs"].sort()

        # Save index
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)


# =============================================================================
# Model Persistence
# =============================================================================


def save_model(model: nn.Module, path: Path, config: Optional[TrainingConfig] = None):
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
# Trace Management
# =============================================================================


def save_trace(traces_dir: Path, preset: str, problem: str, trace_json: str):
    """Save proof trace for training in structured JSON format.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Preset name for trace subdirectory
        problem: Problem file name
        trace_json: Structured JSON string from extract_structured_trace()
    """
    try:
        preset_dir = Path(traces_dir) / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        problem_name = Path(problem).stem
        json_path = preset_dir / f"{problem_name}.json"
        with open(json_path, "w") as f:
            f.write(trace_json)
    except Exception:
        pass


def load_trace_files(
    traces_dir: Path,
    preset: str,
    problem_names: Optional[set] = None,
) -> List[Path]:
    """Get list of trace files for a preset.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Trace preset name (subdirectory in traces_dir)
        problem_names: Optional set of problem names to include. If None, loads all.

    Returns:
        List of trace file paths.
    """
    preset_dir = Path(traces_dir) / preset
    if not preset_dir.exists():
        return []

    trace_files = sorted(preset_dir.glob("*.json"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.stem in problem_names]

    return trace_files


# =============================================================================
# Training Loop
# =============================================================================


def run_training(
    preset: dict,
    trace_dir: Path,
    weights_dir: Path,
    configs_dir: Path,
    problem_names: Optional[set] = None,
    log_callback: Optional[callable] = None,
    web_data_dir: Optional[Path] = None,
    log_file: Optional[Any] = None,
) -> Path:
    """Train a model and return the weights path.

    Args:
        preset: Preset config dict with embedding/scorer fields
        trace_dir: Directory containing trace JSON files
        weights_dir: Directory to save weights (.weights/)
        configs_dir: Directory containing config files (embeddings.json, etc.)
        problem_names: Optional set of problem names to filter traces
        log_callback: Optional callback(epoch, max_epochs, train_loss) for logging
        web_data_dir: Directory for web data (web/data/) - enables live web updates
        log_file: File object for logging (e.g., bench.log)

    Returns:
        Path to saved weights file.
    """
    import random
    import time

    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name

    start_time = time.time()

    def log_msg(msg: str):
        """Log message with timestamp to stdout and optionally to log_file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if log_file and log_file is not sys.stdout:
            # Write to both stdout and log file
            print(line)
            log_file.write(line + "\n")
            log_file.flush()
        else:
            # Just print to stdout (log_file is None or is stdout)
            print(line)

    # Load configs
    configs_dir = Path(configs_dir)
    with open(configs_dir / "embeddings.json") as f:
        embeddings_config = json.load(f)
    with open(configs_dir / "scorers.json") as f:
        scorers_config = json.load(f)
    with open(configs_dir / "training.json") as f:
        training_config = json.load(f)

    # Get model and training config from preset
    embedding_name = preset.get("embedding") or preset.get("model")
    scorer_name = preset.get("scorer", "mlp")
    training_name = preset.get("training", "standard")

    # Get embedding architecture
    embedding_arch = embeddings_config["architectures"].get(embedding_name)
    if not embedding_arch:
        raise ValueError(f"Unknown embedding: {embedding_name}")

    # Get scorer architecture
    scorer_arch = scorers_config["architectures"].get(scorer_name)
    if not scorer_arch:
        raise ValueError(f"Unknown scorer: {scorer_name}")

    # Get training config
    training_defaults = training_config.get("defaults", {})
    training_overrides = training_config.get("configs", {}).get(training_name, {})

    # Merge configs
    config = {**training_defaults, **training_overrides}
    config["embedding"] = embedding_arch
    config["scorer"] = scorer_arch
    config["input_dim"] = embeddings_config.get("input_dim", 8)

    # Get trace files
    trace_dir = Path(trace_dir)
    trace_files = sorted(trace_dir.glob("*.json"))
    if problem_names is not None:
        trace_files = [f for f in trace_files if f.stem in problem_names]

    if not trace_files:
        raise ValueError(f"No trace files found in {trace_dir}")

    log_msg(f"Found {len(trace_files)} trace files")

    # Problem-level split
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    random.shuffle(trace_files)

    if val_ratio > 0:
        val_count = max(1, int(len(trace_files) * val_ratio))
        train_files = trace_files[val_count:]
        val_files = trace_files[:val_count]
    else:
        train_files = trace_files
        val_files = []

    # Create datasets (loads all data upfront)
    max_clauses = config.get("max_clauses_per_proof", 512)
    log_msg("Loading training data...")
    train_ds = ProofDataset(
        trace_dir=None,
        trace_files=train_files,
        output_type="graph",
        max_clauses=max_clauses,
    )
    val_ds = None
    if val_files:
        log_msg("Loading validation data...")
        val_ds = ProofDataset(
            trace_dir=None,
            trace_files=val_files,
            output_type="graph",
            max_clauses=max_clauses,
        )

    log_msg(f"Train: {len(train_ds)} traces, Val: {len(val_ds) if val_ds else 0} traces")

    model_name = get_model_name(preset)
    log_msg(f"Training {model_name}: {len(train_ds)} traces")

    # Create model
    model_type = config.get("type", "gcn")
    model = create_model(
        model_type=model_type,
        node_feature_dim=config.get("input_dim", 13),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 3),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )

    needs_adj = model_type in ["gcn", "gat", "graphsage", "gnn_transformer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_proof_batch, num_workers=0
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_proof_batch, num_workers=0)
        if val_ds else None
    )

    # Optimizer
    optimizer_type = config.get("optimizer", "adamw").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-5)

    if optimizer_type == "adamw":
        betas = tuple(config.get("betas", [0.9, 0.999]))
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "adam":
        betas = tuple(config.get("betas", [0.9, 0.999]))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    max_epochs = config.get("max_epochs", 100)

    # Initialize web logger for live updates
    web_logger = None
    if web_data_dir:
        web_logger = JSONLogger(web_data_dir, model_name)
        web_logger.log_config(
            model_config={
                "type": model_type,
                "hidden_dim": config.get("hidden_dim", 64),
                "num_layers": config.get("num_layers", 3),
                "input_dim": config.get("input_dim", 13),
                "scorer_type": scorer_name,
            },
            training_config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "max_epochs": max_epochs,
                "optimizer": optimizer_type,
                "weight_decay": weight_decay,
                "margin": config.get("margin", 0.1),
            },
        )

    # Track best model and early stopping
    best_val_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    patience = config.get("patience", 10)
    patience_counter = 0

    # Margin ranking loss for pairwise learning
    margin = config.get("margin", 0.1)
    ranking_loss = nn.MarginRankingLoss(margin=margin)

    def compute_pairwise_loss(scores, labels):
        """Compute pairwise margin ranking loss between positive and negative examples."""
        pos_mask = labels == 1
        neg_mask = labels == 0

        if not pos_mask.any() or not neg_mask.any():
            # Fallback to BCE if no pairs available
            return F.binary_cross_entropy_with_logits(scores, labels.float())

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # Sample pairs: for each positive, sample a random negative
        n_pos = pos_scores.size(0)
        n_neg = neg_scores.size(0)

        # Random pairing: sample n_pos negatives (with replacement if needed)
        neg_indices = torch.randint(0, n_neg, (n_pos,), device=scores.device)
        neg_sampled = neg_scores[neg_indices]

        # Target: +1 means first input should be ranked higher than second
        target = torch.ones(n_pos, device=scores.device)
        return ranking_loss(pos_scores, neg_sampled, target)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["node_features"].to(device)
            adj = batch["adj"].to(device)
            pool = batch["pool_matrix"].to(device)
            labels = batch["labels"].to(device)

            scores = model(x, adj, pool) if needs_adj else model(x, pool)
            loss = compute_pairwise_loss(scores, labels)
            loss.backward()

            if config.get("gradient_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["node_features"].to(device)
                    adj = batch["adj"].to(device)
                    pool = batch["pool_matrix"].to(device)
                    labels = batch["labels"].to(device)
                    scores = model(x, adj, pool) if needs_adj else model(x, pool)
                    val_loss += compute_pairwise_loss(scores, labels).item()
            val_loss /= len(val_loader)

        # Track best model and early stopping
        current_val = val_loss if val_loader else train_loss
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log to console and file
        if val_loader:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            log_msg(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f}")

        # Log to web for live updates
        if web_logger:
            web_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss if val_loader else None,
                val_acc=None,
                lr=lr,
            )

        if log_callback:
            log_callback(epoch, max_epochs, train_loss)

        # Early stopping check
        if patience_counter >= patience:
            log_msg(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log_msg(f"Restored best model from epoch {best_epoch}")

    # Log final results
    total_time = time.time() - start_time
    early_stopped = patience_counter >= patience
    if web_logger:
        web_logger.log_final(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            total_time=total_time,
            termination_reason="early_stopped" if early_stopped else "completed",
        )

    # Export to TorchScript for Rust inference
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    model.eval()
    model.cpu()

    # Create example inputs for tracing
    hidden_dim = config.get("hidden_dim", 64)
    example_x = torch.randn(10, config.get("input_dim", 13))
    example_adj = torch.eye(10)
    example_pool = torch.ones(3, 10) / 10

    if needs_adj:
        traced = torch.jit.trace(model, (example_x, example_adj, example_pool))
    else:
        traced = torch.jit.trace(model, (example_x, example_pool))

    traced.save(str(weights_path))
    log_msg(f"TorchScript model saved: {weights_path}")
    log_msg(f"Training completed in {total_time:.1f}s (best epoch: {best_epoch})")

    return weights_path

