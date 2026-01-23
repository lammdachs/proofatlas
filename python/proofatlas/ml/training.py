"""
Training infrastructure for clause selection models.

Provides datasets, collate functions, and loss functions for training
clause selection models. The training loop is in bench.py (run_training).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# =============================================================================
# Model Persistence
# =============================================================================


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


def load_traces(
    traces_dir: Path,
    preset: str,
    problem_names: Optional[set] = None,
) -> Dict[str, Any]:
    """Load traces for training, optionally filtered by problem names.

    Args:
        traces_dir: Base directory for traces (e.g., .data/traces/)
        preset: Trace preset name (subdirectory in traces_dir)
        problem_names: Optional set of problem names to include. If None, loads all.

    Returns:
        Dict with 'problems' list and 'num_problems' count.
    """
    from .structured import clause_to_graph

    preset_dir = Path(traces_dir) / preset
    if not preset_dir.exists():
        return {"problems": [], "num_problems": 0}

    problems = []
    for trace_file in sorted(preset_dir.glob("*.json")):
        # Filter by problem set if specified
        if problem_names is not None and trace_file.stem not in problem_names:
            continue

        try:
            with open(trace_file) as f:
                trace = json.load(f)
        except Exception:
            continue

        if not trace.get("proof_found") or not trace.get("clauses"):
            continue

        # Convert structured clauses to graph tensors
        clauses = trace["clauses"]
        max_age = len(clauses)
        graphs = [clause_to_graph(c, max_age) for c in clauses]
        labels = [c.get("label", 0) for c in clauses]

        problems.append({
            "name": trace_file.stem,
            "graphs": graphs,
            "labels": labels,
        })

    return {"problems": problems, "num_problems": len(problems)}


# =============================================================================
# Training Loop
# =============================================================================


def run_training(
    preset: dict,
    data: Dict[str, Any],
    weights_dir: Path,
    configs_dir: Path,
    init_weights: Optional[Path] = None,
    log_callback: Optional[callable] = None,
) -> Path:
    """Train a model and return the weights path.

    Args:
        preset: Preset config dict with embedding/scorer fields
        data: Training data from load_traces()
        weights_dir: Directory to save weights (.weights/)
        configs_dir: Directory containing config files (embeddings.json, etc.)
        init_weights: Optional path to weights file to initialize from
        log_callback: Optional callback(epoch, max_epochs, train_loss) for logging

    Returns:
        Path to saved weights file.
    """
    import random

    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name

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

    problems = data["problems"]
    if not problems:
        raise ValueError("No training data")

    # Problem-level split
    val_ratio = config.get("val_ratio", 0.0)
    random.seed(42)
    problem_indices = list(range(len(problems)))
    random.shuffle(problem_indices)

    if val_ratio > 0:
        val_count = max(1, int(len(problems) * val_ratio))
        train_indices = problem_indices[val_count:]
        val_indices = problem_indices[:val_count]
    else:
        train_indices = problem_indices
        val_indices = []

    def make_dataset(prob_indices):
        all_graphs, all_labels = [], []
        for idx in prob_indices:
            p = problems[idx]
            all_graphs.extend(p["graphs"])
            all_labels.extend(p["labels"])
        if not all_graphs:
            return None
        return ClauseDataset(
            node_features=[g["x"] for g in all_graphs],
            edge_indices=[g["edge_index"] for g in all_graphs],
            labels=all_labels,
        )

    train_ds = make_dataset(train_indices)
    val_ds = make_dataset(val_indices) if val_indices else None

    if not train_ds:
        raise ValueError("No training examples")

    model_name = get_model_name(preset)
    print(f"Training {model_name}: {len(train_indices)} problems, {len(train_ds)} examples")

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

    # Initialize from existing weights if provided
    if init_weights and Path(init_weights).exists():
        print(f"Initializing from {init_weights}")
        from safetensors.torch import load_file
        state_dict = load_file(init_weights)
        model.load_state_dict(state_dict)
        model = model.to(device)

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_clause_batch
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_clause_batch)
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

        if val_loader:
            print(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{max_epochs} | train={train_loss:.4f}")

        if log_callback:
            log_callback(epoch, max_epochs, train_loss)

    # Save weights using modular naming: {embedding}_{scorer}
    from safetensors.torch import save_file

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.safetensors"

    metadata = {
        "model_type": model_type,
        "hidden_dim": str(config.get("hidden_dim", 64)),
        "num_layers": str(config.get("num_layers", 3)),
        "num_heads": str(config.get("num_heads", 4)),
        "input_dim": str(config.get("input_dim", 13)),
    }
    save_file(model.state_dict(), weights_path, metadata=metadata)
    print(f"Weights saved: {weights_path}")

    return weights_path


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


