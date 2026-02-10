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


def _pool_init():
    """Initializer for pool workers - ignore SIGTERM so parent handles it."""
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _pool_init_tokenizer():
    """Initializer for pool workers that preloads the tokenizer."""
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # Preload tokenizer in each worker to avoid repeated loading
    _get_tokenizer()


def _load_trace_graph(trace_file: Path) -> Optional[Dict]:
    """Load trace file and convert to graphs with selection states.

    Returns graphs for all clauses plus selection_states for U/P sampling.
    Skips traces without selection_states.
    """
    from .structured import clause_to_graph

    try:
        trace = _load_json(trace_file)
    except Exception:
        return None

    if not trace.get("proof_found") or not trace.get("clauses"):
        return None
    if not trace.get("selection_states"):
        return None

    clauses = trace["clauses"]
    labels = [c.get("label", 0) for c in clauses]
    graphs = [clause_to_graph(c) for c in clauses]

    return {
        "graphs": graphs,
        "labels": labels,
        "selection_states": trace["selection_states"],
        "problem": trace_file.stem,
    }


# Global tokenizer for pre-tokenization (loaded lazily)
_tokenizer = None


def _get_tokenizer():
    """Get or load the tokenizer for pre-tokenization."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def _load_trace_tokenized(trace_file: Path) -> Optional[Dict]:
    """Load trace file and pre-tokenize strings with selection states.

    Returns tokenized inputs for all clauses plus selection_states for U/P sampling.
    Skips traces without selection_states.
    """
    from .structured import clause_to_string

    try:
        trace = _load_json(trace_file)
    except Exception:
        return None

    if not trace.get("proof_found") or not trace.get("clauses"):
        return None
    if not trace.get("selection_states"):
        return None

    clauses = trace["clauses"]
    labels = [c.get("label", 0) for c in clauses]
    strings = [clause_to_string(c) for c in clauses]

    # Pre-tokenize
    tokenizer = _get_tokenizer()
    encoded = tokenizer(
        strings,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "selection_states": trace["selection_states"],
        "problem": trace_file.stem,
    }


class DynamicBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that creates batches with approximately equal total size.

    Groups proofs together until max_clauses is reached, allowing efficient
    GPU utilization while avoiding OOM on large proofs.
    """

    def __init__(self, dataset, max_clauses: int = 8192, shuffle: bool = True):
        """
        Args:
            dataset: ProofDataset with items containing size info
            max_clauses: Maximum total clauses per batch
            shuffle: Whether to shuffle between epochs
        """
        self.dataset = dataset
        self.max_clauses = max_clauses
        self.shuffle = shuffle

        # Get sizes for each item
        self.sizes = []
        for item in dataset.items:
            if "graphs" in item:
                self.sizes.append(len(item["graphs"]))
            elif "labels" in item:
                self.sizes.append(len(item["labels"]))
            else:
                self.sizes.append(1)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        batch = []
        batch_size = 0

        for idx in indices:
            size = self.sizes[idx]

            # If single item exceeds max, yield it alone
            if size > self.max_clauses:
                if batch:
                    yield batch
                    batch = []
                    batch_size = 0
                yield [idx]
                continue

            # If adding this item would exceed max, yield current batch
            if batch_size + size > self.max_clauses and batch:
                yield batch
                batch = []
                batch_size = 0

            batch.append(idx)
            batch_size += size

        # Yield remaining batch
        if batch:
            yield batch

    def __len__(self):
        # Approximate number of batches
        total = sum(self.sizes)
        return max(1, total // self.max_clauses)


class ProofDataset(torch.utils.data.Dataset):
    """
    Dataset that loads structured JSON traces and converts to graphs/strings.

    Data is loaded in parallel using multiprocessing for speed.
    """

    def __init__(
        self,
        trace_dir: Path,
        output_type: str = "graph",  # "graph" or "tokenized"
        problem_names: Optional[set] = None,
        trace_files: Optional[List[Path]] = None,
        n_workers: int = None,
    ):
        """
        Args:
            trace_dir: Directory containing .json trace files
            output_type: "graph" for GNN models, "tokenized" for sentence models
            problem_names: Optional set of problem names to include
            trace_files: Optional explicit list of trace files (overrides trace_dir)
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.output_type = output_type

        # Get trace files
        if trace_files is not None:
            files = list(trace_files)
        elif trace_dir:
            trace_dir = Path(trace_dir)
            files = sorted(trace_dir.glob("*.json"))
            if problem_names is not None:
                files = [f for f in files if f.stem in problem_names]
        else:
            files = []

        if not files:
            raise ValueError("No JSON trace files found")

        # Load data in parallel
        if output_type == "tokenized":
            load_fn = _load_trace_tokenized
        else:
            load_fn = _load_trace_graph

        if n_workers is None:
            import os
            n_workers = os.cpu_count() or 4

        if n_workers > 1 and len(files) > 10:
            from multiprocessing import Pool
            # Use tokenizer-aware initializer for pre-tokenized data
            init_fn = _pool_init_tokenizer if output_type == "tokenized" else _pool_init
            with Pool(n_workers, initializer=init_fn) as pool:
                results = pool.map(load_fn, files)
        else:
            # For single-threaded loading, preload tokenizer if needed
            if output_type == "tokenized":
                _get_tokenizer()
            results = [load_fn(f) for f in files]

        self.items = [r for r in results if r is not None]

        if not self.items:
            raise ValueError("No valid traces found (all filtered or empty)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_tokenized_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for pre-tokenized sentence batches with state sampling.

    For each proof, randomly samples one selection state (U/P snapshot).
    Splits tokens into U and P sets with proof_ids for per-proof loss.
    """
    import random

    all_u_input_ids = []
    all_u_attention_mask = []
    all_u_labels = []
    all_p_input_ids = []
    all_p_attention_mask = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]
        states = item["selection_states"]

        # Sample a random selection state
        state = random.choice(states)
        u_indices = state["unprocessed"]
        p_indices = state["processed"]

        for idx in u_indices:
            if idx < len(labels):
                all_u_input_ids.append(input_ids[idx])
                all_u_attention_mask.append(attention_mask[idx])
                all_u_labels.append(labels[idx])
                all_proof_ids.append(proof_idx)

        for idx in p_indices:
            if idx < len(labels):
                all_p_input_ids.append(input_ids[idx])
                all_p_attention_mask.append(attention_mask[idx])

    if not all_u_input_ids:
        return None

    # Pad to uniform length across traces (each trace may have different seq lengths)
    from torch.nn.utils.rnn import pad_sequence
    result = {
        "u_input_ids": pad_sequence(all_u_input_ids, batch_first=True, padding_value=0),
        "u_attention_mask": pad_sequence(all_u_attention_mask, batch_first=True, padding_value=0),
        "labels": torch.tensor(all_u_labels, dtype=torch.float32),
        "proof_ids": torch.tensor(all_proof_ids, dtype=torch.long),
    }

    if all_p_input_ids:
        result["p_input_ids"] = pad_sequence(all_p_input_ids, batch_first=True, padding_value=0)
        result["p_attention_mask"] = pad_sequence(all_p_attention_mask, batch_first=True, padding_value=0)

    return result


def collate_proof_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for graph proof batches with state sampling.

    For each proof, randomly samples one selection state (U/P snapshot).
    Builds separate graph batches for U and P clause sets.
    Labels are only for U clauses (what to select from the unprocessed set).
    """
    import random
    from .structured import batch_graphs

    all_u_graphs = []
    all_p_graphs = []
    all_u_labels = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        graphs = item["graphs"]
        labels = item["labels"]
        states = item["selection_states"]

        # Sample a random selection state
        state = random.choice(states)
        u_indices = state["unprocessed"]
        p_indices = state["processed"]

        # Collect graphs and labels for U and P
        for idx in u_indices:
            if idx < len(graphs):
                all_u_graphs.append(graphs[idx])
                all_u_labels.append(labels[idx])
                all_proof_ids.append(proof_idx)

        for idx in p_indices:
            if idx < len(graphs):
                all_p_graphs.append(graphs[idx])

    if not all_u_graphs:
        return None

    # Batch U graphs (with labels)
    u_batched = batch_graphs(all_u_graphs, labels=all_u_labels)

    result = {
        "u_node_features": u_batched["x"],
        "u_adj": u_batched["adj"],
        "u_pool_matrix": u_batched["pool_matrix"],
        "u_clause_features": u_batched.get("clause_features"),
        "labels": u_batched["y"],
        "proof_ids": torch.tensor(all_proof_ids, dtype=torch.long),
    }

    # Batch P graphs (no labels needed)
    if all_p_graphs:
        p_batched = batch_graphs(all_p_graphs)
        result["p_node_features"] = p_batched["x"]
        result["p_adj"] = p_batched["adj"]
        result["p_pool_matrix"] = p_batched["pool_matrix"]
        result["p_clause_features"] = p_batched.get("clause_features")

    return result


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
        Path to weights directory (not the file) to match find_weights().
    """
    import random
    import time

    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .weights import get_model_name, get_embedding_type

    start_time = time.time()

    def log_msg(msg: str):
        """Log message with timestamp to stdout and optionally to log_file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if log_file and log_file is not sys.stdout:
            # Write to both stdout and log file
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        else:
            # Just print to stdout (log_file is None or is stdout)
            print(line, flush=True)

    # Load configs
    configs_dir = Path(configs_dir)
    with open(configs_dir / "embeddings.json") as f:
        embeddings_config = json.load(f)
    with open(configs_dir / "scorers.json") as f:
        scorers_config = json.load(f)
    with open(configs_dir / "training.json") as f:
        training_config = json.load(f)

    # Get model and training config from preset
    embedding_name = preset.get("encoder") or preset.get("embedding") or preset.get("model")
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

    # Determine output type based on embedding
    embedding_type = get_embedding_type(preset)
    if embedding_type == "string":
        output_type = "tokenized"
    else:
        output_type = "graph"

    # Create datasets (loads all data in parallel)
    log_msg(f"Loading training data ({output_type} format)...")
    train_ds = ProofDataset(
        trace_dir=None,
        trace_files=train_files,
        output_type=output_type,
    )

    val_ds = None
    if val_files:
        log_msg(f"Loading validation data ({output_type} format)...")
        val_ds = ProofDataset(
            trace_dir=None,
            trace_files=val_files,
            output_type=output_type,
        )

    log_msg(f"Train: {len(train_ds)} traces, Val: {len(val_ds) if val_ds else 0} traces")

    model_name = get_model_name(preset)
    log_msg(f"Training {model_name}: {len(train_ds)} traces")

    # Create model
    emb_config = config.get("embedding", {})
    scorer_config = config.get("scorer", {})
    model_type = emb_config.get("type", config.get("type", "gcn"))
    hidden_dim = emb_config.get("hidden_dim", config.get("hidden_dim", 64))
    model = create_model(
        model_type=model_type,
        node_feature_dim=config.get("input_dim", 13),
        hidden_dim=hidden_dim,
        num_layers=emb_config.get("num_layers", config.get("num_layers", 3)),
        num_heads=emb_config.get("num_heads", config.get("num_heads", 4)),
        scorer_type=scorer_config.get("type", "mlp"),
        scorer_num_heads=scorer_config.get("num_heads", 4),
        scorer_num_layers=scorer_config.get("num_layers", 2),
        freeze_encoder=emb_config.get("freeze_encoder", False),
    )

    needs_adj = model_type in ["gcn", "gat", "graphsage", "gnn_transformer"]
    is_sentence_model = model_type == "sentence"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dynamic batching: group proofs up to max_clauses per batch
    # Sentence models use micro-batching inside encode_tokens() to avoid OOM,
    # but we still want smaller proof groups to keep gradient updates frequent.
    default_max_clauses = 512 if is_sentence_model else 8192
    max_clauses = config.get("max_clauses_per_batch", default_max_clauses)
    default_accumulate = 4 if is_sentence_model else 1
    accumulate_steps = config.get("accumulate_steps", default_accumulate)

    # Choose collate function based on output type
    if output_type == "tokenized":
        collate_fn = collate_tokenized_batch
    else:
        collate_fn = collate_proof_batch

    train_sampler = DynamicBatchSampler(train_ds, max_clauses=max_clauses, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0
    )
    val_loader = None
    if val_ds:
        val_sampler = DynamicBatchSampler(val_ds, max_clauses=max_clauses, shuffle=False)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0
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
                "batch_size": accumulate_steps,
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

    # Loss configuration
    loss_type = config.get("loss_type", "info_nce")
    temperature = config.get("temperature", 1.0)
    margin = config.get("margin", 0.1)

    def compute_loss(scores, labels, proof_ids=None):
        """Compute loss based on configured loss type."""
        if loss_type == "info_nce":
            if proof_ids is not None:
                return info_nce_loss_per_proof(scores, labels, proof_ids, temperature)
            else:
                return info_nce_loss(scores, labels, temperature)
        elif loss_type == "margin":
            return margin_ranking_loss(scores, labels, margin=margin)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            labels = batch["labels"].to(device)
            proof_ids = batch.get("proof_ids")
            if proof_ids is not None:
                proof_ids = proof_ids.to(device)

            if is_sentence_model:
                # Sentence model: encode U and P token sets separately
                u_ids = batch["u_input_ids"].to(device)
                u_mask = batch["u_attention_mask"].to(device)
                u_emb = model.encode_tokens(u_ids, u_mask)

                p_emb = None
                if "p_input_ids" in batch:
                    p_ids = batch["p_input_ids"].to(device)
                    p_mask = batch["p_attention_mask"].to(device)
                    p_emb = model.encode_tokens(p_ids, p_mask)

                scores = model.scorer(u_emb, p_emb)
            else:
                # GNN model: encode U and P graph sets separately
                u_x = batch["u_node_features"].to(device)
                u_adj = batch["u_adj"].to(device)
                u_pool = batch["u_pool_matrix"].to(device)
                u_cf = batch.get("u_clause_features")
                if u_cf is not None:
                    u_cf = u_cf.to(device)

                u_emb = model.encode(u_x, u_adj, u_pool, u_cf)

                p_emb = None
                if "p_node_features" in batch:
                    p_x = batch["p_node_features"].to(device)
                    p_adj = batch["p_adj"].to(device)
                    p_pool = batch["p_pool_matrix"].to(device)
                    p_cf = batch.get("p_clause_features")
                    if p_cf is not None:
                        p_cf = p_cf.to(device)
                    p_emb = model.encode(p_x, p_adj, p_pool, p_cf)

                scores = model.scorer(u_emb, p_emb)

            loss = compute_loss(scores, labels, proof_ids)
            if accumulate_steps > 1:
                loss = loss / accumulate_steps
            loss.backward()
            train_loss += loss.item() * (accumulate_steps if accumulate_steps > 1 else 1)
            num_batches += 1

            # Step optimizer after accumulating gradients
            if accumulate_steps <= 1 or (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                gradient_clip = config.get("gradient_clip", 1.0)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= num_batches if num_batches > 0 else 1

        val_loss = 0
        if val_loader:
            model.eval()
            num_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue

                    labels = batch["labels"].to(device)
                    val_proof_ids = batch.get("proof_ids")
                    if val_proof_ids is not None:
                        val_proof_ids = val_proof_ids.to(device)

                    if is_sentence_model:
                        u_ids = batch["u_input_ids"].to(device)
                        u_mask = batch["u_attention_mask"].to(device)
                        u_emb = model.encode_tokens(u_ids, u_mask)

                        p_emb = None
                        if "p_input_ids" in batch:
                            p_ids = batch["p_input_ids"].to(device)
                            p_mask = batch["p_attention_mask"].to(device)
                            p_emb = model.encode_tokens(p_ids, p_mask)

                        scores = model.scorer(u_emb, p_emb)
                    else:
                        u_x = batch["u_node_features"].to(device)
                        u_adj = batch["u_adj"].to(device)
                        u_pool = batch["u_pool_matrix"].to(device)
                        u_cf = batch.get("u_clause_features")
                        if u_cf is not None:
                            u_cf = u_cf.to(device)
                        u_emb = model.encode(u_x, u_adj, u_pool, u_cf)

                        p_emb = None
                        if "p_node_features" in batch:
                            p_x = batch["p_node_features"].to(device)
                            p_adj = batch["p_adj"].to(device)
                            p_pool = batch["p_pool_matrix"].to(device)
                            p_cf = batch.get("p_clause_features")
                            if p_cf is not None:
                                p_cf = p_cf.to(device)
                            p_emb = model.encode(p_x, p_adj, p_pool, p_cf)

                        scores = model.scorer(u_emb, p_emb)

                    val_loss += compute_loss(scores, labels, val_proof_ids).item()
                    num_val_batches += 1
            val_loss /= num_val_batches if num_val_batches > 0 else 1

        # Track best model and early stopping (use model for state dict)
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

    # Restore best model (use model to handle DataParallel)
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

    # Export to TorchScript for Rust inference (use model, not DataParallel wrapper)
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    model.eval()

    # Trace on CPU — torch.jit.trace bakes in device placement for tensor
    # creation ops. CPU is the default eval backend (bench --backend-eval cpu).
    # Modules with data-dependent shapes (GraphNorm) are scripted below to
    # preserve dynamic behavior regardless of trace device.
    model.cpu()

    if is_sentence_model:
        # Sentence model has its own export method that handles tokenizer
        model.cpu()
        model.export_torchscript(str(weights_path), save_tokenizer=True)
    else:
        # GNN models: trace with example inputs (must match Rust call signature)
        # Script GraphNorm modules before tracing — their forward() uses
        # data-dependent shapes (batch.max().item()) that trace bakes as constants
        if hasattr(model, 'norms'):
            for i in range(len(model.norms)):
                model.norms[i] = torch.jit.script(model.norms[i])

        # Rust sends sparse COO tensors for adj and pool_matrix
        num_nodes, num_clauses = 10, 3
        example_x = torch.randn(num_nodes, config.get("input_dim", 13), device=trace_device)
        example_adj = torch.eye(num_nodes, device=trace_device).to_sparse()
        example_pool = (torch.ones(num_clauses, num_nodes, device=trace_device) / num_nodes).to_sparse()
        example_clause_features = torch.randn(num_clauses, 3, device=trace_device)

        if needs_adj:
            traced = torch.jit.trace(model, (example_x, example_adj, example_pool, example_clause_features), check_trace=False)
        else:
            traced = torch.jit.trace(model, (example_x, example_pool), check_trace=False)

        traced.save(str(weights_path))

        # Also export split encoder + scorer for cross-attention in Rust
        if needs_adj and hasattr(model, 'encode'):
            emb_cfg = config.get("embedding", {})
            hidden_dim = emb_cfg.get("hidden_dim", config.get("hidden_dim", 64))

            # Export encoder (model.encode → embeddings)
            class _EncoderWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x, adj, pool, cf):
                    return self.m.encode(x, adj, pool, cf)

            encoder_wrapper = _EncoderWrapper(model)
            encoder_wrapper.eval()
            with torch.no_grad():
                encoder_traced = torch.jit.trace(
                    encoder_wrapper,
                    (example_x, example_adj, example_pool, example_clause_features),
                    check_trace=False,
                )
            encoder_path = weights_dir / f"{model_name}_encoder.pt"
            encoder_traced.save(str(encoder_path))
            log_msg(f"Encoder saved: {encoder_path}")

            # Export scorer (model.scorer → scores)
            example_u_emb = torch.randn(num_clauses, hidden_dim, device=trace_device)
            example_p_emb = torch.randn(2, hidden_dim, device=trace_device)

            class _ScorerWrapper(nn.Module):
                def __init__(self, scorer):
                    super().__init__()
                    self.scorer = scorer
                def forward(self, u_emb, p_emb):
                    return self.scorer(u_emb, p_emb)

            scorer_wrapper = _ScorerWrapper(model.scorer)
            scorer_wrapper.eval()
            with torch.no_grad():
                scorer_traced = torch.jit.trace(
                    scorer_wrapper,
                    (example_u_emb, example_p_emb),
                    check_trace=False,
                )
            scorer_path = weights_dir / f"{model_name}_scorer.pt"
            scorer_traced.save(str(scorer_path))
            log_msg(f"Scorer saved: {scorer_path}")

    log_msg(f"TorchScript model saved: {weights_path}")
    log_msg(f"Training completed in {total_time:.1f}s (best epoch: {best_epoch})")

    # Return the weights directory (not the file path) to match find_weights()
    # Rust expects a directory and constructs the full path itself
    return weights_dir

