"""Data loading and batching for clause selection training.

Provides ProofDataset, DynamicBatchSampler, and collate functions for
both graph (GNN) and tokenized (sentence) model formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

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
            # Cap at 2 to avoid OOM: fork duplicates libtorch address space
            n_workers = min(os.cpu_count() or 4, 2)

        if n_workers > 1 and len(files) > 10:
            from multiprocessing import get_context
            # Use spawn to avoid duplicating libtorch memory via fork
            ctx = get_context("spawn")
            init_fn = _pool_init_tokenizer if output_type == "tokenized" else _pool_init
            with ctx.Pool(n_workers, initializer=init_fn) as pool:
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
