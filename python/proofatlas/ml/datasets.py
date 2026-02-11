"""Data loading and batching for clause selection training.

Provides ProofBatchDataset (IterableDataset) that produces complete batch
tensors in workers, and collate functions for graph/tokenized formats.

Each CPU worker does all the work: load trace, sample state, collate per-proof,
measure bytes, accumulate into batches, and yield complete batch tensors.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import IterableDataset

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


def _scan_trace(trace_file: Path) -> Optional[int]:
    """Quick scan to validate a trace and return its clause count.

    Returns clause count if valid, None if trace should be skipped.
    """
    try:
        trace = _load_json(trace_file)
    except Exception:
        return None

    if not trace.get("proof_found") or not trace.get("clauses"):
        return None
    if not trace.get("selection_states"):
        return None

    return len(trace["clauses"])


def _batch_bytes(batch: dict) -> int:
    """Measure actual byte size of collated batch tensors.

    Handles both dense and sparse tensors.
    """
    total = 0
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            if v.is_sparse:
                total += v._indices().nelement() * v._indices().element_size()
                total += v._values().nelement() * v._values().element_size()
            else:
                total += v.nelement() * v.element_size()
    return total


def _load_and_presample_graph(trace_file: Path) -> Optional[Dict]:
    """Load trace, sample a random selection state, return pre-sampled graph data.

    Returns U/P clause subsets already split and ready for collation.
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
    states = trace["selection_states"]

    # Sample a random selection state
    state = random.choice(states)
    u_indices = state["unprocessed"]
    p_indices = state["processed"]

    # Build graphs and labels for U and P
    u_graphs = []
    u_labels = []
    for idx in u_indices:
        if idx < len(clauses):
            u_graphs.append(clause_to_graph(clauses[idx]))
            u_labels.append(labels[idx])

    p_graphs = []
    for idx in p_indices:
        if idx < len(clauses):
            p_graphs.append(clause_to_graph(clauses[idx]))

    if not u_graphs:
        return None

    return {
        "u_graphs": u_graphs,
        "p_graphs": p_graphs,
        "u_labels": u_labels,
        "problem": trace_file.stem,
    }


def _load_and_presample_tokenized(trace_file: Path) -> Optional[Dict]:
    """Load trace, sample a random selection state, return pre-sampled tokenized data."""
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
    states = trace["selection_states"]

    # Sample a random selection state
    state = random.choice(states)
    u_indices = state["unprocessed"]
    p_indices = state["processed"]

    # Build strings and labels for U and P
    u_strings = []
    u_labels = []
    for idx in u_indices:
        if idx < len(clauses):
            u_strings.append(clause_to_string(clauses[idx]))
            u_labels.append(labels[idx])

    p_strings = []
    for idx in p_indices:
        if idx < len(clauses):
            p_strings.append(clause_to_string(clauses[idx]))

    if not u_strings:
        return None

    # Pre-tokenize
    tokenizer = _get_tokenizer()

    u_encoded = tokenizer(u_strings, padding=True, truncation=True, return_tensors="pt")
    result = {
        "u_input_ids": u_encoded["input_ids"],
        "u_attention_mask": u_encoded["attention_mask"],
        "u_labels": u_labels,
        "problem": trace_file.stem,
    }

    if p_strings:
        p_encoded = tokenizer(p_strings, padding=True, truncation=True, return_tensors="pt")
        result["p_input_ids"] = p_encoded["input_ids"]
        result["p_attention_mask"] = p_encoded["attention_mask"]

    return result


# Global tokenizer for pre-tokenization (loaded lazily)
_tokenizer = None


def _get_tokenizer():
    """Get or load the tokenizer for pre-tokenization."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


class ProofBatchDataset(IterableDataset):
    """IterableDataset that produces complete batch tensors with exact byte measurement.

    Each worker loads traces, samples selection states, collates per-proof,
    measures actual tensor bytes, and yields complete batches. The main process
    just receives ready-to-use batch dicts.

    Args:
        files: List of trace file paths
        output_type: "graph" for GNN models, "tokenized" for sentence models
        max_batch_bytes: Maximum collated tensor bytes per batch
        shuffle: Whether to shuffle files each epoch
    """

    def __init__(
        self,
        files: List[Path],
        output_type: str = "graph",
        max_batch_bytes: int = 16 * 1024 * 1024,  # 16 MB default
        shuffle: bool = True,
    ):
        self.files = list(files)
        self.output_type = output_type
        self.max_batch_bytes = max_batch_bytes
        self.shuffle = shuffle

        if not self.files:
            raise ValueError("No trace files provided")

    def _shard_files(self, worker_info):
        """Shard files across DataLoader workers."""
        if worker_info is None:
            return list(self.files)
        per_worker = len(self.files) // worker_info.num_workers
        remainder = len(self.files) % worker_info.num_workers
        start = worker_info.id * per_worker + min(worker_info.id, remainder)
        end = start + per_worker + (1 if worker_info.id < remainder else 0)
        return list(self.files[start:end])

    def _load_and_presample(self, f: Path) -> Optional[Dict]:
        """Load and presample a single trace file."""
        if self.output_type == "tokenized":
            return _load_and_presample_tokenized(f)
        else:
            return _load_and_presample_graph(f)

    def _collate(self, items: List[Dict]) -> Optional[Dict]:
        """Collate pre-sampled items into a batch."""
        if self.output_type == "tokenized":
            return collate_tokenized_batch(items)
        else:
            return collate_proof_batch(items)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self._shard_files(worker_info)
        if self.shuffle:
            random.shuffle(files)

        buffer = []
        buffer_bytes = 0

        for f in files:
            item = self._load_and_presample(f)
            if item is None:
                continue

            # Collate single proof to measure actual tensor bytes
            single = self._collate([item])
            if single is None:
                continue
            proof_bytes = _batch_bytes(single)

            if buffer_bytes + proof_bytes > self.max_batch_bytes and buffer:
                batch = self._collate(buffer)
                if batch is not None:
                    yield batch
                buffer, buffer_bytes = [], 0

            buffer.append(item)
            buffer_bytes += proof_bytes

        if buffer:
            batch = self._collate(buffer)
            if batch is not None:
                yield batch

    @property
    def num_files(self) -> int:
        """Number of trace files in this dataset."""
        return len(self.files)


def scan_trace_files(files: List[Path], max_workers: int = 4) -> List[Path]:
    """Validate trace files in parallel and return valid ones.

    Args:
        files: List of trace file paths to validate
        max_workers: Number of threads for parallel I/O

    Returns:
        List of valid trace file paths
    """
    if max_workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda f: (f, _scan_trace(f)), files))
    else:
        results = [(f, _scan_trace(f)) for f in files]

    return [f for f, n in results if n is not None]


def collate_tokenized_batch(batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """Collate pre-sampled tokenized items into a batch.

    Items are already split into U/P sets. This just concatenates across proofs
    and assigns proof_ids.
    """
    from torch.nn.utils.rnn import pad_sequence

    all_u_input_ids = []
    all_u_attention_mask = []
    all_u_labels = []
    all_p_input_ids = []
    all_p_attention_mask = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        if item is None:
            continue

        u_ids = item["u_input_ids"]
        u_mask = item["u_attention_mask"]
        u_labels = item["u_labels"]

        for i in range(len(u_labels)):
            all_u_input_ids.append(u_ids[i])
            all_u_attention_mask.append(u_mask[i])
            all_u_labels.append(u_labels[i])
            all_proof_ids.append(proof_idx)

        if "p_input_ids" in item:
            p_ids = item["p_input_ids"]
            p_mask = item["p_attention_mask"]
            for i in range(p_ids.size(0)):
                all_p_input_ids.append(p_ids[i])
                all_p_attention_mask.append(p_mask[i])

    if not all_u_input_ids:
        return None

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


def collate_proof_batch(batch: List[Dict]) -> Optional[Dict[str, Any]]:
    """Collate pre-sampled graph items into a batch.

    Items are already split into U/P graph sets. This just concatenates across
    proofs and assigns proof_ids.
    """
    from .structured import batch_graphs

    all_u_graphs = []
    all_p_graphs = []
    all_u_labels = []
    all_proof_ids = []

    for proof_idx, item in enumerate(batch):
        if item is None:
            continue

        u_graphs = item["u_graphs"]
        u_labels = item["u_labels"]
        p_graphs = item["p_graphs"]

        for i, g in enumerate(u_graphs):
            all_u_graphs.append(g)
            all_u_labels.append(u_labels[i])
            all_proof_ids.append(proof_idx)

        for g in p_graphs:
            all_p_graphs.append(g)

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


# Keep old classes for backward compatibility
class DynamicBatchSampler(torch.utils.data.Sampler):
    """Sampler that creates batches with approximately equal total size.

    Deprecated: use ProofBatchDataset instead.
    """

    def __init__(self, dataset, max_clauses: int = 8192, shuffle: bool = True):
        self.dataset = dataset
        self.max_clauses = max_clauses
        self.shuffle = shuffle
        self.sizes = dataset.sizes

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        batch_size = 0

        for idx in indices:
            size = self.sizes[idx]

            if size > self.max_clauses:
                if batch:
                    yield batch
                    batch = []
                    batch_size = 0
                yield [idx]
                continue

            if batch_size + size > self.max_clauses and batch:
                yield batch
                batch = []
                batch_size = 0

            batch.append(idx)
            batch_size += size

        if batch:
            yield batch

    def __len__(self):
        total = sum(self.sizes)
        return max(1, total // self.max_clauses)


class ProofDataset(torch.utils.data.Dataset):
    """Dataset that lazily loads structured JSON traces on access.

    Deprecated: use ProofBatchDataset instead.
    """

    def __init__(
        self,
        trace_dir: Path,
        output_type: str = "graph",
        problem_names: Optional[set] = None,
        trace_files: Optional[List[Path]] = None,
        n_workers: int = None,
    ):
        self.output_type = output_type

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

        self.files = []
        self.sizes = []
        for f in files:
            n_clauses = _scan_trace(f)
            if n_clauses is not None:
                self.files.append(f)
                self.sizes.append(n_clauses)

        if not self.files:
            raise ValueError("No valid traces found (all filtered or empty)")

        self.items = self

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.output_type == "tokenized":
            return _load_trace_tokenized(self.files[idx])
        else:
            return _load_trace_graph(self.files[idx])


def _load_trace_graph(trace_file: Path) -> Optional[Dict]:
    """Load trace file and convert to graphs with selection states.

    Deprecated: used by ProofDataset. New code uses _load_and_presample_graph.
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


def _load_trace_tokenized(trace_file: Path) -> Optional[Dict]:
    """Load trace file and pre-tokenize strings with selection states.

    Deprecated: used by ProofDataset. New code uses _load_and_presample_tokenized.
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
