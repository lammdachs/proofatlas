"""Data loading and batching for clause selection training.

Provides ProofBatchDataset (IterableDataset) that produces complete batch
tensors in workers, and collate functions for graph/tokenized formats.

Trace files are .npz (numpy compressed archives) produced by Rust's
extract_tensor_trace(). Each worker does: load .npz, sample state, slice
arrays, collate into batch, yield.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _scan_trace(trace_file: Path) -> Optional[int]:
    """Quick scan to validate a trace and return its clause count.

    Returns clause count if valid, None if trace should be skipped.
    """
    try:
        with np.load(trace_file, allow_pickle=True) as data:
            if not bool(data["proof_found"][0]):
                return None
            if "state_selected" not in data or len(data["state_selected"]) == 0:
                return None
            if "labels" not in data:
                return None
            return len(data["labels"])
    except Exception:
        return None


def _batch_bytes(batch: dict) -> int:
    """Measure actual byte size of collated batch tensors."""
    total = 0
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            if v.is_sparse:
                total += v._indices().nelement() * v._indices().element_size()
                total += v._values().nelement() * v._values().element_size()
            else:
                total += v.nelement() * v.element_size()
    return total


def _estimate_graph_bytes(item: dict) -> int:
    """Estimate collated tensor bytes from pre-sampled graph data (numpy arrays)."""
    total = 0
    for graphs in (item.get("u_graphs", []), item.get("p_graphs", [])):
        for g in graphs:
            num_nodes = g["num_nodes"]
            num_edges = g["num_edges"]
            # x: [num_nodes, 3] float32
            total += num_nodes * 3 * 4
            # adj sparse: indices [2, num_nodes + num_edges] int64 + values float32
            total += (num_nodes + num_edges) * 2 * 8  # indices
            total += (num_nodes + num_edges) * 4       # values
            # pool_matrix sparse: indices [2, num_nodes] int64 + values float32
            total += num_nodes * 2 * 8
            total += num_nodes * 4
            # clause_features: [CLAUSE_FEATURE_DIM] float32
            total += 9 * 4
    # labels: float32 per u_graph + proof_ids: int64 per u_graph
    n_u = len(item.get("u_graphs", []))
    total += n_u * 4 + n_u * 8
    return total


def _estimate_tokenized_bytes(item: dict) -> int:
    """Estimate collated tensor bytes from pre-sampled tokenized data."""
    total = 0
    for key in ("u_input_ids", "u_attention_mask", "p_input_ids", "p_attention_mask"):
        t = item.get(key)
        if t is not None and isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    n_u = len(item.get("u_labels", []))
    total += n_u * 4 + n_u * 8  # labels float32 + proof_ids int64
    return total


def _load_and_presample_graph(trace_file: Path) -> Optional[Dict]:
    """Load .npz trace, sample a random selection state, return pre-sampled graph data.

    Returns U/P clause subsets with per-clause graph dicts ready for batch_graphs().
    """
    try:
        data = dict(np.load(trace_file, allow_pickle=True))
    except Exception:
        return None

    if not bool(data["proof_found"][0]):
        return None
    if len(data.get("state_selected", [])) == 0:
        return None

    node_features = data["node_features"]       # [total_nodes, 3]
    edge_src = data["edge_src"]                  # [total_edges]
    edge_dst = data["edge_dst"]                  # [total_edges]
    node_offsets = data["node_offsets"]           # [num_clauses + 1]
    edge_offsets = data["edge_offsets"]           # [num_clauses + 1]
    clause_features = data["clause_features"]    # [num_clauses, 9]
    labels = data["labels"]                      # [num_clauses]

    state_selected = data["state_selected"]
    state_u_offsets = data["state_u_offsets"]
    state_u_indices = data["state_u_indices"]
    state_p_offsets = data["state_p_offsets"]
    state_p_indices = data["state_p_indices"]

    num_states = len(state_selected)
    si = random.randrange(num_states)

    # Get U and P clause indices for this state
    u_start, u_end = int(state_u_offsets[si]), int(state_u_offsets[si + 1])
    p_start, p_end = int(state_p_offsets[si]), int(state_p_offsets[si + 1])
    u_indices = state_u_indices[u_start:u_end].astype(np.int64)
    p_indices = state_p_indices[p_start:p_end].astype(np.int64)

    if len(u_indices) == 0:
        return None

    def _extract_clause_graph(idx):
        """Extract per-clause graph dict from flat arrays."""
        i = int(idx)
        n_start, n_end = int(node_offsets[i]), int(node_offsets[i + 1])
        e_start, e_end = int(edge_offsets[i]), int(edge_offsets[i + 1])

        x = node_features[n_start:n_end].copy()
        num_nodes = n_end - n_start
        num_edges = e_end - e_start

        if num_edges > 0:
            src = edge_src[e_start:e_end].astype(np.int64) - n_start
            dst = edge_dst[e_start:e_end].astype(np.int64) - n_start
            edge_index = np.stack([src, dst])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "clause_features": clause_features[i].copy(),
        }

    u_graphs = [_extract_clause_graph(idx) for idx in u_indices]
    u_labels = [int(labels[int(idx)]) for idx in u_indices]

    p_graphs = [_extract_clause_graph(idx) for idx in p_indices]

    return {
        "u_graphs": u_graphs,
        "p_graphs": p_graphs,
        "u_labels": u_labels,
        "problem": trace_file.name.rsplit(".", 2)[0],
    }


def _load_and_presample_tokenized(trace_file: Path) -> Optional[Dict]:
    """Load .npz trace, sample a random selection state, return pre-sampled tokenized data."""
    try:
        data = dict(np.load(trace_file, allow_pickle=True))
    except Exception:
        return None

    if not bool(data["proof_found"][0]):
        return None
    if len(data.get("state_selected", [])) == 0:
        return None

    clause_strings = data["clause_strings"]    # [num_clauses] object array
    labels = data["labels"]                    # [num_clauses]

    state_selected = data["state_selected"]
    state_u_offsets = data["state_u_offsets"]
    state_u_indices = data["state_u_indices"]
    state_p_offsets = data["state_p_offsets"]
    state_p_indices = data["state_p_indices"]

    num_states = len(state_selected)
    si = random.randrange(num_states)

    u_start, u_end = int(state_u_offsets[si]), int(state_u_offsets[si + 1])
    p_start, p_end = int(state_p_offsets[si]), int(state_p_offsets[si + 1])
    u_indices = state_u_indices[u_start:u_end].astype(np.int64)
    p_indices = state_p_indices[p_start:p_end].astype(np.int64)

    if len(u_indices) == 0:
        return None

    u_strings = [str(clause_strings[int(idx)]) for idx in u_indices]
    u_labels = [int(labels[int(idx)]) for idx in u_indices]
    p_strings = [str(clause_strings[int(idx)]) for idx in p_indices]

    # Pre-tokenize
    tokenizer = _get_tokenizer()

    u_encoded = tokenizer(u_strings, padding=True, truncation=True, return_tensors="pt")
    result = {
        "u_input_ids": u_encoded["input_ids"],
        "u_attention_mask": u_encoded["attention_mask"],
        "u_labels": u_labels,
        "problem": trace_file.name.rsplit(".", 2)[0],
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

    Each worker loads .npz traces, samples selection states, collates per-proof,
    measures actual tensor bytes, and yields complete batches.

    Args:
        files: List of trace file paths (.graph.npz or .sentence.npz)
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

            # Estimate tensor bytes without creating throwaway sparse tensors
            if self.output_type == "tokenized":
                proof_bytes = _estimate_tokenized_bytes(item)
            else:
                proof_bytes = _estimate_graph_bytes(item)

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
    """Collate pre-sampled tokenized items into a batch."""
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
    """Collate pre-sampled graph items into a batch."""
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
