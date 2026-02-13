"""Data loading and batching for clause selection training.

Provides ProofBatchDataset (IterableDataset) that produces complete batch
tensors in workers, and collate functions for graph/sentence formats.

Trace files are per-problem NPZ files with lifecycle encoding:
  traces/age_weight/PUZ001-1.graph.npz    (one file per problem)
  traces/age_weight/PUZ001-1.sentence.npz

Each epoch samples a random step k per problem and reconstructs U_k/P_k
from per-clause lifecycle arrays (transfer_step, activate_step, simplify_step).
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _reconstruct_sets(npz, k: int):
    """Reconstruct U_k and P_k from lifecycle arrays at step k.

    Returns (u_idx, p_idx, selected_idx) as numpy arrays, or None if U is empty.
    """
    transfer_step = npz["transfer_step"]
    activate_step = npz["activate_step"]
    simplify_step = npz["simplify_step"]

    # U_k: transferred, not yet activated at step k, not simplified
    u_mask = (
        (transfer_step != -1) & (transfer_step <= k)
        & ((activate_step == -1) | (activate_step >= k))
        & ((simplify_step == -1) | (simplify_step > k))
    )

    # P_k: activated before step k, not simplified
    p_mask = (
        (activate_step != -1) & (activate_step < k)
        & ((simplify_step == -1) | (simplify_step > k))
    )

    u_idx = np.where(u_mask)[0]
    p_idx = np.where(p_mask)[0]

    if len(u_idx) == 0:
        return None

    # Selected clause: the one with activate_step == k
    selected_mask = activate_step == k
    selected_idx = np.where(selected_mask)[0]

    return u_idx, p_idx, selected_idx


def _load_and_sample_graph(trace_file: Path) -> Optional[Dict]:
    """Load per-problem graph NPZ, sample random step k, reconstruct U_k/P_k."""
    try:
        npz = np.load(trace_file)
    except Exception:
        return None

    try:
        num_steps = int(npz["num_steps"][0])
        if num_steps == 0:
            return None

        k = random.randint(0, num_steps - 1)
        result = _reconstruct_sets(npz, k)
        if result is None:
            return None
        u_idx, p_idx, _selected_idx = result

        node_features = npz["node_features"]
        edge_src = npz["edge_src"]
        edge_dst = npz["edge_dst"]
        node_offsets = npz["node_offsets"]
        edge_offsets = npz["edge_offsets"]
        clause_features = npz["clause_features"]
        labels = npz["labels"]

        has_node_emb = "node_embeddings" in npz
        has_sentinel = "node_sentinel_type" in npz

        def _extract_clause_graph(i):
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

            result = {
                "x": x,
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "clause_features": clause_features[i].copy(),
            }

            if has_node_emb:
                result["node_embeddings"] = npz["node_embeddings"][n_start:n_end].copy()
            if has_sentinel:
                result["node_sentinel_type"] = npz["node_sentinel_type"][n_start:n_end].copy()

            return result

        u_graphs = [_extract_clause_graph(i) for i in u_idx]
        u_labels = [int(labels[i]) for i in u_idx]
        p_graphs = [_extract_clause_graph(i) for i in p_idx]

        problem = trace_file.stem.rsplit(".", 1)[0]  # Remove .graph suffix

        return {
            "u_graphs": u_graphs,
            "p_graphs": p_graphs,
            "u_labels": u_labels,
            "problem": problem,
        }
    finally:
        npz.close()


def _load_and_sample_sentence(trace_file: Path) -> Optional[Dict]:
    """Load per-problem sentence NPZ, sample step k, gather embeddings by index."""
    try:
        npz = np.load(trace_file)
    except Exception:
        return None

    try:
        num_steps = int(npz["num_steps"][0])
        if num_steps == 0:
            return None

        k = random.randint(0, num_steps - 1)
        result = _reconstruct_sets(npz, k)
        if result is None:
            return None
        u_idx, p_idx, _selected_idx = result

        clause_features = npz["clause_features"]
        labels = npz["labels"]

        has_embeddings = "clause_embeddings" in npz

        u_cf = clause_features[u_idx].copy()
        u_labels = [int(labels[i]) for i in u_idx]
        p_cf = clause_features[p_idx].copy() if len(p_idx) > 0 else None

        problem = trace_file.stem.rsplit(".", 1)[0]

        result_dict = {
            "u_clause_features": u_cf,
            "u_labels": u_labels,
            "problem": problem,
        }

        if p_cf is not None:
            result_dict["p_clause_features"] = p_cf

        if has_embeddings:
            embs = npz["clause_embeddings"]
            result_dict["u_embeddings"] = embs[u_idx].copy()
            if len(p_idx) > 0:
                result_dict["p_embeddings"] = embs[p_idx].copy()

        return result_dict
    finally:
        npz.close()


class ProofBatchDataset(IterableDataset):
    """IterableDataset that loads per-problem NPZ traces with lifecycle step sampling.

    Each worker loads per-problem .npz files, samples a random step per problem,
    reconstructs U_k/P_k, buffers until clause count exceeds max_clauses,
    then collates and yields a complete batch.

    Each epoch samples new random steps per problem (natural augmentation).

    Args:
        files: List of per-problem trace file paths (STEM.graph.npz or STEM.sentence.npz)
        output_type: "graph" for GNN models, "sentence" for sentence models
        max_clauses: Maximum total U clauses per batch
        shuffle: Whether to shuffle files each epoch
    """

    def __init__(
        self,
        files: List[Path],
        output_type: str = "graph",
        max_clauses: int = 8192,
        shuffle: bool = True,
    ):
        self.files = list(files)
        self.output_type = output_type
        self.max_clauses = max_clauses
        self.shuffle = shuffle

        if not self.files:
            raise ValueError("No trace files provided")

    def _shard_files(self, worker_info, files):
        """Shard files across DataLoader workers."""
        if worker_info is None:
            return files
        per_worker = len(files) // worker_info.num_workers
        remainder = len(files) % worker_info.num_workers
        start = worker_info.id * per_worker + min(worker_info.id, remainder)
        end = start + per_worker + (1 if worker_info.id < remainder else 0)
        return files[start:end]

    def _load_and_sample(self, f: Path) -> Optional[Dict]:
        """Load and sample a single trace file."""
        if self.output_type == "sentence":
            return _load_and_sample_sentence(f)
        else:
            return _load_and_sample_graph(f)

    def _collate(self, items: List[Dict]) -> Optional[Dict]:
        """Collate pre-sampled items into a batch."""
        if self.output_type == "sentence":
            return collate_sentence_batch(items)
        else:
            return collate_proof_batch(items)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)
        files = self._shard_files(worker_info, files)

        buffer = []
        buffer_clauses = 0

        for f in files:
            item = self._load_and_sample(f)
            if item is None:
                continue

            if self.output_type == "sentence":
                n_u = len(item["u_labels"])
            else:
                n_u = len(item.get("u_graphs", []))

            if buffer_clauses + n_u > self.max_clauses:
                if buffer:
                    batch = self._collate(buffer)
                    if batch is not None:
                        yield batch
                    buffer, buffer_clauses = [], 0
                if n_u > self.max_clauses:
                    continue  # drop oversized item

            buffer.append(item)
            buffer_clauses += n_u

        if buffer:
            batch = self._collate(buffer)
            if batch is not None:
                yield batch

    @property
    def num_problems(self) -> int:
        """Number of distinct problems (same as num_files for per-problem format)."""
        return len(self.files)


def collate_sentence_batch(batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """Collate pre-sampled sentence items into a batch.

    Concatenates pre-computed embeddings (no padding needed) and clause features.
    """
    all_u_embeddings = []
    all_u_cf = []
    all_u_labels = []
    all_proof_ids = []
    all_p_embeddings = []
    all_p_cf = []

    for proof_idx, item in enumerate(batch):
        if item is None:
            continue

        u_labels = item["u_labels"]
        u_cf = item["u_clause_features"]
        all_u_cf.append(u_cf)
        all_u_labels.extend(u_labels)
        all_proof_ids.extend([proof_idx] * len(u_labels))

        if "u_embeddings" in item:
            all_u_embeddings.append(item["u_embeddings"])

        if "p_clause_features" in item:
            all_p_cf.append(item["p_clause_features"])
        if "p_embeddings" in item:
            all_p_embeddings.append(item["p_embeddings"])

    if not all_u_labels:
        return None

    result = {
        "u_clause_features": torch.tensor(np.concatenate(all_u_cf), dtype=torch.float32),
        "labels": torch.tensor(all_u_labels, dtype=torch.float32),
        "proof_ids": torch.tensor(all_proof_ids, dtype=torch.long),
    }

    if all_u_embeddings:
        result["u_embeddings"] = torch.tensor(
            np.concatenate(all_u_embeddings), dtype=torch.float32
        )

    if all_p_cf:
        result["p_clause_features"] = torch.tensor(
            np.concatenate(all_p_cf), dtype=torch.float32
        )

    if all_p_embeddings:
        result["p_embeddings"] = torch.tensor(
            np.concatenate(all_p_embeddings), dtype=torch.float32
        )

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
    if "node_names" in u_batched:
        result["u_node_names"] = u_batched["node_names"]
    if "node_embeddings" in u_batched:
        result["u_node_embeddings"] = u_batched["node_embeddings"]
    if "node_sentinel_type" in u_batched:
        result["u_node_sentinel_type"] = u_batched["node_sentinel_type"]

    # Batch P graphs (no labels needed)
    if all_p_graphs:
        p_batched = batch_graphs(all_p_graphs)
        result["p_node_features"] = p_batched["x"]
        result["p_adj"] = p_batched["adj"]
        result["p_pool_matrix"] = p_batched["pool_matrix"]
        result["p_clause_features"] = p_batched.get("clause_features")
        if "node_names" in p_batched:
            result["p_node_names"] = p_batched["node_names"]
        if "node_embeddings" in p_batched:
            result["p_node_embeddings"] = p_batched["node_embeddings"]
        if "node_sentinel_type" in p_batched:
            result["p_node_sentinel_type"] = p_batched["node_sentinel_type"]

    return result


def scan_trace_files(files: List[Path], max_workers: int = 4) -> List[Path]:
    """Validate trace files in parallel and return valid ones."""
    from concurrent.futures import ThreadPoolExecutor

    def _check(f):
        try:
            with np.load(f) as data:
                if "labels" not in data or "num_steps" not in data:
                    return None
                if int(data["num_steps"][0]) == 0:
                    return None
                return f
        except Exception:
            return None

    if max_workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_check, files))
    else:
        results = [_check(f) for f in files]

    return [f for f in results if f is not None]
