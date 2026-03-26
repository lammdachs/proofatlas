"""Tests for dataset loading and deterministic validation sampling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from proofatlas.ml.datasets import (
    _load_and_sample_graph,
    _load_and_sample_sentence,
    ProofBatchDataset,
)


def _make_graph_trace(path: Path, num_clauses: int = 20, num_steps: int = 10):
    """Create a minimal graph trace NPZ for testing."""
    # Each clause has 3 nodes, 2 edges (within the clause's node range)
    n_nodes = num_clauses * 3
    n_edges = num_clauses * 2
    node_features = np.random.randn(n_nodes, 3).astype(np.float32)
    # Build valid edges: within each clause's 3-node subgraph
    edge_src = np.zeros(n_edges, dtype=np.int32)
    edge_dst = np.zeros(n_edges, dtype=np.int32)
    for c in range(num_clauses):
        base = c * 3
        edge_src[c * 2] = base
        edge_dst[c * 2] = base + 1
        edge_src[c * 2 + 1] = base
        edge_dst[c * 2 + 1] = base + 2
    node_offsets = np.arange(0, n_nodes + 1, 3, dtype=np.int32)
    edge_offsets = np.arange(0, n_edges + 1, 2, dtype=np.int32)
    clause_features = np.random.randn(num_clauses, 9).astype(np.float32)

    # Lifecycle: clauses transfer at step i, activate at step i+2
    transfer_step = np.array([i % num_steps for i in range(num_clauses)], dtype=np.int32)
    activate_step = np.array(
        [min(i % num_steps + 2, num_steps - 1) if i < num_clauses // 2 else -1
         for i in range(num_clauses)],
        dtype=np.int32,
    )
    simplify_step = np.full(num_clauses, -1, dtype=np.int32)
    labels = np.zeros(num_clauses, dtype=np.uint8)
    labels[:3] = 1  # First 3 clauses in proof

    np.savez(
        path,
        node_features=node_features,
        edge_src=edge_src,
        edge_dst=edge_dst,
        node_offsets=node_offsets,
        edge_offsets=edge_offsets,
        clause_features=clause_features,
        transfer_step=transfer_step,
        activate_step=activate_step,
        simplify_step=simplify_step,
        labels=labels,
        num_steps=np.array([num_steps], dtype=np.int32),
    )


class TestDeterministicSampling:
    """Validation sampling should be deterministic with fixed_seed."""

    def test_graph_fixed_seed_is_deterministic(self, tmp_path):
        trace_file = tmp_path / "TEST001-1.graph.npz"
        _make_graph_trace(trace_file)

        results = []
        for _ in range(5):
            item = _load_and_sample_graph(trace_file, need_node_emb=False, fixed_seed=42)
            assert item is not None
            results.append(len(item["u_labels"]))

        # All 5 calls should produce identical results
        assert all(r == results[0] for r in results), f"Non-deterministic: {results}"

    def test_graph_different_seeds_differ(self, tmp_path):
        trace_file = tmp_path / "TEST001-1.graph.npz"
        _make_graph_trace(trace_file, num_clauses=50, num_steps=20)

        results = set()
        for seed in range(20):
            item = _load_and_sample_graph(trace_file, need_node_emb=False, fixed_seed=seed)
            if item is not None:
                results.add(len(item["u_labels"]))

        # Different seeds should sometimes produce different step k → different U_k sizes
        assert len(results) > 1, "All seeds produced identical results"

    def test_graph_no_seed_is_random(self, tmp_path):
        trace_file = tmp_path / "TEST001-1.graph.npz"
        _make_graph_trace(trace_file, num_clauses=50, num_steps=20)

        results = set()
        for _ in range(20):
            item = _load_and_sample_graph(trace_file, need_node_emb=False, fixed_seed=None)
            if item is not None:
                results.add(len(item["u_labels"]))

        # Without fixed seed, random sampling should produce variation
        assert len(results) > 1, "Random sampling produced identical results 20 times"

    def test_dataset_fixed_seed_stable(self, tmp_path):
        """ProofBatchDataset with fixed_seed should yield same batches."""
        for i in range(5):
            _make_graph_trace(tmp_path / f"P{i:03d}.graph.npz", num_clauses=10 + i * 5)

        files = sorted(tmp_path.glob("*.graph.npz"))
        ds1 = ProofBatchDataset(files, output_type="graph", batch_size=1024 * 1024,
                                shuffle=False, need_node_emb=False, fixed_seed=42)
        ds2 = ProofBatchDataset(files, output_type="graph", batch_size=1024 * 1024,
                                shuffle=False, need_node_emb=False, fixed_seed=42)

        batches1 = list(ds1)
        batches2 = list(ds2)
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert (b1["labels"] == b2["labels"]).all()
