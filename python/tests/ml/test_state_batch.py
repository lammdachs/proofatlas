"""Tests for state-sampling collation.

collate_proof_batch expects pre-sampled items with u_graphs, p_graphs, u_labels keys.
"""

import pytest
import torch

try:
    from proofatlas.ml.structured import clause_to_graph, batch_graphs
    from proofatlas.ml.datasets import collate_proof_batch
except ImportError:
    pytestmark = pytest.mark.skip(reason="Depends on removed clause_to_graph API")
    clause_to_graph = None
    batch_graphs = None
    collate_proof_batch = None


def _make_clause(predicate="p", args_type="Variable", args_name="X", label=0, age=0, role="derived"):
    """Create a minimal structured clause dict."""
    return {
        "literals": [
            {
                "polarity": True,
                "atom": {
                    "predicate": predicate,
                    "args": [{"type": args_type, "name": args_name}],
                },
            }
        ],
        "label": label,
        "age": age,
        "role": role,
    }


def _make_batch_item(num_clauses=5, num_positive=2, u_indices=None, p_indices=None):
    """Create a pre-sampled batch item with u_graphs, p_graphs, u_labels."""
    clauses = []
    for i in range(num_clauses):
        clauses.append(_make_clause(
            predicate=f"p{i}",
            args_name=f"X{i}",
            label=1 if i < num_positive else 0,
            age=i,
        ))

    graphs = [clause_to_graph(c) for c in clauses]
    labels = [c["label"] for c in clauses]

    # Default U/P split: even indices are U, odd are P
    if u_indices is None:
        u_indices = list(range(0, num_clauses, 2))
    if p_indices is None:
        p_indices = list(range(1, num_clauses, 2))

    return {
        "u_graphs": [graphs[i] for i in u_indices],
        "p_graphs": [graphs[i] for i in p_indices],
        "u_labels": [labels[i] for i in u_indices],
        "problem": "test_problem",
    }


class TestCollateStateBatch:
    """Tests for collate_proof_batch."""

    def test_basic_collation(self):
        batch = [_make_batch_item(num_clauses=6, num_positive=2)]
        result = collate_proof_batch(batch)

        assert result is not None
        assert "u_node_features" in result
        assert "u_adj" in result
        assert "u_pool_matrix" in result
        assert "labels" in result
        assert "proof_ids" in result

    def test_p_graphs_included(self):
        """When processed indices exist, p_node_features should be present."""
        batch = [_make_batch_item(num_clauses=6)]
        result = collate_proof_batch(batch)

        assert result is not None
        assert "p_node_features" in result
        assert "p_adj" in result
        assert "p_pool_matrix" in result

    def test_empty_processed(self):
        """No processed clauses → no p_node_features key."""
        item = _make_batch_item(
            num_clauses=4,
            u_indices=[0, 1, 2, 3],
            p_indices=[],
        )
        result = collate_proof_batch([item])

        assert result is not None
        assert "p_node_features" not in result

    def test_proof_ids_correct(self):
        """Multiple proofs should have correct proof_id assignment."""
        batch = [
            _make_batch_item(num_clauses=4),
            _make_batch_item(num_clauses=4),
        ]
        result = collate_proof_batch(batch)

        assert result is not None
        proof_ids = result["proof_ids"]
        # Should have IDs 0 and 1
        assert 0 in proof_ids.tolist()
        assert 1 in proof_ids.tolist()

    def test_returns_none_for_empty_u(self):
        """Pre-sampled item with no U graphs should return None."""
        item = _make_batch_item(
            num_clauses=4,
            u_indices=[],
            p_indices=[0, 1, 2, 3],
        )
        result = collate_proof_batch([item])
        assert result is None
