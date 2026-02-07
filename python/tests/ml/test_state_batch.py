"""Tests for state-sampling collation."""

import pytest
import torch

from proofatlas.ml.structured import clause_to_graph, batch_graphs
from proofatlas.ml.training import collate_state_batch


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


def _make_batch_item(num_clauses=5, num_positive=2, num_states=2):
    """Create a batch item with synthetic graphs and selection_states."""
    clauses = []
    for i in range(num_clauses):
        clauses.append(_make_clause(
            predicate=f"p{i}",
            args_name=f"X{i}",
            label=1 if i < num_positive else 0,
            age=i,
        ))

    graphs = [clause_to_graph(c, max_age=num_clauses) for c in clauses]
    labels = [c["label"] for c in clauses]

    # Create selection states with U and P sets
    states = []
    for s in range(num_states):
        # Split clauses between U and P
        u_indices = list(range(s, num_clauses, 2))  # odd or even indices
        p_indices = list(range(1 - s, num_clauses, 2))
        states.append({
            "selected": u_indices[0] if u_indices else 0,
            "unprocessed": u_indices,
            "processed": p_indices,
        })

    return {
        "graphs": graphs,
        "labels": labels,
        "selection_states": states,
        "problem": "test_problem",
    }


class TestCollateStateBatch:
    """Tests for collate_state_batch."""

    def test_basic_collation(self):
        batch = [_make_batch_item(num_clauses=6, num_positive=2)]
        result = collate_state_batch(batch)

        assert result is not None
        assert "u_node_features" in result
        assert "u_adj" in result
        assert "u_pool_matrix" in result
        assert "labels" in result
        assert "proof_ids" in result
        assert result["is_state_batch"] is True

    def test_p_graphs_included(self):
        """When processed indices exist, p_node_features should be present."""
        batch = [_make_batch_item(num_clauses=6)]
        result = collate_state_batch(batch)

        assert result is not None
        assert "p_node_features" in result
        assert "p_adj" in result
        assert "p_pool_matrix" in result

    def test_empty_processed(self):
        """No processed clauses → no p_node_features key."""
        item = _make_batch_item(num_clauses=4)
        # Override states to have empty processed sets
        item["selection_states"] = [{
            "selected": 0,
            "unprocessed": [0, 1, 2, 3],
            "processed": [],
        }]
        result = collate_state_batch([item])

        assert result is not None
        assert "p_node_features" not in result

    def test_proof_ids_correct(self):
        """Multiple proofs should have correct proof_id assignment."""
        batch = [
            _make_batch_item(num_clauses=4),
            _make_batch_item(num_clauses=4),
        ]
        result = collate_state_batch(batch)

        assert result is not None
        proof_ids = result["proof_ids"]
        # Should have IDs 0 and 1
        assert 0 in proof_ids.tolist()
        assert 1 in proof_ids.tolist()

    def test_returns_none_for_empty_u(self):
        """State with no valid U should return None."""
        item = _make_batch_item(num_clauses=4)
        # Override to have empty unprocessed set
        item["selection_states"] = [{
            "selected": 0,
            "unprocessed": [],
            "processed": [0, 1, 2, 3],
        }]
        result = collate_state_batch([item])
        assert result is None
