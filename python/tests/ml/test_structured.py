"""Tests for structured clause format conversion."""

import pytest
import json
from pathlib import Path

# Skip if proofatlas not available
pytest.importorskip("proofatlas")


@pytest.fixture
def sample_clause():
    """A sample structured clause."""
    return {
        "literals": [
            {
                "polarity": True,
                "atom": {
                    "predicate": "=",
                    "args": [
                        {
                            "type": "Function",
                            "name": "mult",
                            "args": [
                                {"type": "Variable", "name": "X"},
                                {"type": "Variable", "name": "Y"},
                            ],
                        },
                        {"type": "Variable", "name": "Z"},
                    ],
                },
            }
        ],
        "label": 1,
        "age": 0,
        "role": "axiom",
    }


@pytest.fixture
def sample_multi_literal_clause():
    """A clause with multiple literals."""
    return {
        "literals": [
            {
                "polarity": True,
                "atom": {
                    "predicate": "p",
                    "args": [{"type": "Variable", "name": "X"}],
                },
            },
            {
                "polarity": False,
                "atom": {
                    "predicate": "q",
                    "args": [
                        {"type": "Variable", "name": "X"},
                        {"type": "Constant", "name": "c"},
                    ],
                },
            },
        ],
        "label": 0,
        "age": 5,
        "role": "derived",
    }


class TestClauseToString:
    """Test structured clause to string conversion."""

    def test_equality_clause(self, sample_clause):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string(sample_clause)
        assert result == "mult(X, Y) = Z"

    def test_multi_literal_clause(self, sample_multi_literal_clause):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string(sample_multi_literal_clause)
        assert result == "p(X) | ~q(X, c)"

    def test_empty_clause(self):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string({"literals": []})
        assert result == "[]"


class TestClauseToGraph:
    """Test structured clause to graph tensor conversion."""

    def test_graph_structure(self, sample_clause):
        from proofatlas.ml.structured import clause_to_graph
        import numpy as np

        result = clause_to_graph(sample_clause)

        # Check numpy array types (torch tensors created during batching)
        assert isinstance(result["edge_index"], np.ndarray)
        assert isinstance(result["x"], np.ndarray)
        assert isinstance(result["node_types"], np.ndarray)
        assert isinstance(result["clause_features"], np.ndarray)

        # Check shapes
        assert result["edge_index"].shape[0] == 2
        assert result["x"].shape[0] == result["num_nodes"]
        assert result["node_types"].shape[0] == result["num_nodes"]

        # Check node feature dimension (3 features: type, arity, arg_pos)
        assert result["x"].shape[1] == 3

        # Check clause feature dimension (3 features: age, role, size)
        assert result["clause_features"].shape[0] == 3

    def test_node_types(self, sample_clause):
        from proofatlas.ml.structured import clause_to_graph

        result = clause_to_graph(sample_clause)
        node_types = result["node_types"].tolist()

        # Should have: clause(0), literal(1), predicate(2), function(3), variables(4)
        assert 0 in node_types  # clause
        assert 1 in node_types  # literal
        assert 2 in node_types  # predicate
        assert 3 in node_types  # function (mult)
        assert 4 in node_types  # variable (X, Y, Z)

    def test_graph_edges(self, sample_clause):
        from proofatlas.ml.structured import clause_to_graph

        result = clause_to_graph(sample_clause)

        # Edges should be non-empty
        assert result["num_edges"] > 0


class TestClausesConversion:
    """Test batch conversion functions."""

    def test_clauses_to_strings(self, sample_clause, sample_multi_literal_clause):
        from proofatlas.ml.structured import clauses_to_strings

        clauses = [sample_clause, sample_multi_literal_clause]
        results = clauses_to_strings(clauses)

        assert len(results) == 2
        assert results[0] == "mult(X, Y) = Z"
        assert results[1] == "p(X) | ~q(X, c)"

    def test_clauses_to_graphs(self, sample_clause, sample_multi_literal_clause):
        from proofatlas.ml.structured import clauses_to_graphs

        clauses = [sample_clause, sample_multi_literal_clause]
        results = clauses_to_graphs(clauses)

        assert len(results) == 2
        assert all("edge_index" in r for r in results)
        assert all("x" in r for r in results)


class TestBatchGraphs:
    """Test graph batching."""

    def test_batch_single_graph(self, sample_clause):
        from proofatlas.ml.structured import clause_to_graph, batch_graphs

        graph = clause_to_graph(sample_clause)
        batched = batch_graphs([graph])

        assert batched["num_graphs"] == 1
        assert batched["batch"].shape[0] == graph["num_nodes"]

    def test_batch_multiple_graphs(self, sample_clause, sample_multi_literal_clause):
        from proofatlas.ml.structured import clause_to_graph, batch_graphs
        import torch

        g1 = clause_to_graph(sample_clause)
        g2 = clause_to_graph(sample_multi_literal_clause)

        batched = batch_graphs([g1, g2], labels=[1, 0])

        assert batched["num_graphs"] == 2
        assert batched["batch"].shape[0] == g1["num_nodes"] + g2["num_nodes"]
        assert torch.equal(batched["y"], torch.tensor([1.0, 0.0]))


class TestRustIntegration:
    """Test integration with Rust extract_structured_trace."""

    def test_extract_structured_trace(self):
        """Test that extract_structured_trace produces valid JSON."""
        from proofatlas import ProofState

        # Create a simple proof
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(ax1, axiom, p(X) | q(X)).
            cnf(ax2, axiom, ~p(a)).
            cnf(ax3, axiom, ~q(a)).
        """)

        # Run saturation to find proof
        proof_found, status, _, _ = state.run_saturation(timeout=10.0, max_iterations=1000)

        if proof_found:
            # Extract structured trace
            trace_json = state.extract_structured_trace(1.0)
            trace = json.loads(trace_json)

            # Check structure
            assert "proof_found" in trace
            assert "time_seconds" in trace
            assert "clauses" in trace
            assert trace["proof_found"] is True

            # Check clauses have required fields
            for clause in trace["clauses"]:
                assert "literals" in clause
                assert "label" in clause
                assert "age" in clause
                assert "role" in clause

    def test_roundtrip_conversion(self):
        """Test that structured JSON can be converted back to graphs/strings."""
        from proofatlas import ProofState
        from proofatlas.ml.structured import clause_to_string, clause_to_graph

        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(mult_assoc, axiom, mult(mult(X, Y), Z) = mult(X, mult(Y, Z))).
            cnf(identity, axiom, mult(X, e) = X).
        """)

        proof_found, status, _, _ = state.run_saturation(timeout=10.0, max_iterations=1000)

        if proof_found:
            trace_json = state.extract_structured_trace(1.0)
            trace = json.loads(trace_json)

            # Convert each clause
            for clause in trace["clauses"]:
                # To string
                s = clause_to_string(clause)
                assert isinstance(s, str)
                assert len(s) > 0

                # To graph
                g = clause_to_graph(clause)
                assert g["num_nodes"] > 0
                assert g["num_edges"] >= 0


class TestPoolValues:
    """Test pool matrix normalization."""

    def test_pool_values_sqrt_normalization(self, sample_clause, sample_multi_literal_clause):
        """Verify pool_values == 1/sqrt(count) (not 1/count)."""
        from proofatlas.ml.structured import clause_to_graph, batch_graphs
        import torch

        g1 = clause_to_graph(sample_clause)
        g2 = clause_to_graph(sample_multi_literal_clause)

        batched = batch_graphs([g1, g2])
        pool = batched["pool_matrix"]

        # Pool matrix is sparse, coalesce and check values
        pool = pool.coalesce()
        values = pool.values()
        indices = pool.indices()

        # For each clause (row), the values should be 1/sqrt(num_nodes_in_clause)
        batch = batched["batch"]
        for clause_idx in range(2):
            mask = indices[0] == clause_idx
            if mask.any():
                clause_values = values[mask]
                num_nodes = (batch == clause_idx).sum().item()
                expected = 1.0 / (num_nodes ** 0.5)
                assert torch.allclose(
                    clause_values,
                    torch.full_like(clause_values, expected),
                    atol=1e-5,
                ), f"Pool values for clause {clause_idx}: expected {expected}, got {clause_values}"

    def test_node_names_in_graph(self, sample_clause):
        """clause_to_graph should include node_names list."""
        from proofatlas.ml.structured import clause_to_graph

        result = clause_to_graph(sample_clause)
        assert "node_names" in result
        assert isinstance(result["node_names"], list)
        assert len(result["node_names"]) == result["num_nodes"]
        # Should contain known names
        assert "CLAUSE" in result["node_names"]
        assert "LIT" in result["node_names"]

    def test_batch_preserves_node_names(self, sample_clause, sample_multi_literal_clause):
        """batch_graphs doesn't strip node_names (they stay on individual graphs)."""
        from proofatlas.ml.structured import clause_to_graph

        g1 = clause_to_graph(sample_clause)
        g2 = clause_to_graph(sample_multi_literal_clause)

        # Verify individual graph node_names
        assert len(g1["node_names"]) == g1["num_nodes"]
        assert len(g2["node_names"]) == g2["num_nodes"]

        # Combined list should have all names
        combined = g1["node_names"] + g2["node_names"]
        assert len(combined) == g1["num_nodes"] + g2["num_nodes"]


class TestProofDataset:
    """Test the ProofDataset class."""

    def test_dataset_with_json_traces(self, tmp_path):
        """Test loading JSON trace files."""
        from proofatlas.ml.training import ProofDataset

        # Create sample traces (must include selection_states)
        trace1 = {
            "proof_found": True,
            "time_seconds": 1.0,
            "clauses": [
                {
                    "literals": [{"polarity": True, "atom": {"predicate": "p", "args": []}}],
                    "label": 1,
                    "age": 0,
                    "role": "axiom",
                },
                {
                    "literals": [{"polarity": False, "atom": {"predicate": "p", "args": []}}],
                    "label": 1,
                    "age": 1,
                    "role": "derived",
                },
            ],
            "selection_states": [
                {"selected": 0, "unprocessed": [0, 1], "processed": []},
            ],
        }

        trace2 = {
            "proof_found": True,
            "time_seconds": 0.5,
            "clauses": [
                {
                    "literals": [{"polarity": True, "atom": {"predicate": "q", "args": []}}],
                    "label": 0,
                    "age": 0,
                    "role": "axiom",
                },
            ],
            "selection_states": [
                {"selected": 0, "unprocessed": [0], "processed": []},
            ],
        }

        # Write traces
        with open(tmp_path / "trace1.json", "w") as f:
            json.dump(trace1, f)
        with open(tmp_path / "trace2.json", "w") as f:
            json.dump(trace2, f)

        # Test graph output (returns pre-converted graphs with selection_states)
        dataset = ProofDataset(tmp_path, output_type="graph")
        assert len(dataset) == 2

        item = dataset[0]
        assert "graphs" in item
        assert "labels" in item
        assert "selection_states" in item
        assert len(item["graphs"]) == 2
