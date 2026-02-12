#!/usr/bin/env python3
"""Test Python-side graph conversion from structured traces"""

import json
import pytest
import numpy as np

try:
    from proofatlas.ml.structured import clause_to_graph, clause_to_string
except ImportError:
    pytestmark = pytest.mark.skip(reason="Depends on removed clause_to_graph API")
    clause_to_graph = None
    clause_to_string = None

from proofatlas import ProofAtlas


def _get_trace_clauses(tptp: str):
    """Helper: prove and return structured trace clauses."""
    state = ProofAtlas()
    state.add_clauses_from_tptp(tptp)
    proof_found, _ = state.prove(timeout=10.0)
    assert proof_found
    trace_json = state.extract_structured_trace(1.0)
    trace = json.loads(trace_json)
    return trace["clauses"]


class TestClauseToGraph:
    """Test clause_to_graph() from ml/structured.py"""

    def test_simple_clause(self):
        """Graph for P(a) has correct structure"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(a)).
        """)

        # Find p(a) input clause
        graph = clause_to_graph(clauses[0])

        assert graph["num_nodes"] > 0
        assert graph["num_edges"] >= 0
        assert graph["x"].shape[0] == graph["num_nodes"]
        assert graph["node_types"].shape[0] == graph["num_nodes"]

    def test_multi_literal_clause(self):
        """Graph for clause with multiple literals"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(X) | q(X)).
            cnf(c2, axiom, ~p(a)).
            cnf(c3, axiom, ~q(a)).
        """)

        # First clause has two literals
        graph = clause_to_graph(clauses[0])
        # Should have more nodes than single-literal clause
        assert graph["num_nodes"] >= 5  # clause + 2*(literal + predicate + term)

    def test_function_terms(self):
        """Graph for clause with nested functions"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(f(g(a)))).
            cnf(c2, axiom, ~p(f(g(a)))).
        """)

        graph = clause_to_graph(clauses[0])
        # Should have function nodes
        assert graph["num_nodes"] >= 5

    def test_equality_clause(self):
        """Graph for equality clause"""
        clauses = _get_trace_clauses("""
            cnf(eq1, axiom, a = b).
            cnf(eq2, axiom, f(a) != f(b)).
        """)

        graph = clause_to_graph(clauses[0])
        assert graph["num_nodes"] > 0

    def test_edge_index_shape(self):
        """Edge index has shape [2, num_edges]"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(X)).
            cnf(c2, axiom, ~p(X)).
        """)

        graph = clause_to_graph(clauses[0])
        edge_index = graph["edge_index"]
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == graph["num_edges"]

    def test_node_features_dimension(self):
        """Node features have expected dimension"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(a)).
        """)

        graph = clause_to_graph(clauses[0])
        # Node features: type, arity, arg_pos (3 dims)
        assert graph["x"].shape[1] == 3

    def test_node_types_valid(self):
        """Node types are valid integers"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(X, a)).
            cnf(c2, axiom, ~p(X, a)).
        """)

        graph = clause_to_graph(clauses[0])
        node_types = graph["node_types"]
        # All types should be non-negative
        assert np.all(node_types >= 0)
        assert np.all(node_types <= 5)

    def test_clause_features(self):
        """Clause-level features are present"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(a)).
        """)

        graph = clause_to_graph(clauses[0])
        assert "clause_features" in graph
        # clause_features: age, role, size (3 dims)
        assert graph["clause_features"].shape == (3,)


class TestClauseToString:
    """Test clause_to_string() from ml/structured.py"""

    def test_simple_clause_string(self):
        """String representation is non-empty"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(a)).
        """)

        s = clause_to_string(clauses[0])
        assert isinstance(s, str)
        assert len(s) > 0

    def test_equality_string(self):
        """Equality clause produces valid string"""
        clauses = _get_trace_clauses("""
            cnf(eq1, axiom, f(a) = g(b)).
            cnf(eq2, axiom, f(a) != g(b)).
        """)

        s = clause_to_string(clauses[0])
        assert isinstance(s, str)
        assert len(s) > 0


class TestGraphsFromTrace:
    """Test converting all clauses in a trace to graphs"""

    def test_all_clauses_convertible(self):
        """All clauses from a proof trace can be converted to graphs"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(X) | q(X)).
            cnf(c3, axiom, ~q(a)).
        """)

        for clause in clauses:
            graph = clause_to_graph(clause)
            assert graph["num_nodes"] > 0
            assert graph["x"].shape[0] == graph["num_nodes"]

    def test_derived_clauses_convertible(self):
        """Derived clauses (from inference) are also valid graphs"""
        clauses = _get_trace_clauses("""
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(X) | q(X)).
            cnf(c3, axiom, ~q(a)).
        """)

        # Some clauses should be derived (rule != "input")
        derived = [c for c in clauses if c.get("rule", "input") != "input"]
        for clause in derived:
            graph = clause_to_graph(clause)
            assert graph["num_nodes"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
