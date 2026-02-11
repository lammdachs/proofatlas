"""Tests for ML data collection from proofs"""

import json
import pytest
from proofatlas import ProofAtlas


class TestStructuredTrace:
    """Test extract_structured_trace() output"""

    def test_trace_contains_expected_fields(self):
        """Structured trace has proof_found, time_seconds, clauses"""
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        assert "proof_found" in trace
        assert "time_seconds" in trace
        assert "clauses" in trace
        assert trace["proof_found"] is True
        assert trace["time_seconds"] == 1.0

    def test_trace_clauses_have_labels(self):
        """Each clause has a label (1=in proof, 0=not)"""
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        labels = [c["label"] for c in trace["clauses"]]
        assert 1 in labels  # At least one in proof
        assert 0 in labels  # q(b) should not be in proof

    def test_trace_clause_structure(self):
        """Each clause has literals, age, role, derivation info"""
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        for clause in trace["clauses"]:
            assert "literals" in clause
            assert "label" in clause
            assert "age" in clause
            assert "role" in clause

    def test_trace_has_selection_states(self):
        """Structured trace includes selection state snapshots"""
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_x_or_q, axiom, ~p(X) | q(X)).
            cnf(not_q_a, negated_conjecture, ~q(a)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        assert "selection_states" in trace
        # Should have at least one selection state (one iteration)
        assert len(trace["selection_states"]) > 0

        for ss in trace["selection_states"]:
            assert "selected" in ss
            assert "unprocessed" in ss
            assert "processed" in ss

    def test_trace_parseable_by_structured_loader(self):
        """Structured trace can be loaded by load_structured_trace"""
        import tempfile
        import os
        from proofatlas.ml.structured import load_structured_trace

        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)

        # Write to temp file and load back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(trace_json)
            tmp_path = f.name

        try:
            loaded = load_structured_trace(tmp_path)
            assert "proof_found" in loaded
            assert "clauses" in loaded
            assert loaded["proof_found"] is True
        finally:
            os.unlink(tmp_path)

    def test_no_proof_trace(self):
        """Structured trace with no proof has proof_found=False"""
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        proof_found, _ = state.prove(timeout=10.0, max_iterations=100)

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        if not proof_found:
            assert trace["proof_found"] is False
            # All labels should be 0 (no proof)
            labels = [c["label"] for c in trace["clauses"]]
            assert all(l == 0 for l in labels)

    def test_trace_clause_graphs(self):
        """Clauses from structured trace can be converted to graphs"""
        from proofatlas.ml.structured import clause_to_graph

        state = ProofAtlas()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found, _ = state.prove(timeout=10.0)
        assert proof_found

        trace_json = state.extract_structured_trace(1.0)
        trace = json.loads(trace_json)

        for clause in trace["clauses"]:
            graph = clause_to_graph(clause)
            assert "edge_index" in graph
            assert "x" in graph
            assert "node_types" in graph
            assert "num_nodes" in graph
            assert graph["num_nodes"] > 0
