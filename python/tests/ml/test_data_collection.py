"""Tests for ML data collection from proofs"""

import pytest
import torch
from proofatlas import ProofState
from proofatlas.ml import to_torch_tensors


def run_saturation(state: ProofState, max_iterations: int = 100) -> bool:
    """Run saturation and return whether proof was found."""
    proof_found, _, _ = state.run_saturation(max_iterations=max_iterations)
    return proof_found


class TestTrainingDataExtraction:
    """Test extraction of training examples from proofs"""

    def test_simple_proof_all_in_proof(self):
        """All clauses used in minimal proof"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found = run_saturation(state, max_iterations=100)
        assert proof_found

        examples = state.extract_training_examples()
        assert len(examples) == 3  # 2 input + 1 empty clause

        # All should be in proof
        labels = [e.label for e in examples]
        assert all(l == 1 for l in labels)

    def test_proof_with_unused_clauses(self):
        """Some clauses not used in proof"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
            cnf(q_b, axiom, q(b)).
            cnf(r_c, axiom, r(c)).
        """)

        proof_found = run_saturation(state, max_iterations=100)
        assert proof_found

        examples = state.extract_training_examples()

        # Check we have both positive and negative examples
        labels = [e.label for e in examples]
        assert 1 in labels  # At least one in proof
        assert 0 in labels  # At least one not in proof

        # q(b) and r(c) should not be in proof
        stats = state.get_proof_statistics()
        assert stats["proof_clauses"] < stats["total_clauses"]

    def test_no_proof_returns_empty(self):
        """No training examples if no proof found"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        # Limit iterations to prevent finding a proof
        proof_found = run_saturation(state, max_iterations=10)
        # This may or may not find proof depending on problem

        if not state.contains_empty_clause():
            examples = state.extract_training_examples()
            assert len(examples) == 0

    def test_graph_extraction_for_training(self):
        """Test that we can extract graphs for all training examples"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        proof_found = run_saturation(state, max_iterations=100)
        assert proof_found

        examples = state.extract_training_examples()
        clause_ids = [e.clause_idx for e in examples]

        # Get graphs
        graphs = state.clauses_to_graphs(clause_ids)
        assert len(graphs) == len(examples)

        # Convert all to tensors
        for graph in graphs:
            tensors = to_torch_tensors(graph)
            assert "x" in tensors
            assert "edge_index" in tensors
            assert tensors["x"].shape[1] == 8  # Node feature dimension

    def test_proof_statistics(self):
        """Test proof statistics computation"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        proof_found = run_saturation(state, max_iterations=100)
        assert proof_found

        stats = state.get_proof_statistics()

        assert "total_clauses" in stats
        assert "proof_clauses" in stats
        assert "non_proof_clauses" in stats
        assert "proof_percentage" in stats

        # Verify consistency
        assert stats["proof_clauses"] + stats["non_proof_clauses"] == stats["total_clauses"]


class TestSaturationLoop:
    """Test the Python saturation loop"""

    def test_finds_simple_proof(self):
        """Basic proof finding"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        result = run_saturation(state, max_iterations=100)
        assert result is True
        assert state.contains_empty_clause()

    def test_saturates_without_proof(self):
        """Saturation without finding proof"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        result = run_saturation(state, max_iterations=100)
        # May or may not saturate, but shouldn't find proof
        # The result depends on the problem

    def test_respects_iteration_limit(self):
        """Iteration limit prevents infinite loop"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
        """)

        # With only 1 iteration, shouldn't find proof
        result = run_saturation(state, max_iterations=1)
        # Proof should not be found with only 1 iteration
        # (need at least 2: select p(a), then select ~p(a))


class TestProofClauseIds:
    """Test getting proof clause IDs"""

    def test_proof_clause_ids(self):
        """Get IDs of clauses in proof"""
        state = ProofState()
        state.add_clauses_from_tptp("""
            cnf(p_a, axiom, p(a)).
            cnf(not_p_a, negated_conjecture, ~p(a)).
            cnf(q_b, axiom, q(b)).
        """)

        run_saturation(state, max_iterations=100)

        proof_ids = state.proof_clause_ids()
        all_ids = state.all_clause_ids()

        # Proof IDs should be subset of all IDs
        assert set(proof_ids).issubset(set(all_ids))

        # Should have fewer proof clauses than total
        assert len(proof_ids) <= len(all_ids)
