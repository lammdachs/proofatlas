"""Test Rust-Python integration for proof types."""

import os
import pytest

# Set environment variable to use Rust
os.environ['PROOFATLAS_USE_RUST'] = 'true'

from proofatlas.core.logic import Constant, Predicate, Literal, Clause
from proofatlas.proofs import Proof, ProofState


class TestRustProofIntegration:
    """Test that Rust proof implementations work with Python code."""
    
    def test_proof_state_creation(self):
        """Test creating a ProofState with Rust backend."""
        # Create some clauses
        P = Predicate('P', 1)
        a = Constant('a')
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        
        # Create ProofState
        state = ProofState(processed=[c1], unprocessed=[c2])
        
        assert len(state.processed) == 1
        assert len(state.unprocessed) == 1
        assert len(state.all_clauses) == 2
        assert not state.contains_empty_clause
    
    def test_proof_state_operations(self):
        """Test ProofState operations with Rust backend."""
        P = Predicate('P', 1)
        a = Constant('a')
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        c3 = Clause()  # Empty clause
        
        state = ProofState([], [c1, c2])
        
        # Test add operations
        state.add_unprocessed(c3)
        assert len(state.unprocessed) == 3
        assert state.contains_empty_clause
        
        # Test move to processed
        state.move_to_processed(c1)
        assert len(state.processed) == 1
        assert len(state.unprocessed) == 2
    
    def test_proof_creation(self):
        """Test creating a Proof with Rust backend."""
        P = Predicate('P', 1)
        a = Constant('a')
        c1 = Clause(Literal(P(a), True))
        
        initial_state = ProofState([], [c1])
        proof = Proof(initial_state)
        
        assert proof.initial_state is not None
        assert proof.final_state is not None
        assert len(proof.steps) == 1
        assert not proof.is_complete
    
    def test_proof_steps(self):
        """Test adding steps to a Proof with Rust backend."""
        P = Predicate('P', 1)
        a = Constant('a')
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        empty = Clause()
        
        # Create proof
        initial_state = ProofState([], [c1, c2])
        proof = Proof(initial_state)
        
        # Add step
        new_state = ProofState([c1], [c2])
        proof.add_step(new_state, selected_clause=0, rule="given_clause")
        
        assert len(proof.steps) == 2
        assert proof.steps[-1].selected_clause == 0
        
        # Add step with empty clause
        final_state = ProofState([c1, c2], [empty])
        proof.add_step(final_state, selected_clause=1, rule="resolution")
        
        assert proof.final_state.contains_empty_clause
        assert proof.is_complete
    
    def test_proof_finalize(self):
        """Test finalizing a proof with Rust backend."""
        P = Predicate('P', 1)
        a = Constant('a')
        empty = Clause()
        
        initial_state = ProofState([], [])
        proof = Proof(initial_state)
        
        # Finalize with empty clause
        final_state = ProofState([empty], [])
        proof.finalize(final_state)
        
        assert proof.is_complete
        assert proof.steps[-1].selected_clause is None
    
    def test_metadata_handling(self):
        """Test metadata in proof steps."""
        initial_state = ProofState([], [])
        proof = Proof(initial_state)
        
        # Add step with metadata
        new_state = ProofState([], [])
        proof.add_step(
            new_state,
            selected_clause=0,
            rule="test_rule",
            score=0.95,
            custom_data={"key": "value"}
        )
        
        # Check metadata history
        rule_history = proof.get_metadata_history("rule")
        assert len(rule_history) == 1
        assert rule_history[0] == "test_rule"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])