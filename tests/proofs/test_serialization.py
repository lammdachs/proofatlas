"""Tests for proofs JSON serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from proofatlas.core import (
    Variable, Constant, Predicate,
    Literal, Clause, Problem
)
from proofatlas.proofs import (
    ProofState, Proof, ProofStep,
    ProofJSONEncoder, ProofJSONDecoder,
    proof_to_json, proof_from_json,
    save_proof, load_proof
)


class TestProofsJSONSerialization:
    """Test JSON serialization of proof objects."""
    
    def test_proof_state_serialization(self):
        """Test ProofState serialization."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        
        state = ProofState(processed=[c1], unprocessed=[c2])
        
        json_str = json.dumps(state, cls=ProofJSONEncoder)
        state2 = json.loads(json_str, cls=ProofJSONDecoder)
        
        assert isinstance(state2, ProofState)
        assert len(state2.processed) == 1
        assert len(state2.unprocessed) == 1
        assert str(state2.processed[0]) == str(c1)
        assert str(state2.unprocessed[0]) == str(c2)
    
    def test_proof_with_state_roundtrip(self):
        """Test that proofs with ProofState serialize correctly."""
        P = Predicate("P", 1)
        a = Constant("a")
        
        c1 = Clause(Literal(P(a), True))
        c2 = Clause(Literal(P(a), False))
        
        # Create a proof
        initial_state = ProofState([], [c1, c2])
        proof = Proof(initial_state)
        
        # Add a step
        new_state = ProofState([c1], [c2])
        proof.add_step(new_state, selected_clause=0, rule="given_clause")
        
        # Serialize and deserialize
        json_str = proof_to_json(proof)
        proof2 = proof_from_json(json_str)
        
        assert isinstance(proof2, Proof)
        assert len(proof2.steps) == 2
        assert isinstance(proof2.steps[0].state, ProofState)
        # First step should have the selection and updated state
        assert proof2.steps[0].selected_clause == 0
        assert len(proof2.steps[0].state.processed) == 1
        assert len(proof2.steps[0].state.unprocessed) == 1
        # Second step should be the final state with no selection
        assert proof2.steps[1].selected_clause is None