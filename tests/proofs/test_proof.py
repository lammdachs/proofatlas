"""Tests for proofs.proof module."""

import unittest
from proofatlas.core.logic import Predicate, Constant, Variable, Literal, Clause
from proofatlas.proofs.state import ProofState
from proofatlas.proofs import Proof, ProofStep


class TestProofStep(unittest.TestCase):
    """Test ProofStep dataclass."""
    
    def setUp(self):
        """Create test data."""
        P = Predicate("P", 1)
        a = Constant("a")
        self.clause = Clause(Literal(P(a), True))
        self.state = ProofState([], [self.clause])
    
    def test_proof_step_creation(self):
        """Test creating proof steps."""
        # Minimal step
        step1 = ProofStep(self.state)
        self.assertEqual(step1.state, self.state)
        self.assertIsNone(step1.selected_clause)
        self.assertEqual(len(step1.metadata), 0)
        
        # Step with selected clause
        step2 = ProofStep(self.state, selected_clause=0)
        self.assertEqual(step2.selected_clause, 0)
        
        # Step with metadata
        step3 = ProofStep(
            state=self.state,
            selected_clause=1,
            metadata={"rule": "resolution", "parent_clauses": [0, 1]}
        )
        self.assertEqual(step3.selected_clause, 1)
        self.assertEqual(step3.metadata["rule"], "resolution")
        self.assertEqual(step3.metadata["parent_clauses"], [0, 1])


class TestProof(unittest.TestCase):
    """Test Proof class."""
    
    def setUp(self):
        """Create test data."""
        # Symbols
        self.P = Predicate("P", 1)
        self.Q = Predicate("Q", 1)
        self.a = Constant("a")
        self.x = Variable("X")
        
        # Clauses
        self.clause1 = Clause(Literal(self.P(self.a), True))  # P(a)
        self.clause2 = Clause(Literal(self.P(self.x), False), Literal(self.Q(self.x), True))  # ~P(X) | Q(X)
        self.clause3 = Clause(Literal(self.Q(self.a), True))  # Q(a)
        self.empty = Clause()  # Empty clause
        
        # Initial state
        self.initial_state = ProofState([], [self.clause1, self.clause2])
    
    def test_proof_creation(self):
        """Test creating proofs."""
        # With initial state
        proof1 = Proof(self.initial_state)
        self.assertEqual(len(proof1.steps), 1)
        self.assertEqual(proof1.initial_state, self.initial_state)
        self.assertEqual(proof1.final_state, self.initial_state)
        self.assertEqual(proof1.length, 0)  # No inference steps yet
        self.assertIsNone(proof1.steps[0].selected_clause)
        
        # Without initial state (default empty)
        proof2 = Proof()
        self.assertEqual(len(proof2.steps), 1)
        self.assertEqual(len(proof2.initial_state.processed), 0)
        self.assertEqual(len(proof2.initial_state.unprocessed), 0)
    
    def test_add_step_behavior(self):
        """Test the behavior of add_step with the last step rule."""
        proof = Proof(self.initial_state)
        
        # Initially, last step has no selection
        self.assertIsNone(proof.steps[-1].selected_clause)
        
        # Add a step with selection - should replace the last step
        new_state = ProofState([self.clause1], [self.clause2])
        proof.add_step(
            state=new_state,
            selected_clause=0,
            rule="given_clause"
        )
        
        # Should still have 1 step (replaced), but now needs a final step
        self.assertEqual(len(proof.steps), 2)
        self.assertEqual(proof.steps[0].selected_clause, 0)
        self.assertIsNone(proof.steps[-1].selected_clause)
        self.assertEqual(proof.length, 1)
        
        # Add another step with selection
        newer_state = ProofState([self.clause1, self.clause2], [])
        proof.add_step(newer_state, selected_clause=1)
        
        # Last step should be replaced, then a new final step added
        self.assertEqual(len(proof.steps), 3)
        self.assertEqual(proof.steps[1].selected_clause, 1)
        self.assertIsNone(proof.steps[-1].selected_clause)
        self.assertEqual(proof.length, 2)
    
    def test_add_step_without_selection(self):
        """Test adding a step without selection."""
        proof = Proof(self.initial_state)
        
        # Add step without selection - should just replace last step
        new_state = ProofState([self.clause1], [self.clause2])
        proof.add_step(new_state)
        
        self.assertEqual(len(proof.steps), 1)
        self.assertIsNone(proof.steps[-1].selected_clause)
        self.assertEqual(proof.final_state, new_state)
    
    def test_proof_properties(self):
        """Test proof properties."""
        proof = Proof(self.initial_state)
        
        # Not complete initially
        self.assertFalse(proof.is_complete)
        self.assertFalse(proof.is_saturated)
        
        # Add empty clause
        state_with_empty = ProofState([self.clause1], [self.empty])
        proof.add_step(state_with_empty, selected_clause=0)
        
        # Now complete
        self.assertTrue(proof.is_complete)
        
        # Test saturated (no unprocessed)
        saturated_state = ProofState([self.clause1, self.clause2], [])
        proof2 = Proof(saturated_state)
        self.assertTrue(proof2.is_saturated)
        self.assertFalse(proof2.is_complete)  # Saturated but no contradiction
    
    def test_get_selected_clauses(self):
        """Test getting selected clause history."""
        proof = Proof(self.initial_state)
        
        # Add steps with selections
        state1 = ProofState([self.clause1], [self.clause2])
        proof.add_step(state1, selected_clause=0)
        
        state2 = ProofState([self.clause1, self.clause2], [])
        proof.add_step(state2, selected_clause=1)
        
        selected = proof.get_selected_clauses()
        self.assertEqual(selected, [0, 1])
        
        # Last step should have no selection
        self.assertIsNone(proof.steps[-1].selected_clause)
    
    def test_finalize(self):
        """Test finalizing a proof."""
        proof = Proof(self.initial_state)
        
        # Add step with selection
        state1 = ProofState([self.clause1], [self.clause2])
        proof.add_step(state1, selected_clause=0)
        
        # Already has final step
        self.assertIsNone(proof.steps[-1].selected_clause)
        
        # Finalize with new state
        final_state = ProofState([self.clause1, self.clause2], [self.empty])
        proof.finalize(final_state)
        
        # Should have replaced the last step
        self.assertEqual(proof.final_state, final_state)
        self.assertIsNone(proof.steps[-1].selected_clause)
        
        # Test finalize without new state (should do nothing if already finalized)
        num_steps = len(proof.steps)
        proof.finalize()
        self.assertEqual(len(proof.steps), num_steps)
    
    def test_get_step(self):
        """Test getting specific steps."""
        proof = Proof(self.initial_state)
        
        # Add some steps
        state1 = ProofState([self.clause1], [self.clause2])
        proof.add_step(state1, selected_clause=0)
        
        # Get steps
        step0 = proof.get_step(0)
        self.assertEqual(step0.selected_clause, 0)
        
        step1 = proof.get_step(1)
        self.assertIsNone(step1.selected_clause)  # Final step
        
        # Out of bounds
        self.assertIsNone(proof.get_step(-1))
        self.assertIsNone(proof.get_step(10))
    
    def test_get_metadata_history(self):
        """Test getting metadata history."""
        proof = Proof(self.initial_state)
        
        # Add steps with metadata
        state1 = ProofState([self.clause1], [self.clause2])
        proof.add_step(state1, selected_clause=0, rule="given_clause", score=0.5)
        
        state2 = ProofState([self.clause1, self.clause2], [self.clause3])
        proof.add_step(state2, selected_clause=1, rule="resolution", score=0.8, generated=[2])
        
        # Get histories
        rules = proof.get_metadata_history("rule")
        self.assertEqual(rules, ["given_clause", "resolution"])
        
        scores = proof.get_metadata_history("score")
        self.assertEqual(scores, [0.5, 0.8])
    
    def test_repr(self):
        """Test string representation."""
        proof = Proof(self.initial_state)
        self.assertEqual(str(proof), "Proof(steps=1, complete=False)")
        
        # Add steps
        proof.add_step(ProofState([self.clause1], [self.clause2]), selected_clause=0)
        self.assertEqual(str(proof), "Proof(steps=2, complete=False)")


class TestProofScenarios(unittest.TestCase):
    """Test realistic proof scenarios."""
    
    def setUp(self):
        """Set up a simple resolution problem."""
        # Symbols
        P = Predicate("P", 1)
        Q = Predicate("Q", 1)
        a = Constant("a")
        x = Variable("X")
        
        # Initial clauses
        self.c1 = Clause(Literal(P(a), True))  # P(a)
        self.c2 = Clause(Literal(P(x), False), Literal(Q(x), True))  # ~P(X) | Q(X)
        self.c3 = Clause(Literal(Q(a), False))  # ~Q(a) (goal)
        
        # Expected resolvents
        self.r1 = Clause(Literal(Q(a), True))  # Q(a) from c1 and c2
        self.r2 = Clause()  # Empty from r1 and c3
    
    def test_complete_proof(self):
        """Test building a complete proof."""
        # Initial state
        initial = ProofState([], [self.c1, self.c2, self.c3])
        proof = Proof(initial)
        
        # Step 1: Select c1
        state1 = ProofState([self.c1], [self.c2, self.c3])
        proof.add_step(state1, selected_clause=0, rule="given_clause")
        
        # Step 2: Select c2, resolve with c1 to get Q(a)
        state2 = ProofState([self.c1, self.c2], [self.c3, self.r1])
        proof.add_step(
            state2, 
            selected_clause=1,
            rule="resolution",
            parent_clauses=[0, 1],
            generated_clause=self.r1
        )
        
        # Step 3: Select the resolvent Q(a)
        state3 = ProofState([self.c1, self.c2, self.r1], [self.c3])
        proof.add_step(state3, selected_clause=3, rule="given_clause")
        
        # Step 4: Select c3, resolve with Q(a) to get empty clause
        state4 = ProofState([self.c1, self.c2, self.r1, self.c3], [self.r2])
        proof.add_step(
            state4,
            selected_clause=0,
            rule="resolution", 
            parent_clauses=[2, 3],
            generated_clause=self.r2
        )
        
        # Check proof is complete
        self.assertTrue(proof.is_complete)
        self.assertEqual(proof.length, 4)
        
        # Last step should have no selection
        self.assertIsNone(proof.steps[-1].selected_clause)
        
        # Check selected clauses
        selected = proof.get_selected_clauses()
        self.assertEqual(selected, [0, 1, 3, 0])
        
        # Check metadata
        rules = proof.get_metadata_history("rule")
        self.assertEqual(rules, ["given_clause", "resolution", "given_clause", "resolution"])
        
        # Verify the empty clause is in final state
        has_empty = any(len(c.literals) == 0 for c in proof.final_state.all_clauses)
        self.assertTrue(has_empty)
    
    def test_saturated_proof(self):
        """Test a proof that saturates without finding contradiction."""
        # Only positive unit clauses - will saturate
        p_a = Clause(Literal(Predicate("P", 1)(Constant("a")), True))
        q_b = Clause(Literal(Predicate("Q", 1)(Constant("b")), True))
        
        initial = ProofState([], [p_a, q_b])
        proof = Proof(initial)
        
        # Process both clauses
        state1 = ProofState([p_a], [q_b])
        proof.add_step(state1, selected_clause=0)
        
        state2 = ProofState([p_a, q_b], [])
        proof.add_step(state2, selected_clause=0)
        
        # Should be saturated but not complete
        self.assertTrue(proof.is_saturated)
        self.assertFalse(proof.is_complete)
        
        # Last step should have no selection
        self.assertIsNone(proof.steps[-1].selected_clause)


if __name__ == '__main__':
    unittest.main()