"""Tests for the basic saturation loop implementation."""

import unittest
from proofatlas.core.logic import Predicate, Constant, Variable, Literal, Clause
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.loops.basic import BasicLoop


class TestBasicLoop(unittest.TestCase):
    """Test the BasicLoop implementation."""
    
    def setUp(self):
        """Set up test data."""
        # Predicates and constants
        self.P = Predicate("P", 1)
        self.Q = Predicate("Q", 1)
        self.R = Predicate("R", 1)
        self.a = Constant("a")
        self.b = Constant("b")
        self.x = Variable("X")
        
        # Test clauses
        self.c1 = Clause(Literal(self.P(self.a), True))  # P(a)
        self.c2 = Clause(Literal(self.P(self.a), False), Literal(self.Q(self.a), True))  # ~P(a) | Q(a)
        self.c3 = Clause(Literal(self.Q(self.a), False), Literal(self.R(self.a), True))  # ~Q(a) | R(a)
        self.c4 = Clause(Literal(self.P(self.a), False))  # ~P(a) (for contradiction)
        
        # Clauses for tautology testing
        self.taut = Clause(Literal(self.P(self.a), True), Literal(self.P(self.a), False))  # P(a) | ~P(a)
    
    def test_basic_loop_initialization(self):
        """Test BasicLoop initialization with different parameters."""
        # Default initialization
        loop1 = BasicLoop()
        self.assertEqual(loop1.max_clause_size, 100)
        self.assertTrue(loop1.forward_simplify)
        self.assertTrue(loop1.backward_simplify)
        
        # Custom initialization
        loop2 = BasicLoop(max_clause_size=50, forward_simplify=False, backward_simplify=False)
        self.assertEqual(loop2.max_clause_size, 50)
        self.assertFalse(loop2.forward_simplify)
        self.assertFalse(loop2.backward_simplify)
    
    def test_single_step_basic(self):
        """Test a single step of the saturation loop."""
        loop = BasicLoop()
        
        # Initial state: c1 in unprocessed
        initial_state = ProofState(processed=[], unprocessed=[self.c1])
        proof = Proof(initial_state=initial_state)
        
        # Select first (and only) clause
        proof = loop.step(proof, given_clause=0)
        
        # Check the result
        # Initial state + result step = 2
        self.assertEqual(len(proof.steps), 2)
        final_state = proof.final_state
        
        # c1 should be moved to processed
        self.assertEqual(len(final_state.processed), 1)
        self.assertEqual(final_state.processed[0], self.c1)
        self.assertEqual(len(final_state.unprocessed), 0)
        
        # Check result step (step 0)
        result_step = proof.steps[0]
        self.assertEqual(result_step.selected_clause, 0)
    
    def test_resolution_generation(self):
        """Test that resolution generates new clauses."""
        loop = BasicLoop()
        
        # Initial state: c1 processed, c2 unprocessed
        initial_state = ProofState(processed=[self.c1], unprocessed=[self.c2])
        proof = Proof(initial_state=initial_state)
        
        # Select c2 (which can resolve with c1)
        proof = loop.step(proof, given_clause=0)
        
        final_state = proof.final_state
        
        # c2 should be moved to processed
        self.assertEqual(len(final_state.processed), 2)
        self.assertIn(self.c1, final_state.processed)
        self.assertIn(self.c2, final_state.processed)
        
        # Should generate Q(a) from resolving P(a) and ~P(a) | Q(a)
        self.assertEqual(len(final_state.unprocessed), 1)
        generated = final_state.unprocessed[0]
        self.assertEqual(len(generated.literals), 1)
        self.assertEqual(generated.literals[0].predicate.symbol, self.Q)
        self.assertTrue(generated.literals[0].polarity)
        
        # Check rule applications were tracked
        # The step with rules is at index 0 (the initial step was replaced)
        step = proof.steps[0]
        self.assertTrue(len(step.applied_rules) > 0)
        self.assertTrue(any(rule.rule_name == "resolution" for rule in step.applied_rules))
    
    def test_contradiction_detection(self):
        """Test generating empty clause (contradiction)."""
        loop = BasicLoop()
        
        # Initial state: P(a) processed, ~P(a) unprocessed
        initial_state = ProofState(processed=[self.c1], unprocessed=[self.c4])
        proof = Proof(initial_state=initial_state)
        
        # Select ~P(a)
        proof = loop.step(proof, given_clause=0)
        
        final_state = proof.final_state
        
        # Should generate empty clause
        self.assertTrue(any(len(clause.literals) == 0 for clause in final_state.unprocessed))
        
        # Verify empty clause detection
        empty_clause = next(c for c in final_state.unprocessed if len(c.literals) == 0)
        self.assertTrue(loop.is_contradiction(empty_clause))
    
    def test_tautology_filtering(self):
        """Test that tautologies are filtered out with forward simplification."""
        loop = BasicLoop(forward_simplify=True)
        
        # Initial state with tautology
        initial_state = ProofState(processed=[], unprocessed=[self.taut])
        proof = Proof(initial_state=initial_state)
        
        # Select the tautology
        proof = loop.step(proof, given_clause=0)
        
        final_state = proof.final_state
        
        # Tautology should be moved to processed but not generate new clauses
        self.assertEqual(len(final_state.processed), 1)
        self.assertEqual(len(final_state.unprocessed), 0)
        
        # Verify tautology detection
        self.assertTrue(loop.is_tautology(self.taut))
    
    def test_clause_size_limit(self):
        """Test that clauses exceeding size limit are filtered."""
        loop = BasicLoop(max_clause_size=2)
        
        # Create a clause that will generate large resolvents
        large_clause = Clause(
            Literal(self.P(self.a), False),
            Literal(self.Q(self.a), True),
            Literal(self.R(self.a), True)
        )  # ~P(a) | Q(a) | R(a)
        
        initial_state = ProofState(processed=[self.c1], unprocessed=[large_clause])
        proof = Proof(initial_state=initial_state)
        
        proof = loop.step(proof, given_clause=0)
        
        # Any generated clause should have at most 2 literals
        for clause in proof.final_state.unprocessed:
            self.assertLessEqual(len(clause.literals), 2)
    
    def test_multiple_steps(self):
        """Test multiple steps of the saturation loop."""
        loop = BasicLoop()
        
        # Initial state with chain of clauses
        initial_state = ProofState(processed=[], unprocessed=[self.c1, self.c2, self.c3])
        proof = Proof(initial_state=initial_state)
        
        # Step 1: Process P(a)
        proof = loop.step(proof, given_clause=0)
        # Initial step is replaced, plus final step = 2
        self.assertEqual(len(proof.steps), 2)
        
        # Step 2: Process ~P(a) | Q(a)
        proof = loop.step(proof, given_clause=0)
        # Previous final step is replaced, new final added = 3
        self.assertEqual(len(proof.steps), 3)
        
        # Should have generated Q(a)
        state_after_2 = proof.final_state
        self.assertTrue(any(
            len(c.literals) == 1 and c.literals[0].predicate.symbol == self.Q 
            for c in state_after_2.unprocessed
        ))
        
        # Step 3: Process ~Q(a) | R(a) (the original c3)
        proof = loop.step(proof, given_clause=0)
        # Previous final replaced, new final added = 4
        self.assertEqual(len(proof.steps), 4)
        
        # Now we have Q(a) in unprocessed (from step 2) and ~Q(a) | R(a) in processed
        # Step 4: Process Q(a) to generate R(a)
        proof = loop.step(proof, given_clause=0)
        # Previous final replaced, new final added = 5
        self.assertEqual(len(proof.steps), 5)
        
        # Should generate R(a) from Q(a) and ~Q(a) | R(a)
        final_state = proof.final_state
        
        self.assertTrue(any(
            len(c.literals) == 1 and c.literals[0].predicate.symbol == self.R
            for c in final_state.unprocessed
        ))
    
    def test_invalid_clause_index(self):
        """Test error handling for invalid clause indices."""
        loop = BasicLoop()
        
        initial_state = ProofState(processed=[], unprocessed=[self.c1])
        proof = Proof(initial_state=initial_state)
        
        # Test negative index
        with self.assertRaises(ValueError):
            loop.step(proof, given_clause=-1)
        
        # Test index out of bounds
        with self.assertRaises(ValueError):
            loop.step(proof, given_clause=5)
    
    def test_subsumption_checking(self):
        """Test subsumption in forward simplification."""
        loop = BasicLoop(forward_simplify=True)
        
        # c1 subsumes any clause containing P(a)
        subsumed = Clause(Literal(self.P(self.a), True), Literal(self.Q(self.b), True))  # P(a) | Q(b)
        
        initial_state = ProofState(processed=[self.c1], unprocessed=[subsumed])
        proof = Proof(initial_state=initial_state)
        
        # Process the subsumed clause
        proof = loop.step(proof, given_clause=0)
        
        # The subsumed clause should be in processed but shouldn't generate
        # new subsumed clauses through resolution
        final_state = proof.final_state
        self.assertEqual(len(final_state.processed), 2)
        
        # Check subsumption detection
        self.assertTrue(loop.subsumes(self.c1, subsumed))
        self.assertTrue(loop.is_subsumed(subsumed, [self.c1]))
    
    def test_no_forward_simplification(self):
        """Test loop without forward simplification."""
        loop = BasicLoop(forward_simplify=False)
        
        # Without simplification, tautologies should be kept
        initial_state = ProofState(processed=[self.c1], unprocessed=[self.taut])
        proof = Proof(initial_state=initial_state)
        
        proof = loop.step(proof, given_clause=0)
        
        # If resolution generates tautologies, they should be kept
        # (The exact behavior depends on what the resolution rule generates)
        final_state = proof.final_state
        self.assertEqual(len(final_state.processed), 2)
    
    def test_proof_metadata_tracking(self):
        """Test that metadata is properly tracked in proof steps."""
        loop = BasicLoop()
        
        initial_state = ProofState(processed=[self.c1], unprocessed=[self.c2])
        proof = Proof(initial_state=initial_state)
        
        proof = loop.step(proof, given_clause=0)
        
        # The step with metadata is at index 0 (the initial step was replaced)
        step = proof.steps[0]
        
        # Check metadata is empty (we removed redundant metadata)
        self.assertEqual(step.metadata, {})


if __name__ == '__main__':
    unittest.main()