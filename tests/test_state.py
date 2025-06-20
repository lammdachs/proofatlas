"""Tests for core.state module."""

import unittest
from proofatlas.core.logic import Predicate, Constant, Variable, Literal, Clause
from proofatlas.core.state import ProofState


class TestProofState(unittest.TestCase):
    """Test ProofState class."""
    
    def setUp(self):
        """Create test clauses."""
        # Create symbols
        self.P = Predicate("P", 1)
        self.Q = Predicate("Q", 2)
        self.a = Constant("a")
        self.b = Constant("b")
        self.x = Variable("X")
        
        # Create test clauses
        self.clause1 = Clause(Literal(self.P(self.a), True))  # P(a)
        self.clause2 = Clause(Literal(self.P(self.x), False))  # ~P(X)
        self.clause3 = Clause(Literal(self.Q(self.a, self.b), True))  # Q(a,b)
        self.empty_clause = Clause()  # Empty clause (contradiction)
    
    def test_state_creation(self):
        """Test creating proof states."""
        # Empty state
        state1 = ProofState([], [])
        self.assertEqual(len(state1.processed), 0)
        self.assertEqual(len(state1.unprocessed), 0)
        self.assertEqual(len(state1.all_clauses), 0)
        
        # State with unprocessed clauses
        state2 = ProofState([], [self.clause1, self.clause2])
        self.assertEqual(len(state2.processed), 0)
        self.assertEqual(len(state2.unprocessed), 2)
        self.assertEqual(len(state2.all_clauses), 2)
        
        # State with both processed and unprocessed
        state3 = ProofState([self.clause1], [self.clause2, self.clause3])
        self.assertEqual(len(state3.processed), 1)
        self.assertEqual(len(state3.unprocessed), 2)
        self.assertEqual(len(state3.all_clauses), 3)
    
    def test_post_init_copies(self):
        """Test that __post_init__ creates mutable copies."""
        original_list = [self.clause1, self.clause2]
        state = ProofState([], original_list)
        
        # Modify the state's list
        state.unprocessed.append(self.clause3)
        
        # Original list should be unchanged
        self.assertEqual(len(original_list), 2)
        self.assertEqual(len(state.unprocessed), 3)
    
    def test_add_clauses(self):
        """Test adding clauses to state."""
        state = ProofState([], [])
        
        # Add to unprocessed
        state.add_unprocessed(self.clause1)
        self.assertEqual(len(state.unprocessed), 1)
        self.assertIn(self.clause1, state.unprocessed)
        
        # Add to processed
        state.add_processed(self.clause2)
        self.assertEqual(len(state.processed), 1)
        self.assertIn(self.clause2, state.processed)
        
        # Check all_clauses
        self.assertEqual(len(state.all_clauses), 2)
    
    def test_move_to_processed(self):
        """Test moving clauses from unprocessed to processed."""
        state = ProofState([], [self.clause1, self.clause2, self.clause3])
        
        # Move clause1
        state.move_to_processed(self.clause1)
        self.assertEqual(len(state.processed), 1)
        self.assertEqual(len(state.unprocessed), 2)
        self.assertIn(self.clause1, state.processed)
        self.assertNotIn(self.clause1, state.unprocessed)
        
        # Move clause3
        state.move_to_processed(self.clause3)
        self.assertEqual(len(state.processed), 2)
        self.assertEqual(len(state.unprocessed), 1)
        self.assertEqual(state.unprocessed[0], self.clause2)
        
        # Try to move non-existent clause (should do nothing)
        fake_clause = Clause(Literal(self.P(self.b), True))
        state.move_to_processed(fake_clause)
        self.assertEqual(len(state.processed), 2)
        self.assertEqual(len(state.unprocessed), 1)
    
    def test_all_clauses_property(self):
        """Test that all_clauses returns combined list."""
        state = ProofState([self.clause1], [self.clause2, self.clause3])
        all_clauses = state.all_clauses
        
        self.assertEqual(len(all_clauses), 3)
        self.assertEqual(all_clauses[0], self.clause1)
        self.assertEqual(all_clauses[1], self.clause2)
        self.assertEqual(all_clauses[2], self.clause3)
        
        # Test that it's a new list each time
        all_clauses2 = state.all_clauses
        self.assertIsNot(all_clauses, all_clauses2)
        self.assertEqual(all_clauses, all_clauses2)
    
    def test_state_with_empty_clause(self):
        """Test state containing empty clause."""
        # State with empty clause in unprocessed
        state1 = ProofState([self.clause1], [self.empty_clause])
        self.assertIn(self.empty_clause, state1.unprocessed)
        
        # State with empty clause in processed
        state2 = ProofState([self.empty_clause], [self.clause1])
        self.assertIn(self.empty_clause, state2.processed)
        
        # Empty clause should be detectable
        for clause in state1.all_clauses:
            if len(clause.literals) == 0:
                self.assertEqual(clause, self.empty_clause)
                break
        else:
            self.fail("Empty clause not found")


class TestProofStateScenarios(unittest.TestCase):
    """Test ProofState in realistic theorem proving scenarios."""
    
    def setUp(self):
        """Create a realistic problem."""
        # Symbols
        P = Predicate("P", 1)
        Q = Predicate("Q", 1)
        R = Predicate("R", 1)
        a = Constant("a")
        x = Variable("X")
        
        # Initial clauses (axioms)
        self.axiom1 = Clause(Literal(P(a), True))  # P(a)
        self.axiom2 = Clause(Literal(P(x), False), Literal(Q(x), True))  # ~P(X) | Q(X)
        self.axiom3 = Clause(Literal(Q(x), False), Literal(R(x), True))  # ~Q(X) | R(X)
        self.goal = Clause(Literal(R(a), False))  # ~R(a) (negated goal)
        
        # Derived clauses
        self.derived1 = Clause(Literal(Q(a), True))  # Q(a) from axiom1 and axiom2
        self.derived2 = Clause(Literal(R(a), True))  # R(a) from derived1 and axiom3
        self.empty = Clause()  # Empty clause from derived2 and goal
    
    def test_initial_state(self):
        """Test setting up initial proof state."""
        # Start with all axioms and negated goal as unprocessed
        initial = ProofState([], [self.axiom1, self.axiom2, self.axiom3, self.goal])
        
        self.assertEqual(len(initial.processed), 0)
        self.assertEqual(len(initial.unprocessed), 4)
        
        # No contradictions yet
        has_empty = any(len(c.literals) == 0 for c in initial.all_clauses)
        self.assertFalse(has_empty)
    
    def test_proof_progression(self):
        """Test a sequence of proof steps."""
        # Initial state
        state = ProofState([], [self.axiom1, self.axiom2, self.axiom3, self.goal])
        
        # Step 1: Select and process axiom1
        state.move_to_processed(self.axiom1)
        self.assertEqual(len(state.processed), 1)
        self.assertEqual(len(state.unprocessed), 3)
        
        # Step 2: Select and process axiom2, derive Q(a)
        state.move_to_processed(self.axiom2)
        state.add_unprocessed(self.derived1)
        self.assertEqual(len(state.processed), 2)
        self.assertEqual(len(state.unprocessed), 3)  # goal, axiom3, derived1
        
        # Continue until contradiction
        state.move_to_processed(self.derived1)
        state.move_to_processed(self.axiom3)
        state.add_unprocessed(self.derived2)
        state.move_to_processed(self.derived2)
        state.move_to_processed(self.goal)
        state.add_unprocessed(self.empty)
        
        # Check final state
        self.assertEqual(len(state.processed), 6)
        self.assertEqual(len(state.unprocessed), 1)
        self.assertEqual(state.unprocessed[0], self.empty)
        
        # Found contradiction
        has_empty = any(len(c.literals) == 0 for c in state.all_clauses)
        self.assertTrue(has_empty)


if __name__ == '__main__':
    unittest.main()