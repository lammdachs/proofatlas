"""Tests for RandomSelector."""

import unittest

from proofatlas.core.logic import Constant, Predicate, Literal, Clause
from proofatlas.proofs.state import ProofState
from proofatlas.selectors.random import RandomSelector


class TestRandomSelector(unittest.TestCase):
    """Test cases for RandomSelector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create some test clauses
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        b = Constant('b')
        
        self.clauses = [
            Clause(Literal(P(a), True)),
            Clause(Literal(P(b), False)),
            Clause(Literal(Q(a), True), Literal(Q(b), False)),
        ]
    
    def test_random_selector_empty_state(self):
        """Test that RandomSelector returns None for empty state."""
        selector = RandomSelector(seed=42)
        empty_state = ProofState(processed=[], unprocessed=[])
        
        result = selector.select(empty_state)
        self.assertIsNone(result)
    
    def test_random_selector_single_clause(self):
        """Test RandomSelector with single unprocessed clause."""
        selector = RandomSelector(seed=42)
        state = ProofState(processed=[], unprocessed=[self.clauses[0]])
        
        result = selector.select(state)
        self.assertEqual(result, 0)
    
    def test_random_selector_multiple_clauses(self):
        """Test RandomSelector with multiple unprocessed clauses."""
        selector = RandomSelector(seed=42)
        state = ProofState(processed=[], unprocessed=self.clauses)
        
        # Select multiple times to check randomness
        selections = []
        for _ in range(100):
            idx = selector.select(state)
            self.assertIsNotNone(idx)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(self.clauses))
            selections.append(idx)
        
        # Check that we selected different indices
        unique_selections = set(selections)
        self.assertGreater(len(unique_selections), 1)
        
        # Check all valid indices were selected at least once
        self.assertEqual(unique_selections, {0, 1, 2})
    
    def test_random_selector_deterministic_with_seed(self):
        """Test that RandomSelector is deterministic with same seed."""
        state = ProofState(processed=[], unprocessed=self.clauses)
        
        # Create two selectors with same seed
        selector1 = RandomSelector(seed=12345)
        selector2 = RandomSelector(seed=12345)
        
        # They should produce same sequence
        for _ in range(10):
            idx1 = selector1.select(state)
            idx2 = selector2.select(state)
            self.assertEqual(idx1, idx2)
    
    def test_random_selector_different_with_different_seeds(self):
        """Test that different seeds produce different sequences."""
        state = ProofState(processed=[], unprocessed=self.clauses * 10)  # More clauses for variety
        
        selector1 = RandomSelector(seed=111)
        selector2 = RandomSelector(seed=222)
        
        # Collect sequences
        seq1 = [selector1.select(state) for _ in range(20)]
        seq2 = [selector2.select(state) for _ in range(20)]
        
        # Sequences should be different (with high probability)
        self.assertNotEqual(seq1, seq2)
    
    def test_random_selector_name(self):
        """Test that selector reports correct name."""
        selector = RandomSelector()
        self.assertEqual(selector.name, "random")
    
    def test_random_selector_run_method(self):
        """Test that run() method works as alias for select()."""
        selector = RandomSelector(seed=42)
        state = ProofState(processed=[], unprocessed=self.clauses)
        
        # Reset seed for consistency
        selector1 = RandomSelector(seed=42)
        selector2 = RandomSelector(seed=42)
        
        # select() and run() should give same results
        idx1 = selector1.select(state)
        idx2 = selector2.run(state)
        self.assertEqual(idx1, idx2)


if __name__ == '__main__':
    unittest.main()