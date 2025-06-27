"""Integration tests for RandomSelector with BasicLoop."""

import unittest

from proofatlas.core.logic import Constant, Predicate, Literal, Clause
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.loops.basic import BasicLoop
from proofatlas.selectors.random import RandomSelector


class TestRandomSelectorWithLoop(unittest.TestCase):
    """Test RandomSelector integration with BasicLoop."""
    
    def test_simple_proof_search(self):
        """Test that RandomSelector can guide a simple proof to completion."""
        # Create predicates and constants
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        
        # Create clauses that lead to contradiction
        clauses = [
            Clause(Literal(P(a), True)),                        # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)), # ~P(a) ∨ Q(a)
            Clause(Literal(Q(a), False))                        # ~Q(a)
        ]
        
        # Create initial state and proof
        initial_state = ProofState(processed=[], unprocessed=clauses)
        proof = Proof(initial_state)
        
        # Create loop and selector
        loop = BasicLoop(max_clause_size=10, forward_simplify=True)
        selector = RandomSelector(seed=12345)
        
        # Run proof search
        max_steps = 10
        contradiction_found = False
        
        for step in range(max_steps):
            idx = selector.select(proof.final_state)
            if idx is None:
                break
            
            proof = loop.step(proof, idx)
            
            # Check for empty clause
            all_clauses = proof.final_state.processed + proof.final_state.unprocessed
            if any(len(c.literals) == 0 for c in all_clauses):
                contradiction_found = True
                break
        
        self.assertTrue(contradiction_found, "Should find contradiction")
        self.assertLess(step, max_steps - 1, "Should find proof before max steps")
    
    def test_random_exploration(self):
        """Test that RandomSelector explores different proof paths with different seeds."""
        # Create a more complex problem
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        a = Constant('a')
        
        clauses = [
            Clause(Literal(P(a), True)),                          # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)),   # ~P(a) ∨ Q(a)
            Clause(Literal(Q(a), False), Literal(R(a), True)),   # ~Q(a) ∨ R(a)
            Clause(Literal(R(a), False)),                         # ~R(a)
            Clause(Literal(P(a), False), Literal(R(a), True)),   # ~P(a) ∨ R(a) (redundant path)
        ]
        
        # Try with different seeds
        paths = []
        for seed in [111, 222, 333]:
            initial_state = ProofState(processed=[], unprocessed=clauses)
            proof = Proof(initial_state)
            
            loop = BasicLoop(max_clause_size=10, forward_simplify=True)
            selector = RandomSelector(seed=seed)
            
            # Track selection sequence
            selections = []
            for _ in range(10):
                idx = selector.select(proof.final_state)
                if idx is None:
                    break
                selections.append(idx)
                proof = loop.step(proof, idx)
                
                # Stop if contradiction found
                all_clauses = proof.final_state.processed + proof.final_state.unprocessed
                if any(len(c.literals) == 0 for c in all_clauses):
                    break
            
            paths.append(selections)
        
        # Different seeds should explore different paths (with high probability)
        self.assertFalse(all(p == paths[0] for p in paths), 
                        "Different seeds should produce different exploration paths")
    
    def test_selector_with_empty_result(self):
        """Test RandomSelector behavior when no clauses remain."""
        # Start with one clause that leads nowhere
        P = Predicate('P', 1)
        a = Constant('a')
        
        initial_state = ProofState(processed=[], unprocessed=[Clause(Literal(P(a), True))])
        proof = Proof(initial_state)
        
        loop = BasicLoop()
        selector = RandomSelector()
        
        # Process the only clause
        idx = selector.select(proof.final_state)
        self.assertEqual(idx, 0)
        
        proof = loop.step(proof, idx)
        
        # Now no clauses remain in unprocessed
        idx = selector.select(proof.final_state)
        self.assertIsNone(idx)


if __name__ == '__main__':
    unittest.main()