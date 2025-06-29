"""Integration tests for BasicLoop with RandomSelector."""

import json
import os
from pathlib import Path
import unittest

from proofatlas.core.logic import Constant, Variable, Predicate, Literal, Clause
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.proofs.serialization import ProofJSONEncoder
from proofatlas.loops.basic import BasicLoop
from proofatlas.selectors.random import RandomSelector


class TestBasicLoopRandomSelector(unittest.TestCase):
    """Test BasicLoop with RandomSelector integration."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data directory for saved proofs."""
        # Use new .data/proofs structure
        cls.test_dir = Path(__file__).parent.parent.parent / ".data" / "proofs" / "test_examples" / "random_selector"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
    
    def save_proof(self, proof: Proof, filename: str, description: str):
        """Save proof to JSON file with metadata."""
        filepath = self.test_dir / filename
        
        # Wrap proof with metadata
        data = {
            "description": description,
            "generator": "BasicLoop with RandomSelector",
            "proof": proof
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=ProofJSONEncoder, indent=2)
        
        print(f"\nSaved proof to: {filepath}")
        print(f"Steps: {len(proof.steps)}")
        print(f"Final processed: {len(proof.final_state.processed)}")
        print(f"Final unprocessed: {len(proof.final_state.unprocessed)}")
    
    def run_proof_search(self, clauses, max_steps=50, seed=42):
        """Run proof search with RandomSelector."""
        # Create initial state
        initial_state = ProofState(processed=[], unprocessed=clauses)
        proof = Proof(initial_state)
        
        # Create loop and selector
        loop = BasicLoop(max_clause_size=10, forward_simplify=True)
        selector = RandomSelector(seed=seed)
        
        # Run proof search
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
        
        return proof, contradiction_found
    
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
        
        proof, found = self.run_proof_search(clauses, max_steps=10, seed=12345)
        
        # Should find contradiction
        self.assertTrue(found, "Should find contradiction")
        
        # Save the proof
        self.save_proof(proof, "simple_proof.json", "Simple propositional proof")
    
    def test_propositional_proof(self):
        """Test BasicLoop with RandomSelector on a propositional problem and save proof."""
        # Create predicates and constants
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        a = Constant('a')
        
        # Create a more complex problem
        clauses = [
            Clause(Literal(P(a), True)),                         # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)),  # ~P(a) ∨ Q(a)
            Clause(Literal(Q(a), False), Literal(R(a), True)),  # ~Q(a) ∨ R(a)
            Clause(Literal(R(a), False))                         # ~R(a)
        ]
        
        proof, found = self.run_proof_search(clauses, max_steps=20, seed=42)
        
        # Should find contradiction
        self.assertTrue(found, "Should find contradiction")
        
        # Save the proof
        self.save_proof(proof, "propositional_proof.json", "Propositional logic proof with chain of implications")
    
    def test_fol_proof(self):
        """Test BasicLoop with RandomSelector on a first-order problem and save proof."""
        # Create predicates and constants
        P = Predicate('P', 1)
        Q = Predicate('Q', 2)
        a = Constant('a')
        b = Constant('b')
        x = Variable('X')
        y = Variable('Y')
        
        # Create clauses
        clauses = [
            Clause(Literal(P(a), True)),                         # P(a)
            Clause(Literal(P(x), False), Literal(Q(x, b), True)), # ∀x. ~P(x) ∨ Q(x,b)
            Clause(Literal(Q(y, b), False))                      # ∀y. ~Q(y,b)
        ]
        
        proof, found = self.run_proof_search(clauses, max_steps=20, seed=123)
        
        # Should find contradiction
        self.assertTrue(found, "Should find contradiction")
        
        # Save the proof
        self.save_proof(proof, "fol_proof.json", "First-order logic proof with unification")
    
    def test_larger_problem(self):
        """Test on a slightly larger problem that requires more search."""
        # Create predicates
        P = Predicate('P', 2)
        Q = Predicate('Q', 2)
        R = Predicate('R', 1)
        S = Predicate('S', 1)
        
        # Create constants and variables
        a = Constant('a')
        b = Constant('b')
        c = Constant('c')
        x = Variable('X')
        y = Variable('Y')
        
        # Create clauses - a more complex problem
        clauses = [
            Clause(Literal(P(a, b), True)),                      # P(a,b)
            Clause(Literal(P(b, c), True)),                      # P(b,c)
            Clause(Literal(P(x, y), False), Literal(Q(x, y), True)), # ∀x,y. ~P(x,y) ∨ Q(x,y)
            Clause(Literal(Q(a, x), False), Literal(R(x), True)),    # ∀x. ~Q(a,x) ∨ R(x)
            Clause(Literal(Q(y, c), False), Literal(S(y), True)),    # ∀y. ~Q(y,c) ∨ S(y)
            Clause(Literal(R(b), False)),                            # ~R(b)
            Clause(Literal(S(b), False))                             # ~S(b)
        ]
        
        proof, found = self.run_proof_search(clauses, max_steps=50, seed=999)
        
        # Should find contradiction
        self.assertTrue(found, "Should find contradiction")
        
        # Save the proof
        self.save_proof(proof, "larger_proof.json", "Larger problem requiring more search steps")
    
    def test_search_behavior_consistency(self):
        """Test that RandomSelector with same seed produces consistent results."""
        # Create simple problem
        P = Predicate('P', 1)
        a = Constant('a')
        
        clauses = [
            Clause(Literal(P(a), True)),
            Clause(Literal(P(a), False))
        ]
        
        # Run twice with same seed
        proof1, found1 = self.run_proof_search(clauses, seed=42)
        proof2, found2 = self.run_proof_search(clauses, seed=42)
        
        # Should have same results
        self.assertEqual(found1, found2)
        self.assertEqual(len(proof1.steps), len(proof2.steps))
        
        # Run with different seed - might have different path
        proof3, found3 = self.run_proof_search(clauses, seed=9999)
        self.assertEqual(found3, found1)  # Should still find proof
        # But path might be different (not checking step count)
    
    def test_no_proof_exists(self):
        """Test behavior when no proof exists."""
        # Create satisfiable problem
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        
        clauses = [
            Clause(Literal(P(a), True)),   # P(a)
            Clause(Literal(Q(a), True))    # Q(a)
            # No contradictions possible
        ]
        
        proof, found = self.run_proof_search(clauses, max_steps=10)
        
        # Should not find contradiction
        self.assertFalse(found)
        
        # Should have processed all clauses (saturation)
        self.assertEqual(len(proof.final_state.unprocessed), 0)


if __name__ == "__main__":
    unittest.main()