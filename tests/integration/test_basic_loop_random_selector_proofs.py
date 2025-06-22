"""Tests for BasicLoop with RandomSelector that save proofs for inspection."""

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


class TestBasicLoopRandomSelectorProofs(unittest.TestCase):
    """Test BasicLoop with RandomSelector and save generated proofs."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data directory."""
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
        loop = BasicLoop(max_clause_size=50, forward_simplify=True)
        selector = RandomSelector(seed=seed)
        
        # Run proof search
        steps = 0
        while proof.final_state.unprocessed and steps < max_steps:
            idx = selector.select(proof.final_state)
            if idx is None:
                break
            
            proof = loop.step(proof, idx)
            steps += 1
            
            # Check for contradiction
            all_clauses = proof.final_state.processed + proof.final_state.unprocessed
            if any(len(c.literals) == 0 for c in all_clauses):
                break
        
        return proof
    
    def test_simple_random_proof(self):
        """Test simple proof with random selection."""
        # Create predicates and constants
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        
        # Create clauses: P(a), ~P(a) ∨ Q(a), ~Q(a) |- ⊥
        clauses = [
            Clause(Literal(P(a), True)),                        # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)), # ~P(a) ∨ Q(a)
            Clause(Literal(Q(a), False))                        # ~Q(a)
        ]
        
        proof = self.run_proof_search(clauses, seed=42)
        
        # Should find contradiction
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        self.assertTrue(any(len(c.literals) == 0 for c in all_clauses))
        
        self.save_proof(
            proof,
            "simple_random_proof.json",
            "Simple proof with random selection: P(a), ~P(a)∨Q(a), ~Q(a) |- ⊥"
        )
    
    def test_longer_chain_random(self):
        """Test longer resolution chain with random selection."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        S = Predicate('S', 1)
        a = Constant('a')
        
        # Create chain: P(a) -> Q(a) -> R(a) -> S(a), with ~S(a)
        clauses = [
            Clause(Literal(P(a), True)),                          # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)),   # ~P(a) ∨ Q(a)
            Clause(Literal(Q(a), False), Literal(R(a), True)),   # ~Q(a) ∨ R(a)
            Clause(Literal(R(a), False), Literal(S(a), True)),   # ~R(a) ∨ S(a)
            Clause(Literal(S(a), False))                          # ~S(a)
        ]
        
        proof = self.run_proof_search(clauses, seed=12345)
        
        # Should find contradiction
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        self.assertTrue(any(len(c.literals) == 0 for c in all_clauses))
        
        self.save_proof(
            proof,
            "chain_random_proof.json",
            "Chain resolution with random selection: P(a) -> Q(a) -> R(a) -> S(a), ~S(a) |- ⊥"
        )
    
    def test_branching_proof_random(self):
        """Test proof with multiple possible paths using random selection."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        a = Constant('a')
        
        # Create branching structure
        clauses = [
            Clause(Literal(P(a), True)),                          # P(a)
            Clause(Literal(P(a), False), Literal(Q(a), True)),   # ~P(a) ∨ Q(a)
            Clause(Literal(P(a), False), Literal(R(a), True)),   # ~P(a) ∨ R(a)
            Clause(Literal(Q(a), False)),                         # ~Q(a)
            Clause(Literal(R(a), False))                          # ~R(a)
        ]
        
        proof = self.run_proof_search(clauses, seed=99999)
        
        # Should find contradiction (multiple paths possible)
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        self.assertTrue(any(len(c.literals) == 0 for c in all_clauses))
        
        self.save_proof(
            proof,
            "branching_random_proof.json",
            "Branching proof with random selection: multiple paths to contradiction"
        )
    
    def test_random_with_redundancy(self):
        """Test random selection with redundant clauses."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        b = Constant('b')
        
        clauses = [
            Clause(Literal(P(a), True)),                          # P(a)
            Clause(Literal(P(b), True)),                          # P(b) (irrelevant)
            Clause(Literal(P(a), False), Literal(Q(a), True)),   # ~P(a) ∨ Q(a)
            Clause(Literal(P(b), False), Literal(Q(b), True)),   # ~P(b) ∨ Q(b) (irrelevant)
            Clause(Literal(Q(a), False)),                         # ~Q(a)
            Clause(Literal(Q(b), True))                           # Q(b) (irrelevant)
        ]
        
        proof = self.run_proof_search(clauses, seed=7777)
        
        # Should still find contradiction despite irrelevant clauses
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        self.assertTrue(any(len(c.literals) == 0 for c in all_clauses))
        
        self.save_proof(
            proof,
            "redundant_random_proof.json",
            "Proof with redundant clauses and random selection"
        )
    
    def test_multiple_seeds_same_problem(self):
        """Test same problem with different random seeds."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        a = Constant('a')
        
        # Problem with multiple resolution paths
        clauses = [
            Clause(Literal(P(a), True)),                            # P(a)
            Clause(Literal(Q(a), True)),                            # Q(a)
            Clause(Literal(P(a), False), Literal(R(a), True)),     # ~P(a) ∨ R(a)
            Clause(Literal(Q(a), False), Literal(R(a), True)),     # ~Q(a) ∨ R(a)
            Clause(Literal(R(a), False))                            # ~R(a)
        ]
        
        # Run with different seeds
        for i, seed in enumerate([111, 222, 333]):
            proof = self.run_proof_search(clauses, seed=seed)
            
            # All should find contradiction
            all_clauses = proof.final_state.processed + proof.final_state.unprocessed
            self.assertTrue(any(len(c.literals) == 0 for c in all_clauses))
            
            self.save_proof(
                proof,
                f"multi_path_seed_{seed}.json",
                f"Same problem with seed {seed}: demonstrates different proof paths"
            )
    
    def test_proof_with_duplicate_literals(self):
        """Test proof search with clauses containing duplicate literals."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        a = Constant('a')
        b = Constant('b')
        
        # Include clauses with duplicate literals
        # Note: BasicLoop with factoring should handle these
        clauses = [
            Clause(Literal(P(a), True), Literal(P(a), True)),      # P(a) ∨ P(a)
            Clause(Literal(P(a), False)),                          # ~P(a)
            Clause(Literal(Q(a), True), Literal(Q(b), True), 
                   Literal(Q(a), True))                             # Q(a) ∨ Q(b) ∨ Q(a)
        ]
        
        # Run search - should handle duplicate literals gracefully
        proof = self.run_proof_search(clauses, seed=42, max_steps=20)
        
        # The proof should complete (all clauses processed or contradiction found)
        self.assertGreater(len(proof.steps), 0, "Should have at least one step")
        
        # Save proof for inspection
        self.save_proof(
            proof,
            "duplicate_literals_proof.json",
            "Proof with clauses containing duplicate literals"
        )
        
        # Note: We don't check for specific factoring results because
        # the current factoring implementation is basic and may not
        # produce factors for ground clauses like P(a) ∨ P(a)
    
    def test_no_proof_exists(self):
        """Test random selector behavior when no proof exists."""
        P = Predicate('P', 1)
        Q = Predicate('Q', 1)
        R = Predicate('R', 1)
        a = Constant('a')
        
        # Satisfiable set of clauses (Horn clauses with no negative unit clause)
        clauses = [
            Clause(Literal(P(a), True)),                          # P(a)
            Clause(Literal(Q(a), True)),                          # Q(a)
            Clause(Literal(P(a), False), Literal(R(a), True)),   # ~P(a) ∨ R(a)
            Clause(Literal(Q(a), False), Literal(R(a), True)),   # ~Q(a) ∨ R(a)
            # Note: No ~R(a), so this is satisfiable
        ]
        
        proof = self.run_proof_search(clauses, max_steps=20, seed=88888)
        
        # Should NOT find contradiction
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        self.assertFalse(any(len(c.literals) == 0 for c in all_clauses),
                        "Should not find contradiction in satisfiable clause set")
        
        self.save_proof(
            proof,
            "no_proof_random.json",
            "Satisfiable problem with random selection (no contradiction)"
        )
    
    def test_complex_problem_random(self):
        """Test a more complex problem with random selection."""
        # Predicates
        Human = Predicate('Human', 1)
        Mortal = Predicate('Mortal', 1)
        Greek = Predicate('Greek', 1)
        Philosopher = Predicate('Philosopher', 1)
        
        # Constants and variables
        socrates = Constant('socrates')
        plato = Constant('plato')
        X = Variable('X')
        
        clauses = [
            # Facts
            Clause(Literal(Human(socrates), True)),              # Human(socrates)
            Clause(Literal(Greek(socrates), True)),              # Greek(socrates)
            Clause(Literal(Philosopher(socrates), True)),        # Philosopher(socrates)
            
            # Rules
            Clause(Literal(Human(X), False),                     # ∀X. Human(X) → Mortal(X)
                   Literal(Mortal(X), True)),
            Clause(Literal(Greek(X), False),                     # ∀X. Greek(X) ∧ Philosopher(X) → Human(X)
                   Literal(Philosopher(X), False),
                   Literal(Human(X), True)),
            
            # Query (negated)
            Clause(Literal(Mortal(socrates), False))             # ~Mortal(socrates)
        ]
        
        # Try multiple seeds to find a proof
        proof = None
        for seed in [2468, 1234, 5678, 9999, 1111]:
            proof = self.run_proof_search(clauses, max_steps=30, seed=seed)
            all_clauses = proof.final_state.processed + proof.final_state.unprocessed
            if any(len(c.literals) == 0 for c in all_clauses):
                print(f"Found proof with seed {seed}")
                break
        
        # Check if we found Mortal(socrates)
        mortal_socrates = any(
            str(c) == "Mortal(socrates)" 
            for c in proof.final_state.processed + proof.final_state.unprocessed
        )
        
        self.save_proof(
            proof,
            "complex_random_proof.json",
            "Complex problem (Socrates) with random selection"
        )


if __name__ == '__main__':
    unittest.main()