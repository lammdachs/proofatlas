#!/usr/bin/env python3
"""Example of using RandomSelector with BasicLoop."""

import sys
sys.path.insert(0, 'src')

from proofatlas.core.logic import Constant, Predicate, Literal, Clause
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.loops.basic import BasicLoop
from proofatlas.selectors.random import RandomSelector


def main():
    """Run a simple proof search with random clause selection."""
    # Create predicates and constants
    P = Predicate('P', 1)
    Q = Predicate('Q', 1)
    a = Constant('a')
    
    # Create clauses: P(a), ~P(a) ∨ Q(a), ~Q(a)
    # This should derive a contradiction
    clauses = [
        Clause(Literal(P(a), True)),           # P(a)
        Clause(Literal(P(a), False), Literal(Q(a), True)),  # ~P(a) ∨ Q(a)
        Clause(Literal(Q(a), False))           # ~Q(a)
    ]
    
    # Create initial state
    initial_state = ProofState(processed=[], unprocessed=clauses)
    proof = Proof(initial_state)
    
    # Create loop and selector
    loop = BasicLoop(max_clause_size=10, forward_simplify=True)
    selector = RandomSelector(seed=42)  # Use seed for reproducibility
    
    print("Initial clauses:")
    for i, clause in enumerate(clauses):
        print(f"  {i}: {clause}")
    print()
    
    # Run proof search
    step_count = 0
    max_steps = 20
    
    while proof.final_state.unprocessed and step_count < max_steps:
        # Select clause
        idx = selector.select(proof.final_state)
        if idx is None:
            print("No more clauses to select")
            break
        
        selected_clause = proof.final_state.unprocessed[idx]
        print(f"Step {step_count}: Selected clause {idx}: {selected_clause}")
        
        # Apply loop step
        proof = loop.step(proof, idx)
        
        # Check for contradiction
        all_clauses = proof.final_state.processed + proof.final_state.unprocessed
        if any(len(c.literals) == 0 for c in all_clauses):
            print("\nContradiction found!")
            print(f"Proof completed in {step_count + 1} steps")
            break
        
        step_count += 1
    
    if step_count >= max_steps:
        print(f"\nReached maximum steps ({max_steps})")
    
    # Print final state
    print("\nFinal state:")
    print("Processed clauses:")
    for i, clause in enumerate(proof.final_state.processed):
        print(f"  {i}: {clause}")
    print("Unprocessed clauses:")
    for i, clause in enumerate(proof.final_state.unprocessed):
        print(f"  {i}: {clause}")


if __name__ == '__main__':
    main()