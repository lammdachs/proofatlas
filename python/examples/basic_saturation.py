#!/usr/bin/env python3
"""
Basic saturation loop example.

This example demonstrates how to use the BasicLoop to find a proof
by saturation using the given clause algorithm.
"""

from proofatlas.core import (
    Constant, Predicate, Literal, Clause, Problem
)
from proofatlas.proofs import Proof, ProofState
from proofatlas.loops import BasicLoop


def create_simple_problem():
    """Create a simple propositional problem: P, P→Q, Q→R ⊢ R"""
    # Constants and predicates
    P = Predicate("P", 0)
    Q = Predicate("Q", 0)
    R = Predicate("R", 0)
    
    # Create clauses:
    # 1. P
    # 2. ¬P ∨ Q  (from P → Q)
    # 3. ¬Q ∨ R  (from Q → R)
    # 4. ¬R      (negated goal)
    
    clause1 = Clause(Literal(P(), True))
    clause2 = Clause(Literal(P(), False), Literal(Q(), True))
    clause3 = Clause(Literal(Q(), False), Literal(R(), True))
    clause4 = Clause(Literal(R(), False))
    
    return Problem(clause1, clause2, clause3, clause4)


def run_saturation(problem):
    """Run the saturation loop on a problem."""
    print(f"Problem with {len(problem.clauses)} clauses:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    print()
    
    # Create initial proof state
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    
    # Create proof object
    proof = Proof(initial_state)
    
    # Create saturation loop
    loop = BasicLoop(max_clause_size=10, forward_simplify=True)
    
    # Run saturation steps
    step = 0
    while proof.final_state.unprocessed:
        # Select the first unprocessed clause (FIFO)
        given_clause_idx = 0
        
        print(f"Step {step}:")
        print(f"  Given clause: {proof.final_state.unprocessed[given_clause_idx]}")
        
        # Apply one step of the loop
        proof = loop.step(proof, given_clause=given_clause_idx)
        
        print(f"  Generated {len(proof.steps[-1].applied_rules)} rule applications")
        
        # Check if we found a contradiction
        if proof.final_state.contains_empty_clause:
            print(f"  Found contradiction! ⊥")
            break
            
        print(f"  Processed: {len(proof.final_state.processed)}")
        print(f"  Unprocessed: {len(proof.final_state.unprocessed)}")
        print()
        
        step += 1
        
        # Prevent infinite loops
        if step > 100:
            print("Reached step limit")
            break
    
    return proof


def main():
    """Main example function."""
    print("=== Basic Saturation Example ===\n")
    
    # Create and solve a simple problem
    problem = create_simple_problem()
    proof = run_saturation(problem)
    
    print("\n=== Final Statistics ===")
    print(f"Total steps: {len(proof.steps)}")
    print(f"Final processed clauses: {len(proof.final_state.processed)}")
    print(f"Proof found: {proof.final_state.contains_empty_clause}")
    
    if proof.final_state.contains_empty_clause:
        print("\nProof trace:")
        for i, step in enumerate(proof.steps):
            if step.applied_rules:
                print(f"  Step {i}: Selected clause {step.selected_clause}")
                for rule_app in step.applied_rules:
                    if rule_app.generated_clauses:
                        for clause in rule_app.generated_clauses:
                            print(f"    {rule_app.rule_name}: {clause}")


if __name__ == "__main__":
    main()