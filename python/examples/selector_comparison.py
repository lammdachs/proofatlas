#!/usr/bin/env python3
"""
Clause selector comparison example.

This example compares different clause selection strategies
on the same problem to see how they affect proof search.
"""

from proofatlas.core import (
    Constant, Predicate, Literal, Clause, Problem
)
from proofatlas.proofs import Proof, ProofState
from proofatlas.loops import BasicLoop
from proofatlas.selectors import RandomSelector


def create_test_problem():
    """Create a problem with multiple proof paths."""
    # Create a problem where selection order matters
    P = Predicate("P", 0)
    Q = Predicate("Q", 0)
    R = Predicate("R", 0)
    S = Predicate("S", 0)
    T = Predicate("T", 0)
    
    clauses = [
        # Multiple ways to derive contradictions
        Clause(Literal(P(), True)),                          # P
        Clause(Literal(P(), False), Literal(Q(), True)),     # ¬P ∨ Q
        Clause(Literal(P(), False), Literal(R(), True)),     # ¬P ∨ R
        Clause(Literal(Q(), False), Literal(S(), True)),     # ¬Q ∨ S
        Clause(Literal(R(), False), Literal(S(), True)),     # ¬R ∨ S
        Clause(Literal(S(), False), Literal(T(), True)),     # ¬S ∨ T
        Clause(Literal(T(), False)),                         # ¬T
        
        # Red herrings that don't lead to proof
        Clause(Literal(Q(), False), Literal(R(), False)),    # ¬Q ∨ ¬R
    ]
    
    return Problem(*clauses)


def run_with_fifo_selector(problem, loop):
    """Run proof search with FIFO (first-in-first-out) selection."""
    print("=== FIFO Selector (First-In-First-Out) ===\n")
    
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    proof = Proof(initial_state)
    
    steps = 0
    while proof.final_state.unprocessed and steps < 50:
        # FIFO: always select the first (oldest) unprocessed clause
        given_clause_idx = 0
        
        selected = proof.final_state.unprocessed[given_clause_idx]
        print(f"Step {steps}: Selected {selected}")
        
        proof = loop.step(proof, given_clause=given_clause_idx)
        
        if proof.final_state.contains_empty_clause:
            print(f"  → Found proof in {steps + 1} steps!")
            break
            
        steps += 1
    
    return proof, steps + 1


def run_with_lifo_selector(problem, loop):
    """Run proof search with LIFO (last-in-first-out) selection."""
    print("\n=== LIFO Selector (Last-In-First-Out) ===\n")
    
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    proof = Proof(initial_state)
    
    steps = 0
    while proof.final_state.unprocessed and steps < 50:
        # LIFO: always select the last (newest) unprocessed clause
        given_clause_idx = len(proof.final_state.unprocessed) - 1
        
        selected = proof.final_state.unprocessed[given_clause_idx]
        print(f"Step {steps}: Selected {selected}")
        
        proof = loop.step(proof, given_clause=given_clause_idx)
        
        if proof.final_state.contains_empty_clause:
            print(f"  → Found proof in {steps + 1} steps!")
            break
            
        steps += 1
    
    return proof, steps + 1


def run_with_shortest_selector(problem, loop):
    """Run proof search selecting shortest clauses first."""
    print("\n=== Shortest Clause Selector ===\n")
    
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    proof = Proof(initial_state)
    
    steps = 0
    while proof.final_state.unprocessed and steps < 50:
        # Select the clause with fewest literals
        min_len = float('inf')
        given_clause_idx = 0
        
        for i, clause in enumerate(proof.final_state.unprocessed):
            if len(clause.literals) < min_len:
                min_len = len(clause.literals)
                given_clause_idx = i
        
        selected = proof.final_state.unprocessed[given_clause_idx]
        print(f"Step {steps}: Selected {selected} (length {len(selected.literals)})")
        
        proof = loop.step(proof, given_clause=given_clause_idx)
        
        if proof.final_state.contains_empty_clause:
            print(f"  → Found proof in {steps + 1} steps!")
            break
            
        steps += 1
    
    return proof, steps + 1


def run_with_random_selector(problem, loop, seed=42):
    """Run proof search with random selection."""
    print(f"\n=== Random Selector (seed={seed}) ===\n")
    
    selector = RandomSelector(seed=seed)
    
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    proof = Proof(initial_state)
    
    steps = 0
    while proof.final_state.unprocessed and steps < 50:
        # Use the random selector
        given_clause_idx = selector.select(proof.final_state)
        
        if given_clause_idx is None:
            break
            
        selected = proof.final_state.unprocessed[given_clause_idx]
        print(f"Step {steps}: Selected {selected}")
        
        proof = loop.step(proof, given_clause=given_clause_idx)
        
        if proof.final_state.contains_empty_clause:
            print(f"  → Found proof in {steps + 1} steps!")
            break
            
        steps += 1
    
    return proof, steps + 1


def compare_selectors():
    """Compare all selectors on the same problem."""
    problem = create_test_problem()
    
    print("Test problem:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    print()
    
    # Create loop with consistent settings
    loop = BasicLoop(max_clause_size=10, forward_simplify=True)
    
    # Track results
    results = {}
    
    # Run with each selector
    proof1, steps1 = run_with_fifo_selector(problem, loop)
    results['FIFO'] = steps1
    
    proof2, steps2 = run_with_lifo_selector(problem, loop)
    results['LIFO'] = steps2
    
    proof3, steps3 = run_with_shortest_selector(problem, loop)
    results['Shortest'] = steps3
    
    proof4, steps4 = run_with_random_selector(problem, loop, seed=42)
    results['Random(42)'] = steps4
    
    proof5, steps5 = run_with_random_selector(problem, loop, seed=123)
    results['Random(123)'] = steps5
    
    # Summary
    print("\n=== Summary ===")
    print("Steps to find proof:")
    for name, steps in results.items():
        print(f"  {name}: {steps} steps")
    
    best = min(results.items(), key=lambda x: x[1])
    print(f"\nBest performer: {best[0]} with {best[1]} steps")


def main():
    """Main example function."""
    print("=== Clause Selector Comparison ===\n")
    print("This example compares different clause selection strategies")
    print("on the same problem to see how they affect proof search.\n")
    
    compare_selectors()


if __name__ == "__main__":
    main()