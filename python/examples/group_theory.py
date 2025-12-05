#!/usr/bin/env python3
"""
Group theory examples using ProofAtlas

This demonstrates proving theorems in group theory using
the superposition calculus for equality reasoning.
"""

from proofatlas import ProofState, saturate_step
import time


def prove_right_identity():
    """Prove that e is a right identity from group axioms"""
    print("Theorem: Right Identity")
    print("=" * 60)
    print("Given group axioms, prove: ∀x. x·e = x")
    print()
    
    state = ProofState()
    state.set_use_superposition(True)
    state.set_literal_selection("20")  # Select maximal literals
    
    # Group axioms
    axioms = """
    % Left identity
    cnf(left_identity, axiom, mult(e, X) = X).
    
    % Left inverse
    cnf(left_inverse, axiom, mult(inv(X), X) = e).
    
    % Associativity
    cnf(associativity, axiom, mult(mult(X, Y), Z) = mult(X, mult(Y, Z))).
    
    % Negated theorem: ∃x. x·e ≠ x
    cnf(not_right_identity, negated_conjecture, mult(c, e) != c).
    """
    
    state.add_clauses_from_tptp(axioms)
    
    print("Axioms:")
    for i in range(state.num_clauses() - 1):
        print(f"  {state.clause_to_string(i)}")
    print("\nNegated goal:")
    print(f"  {state.clause_to_string(state.num_clauses() - 1)}")
    
    # Run proof search
    print("\nRunning proof search...")
    start_time = time.time()
    
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        result = saturate_step(state, clause_selection="smallest")
        steps += 1
        
        if steps % 10 == 0:
            stats = state.get_statistics()
            print(f"  Step {steps}: {stats['total']} clauses, "
                  f"{stats['unprocessed']} in queue")
        
        if result['proof_found']:
            elapsed = time.time() - start_time
            print(f"\n✓ Proof found in {steps} steps ({elapsed:.3f}s)")
            print_proof_summary(state)
            return True
            
        if result['saturated']:
            print(f"\nSaturated after {steps} steps without proof")
            return False
    
    print(f"\nReached step limit ({max_steps})")
    return False


def prove_right_inverse():
    """Prove existence of right inverses"""
    print("\n\nTheorem: Right Inverse")
    print("=" * 60)
    print("Given group axioms, prove: ∀x. x·inv(x) = e")
    print()
    
    state = ProofState()
    state.set_use_superposition(True)
    state.set_literal_selection("20")  # Select maximal literals
    
    axioms = """
    % Standard group axioms
    cnf(left_identity, axiom, mult(e, X) = X).
    cnf(left_inverse, axiom, mult(inv(X), X) = e).
    cnf(associativity, axiom, mult(mult(X, Y), Z) = mult(X, mult(Y, Z))).
    
    % Additional axiom: we've already proven right identity
    cnf(right_identity, axiom, mult(X, e) = X).
    
    % Negated theorem
    cnf(not_right_inverse, negated_conjecture, mult(c, inv(c)) != e).
    """
    
    state.add_clauses_from_tptp(axioms)
    
    print("Starting proof search...")
    steps = 0
    
    while steps < 200:
        result = saturate_step(state, clause_selection="smallest")
        steps += 1
        
        if result['proof_found']:
            print(f"\n✓ Proof found in {steps} steps")
            return True
            
        if result['saturated']:
            print(f"\nSaturated after {steps} steps")
            return False
    
    return False


def prove_inverse_involution():
    """Prove that inv(inv(x)) = x"""
    print("\n\nTheorem: Inverse Involution")
    print("=" * 60)
    print("Prove: ∀x. inv(inv(x)) = x")
    print()
    
    state = ProofState()
    state.set_use_superposition(True)
    
    # Use more complete axioms
    axioms = """
    % Complete group axioms
    cnf(left_identity, axiom, mult(e, X) = X).
    cnf(right_identity, axiom, mult(X, e) = X).
    cnf(left_inverse, axiom, mult(inv(X), X) = e).
    cnf(right_inverse, axiom, mult(X, inv(X)) = e).
    cnf(associativity, axiom, mult(mult(X, Y), Z) = mult(X, mult(Y, Z))).
    
    % Negated theorem
    cnf(not_involution, negated_conjecture, inv(inv(c)) != c).
    """
    
    state.add_clauses_from_tptp(axioms)
    
    # Try different strategies
    for strategy in ["smallest", "age"]:
        print(f"\nTrying with {strategy} clause selection...")
        
        steps = 0
        while steps < 100:
            result = saturate_step(state, clause_selection=strategy)
            steps += 1
            
            if result['proof_found']:
                print(f"✓ Proof found in {steps} steps with {strategy}")
                return True
                
            if result['saturated']:
                print(f"Saturated with {strategy}")
                break
    
    return False


def prove_unique_identity():
    """Prove uniqueness of identity element"""
    print("\n\nTheorem: Unique Identity")
    print("=" * 60)
    print("Prove: If e' is another identity, then e' = e")
    print()
    
    state = ProofState()
    state.set_use_superposition(True)
    
    axioms = """
    % Standard identity axioms for e
    cnf(left_identity_e, axiom, mult(e, X) = X).
    cnf(right_identity_e, axiom, mult(X, e) = X).
    
    % Another element e_prime acts as identity
    cnf(left_identity_ep, axiom, mult(e_prime, X) = X).
    cnf(right_identity_ep, axiom, mult(X, e_prime) = X).
    
    % Negated theorem: e_prime != e
    cnf(not_equal, negated_conjecture, e_prime != e).
    """
    
    state.add_clauses_from_tptp(axioms)
    
    # This should be quick
    for step in range(20):
        result = saturate_step(state)
        
        if result['proof_found']:
            print(f"✓ Proof found in {step + 1} steps")
            print("\nKey insight: e = mult(e, e_prime) = e_prime")
            return True
            
        if result['saturated']:
            break
    
    return False


def print_proof_summary(state: ProofState):
    """Print a summary of the proof"""
    trace = state.get_proof_trace()
    
    print("\nProof summary:")
    print("-" * 40)
    
    # Find key steps
    key_steps = []
    for step in trace:
        if step.rule_name in ["superposition", "equality_resolution"]:
            if len(step.clause_string) < 50:  # Short enough to be interesting
                key_steps.append(step)
    
    # Show last few key steps
    for step in key_steps[-5:]:
        print(f"[{step.clause_id}] {step.clause_string}")
        if step.parent_ids:
            print(f"    by {step.rule_name} from [{','.join(map(str, step.parent_ids))}]")
    
    if trace:
        print(f"\n[{trace[-1].clause_id}] ⊥ (contradiction found)")


def benchmark_group_theorems():
    """Run benchmarks on various group theory theorems"""
    print("\n\nGroup Theory Benchmark")
    print("=" * 60)
    
    theorems = [
        ("Right Identity", prove_right_identity),
        ("Right Inverse", prove_right_inverse),
        ("Inverse Involution", prove_inverse_involution),
        ("Unique Identity", prove_unique_identity),
    ]
    
    results = []
    for name, prove_func in theorems:
        print(f"\nTesting {name}...")
        start = time.time()
        success = prove_func()
        elapsed = time.time() - start
        results.append((name, success, elapsed))
    
    # Summary
    print("\n\nBenchmark Summary")
    print("=" * 60)
    print(f"{'Theorem':<20} {'Result':<10} {'Time (s)':<10}")
    print("-" * 40)
    for name, success, elapsed in results:
        result = "✓ Proved" if success else "✗ Failed"
        print(f"{name:<20} {result:<10} {elapsed:<10.3f}")


if __name__ == "__main__":
    # Run individual proofs
    prove_right_identity()
    prove_right_inverse()
    prove_inverse_involution()
    prove_unique_identity()
    
    # Run benchmark
    benchmark_group_theorems()