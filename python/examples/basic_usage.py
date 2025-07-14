#!/usr/bin/env python3
"""
Basic usage examples for ProofAtlas Python interface
"""

from proofatlas import ProofState, saturate_step


def simple_propositional_proof():
    """Prove a simple propositional theorem"""
    print("Simple Propositional Logic Example")
    print("=" * 50)
    
    # Create proof state
    state = ProofState()
    
    # Add a simple problem: Modus Ponens
    # From P and P→Q, derive Q
    state.add_clauses_from_tptp("""
    cnf(premise1, axiom, p).
    cnf(premise2, axiom, ~p | q).
    cnf(goal, negated_conjecture, ~q).
    """)
    
    print("Problem: From P and P→Q, prove Q")
    print("\nInitial clauses:")
    for i in range(state.num_clauses()):
        print(f"  [{i}] {state.clause_to_string(i)}")
    
    # Run saturation
    steps = 0
    while not state.contains_empty_clause() and steps < 10:
        result = saturate_step(state)
        steps += 1
        
        if result['saturated']:
            print("\nSaturated without proof!")
            break
            
        print(f"\nStep {steps}: Given clause [{result['given_id']}]")
        print(f"  Generated {len(result['new_clauses'])} new clauses")
        
        for new_id in result['new_clauses']:
            print(f"  [{new_id}] {state.clause_to_string(new_id)}")
    
    if state.contains_empty_clause():
        print("\n✓ Proof found!")


def first_order_example():
    """Example with quantifiers"""
    print("\n\nFirst-Order Logic Example")
    print("=" * 50)
    
    state = ProofState()
    
    # All men are mortal, Socrates is a man, therefore Socrates is mortal
    state.add_clauses_from_tptp("""
    cnf(all_men_mortal, axiom, ~man(X) | mortal(X)).
    cnf(socrates_man, axiom, man(socrates)).
    cnf(not_mortal, negated_conjecture, ~mortal(socrates)).
    """)
    
    print("Problem: All men are mortal, Socrates is a man ⊢ Socrates is mortal")
    
    # Show initial state
    print("\nInitial clauses:")
    for i in range(state.num_clauses()):
        info = state.get_clause_info(i)
        print(f"  [{i}] {info.clause_string}")
        if info.variables:
            print(f"      Variables: {', '.join(info.variables)}")
    
    # Run proof search
    steps = 0
    while steps < 10:
        result = saturate_step(state, clause_selection="smallest")
        steps += 1
        
        if result['proof_found']:
            print(f"\n✓ Proof found in {steps} steps!")
            break
        elif result['saturated']:
            print("\nSaturated without proof!")
            break


def equality_example():
    """Example using equality reasoning"""
    print("\n\nEquality Reasoning Example")
    print("=" * 50)
    
    state = ProofState()
    state.set_use_superposition(True)  # Enable equality reasoning
    
    # Prove transitivity of equality
    # ProofAtlas automatically orients equalities for optimal performance
    state.add_clauses_from_tptp("""
    cnf(eq1, axiom, a = b).
    cnf(eq2, axiom, b = c).
    cnf(not_eq, negated_conjecture, a != c).
    """)
    
    print("Problem: From a=b and b=c, prove a=c")
    
    steps = 0
    while steps < 20:
        result = saturate_step(state)
        steps += 1
        
        if result['proof_found']:
            print(f"\n✓ Proof found in {steps} steps!")
            
            # Show the proof
            print("\nProof trace:")
            for step in state.get_proof_trace():
                print(f"[{step.clause_id}] {step.clause_string}")
                if step.parent_ids:
                    print(f"    from {step.parent_ids} by {step.rule_name}")
            break
        elif result['saturated']:
            print("\nSaturated without proof!")
            break


def statistics_demo():
    """Show statistics during proof search"""
    print("\n\nStatistics Demo")
    print("=" * 50)
    
    state = ProofState()
    
    # A slightly harder problem
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(X) | q(X)).
    cnf(c2, axiom, ~p(a) | r(a)).
    cnf(c3, axiom, ~q(b) | r(b)).
    cnf(c4, axiom, a = b).
    cnf(c5, axiom, ~r(a)).
    cnf(c6, axiom, ~r(b)).
    """)
    
    print("Running proof search with statistics...")
    
    state.set_use_superposition(True)
    
    for step in range(1, 21):
        result = saturate_step(state, clause_selection="smallest")
        
        stats = state.get_statistics()
        print(f"\nStep {step}: {stats['total']} clauses "
              f"({stats['processed']} processed, {stats['unprocessed']} queued)")
        
        if result['proof_found']:
            print("\n✓ Proof found!")
            print(f"Final statistics: {stats}")
            break
        elif result['saturated']:
            print("\nSaturated without proof")
            break


if __name__ == "__main__":
    simple_propositional_proof()
    first_order_example()
    equality_example()
    statistics_demo()