#!/usr/bin/env python3
"""
Basic usage examples for ProofAtlas Python interface
"""

from proofatlas import ProofAtlas


def example_simple_proof():
    """Prove a simple contradiction: P(a), ~P(X)|Q(X), ~Q(a) |- false"""
    print("=" * 60)
    print("Example 1: Simple Proof")
    print("=" * 60)

    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(fact, axiom, p(a)).
    cnf(rule, axiom, ~p(X) | q(X)).
    cnf(goal, negated_conjecture, ~q(a)).
    """)

    stats = state.statistics()
    print(f"Loaded {stats['total']} clauses")

    proof_found, status = state.prove(timeout=10.0)

    if proof_found:
        print(f"Proof found!")
        steps = state.proof_steps()
        print(f"Proof has {len(steps)} steps:")
        for step in steps:
            parents = f" (from {step.parent_ids}, {step.rule_name})" if step.parent_ids else ""
            print(f"  [{step.clause_id}] {step.clause_string}{parents}")
    else:
        print(f"No proof found: {status}")


def example_equality_reasoning():
    """Prove using equality: a=b, f(a)!=f(b) |- false"""
    print("\n" + "=" * 60)
    print("Example 2: Equality Reasoning (Superposition)")
    print("=" * 60)

    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(eq1, axiom, a = b).
    cnf(neq, negated_conjecture, f(a) != f(b)).
    """)

    proof_found, status = state.prove(timeout=10.0)

    if proof_found:
        print("Proof found using superposition!")
        for step in state.proof_steps():
            print(f"  [{step.clause_id}] {step.clause_string} ({step.rule_name})")
    else:
        print(f"No proof: {status}")


def example_statistics():
    """Show statistics from a proof attempt"""
    print("\n" + "=" * 60)
    print("Example 3: Statistics")
    print("=" * 60)

    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(X) | r(X)).
    cnf(c4, negated_conjecture, ~r(a)).
    """)

    proof_found, status = state.prove(timeout=10.0)

    stats = state.statistics()
    print(f"Status: {'proof' if proof_found else status}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    # All steps vs proof steps
    all_steps = state.all_steps()
    proof_steps = state.proof_steps()
    print(f"\n  Total derivation steps: {len(all_steps)}")
    print(f"  Steps in proof: {len(proof_steps)}")


def example_literal_selection():
    """Try different literal selection strategies"""
    print("\n" + "=" * 60)
    print("Example 4: Literal Selection Strategies")
    print("=" * 60)

    tptp = """
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(X) | r(X)).
    cnf(c4, negated_conjecture, ~r(a)).
    """

    strategies = {0: "select all", 20: "maximal", 21: "unique/neg/max", 22: "neg/max"}

    for sel_id, name in strategies.items():
        state = ProofAtlas()
        state.add_clauses_from_tptp(tptp)
        proof_found, status = state.prove(timeout=10.0, literal_selection=sel_id)
        steps = len(state.proof_steps()) if proof_found else 0
        result = f"proof ({steps} steps)" if proof_found else status
        print(f"  Strategy {sel_id} ({name}): {result}")


if __name__ == "__main__":
    example_simple_proof()
    example_equality_reasoning()
    example_statistics()
    example_literal_selection()
    print("\nAll examples completed!")
