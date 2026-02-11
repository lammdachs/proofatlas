#!/usr/bin/env python3
"""
Interactive proof exploration with ProofAtlas

This example shows how to prove problems and inspect results.
"""

from proofatlas import ProofAtlas


def prove_and_inspect():
    """Prove a problem and inspect the proof trace"""
    print("=" * 60)
    print("Proof Inspection Demo")
    print("=" * 60)

    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(X) | r(X)).
    cnf(c4, negated_conjecture, ~r(a)).
    """)

    stats = state.statistics()
    print(f"Initial clauses: {stats['total']}")

    proof_found, status = state.prove(timeout=10.0, literal_selection=21)

    if proof_found:
        print(f"\nProof found!")
        stats = state.statistics()
        print(f"Final state: {stats['total']} total, {stats['processed']} processed")

        print("\nProof trace:")
        for step in state.proof_steps():
            if step.parent_ids:
                print(f"  [{step.clause_id}] {step.clause_string}")
                print(f"       rule: {step.rule_name}, parents: {step.parent_ids}")
            else:
                print(f"  [{step.clause_id}] {step.clause_string} (input)")

        print(f"\nAll derivation steps: {len(state.all_steps())}")
    else:
        print(f"No proof: {status}")


def step_by_step_demo():
    """Show how different problems behave"""
    print("\n" + "=" * 60)
    print("Problem Comparison")
    print("=" * 60)

    problems = [
        ("Direct contradiction", """
            cnf(c1, axiom, p(a)).
            cnf(c2, negated_conjecture, ~p(a)).
        """),
        ("Chain reasoning", """
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, ~p(X) | q(X)).
            cnf(c3, axiom, ~q(X) | r(X)).
            cnf(c4, negated_conjecture, ~r(a)).
        """),
        ("Satisfiable (no proof)", """
            cnf(c1, axiom, p(a)).
            cnf(c2, axiom, q(b)).
        """),
    ]

    for name, tptp in problems:
        state = ProofAtlas()
        state.add_clauses_from_tptp(tptp)
        proof_found, status = state.prove(timeout=10.0, max_iterations=100)

        stats = state.statistics()
        if proof_found:
            proof_len = len(state.proof_steps())
            print(f"  {name}: proof ({proof_len} steps, {stats['total']} clauses)")
        else:
            print(f"  {name}: {status} ({stats['total']} clauses)")


if __name__ == "__main__":
    prove_and_inspect()
    step_by_step_demo()
    print("\nDone!")
