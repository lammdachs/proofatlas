#!/usr/bin/env python3
"""Test the Python interface to ProofAtlas"""

import json
import pytest
from proofatlas import ProofAtlas, ProofStep


def test_add_clauses_from_tptp():
    """Test parsing TPTP content"""
    state = ProofAtlas()
    clause_ids = state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)
    assert len(clause_ids) == 3
    assert state.statistics()["total"] == 3


def test_add_clauses_invalid_tptp():
    """Test that malformed cnf() statement raises a parse error"""
    state = ProofAtlas()
    with pytest.raises(Exception):
        state.add_clauses_from_tptp("cnf(bad, axiom, |||).")


def test_prove_finds_proof():
    """Test that prove() finds proof for unsatisfiable problem"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    proof_found, status = state.prove(timeout=10.0)
    assert proof_found
    assert status == "proof"


def test_prove_immediate_contradiction():
    """Test proof with direct contradiction"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(a)).
    """)

    proof_found, status = state.prove(timeout=10.0)
    assert proof_found
    assert status == "proof"


def test_prove_equality():
    """Test equality reasoning with superposition"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(eq1, axiom, a = b).
    cnf(eq2, axiom, f(a) != f(b)).
    """)

    proof_found, status = state.prove(timeout=10.0)
    assert proof_found
    assert status == "proof"


def test_prove_saturated():
    """Test that satisfiable problem saturates"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, q(b)).
    """)

    proof_found, status = state.prove(timeout=10.0, max_iterations=100)
    assert not proof_found
    assert status in ("saturated", "resource_limit")


def test_prove_resource_limit():
    """Test that max_iterations triggers resource_limit"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | p(f(X))).
    cnf(c3, axiom, ~p(f(f(f(a))))).
    """)

    proof_found, status = state.prove(timeout=10.0, max_iterations=2)
    # With only 2 iterations, unlikely to find proof
    if not proof_found:
        assert status == "resource_limit"


def test_prove_returns_two_tuple():
    """Test that prove() returns exactly (bool, str)"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("cnf(c1, axiom, p(a)). cnf(c2, axiom, ~p(a)).")
    result = state.prove(timeout=10.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_proof_steps():
    """Test that proof_steps() returns valid proof chain"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    proof_found, _ = state.prove(timeout=10.0)
    assert proof_found

    steps = state.proof_steps()
    assert len(steps) > 0

    # Last step should be empty clause
    assert any(step.clause_string == "⊥" for step in steps)

    # All steps should be ProofStep objects
    for step in steps:
        assert isinstance(step, ProofStep)
        assert isinstance(step.clause_id, int)
        assert isinstance(step.clause_string, str)
        assert isinstance(step.parent_ids, list)
        assert isinstance(step.rule_name, str)

    # Input clauses should have rule "Input"
    input_steps = [s for s in steps if s.rule_name == "Input"]
    assert len(input_steps) > 0


def test_proof_steps_empty_when_no_proof():
    """Test that proof_steps() returns empty when no proof found"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, q(b)).
    """)

    proof_found, _ = state.prove(timeout=10.0, max_iterations=100)
    if not proof_found:
        steps = state.proof_steps()
        assert len(steps) == 0


def test_all_steps():
    """Test that all_steps() returns all derivation events"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    proof_found, _ = state.prove(timeout=10.0)
    assert proof_found

    all_steps = state.all_steps()
    proof_steps = state.proof_steps()

    # all_steps should have at least as many entries as proof_steps
    assert len(all_steps) >= len(proof_steps)


def test_statistics():
    """Test statistics() returns expected keys"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("""
    cnf(unit1, axiom, p(a)).
    cnf(unit2, axiom, q(b)).
    cnf(binary, axiom, r(X) | s(X)).
    cnf(ternary, axiom, t(X) | u(Y) | v(X,Y)).
    """)

    # Before proving
    stats = state.statistics()
    assert stats["total"] == 4
    assert stats["unit_clauses"] == 2
    assert stats["empty_clauses"] == 0

    # After proving
    state.prove(timeout=10.0, max_iterations=100)
    stats = state.statistics()
    assert "total" in stats
    assert "processed" in stats
    assert "unit_clauses" in stats
    assert "empty_clauses" in stats


def test_profile_json():
    """Test profile_json() accessor"""
    state = ProofAtlas()
    state.add_clauses_from_tptp("cnf(c1, axiom, p(a)). cnf(c2, axiom, ~p(a)).")

    # Without profiling
    state.prove(timeout=10.0)
    assert state.profile_json() is None

    # With profiling
    state2 = ProofAtlas()
    state2.add_clauses_from_tptp("cnf(c1, axiom, p(a)). cnf(c2, axiom, ~p(a)).")
    state2.prove(timeout=10.0, enable_profiling=True)
    profile = state2.profile_json()
    assert profile is not None
    data = json.loads(profile)
    assert isinstance(data, dict)


def test_literal_selection_param():
    """Test literal_selection parameter in prove()"""
    for sel in [0, 20, 21, 22]:
        state = ProofAtlas()
        state.add_clauses_from_tptp("""
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, ~p(a)).
        """)
        proof_found, _ = state.prove(timeout=10.0, literal_selection=sel)
        assert proof_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
