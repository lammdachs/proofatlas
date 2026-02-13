#!/usr/bin/env python3
"""Test the Python interface to ProofAtlas"""

import json
import pytest
from proofatlas import ProofAtlas, ProofStep


def test_prove_finds_proof():
    """Test that prove() finds proof for unsatisfiable problem"""
    atlas = ProofAtlas()
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    assert prover.proof_found
    assert prover.status == "proof"


def test_prove_immediate_contradiction():
    """Test proof with direct contradiction"""
    atlas = ProofAtlas()
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(a)).
    """)

    assert prover.proof_found
    assert prover.status == "proof"


def test_prove_equality():
    """Test equality reasoning with superposition"""
    atlas = ProofAtlas()
    prover = atlas.prove_string("""
    cnf(eq1, axiom, a = b).
    cnf(eq2, axiom, f(a) != f(b)).
    """)

    assert prover.proof_found
    assert prover.status == "proof"


def test_prove_saturated():
    """Test that satisfiable problem saturates"""
    atlas = ProofAtlas(max_iterations=100, timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, q(b)).
    """)

    assert not prover.proof_found
    assert prover.status in ("saturated", "resource_limit")


def test_prove_resource_limit():
    """Test that max_iterations triggers resource_limit"""
    atlas = ProofAtlas(max_iterations=2, timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | p(f(X))).
    cnf(c3, axiom, ~p(f(f(f(a))))).
    """)

    # With only 2 iterations, unlikely to find proof
    if not prover.proof_found:
        assert prover.status == "resource_limit"


def test_prover_properties():
    """Test that Prover has proof_found and status properties"""
    atlas = ProofAtlas(timeout=10.0)
    prover = atlas.prove_string("cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).")
    assert isinstance(prover.proof_found, bool)
    assert isinstance(prover.status, str)
    assert prover.proof_found
    assert prover.status == "proof"


def test_proof_steps():
    """Test that proof_steps() returns valid proof chain"""
    atlas = ProofAtlas(timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    assert prover.proof_found

    steps = prover.proof_steps()
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
    atlas = ProofAtlas(max_iterations=100, timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, q(b)).
    """)

    if not prover.proof_found:
        steps = prover.proof_steps()
        assert len(steps) == 0


def test_all_steps():
    """Test that all_steps() returns all derivation events"""
    atlas = ProofAtlas(timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)

    assert prover.proof_found

    all_steps = prover.all_steps()
    proof_steps = prover.proof_steps()

    # all_steps should have at least as many entries as proof_steps
    assert len(all_steps) >= len(proof_steps)


def test_statistics():
    """Test statistics() returns expected keys"""
    atlas = ProofAtlas(timeout=10.0)
    prover = atlas.prove_string("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(a)).
    """)

    stats = prover.statistics()
    assert "total" in stats
    assert "processed" in stats
    assert "unit_clauses" in stats
    assert "empty_clauses" in stats


def test_profile_json():
    """Test profile_json() accessor"""
    # Without profiling
    atlas = ProofAtlas(timeout=10.0)
    prover = atlas.prove_string("cnf(c1, axiom, p(a)). cnf(c2, axiom, ~p(a)).")
    assert prover.profile_json() is None

    # With profiling
    atlas2 = ProofAtlas(timeout=10.0, enable_profiling=True)
    prover2 = atlas2.prove_string("cnf(c1, axiom, p(a)). cnf(c2, axiom, ~p(a)).")
    profile = prover2.profile_json()
    assert profile is not None
    data = json.loads(profile)
    assert isinstance(data, dict)


def test_literal_selection_param():
    """Test literal_selection parameter"""
    for sel in [0, 20, 21, 22]:
        atlas = ProofAtlas(literal_selection=sel, timeout=10.0)
        prover = atlas.prove_string("""
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, ~p(a)).
        """)
        assert prover.proof_found


def test_reuse_atlas():
    """Test that atlas can be reused for multiple problems"""
    atlas = ProofAtlas(timeout=10.0)

    p1 = atlas.prove_string("cnf(c1, axiom, p(a)).\ncnf(c2, axiom, ~p(a)).")
    assert p1.proof_found

    p2 = atlas.prove_string("cnf(c1, axiom, q(b)).\ncnf(c2, axiom, ~q(b)).")
    assert p2.proof_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
