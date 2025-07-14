#!/usr/bin/env python3
"""Test the Python interface to ProofAtlas"""

import pytest
from proofatlas import ProofState, ClauseInfo, saturate_step


def test_basic_proof():
    """Test a simple proof: P(a), ~P(X) | Q(X), ~Q(a) |- ⊥"""
    
    # Create proof state
    state = ProofState()
    assert state.num_clauses() == 0
    assert state.num_processed() == 0
    assert state.num_unprocessed() == 0
    
    # Add clauses
    tptp = """
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """
    
    clause_ids = state.add_clauses_from_tptp(tptp)
    assert len(clause_ids) == 3
    assert state.num_clauses() == 3
    assert state.num_unprocessed() == 3
    
    # Run saturation
    steps = 0
    while state.num_unprocessed() > 0 and not state.contains_empty_clause():
        steps += 1
        given_id = state.select_given_clause()
        
        if given_id is None:
            break
            
        inferences = state.generate_inferences(given_id)
        for inf in inferences:
            state.add_inference(inf)
        
        state.process_clause(given_id)
        
        if steps > 20:  # Safety limit
            pytest.fail("Too many steps without finding proof")
    
    # Check result
    assert state.contains_empty_clause()
    
    # Verify proof trace
    trace = state.get_proof_trace()
    assert len(trace) > 0
    assert any(step.clause_string == "⊥" for step in trace)


def test_clause_info():
    """Test clause information extraction"""
    state = ProofState()
    
    # Add a test clause
    state.add_clauses_from_tptp("cnf(test, axiom, p(X) | q(f(X)) | r(g(h(X)))).")
    
    info = state.get_clause_info(0)
    assert info.clause_id == 0
    assert info.num_literals == 3
    assert len(info.literal_strings) == 3
    assert info.weight == 9  # p + X + q + f + X + r + g + h + X
    assert 'X' in info.variables
    assert not info.is_unit
    assert info.is_horn  # All positive literals
    assert not info.is_equality


def test_literal_selection():
    """Test literal selection strategies"""
    state = ProofState()
    
    # Test with no selection (default)
    state.set_literal_selection("no_selection")
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(f(X))).
    """)
    
    # Should be able to generate inferences
    inferences = state.generate_inferences(0)
    # With no selection, all literals are eligible
    
    # Test with max weight selection
    state2 = ProofState()
    state2.set_literal_selection("max_weight")
    state2.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(f(X))).
    """)
    
    # Should still generate inferences with weight-based selection
    inferences2 = state2.generate_inferences(0)


def test_saturate_step_helper():
    """Test the saturate_step helper function"""
    state = ProofState()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(X) | q(X)).
    cnf(c3, axiom, ~q(a)).
    """)
    
    steps = 0
    proof_found = False
    
    while steps < 20:
        steps += 1
        result = saturate_step(state, clause_selection="fifo")
        
        if result['saturated']:
            break
            
        if result['proof_found']:
            proof_found = True
            break
    
    assert proof_found
    assert steps < 20


def test_statistics():
    """Test statistics collection"""
    state = ProofState()
    
    # Add some clauses
    state.add_clauses_from_tptp("""
    cnf(unit1, axiom, p(a)).
    cnf(unit2, axiom, q(b)).
    cnf(binary, axiom, r(X) | s(X)).
    cnf(ternary, axiom, t(X) | u(Y) | v(X,Y)).
    """)
    
    stats = state.get_statistics()
    assert stats['total'] == 4
    assert stats['processed'] == 0
    assert stats['unprocessed'] == 4
    assert stats['unit_clauses'] == 2
    assert stats['empty_clauses'] == 0
    
    # Process one clause
    state.select_given_clause()
    state.process_clause(0)
    
    stats = state.get_statistics()
    assert stats['processed'] == 1
    assert stats['unprocessed'] == 3


def test_empty_problem():
    """Test with empty problem"""
    state = ProofState()
    assert state.num_clauses() == 0
    assert not state.contains_empty_clause()
    
    result = saturate_step(state)
    assert result['saturated']
    assert not result['proof_found']


def test_already_contains_empty():
    """Test with immediate contradiction"""
    state = ProofState()
    state.add_clauses_from_tptp("""
    cnf(c1, axiom, p(a)).
    cnf(c2, axiom, ~p(a)).
    """)
    
    # Should find empty clause quickly
    result = saturate_step(state)
    assert not result['saturated']
    
    # After a few steps should find proof
    for _ in range(5):
        result = saturate_step(state)
        if result['proof_found']:
            break
    
    assert result['proof_found']


def test_superposition():
    """Test equality reasoning"""
    state = ProofState()
    state.set_use_superposition(True)
    
    state.add_clauses_from_tptp("""
    cnf(eq1, axiom, a = b).
    cnf(eq2, axiom, f(a) != f(b)).
    """)
    
    # Should find contradiction with superposition
    steps = 0
    while steps < 10:
        result = saturate_step(state)
        if result['proof_found'] or result['saturated']:
            break
        steps += 1
    
    # With superposition enabled, should find proof
    assert state.contains_empty_clause()


if __name__ == "__main__":
    # Run all tests
    test_basic_proof()
    print("✓ Basic proof test passed")
    
    test_clause_info()
    print("✓ Clause info test passed")
    
    test_literal_selection()
    print("✓ Literal selection test passed")
    
    test_saturate_step_helper()
    print("✓ Saturate step helper test passed")
    
    test_statistics()
    print("✓ Statistics test passed")
    
    test_empty_problem()
    print("✓ Empty problem test passed")
    
    test_already_contains_empty()
    print("✓ Immediate contradiction test passed")
    
    test_superposition()
    print("✓ Superposition test passed")
    
    print("\nAll tests passed!")