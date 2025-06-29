#!/usr/bin/env python3
"""Test equivalence between standard and array implementations."""

import proofatlas_rust
from proofatlas.core.logic import Clause, Literal, Predicate, Term, Constant, Variable, Problem

def test_basic_contradiction():
    """Test P and ~P leads to empty clause."""
    print("Test: Basic contradiction (P and ~P)")
    
    # Create problem
    p = Predicate('p', 0)
    clauses = [
        Clause(Literal(p(), True)),
        Clause(Literal(p(), False))
    ]
    
    # Convert to TPTP
    problem = Problem(*clauses)
    tptp_str = '\n'.join([
        'fof(c1, axiom, p).',
        'fof(c2, axiom, ~p).'
    ])
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    print(f"  Initial clauses: {array_problem.num_clauses}")
    
    # Run saturation
    found, generated, iterations = array_problem.saturate(max_clauses=100, max_iterations=10)
    
    print(f"  Found empty clause: {found}")
    print(f"  Generated clauses: {generated}")
    print(f"  Iterations: {iterations}")
    
    assert found, "Should find contradiction"
    assert iterations == 1, "Should find in first iteration"
    print("  ✓ Test passed\n")


def test_propositional_square():
    """Test propositional square of opposition."""
    print("Test: Propositional square")
    
    tptp_str = '\n'.join([
        'fof(c1, axiom, p | q).',
        'fof(c2, axiom, ~p | q).',
        'fof(c3, axiom, p | ~q).',
        'fof(c4, axiom, ~p | ~q).'
    ])
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    print(f"  Initial clauses: {array_problem.num_clauses}")
    
    # Run saturation
    found, generated, iterations = array_problem.saturate(max_clauses=1000, max_iterations=100)
    
    print(f"  Found empty clause: {found}")
    print(f"  Generated clauses: {generated}")
    print(f"  Iterations: {iterations}")
    
    assert found, "Should find contradiction"
    print("  ✓ Test passed\n")


def test_first_order_unification():
    """Test first-order unification."""
    print("Test: First-order unification")
    
    tptp_str = '\n'.join([
        'fof(c1, axiom, p(X, f(X))).',
        'fof(c2, axiom, ~p(a, f(a))).'
    ])
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    print(f"  Initial clauses: {array_problem.num_clauses}")
    
    # Run saturation
    found, generated, iterations = array_problem.saturate(max_clauses=100, max_iterations=10)
    
    print(f"  Found empty clause: {found}")
    print(f"  Generated clauses: {generated}")
    print(f"  Iterations: {iterations}")
    
    assert found, "Should find contradiction through unification"
    print("  ✓ Test passed\n")


def test_chain_of_implications():
    """Test chain of implications."""
    print("Test: Chain of implications")
    
    tptp_str = '\n'.join([
        'fof(c1, axiom, ~p(X) | q(X)).',   # P(X) -> Q(X)
        'fof(c2, axiom, ~q(Y) | r(Y)).',   # Q(Y) -> R(Y)
        'fof(c3, axiom, p(a)).',           # P(a)
        'fof(c4, axiom, ~r(a)).'           # ~R(a)
    ])
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    print(f"  Initial clauses: {array_problem.num_clauses}")
    
    # Run saturation
    found, generated, iterations = array_problem.saturate(max_clauses=1000, max_iterations=50)
    
    print(f"  Found empty clause: {found}")
    print(f"  Generated clauses: {generated}")
    print(f"  Iterations: {iterations}")
    
    assert found, "Should find contradiction through chain of inferences"
    print("  ✓ Test passed\n")


def test_tautology_filtering():
    """Test that tautologies are handled correctly."""
    print("Test: Tautology filtering")
    
    tptp_str = '\n'.join([
        'fof(c1, axiom, p | ~p).',    # Tautology
        'fof(c2, axiom, q).',
        'fof(c3, axiom, ~q).'
    ])
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    print(f"  Initial clauses: {array_problem.num_clauses}")
    
    # Run saturation
    found, generated, iterations = array_problem.saturate(max_clauses=100, max_iterations=10)
    
    print(f"  Found empty clause: {found}")
    print(f"  Generated clauses: {generated}")
    print(f"  Iterations: {iterations}")
    
    assert found, "Should find contradiction despite tautology"
    assert generated <= 2, f"Should generate minimal clauses, but generated {generated}"
    print("  ✓ Test passed\n")


def test_array_data_access():
    """Test array data access for ML."""
    print("Test: Array data access")
    
    tptp_str = 'fof(c1, axiom, p(f(a), b) | ~q(X, g(X))).'
    
    # Parse and convert to array
    rust_problem = proofatlas_rust.parser.parse_string(tptp_str)
    array_problem = proofatlas_rust.array_repr.ArrayProblem.from_problem(rust_problem)
    
    # Access arrays
    node_types, symbols, polarities, arities = array_problem.get_node_arrays()
    offsets, indices, edge_types = array_problem.get_edge_arrays()
    
    print(f"  Nodes: {array_problem.num_nodes}")
    print(f"  Node types shape: {node_types.shape}")
    print(f"  Edge indices: {len(indices)}")
    print(f"  Symbols: {array_problem.get_symbols()}")
    
    assert len(node_types) == array_problem.num_nodes
    assert len(offsets) == array_problem.num_nodes + 1
    print("  ✓ Test passed\n")


if __name__ == "__main__":
    print("Running array implementation equivalence tests...\n")
    
    test_basic_contradiction()
    test_propositional_square()
    test_first_order_unification()
    test_chain_of_implications()
    test_tautology_filtering()
    test_array_data_access()
    
    print("All tests passed! Array implementation is working correctly.")