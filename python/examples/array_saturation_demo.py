#!/usr/bin/env python3
"""
Demo of array-based saturation using Rust backend.

This example shows how to:
1. Convert traditional logic problems to array representation
2. Access array data for ML processing
3. Run saturation on array representation
"""

import numpy as np
import scipy.sparse as sp

# Check if Rust module is available
try:
    import proofatlas_rust
    from proofatlas_rust.array_repr import ArrayProblem
    from proofatlas_rust.core import Problem
    RUST_AVAILABLE = True
except ImportError:
    print("Rust module not available. Please build with: make rust")
    RUST_AVAILABLE = False
    exit(1)

from proofatlas.core.logic import (
    Clause, Literal, Predicate, Constant, Variable
)


def create_simple_problem():
    """Create a simple propositional problem that should find contradiction."""
    # Create propositions
    P = Predicate('P', 0)
    Q = Predicate('Q', 0)
    
    # Create clauses:
    # 1. P ∨ Q
    # 2. ¬P
    # 3. ¬Q
    # Should derive empty clause
    
    clause1 = Clause(Literal(P(), True), Literal(Q(), True))
    clause2 = Clause(Literal(P(), False))
    clause3 = Clause(Literal(Q(), False))
    
    return [clause1, clause2, clause3]


def create_first_order_problem():
    """Create a simple first-order problem."""
    # Predicates
    P = Predicate('P', 1)
    Q = Predicate('Q', 1)
    
    # Terms
    a = Constant('a')
    b = Constant('b')
    X = Variable('X')
    
    # Clauses:
    # 1. ∀X. P(X) → Q(X)  as  ¬P(X) ∨ Q(X)
    # 2. P(a)
    # 3. P(b)
    # 4. ¬Q(a)
    # Should derive contradiction
    
    clause1 = Clause(Literal(P(X), False), Literal(Q(X), True))
    clause2 = Clause(Literal(P(a), True))
    clause3 = Clause(Literal(P(b), True))
    clause4 = Clause(Literal(Q(a), False))
    
    return [clause1, clause2, clause3, clause4]


def visualize_graph(array_problem):
    """Visualize the graph structure using arrays."""
    # Get arrays
    node_types, node_symbols, polarities, arities = array_problem.get_node_arrays()
    row_offsets, col_indices, edge_types = array_problem.get_edge_arrays()
    clause_bounds, literal_bounds, num_clauses, num_literals = array_problem.get_clause_info()
    symbols = array_problem.get_symbols()
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {array_problem.num_nodes}")
    print(f"  Clauses: {num_clauses}")
    print(f"  Literals: {num_literals}")
    print(f"  Symbols: {len(symbols)}")
    
    # Build sparse adjacency matrix
    num_nodes = len(node_types)
    adjacency = sp.csr_matrix(
        (np.ones_like(col_indices), col_indices, row_offsets),
        shape=(num_nodes, num_nodes)
    )
    
    print(f"\nAdjacency matrix shape: {adjacency.shape}")
    print(f"Number of edges: {adjacency.nnz}")
    
    # Show node type distribution
    node_type_names = ['Variable', 'Constant', 'Function', 'Predicate', 'Literal', 'Clause']
    print("\nNode type distribution:")
    for i, name in enumerate(node_type_names):
        count = np.sum(node_types == i)
        if count > 0:
            print(f"  {name}: {count}")
    
    # Show symbols
    print("\nSymbol table:")
    for i, symbol in enumerate(symbols):
        print(f"  {i}: {symbol}")


def run_array_saturation(clauses):
    """Run saturation using array representation."""
    print("Converting to array representation...")
    
    # Create traditional problem first
    problem = Problem()
    for clause in clauses:
        # Need to convert our Python clauses to Rust format
        # For now, we'll use the ArrayProblem directly
        pass
    
    # Create array problem
    array_problem = ArrayProblem()
    
    # For demonstration, let's manually add clauses
    # In practice, we'd have a proper conversion
    print("Note: Direct clause conversion not yet implemented")
    print("      Using empty problem for demonstration")
    
    # Visualize structure
    visualize_graph(array_problem)
    
    # Run saturation
    print("\nRunning saturation...")
    found_empty, clauses_generated, iterations = array_problem.saturate(
        max_clauses=1000,
        max_iterations=100
    )
    
    print(f"\nSaturation results:")
    print(f"  Found empty clause: {found_empty}")
    print(f"  Clauses generated: {clauses_generated}")
    print(f"  Iterations: {iterations}")
    
    return array_problem


def demo_ml_features(array_problem):
    """Demonstrate how to extract features for ML."""
    # Get arrays
    node_types, node_symbols, polarities, arities = array_problem.get_node_arrays()
    row_offsets, col_indices, edge_types = array_problem.get_edge_arrays()
    
    print("\nML-ready features:")
    print(f"  Node features shape: {node_types.shape}")
    print(f"  Edge indices shape: {col_indices.shape}")
    
    # Example: Create node feature matrix
    # Combine multiple node attributes
    node_features = np.column_stack([
        node_types,
        polarities,
        arities,
    ])
    
    print(f"  Combined features shape: {node_features.shape}")
    
    # Example: Graph statistics as features
    if len(node_types) > 0:
        in_degrees = np.diff(row_offsets)
        print(f"  Max in-degree: {np.max(in_degrees)}")
        print(f"  Avg in-degree: {np.mean(in_degrees):.2f}")


def main():
    print("Array-Based Saturation Demo")
    print("=" * 50)
    
    # Create problems
    print("\n1. Simple Propositional Problem")
    prop_clauses = create_simple_problem()
    for i, clause in enumerate(prop_clauses):
        print(f"   {i+1}. {clause}")
    
    print("\n2. First-Order Problem")
    fo_clauses = create_first_order_problem()
    for i, clause in enumerate(fo_clauses):
        print(f"   {i+1}. {clause}")
    
    # Run array saturation
    print("\n3. Array-Based Saturation")
    array_problem = run_array_saturation(prop_clauses)
    
    # Show ML features
    print("\n4. ML Feature Extraction")
    demo_ml_features(array_problem)
    
    print("\n" + "=" * 50)
    print("Demo complete!")


if __name__ == "__main__":
    main()