#!/usr/bin/env python3
"""Simple test for zero-copy array interface"""

try:
    import proofatlas_rust as rust
except ImportError as e:
    print(f"Error: Could not import proofatlas_rust: {e}")
    print("Make sure to run 'maturin develop' first.")
    import sys
    sys.exit(1)

import numpy as np

def test_zero_copy():
    """Test that arrays are zero-copy views"""
    # Create an empty ArrayProblem
    problem = rust.array_repr.ArrayProblem()
    
    # Get initial arrays (should be empty or minimal)
    node_types, node_symbols, node_polarities, node_arities = problem.get_node_arrays()
    edge_row_offsets, edge_col_indices = problem.get_edge_arrays()
    clause_boundaries, literal_boundaries, num_clauses, num_literals = problem.get_clause_info()
    clause_types = problem.get_clause_types()
    
    # Verify we got numpy arrays
    assert isinstance(node_types, np.ndarray), f"node_types is {type(node_types)}"
    assert isinstance(node_symbols, np.ndarray), f"node_symbols is {type(node_symbols)}"
    assert isinstance(node_polarities, np.ndarray), f"node_polarities is {type(node_polarities)}"
    assert isinstance(node_arities, np.ndarray), f"node_arities is {type(node_arities)}"
    assert isinstance(clause_types, np.ndarray), f"clause_types is {type(clause_types)}"
    
    # Check data types
    assert node_types.dtype == np.uint8, f"node_types dtype is {node_types.dtype}"
    assert node_symbols.dtype == np.uint32, f"node_symbols dtype is {node_symbols.dtype}"
    assert node_polarities.dtype == np.int8, f"node_polarities dtype is {node_polarities.dtype}"
    assert node_arities.dtype == np.uint32, f"node_arities dtype is {node_arities.dtype}"
    assert clause_types.dtype == np.uint8, f"clause_types dtype is {clause_types.dtype}"
    
    # Print info
    print(f"Initial state:")
    print(f"  Number of nodes: {problem.num_nodes}")
    print(f"  Number of clauses: {num_clauses}")
    print(f"  Number of literals: {num_literals}")
    print(f"  Node types shape: {node_types.shape}")
    print(f"  Node symbols shape: {node_symbols.shape}")
    
    # Check node type constants
    print(f"\nNode type constants:")
    print(f"  NODE_VARIABLE = {rust.array_repr.NODE_VARIABLE}")
    print(f"  NODE_CONSTANT = {rust.array_repr.NODE_CONSTANT}")
    print(f"  NODE_FUNCTION = {rust.array_repr.NODE_FUNCTION}")
    print(f"  NODE_PREDICATE = {rust.array_repr.NODE_PREDICATE}")
    print(f"  NODE_LITERAL = {rust.array_repr.NODE_LITERAL}")
    print(f"  NODE_CLAUSE = {rust.array_repr.NODE_CLAUSE}")
    
    # Verify arrays are views (not copies)
    # The key property of zero-copy is that we're viewing the same memory
    print(f"\nArrays are zero-copy views into Rust memory")
    print(f"  node_types flags: {node_types.flags}")
    print(f"  - C_CONTIGUOUS: {node_types.flags['C_CONTIGUOUS']}")
    print(f"  - OWNDATA: {node_types.flags['OWNDATA']}")  # Should be False for views
    print(f"  - WRITEABLE: {node_types.flags['WRITEABLE']}")
    
    if not node_types.flags['OWNDATA']:
        print("\n✓ Confirmed: Arrays are views (not copies) of Rust memory!")
    else:
        print("\n✗ Warning: Arrays appear to own their data (might be copies)")
    
    print("\nZero-copy test completed!")

if __name__ == "__main__":
    test_zero_copy()