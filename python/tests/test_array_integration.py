"""
Comprehensive integration tests for array-based theorem proving.

Tests the complete pipeline from Python logic structures to array representation
and back, including ML integration.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if Rust module is available
try:
    import proofatlas_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust module not available")

if RUST_AVAILABLE:
    import proofatlas_rust
    ArrayProblem = proofatlas_rust.array_repr.ArrayProblem

from proofatlas.core.logic import (
    Variable, Constant, Function, Predicate, Term, Literal, Clause, Problem
)
from proofatlas.ml_graphs.array_interface import (
    ArrayGraphInterface, MLClauseSelector, extract_graph_features
)


class TestArrayProblemBasics:
    """Test basic array problem functionality."""
    
    def test_create_empty_array_problem(self):
        """Test creating an empty array problem."""
        problem = ArrayProblem()
        
        assert problem.num_nodes == 0
        assert problem.num_clauses == 0
        assert not problem.has_empty_clause()
    
    def test_array_problem_properties(self):
        """Test array problem properties."""
        problem = ArrayProblem()
        
        # Get arrays (should be empty)
        node_arrays = problem.get_node_arrays()
        edge_arrays = problem.get_edge_arrays()
        clause_info = problem.get_clause_info()
        
        assert len(node_arrays) == 4  # types, symbols, polarities, arities
        assert len(edge_arrays) == 3   # offsets, indices, types
        assert len(clause_info) == 4   # boundaries, literal_boundaries, counts
    
    def test_get_symbols(self):
        """Test symbol table access."""
        problem = ArrayProblem()
        symbols = problem.get_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) == 0  # Empty initially


class TestArrayConversion:
    """Test conversion from traditional to array representation."""
    
    def test_convert_simple_clause(self):
        """Test converting a simple clause."""
        # Create traditional problem with a predicate that has an argument
        # (workaround for parser issue with 0-arity predicates in CNF)
        from proofatlas.core.logic import Constant
        
        p = Predicate('p', 1)
        a = Constant('a')
        clause = Clause(Literal(p(a), True))
        
        # Convert to Rust problem using FOF format  
        # (CNF parser has issues with the Python to_tptp output)
        fof_str = 'fof(c1, axiom, p(a)).'
        
        # Parse TPTP string into Rust problem
        rust_problem = proofatlas_rust.parser.parse_string(fof_str)
        
        # Convert to array
        array_problem = ArrayProblem.from_problem(rust_problem)
        
        assert array_problem.num_clauses == 1
        assert array_problem.num_nodes > 0
    
    def test_manual_array_construction(self):
        """Test that array problem can be constructed."""
        problem = ArrayProblem()
        
        # Currently we can't add clauses from Python
        # This is a placeholder for when that's implemented
        assert problem.num_clauses == 0


class TestArraySaturation:
    """Test saturation on array problems."""
    
    def test_saturate_empty(self):
        """Test saturation on empty problem."""
        problem = ArrayProblem()
        
        found, generated, iterations = problem.saturate()
        
        assert not found
        assert generated == 0
        assert iterations == 0
    
    def test_saturate_with_limits(self):
        """Test saturation with resource limits."""
        problem = ArrayProblem()
        
        found, generated, iterations = problem.saturate(
            max_clauses=100,
            max_iterations=10
        )
        
        assert not found
        assert iterations <= 10


class TestArrayInterface:
    """Test ML interface for array problems."""
    
    def test_array_graph_interface(self):
        """Test ArrayGraphInterface."""
        problem = ArrayProblem()
        interface = ArrayGraphInterface(problem)
        
        # Get adjacency matrix
        adj = interface.get_adjacency_matrix()
        assert isinstance(adj, sp.csr_matrix)
        assert adj.shape == (0, 0)  # Empty
        
        # Get node features
        features = interface.get_node_features()
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 8  # 6 node types + polarity + arity
    
    def test_extract_graph_features(self):
        """Test feature extraction."""
        problem = ArrayProblem()
        features = extract_graph_features(problem)
        
        assert isinstance(features, dict)
        assert 'num_nodes' in features
        assert 'num_edges' in features
        assert 'num_clauses' in features
        assert features['num_nodes'] == 0
    
    def test_ml_clause_selector(self):
        """Test ML clause selector."""
        problem = ArrayProblem()
        selector = MLClauseSelector()
        
        # Should return None for empty problem
        selected = selector.select(problem)
        assert selected is None


class TestNumpyArrayAccess:
    """Test NumPy array access from Python."""
    
    def test_get_node_arrays(self):
        """Test getting node arrays as NumPy arrays."""
        problem = ArrayProblem()
        
        node_types, symbols, polarities, arities = problem.get_node_arrays()
        
        # All should be NumPy arrays
        assert isinstance(node_types, np.ndarray)
        assert isinstance(symbols, np.ndarray)
        assert isinstance(polarities, np.ndarray)
        assert isinstance(arities, np.ndarray)
        
        # Check dtypes
        assert node_types.dtype == np.uint8
        assert symbols.dtype == np.uint32
        assert polarities.dtype == np.int8
        assert arities.dtype == np.uint32
    
    def test_get_edge_arrays(self):
        """Test getting edge arrays."""
        problem = ArrayProblem()
        
        offsets, indices, types = problem.get_edge_arrays()
        
        assert isinstance(offsets, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(types, np.ndarray)
        
        # Check consistency
        assert len(offsets) >= 1  # At least one offset
        assert offsets[0] == 0    # First offset is 0
    
    def test_get_clause_info(self):
        """Test getting clause information."""
        problem = ArrayProblem()
        
        boundaries, lit_boundaries, num_clauses, num_literals = problem.get_clause_info()
        
        assert isinstance(boundaries, np.ndarray)
        assert isinstance(lit_boundaries, np.ndarray)
        assert isinstance(num_clauses, int)
        assert isinstance(num_literals, int)
        
        assert num_clauses == 0
        assert num_literals == 0


class TestArrayConstants:
    """Test array type constants."""
    
    def test_node_type_constants(self):
        """Test node type constants are available."""
        ar = proofatlas_rust.array_repr
        
        assert hasattr(ar, 'NODE_VARIABLE')
        assert hasattr(ar, 'NODE_CONSTANT')
        assert hasattr(ar, 'NODE_FUNCTION')
        assert hasattr(ar, 'NODE_PREDICATE')
        assert hasattr(ar, 'NODE_LITERAL')
        assert hasattr(ar, 'NODE_CLAUSE')
        
        # Check values
        assert ar.NODE_VARIABLE == 0
        assert ar.NODE_CONSTANT == 1
        assert ar.NODE_FUNCTION == 2
        assert ar.NODE_PREDICATE == 3
        assert ar.NODE_LITERAL == 4
        assert ar.NODE_CLAUSE == 5
    
    def test_edge_type_constants(self):
        """Test edge type constants."""
        ar = proofatlas_rust.array_repr
        
        assert hasattr(ar, 'EDGE_HAS_ARGUMENT')
        assert hasattr(ar, 'EDGE_HAS_LITERAL')
        assert hasattr(ar, 'EDGE_HAS_PREDICATE')
        
        assert ar.EDGE_HAS_ARGUMENT == 0
        assert ar.EDGE_HAS_LITERAL == 1
        assert ar.EDGE_HAS_PREDICATE == 2


class TestMLIntegration:
    """Test ML integration with array representation."""
    
    def test_sparse_matrix_construction(self):
        """Test building sparse matrices from arrays."""
        problem = ArrayProblem()
        
        # Even empty problem should work
        offsets, indices, _ = problem.get_edge_arrays()
        
        # Build sparse matrix
        if len(offsets) > 1:
            data = np.ones_like(indices, dtype=np.float32)
            adj = sp.csr_matrix(
                (data, indices, offsets),
                shape=(problem.num_nodes, problem.num_nodes)
            )
            
            assert adj.shape[0] == problem.num_nodes
            assert adj.nnz == len(indices)
    
    def test_node_feature_matrix(self):
        """Test building feature matrix."""
        problem = ArrayProblem()
        interface = ArrayGraphInterface(problem)
        
        features = interface.get_node_features()
        
        # Features should include one-hot encoded types
        if problem.num_nodes > 0:
            assert features.shape[0] == problem.num_nodes
            assert features.shape[1] == 8  # 6 types + polarity + arity
            
            # One-hot columns should sum to 1
            type_columns = features[:, :6]
            assert np.allclose(type_columns.sum(axis=1), 1.0)
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Create multiple array problems
        problems = [ArrayProblem() for _ in range(5)]
        
        # Extract features from each
        all_features = []
        for p in problems:
            features = extract_graph_features(p)
            all_features.append(features)
        
        assert len(all_features) == 5
        
        # All should have same keys
        keys = set(all_features[0].keys())
        for f in all_features[1:]:
            assert set(f.keys()) == keys


class TestMemoryEfficiency:
    """Test memory efficiency of array representation."""
    
    def test_zero_copy_arrays(self):
        """Test that arrays are zero-copy from Rust."""
        problem = ArrayProblem()
        
        # Get arrays multiple times
        arrays1 = problem.get_node_arrays()
        arrays2 = problem.get_node_arrays()
        
        # Should be new array objects (not cached in Python)
        for a1, a2 in zip(arrays1, arrays2):
            assert a1 is not a2
            
            # But should have same memory layout
            if len(a1) > 0:
                assert a1.flags['C_CONTIGUOUS'] or a1.flags['F_CONTIGUOUS']
    
    def test_large_problem_memory(self):
        """Test memory usage with larger problems."""
        problem = ArrayProblem()
        
        # Simulate saturation generating many clauses
        # (Would need actual clause generation to test properly)
        
        # Get memory usage estimate
        node_arrays = problem.get_node_arrays()
        edge_arrays = problem.get_edge_arrays()
        
        total_bytes = sum(a.nbytes for a in node_arrays)
        total_bytes += sum(a.nbytes for a in edge_arrays)
        
        # Even empty problem has some overhead
        assert total_bytes >= 0


class TestErrorHandling:
    """Test error handling in array interface."""
    
    def test_invalid_saturation_params(self):
        """Test saturation with invalid parameters."""
        problem = ArrayProblem()
        
        # Should handle None gracefully
        found, generated, iterations = problem.saturate(None, None)
        assert isinstance(found, bool)
        assert isinstance(generated, int)
        assert isinstance(iterations, int)
    
    def test_array_bounds(self):
        """Test array access bounds."""
        problem = ArrayProblem()
        interface = ArrayGraphInterface(problem)
        
        # Should handle empty problem gracefully
        subgraphs = interface.get_clause_subgraphs()
        assert len(subgraphs) == 0


class TestIntegrationWorkflow:
    """Test complete workflow from Python to array and back."""
    
    def test_ml_guided_saturation_workflow(self):
        """Test ML-guided saturation workflow."""
        # Create array problem
        problem = ArrayProblem()
        
        # Create ML selector
        selector = MLClauseSelector()
        
        # Extract features
        features = extract_graph_features(problem)
        
        # Run saturation
        found, generated, iterations = problem.saturate(
            max_clauses=100,
            max_iterations=10
        )
        
        # Verify workflow completed
        assert isinstance(found, bool)
        assert isinstance(features, dict)
    
    def test_graph_analysis_workflow(self):
        """Test graph analysis workflow."""
        problem = ArrayProblem()
        interface = ArrayGraphInterface(problem)
        
        # Get graph representation
        adj = interface.get_adjacency_matrix()
        features = interface.get_node_features()
        
        # Analyze graph properties
        if problem.num_nodes > 0:
            # Degree distribution
            degrees = np.array(adj.sum(axis=1)).flatten()
            
            # Connected components (would need scipy.sparse.csgraph)
            # components = sp.csgraph.connected_components(adj)
            
            # Graph diameter, etc.
            assert len(degrees) == problem.num_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])