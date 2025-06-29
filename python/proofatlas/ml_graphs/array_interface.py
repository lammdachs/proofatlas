"""
ML interface for array-based theorem proving.

This module provides utilities for working with array representations
for machine learning models.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Optional


class ArrayGraphInterface:
    """Interface for ML models to work with array-based logic representations."""
    
    def __init__(self, array_problem):
        """Initialize with an ArrayProblem from Rust."""
        self.problem = array_problem
        self._cache = {}
        
    def get_adjacency_matrix(self) -> sp.csr_matrix:
        """Get the sparse adjacency matrix of the logic graph."""
        if 'adjacency' in self._cache:
            return self._cache['adjacency']
            
        _, _, edge_types = self.problem.get_edge_arrays()
        row_offsets, col_indices, _ = self.problem.get_edge_arrays()
        
        # Create sparse matrix
        data = np.ones_like(col_indices, dtype=np.float32)
        adjacency = sp.csr_matrix(
            (data, col_indices, row_offsets),
            shape=(self.problem.num_nodes, self.problem.num_nodes)
        )
        
        self._cache['adjacency'] = adjacency
        return adjacency
    
    def get_node_features(self) -> np.ndarray:
        """Get node feature matrix for ML models."""
        if 'node_features' in self._cache:
            return self._cache['node_features']
            
        node_types, node_symbols, polarities, arities = self.problem.get_node_arrays()
        
        # One-hot encode node types
        num_types = 6  # Variable, Constant, Function, Predicate, Literal, Clause
        type_onehot = np.zeros((len(node_types), num_types), dtype=np.float32)
        type_onehot[np.arange(len(node_types)), node_types] = 1
        
        # Combine features
        features = np.column_stack([
            type_onehot,
            polarities.astype(np.float32).reshape(-1, 1),
            arities.astype(np.float32).reshape(-1, 1),
            # Could add symbol embeddings here
        ])
        
        self._cache['node_features'] = features
        return features
    
    def get_clause_subgraphs(self) -> List[Tuple[np.ndarray, sp.csr_matrix]]:
        """Extract subgraphs for each clause."""
        clause_bounds, _, num_clauses, _ = self.problem.get_clause_info()
        adjacency = self.get_adjacency_matrix()
        features = self.get_node_features()
        
        subgraphs = []
        for i in range(num_clauses):
            start = clause_bounds[i]
            end = clause_bounds[i + 1]
            
            # Get nodes in this clause
            clause_nodes = list(range(start, end))
            
            # Extract subgraph
            sub_adj = adjacency[clause_nodes][:, clause_nodes]
            sub_features = features[clause_nodes]
            
            subgraphs.append((sub_features, sub_adj))
            
        return subgraphs
    
    def score_clauses_with_gnn(self, gnn_model) -> np.ndarray:
        """Score all clauses using a GNN model."""
        # This is a placeholder - actual implementation would depend on the GNN framework
        clause_bounds, _, num_clauses, _ = self.problem.get_clause_info()
        
        scores = np.zeros(num_clauses)
        
        # For each clause, extract subgraph and score
        for i in range(num_clauses):
            # Get clause subgraph
            start = clause_bounds[i]
            end = clause_bounds[i + 1] if i + 1 < len(clause_bounds) else self.problem.num_nodes
            
            # Simple heuristic score based on size
            # Real implementation would use GNN
            scores[i] = 1.0 / (end - start + 1)
            
        return scores


class MLClauseSelector:
    """ML-based clause selector using array representation."""
    
    def __init__(self, model=None):
        self.model = model
        
    def select(self, array_problem) -> Optional[int]:
        """Select next clause to process."""
        interface = ArrayGraphInterface(array_problem)
        
        if self.model is None:
            # Fallback to simple heuristic
            return self._heuristic_select(interface)
        
        # Use ML model
        scores = interface.score_clauses_with_gnn(self.model)
        return np.argmax(scores)
    
    def _heuristic_select(self, interface: ArrayGraphInterface) -> Optional[int]:
        """Simple heuristic: select smallest clause."""
        clause_bounds, _, num_clauses, _ = interface.problem.get_clause_info()
        
        if num_clauses == 0:
            return None
            
        # Find smallest clause
        min_size = float('inf')
        best_clause = 0
        
        for i in range(num_clauses):
            start = clause_bounds[i]
            end = clause_bounds[i + 1] if i + 1 < len(clause_bounds) else interface.problem.num_nodes
            size = end - start
            
            if size < min_size:
                min_size = size
                best_clause = i
                
        return best_clause


def extract_graph_features(array_problem) -> dict:
    """Extract various graph features for analysis."""
    interface = ArrayGraphInterface(array_problem)
    adjacency = interface.get_adjacency_matrix()
    features = interface.get_node_features()
    
    # Graph-level statistics
    stats = {
        'num_nodes': array_problem.num_nodes,
        'num_edges': adjacency.nnz,
        'num_clauses': array_problem.num_clauses,
        'density': adjacency.nnz / (array_problem.num_nodes ** 2) if array_problem.num_nodes > 0 else 0,
        'avg_degree': adjacency.nnz / array_problem.num_nodes if array_problem.num_nodes > 0 else 0,
    }
    
    # Node type distribution
    if array_problem.num_nodes > 0:
        node_types = features[:, :6].argmax(axis=1)
        for i, name in enumerate(['var', 'const', 'func', 'pred', 'lit', 'clause']):
            stats[f'num_{name}'] = np.sum(node_types == i)
    else:
        for name in ['var', 'const', 'func', 'pred', 'lit', 'clause']:
            stats[f'num_{name}'] = 0
    
    return stats