"""Batch graphs and constants for training.

Graph construction is done in Rust via GraphBuilder::build_from_clauses() — a single
function shared between trace extraction and inference. This module handles:
- Graph batching from per-clause numpy arrays into training batch tensors
- Constants for node types, feature indices, and role/rule mappings
"""

from typing import Dict, List, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Node type constants (matching Rust GraphBuilder)
TYPE_CLAUSE = 0
TYPE_LITERAL = 1
TYPE_PREDICATE = 2
TYPE_FUNCTION = 3
TYPE_VARIABLE = 4
TYPE_CONSTANT = 5

# Node feature indices (3 features per node for GCN encoder)
FEAT_NODE_TYPE = 0
FEAT_ARITY = 1
FEAT_ARG_POSITION = 2

# Clause feature indices (9 features per clause)
CLAUSE_FEATURE_DIM = 9

CLAUSE_FEAT_AGE = 0
CLAUSE_FEAT_ROLE = 1
CLAUSE_FEAT_RULE = 2
CLAUSE_FEAT_SIZE = 3
CLAUSE_FEAT_DEPTH = 4
CLAUSE_FEAT_SYMBOL_COUNT = 5
CLAUSE_FEAT_DISTINCT_SYMBOLS = 6
CLAUSE_FEAT_VARIABLE_COUNT = 7
CLAUSE_FEAT_DISTINCT_VARIABLES = 8

# Role encoding
ROLE_MAP = {
    "axiom": 0,
    "hypothesis": 1,
    "definition": 2,
    "negated_conjecture": 3,
    "derived": 4,
}

# Rule encoding
RULE_MAP = {
    "input": 0,
    "resolution": 1,
    "factoring": 2,
    "superposition": 3,
    "equality_resolution": 4,
    "equality_factoring": 5,
    "demodulation": 6,
}


def batch_graphs(
    graphs: List[Dict[str, Any]],
    labels: Optional[List[int]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Batch multiple clause graphs with edge-list adjacency and pool matrices.

    Uses dense edge-list format (row, col, val tensors) instead of sparse COO
    tensors to avoid a PyTorch 2.10 bug where large sparse COO tensor
    deallocation corrupts the heap (double free in tcache).

    Args:
        graphs: List of graph dicts with numpy arrays (x, edge_index, num_nodes, num_edges, clause_features)
        labels: Optional list of labels (one per graph)
        device: PyTorch device

    Returns:
        Dict with:
            x: [num_nodes, 3] node features
            adj: (row, col, val, (M, N)) edge-list tuple
            pool_matrix: (row, col, val, (M, N)) edge-list tuple
            clause_features: [num_clauses, CLAUSE_FEATURE_DIM] clause features
            labels: [num_clauses] if provided
            batch: [num_nodes] clause index for each node
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for batching")

    import torch

    if len(graphs) == 0:
        raise ValueError("Cannot batch empty list of graphs")

    all_x = []
    all_edges = []
    all_clause_features = []
    all_node_names = []
    all_node_embeddings = []
    all_node_sentinel_type = []
    batch_indices = []

    node_offset = 0

    for i, g in enumerate(graphs):
        all_x.append(g['x'])
        all_edges.append(g['edge_index'] + node_offset)
        batch_indices.extend([i] * g['num_nodes'])

        if 'clause_features' in g:
            all_clause_features.append(g['clause_features'])
        if 'node_names' in g:
            all_node_names.extend(g['node_names'])
        if 'node_embeddings' in g:
            all_node_embeddings.append(g['node_embeddings'])
        if 'node_sentinel_type' in g:
            all_node_sentinel_type.append(g['node_sentinel_type'])

        node_offset += g['num_nodes']

    # Concatenate node features
    # Use torch.tensor() (copies data) instead of torch.from_numpy() (shares memory)
    # to avoid heap corruption with PyTorch 2.10 when numpy arrays are freed
    x_np = np.concatenate(all_x, axis=0)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    del x_np
    num_nodes = x.size(0)
    num_clauses = len(graphs)

    # Build edge-list adjacency with self-loops and D^{-1/2} A D^{-1/2} normalization
    ei_np = np.concatenate(all_edges, axis=1)
    edge_index = torch.tensor(ei_np, dtype=torch.long, device=device)
    del ei_np, all_x, all_edges
    self_loops = torch.arange(num_nodes, device=device)
    all_row = torch.cat([edge_index[0], self_loops])
    all_col = torch.cat([edge_index[1], self_loops])
    all_val = torch.ones(all_row.size(0), device=device)

    # Coalesce by summing duplicate edges (needed for correct degree computation)
    # Use a hash-based approach: pack (row, col) into a single long
    packed = all_row * num_nodes + all_col
    unique_packed, inverse = packed.unique(return_inverse=True)
    coalesced_val = torch.zeros(unique_packed.size(0), device=device)
    coalesced_val.scatter_add_(0, inverse, all_val)
    adj_row = unique_packed // num_nodes
    adj_col = unique_packed % num_nodes

    # Normalize: D^{-1/2} A D^{-1/2}
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, adj_row, coalesced_val)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_val = deg_inv_sqrt[adj_row] * coalesced_val * deg_inv_sqrt[adj_col]

    adj = (adj_row, adj_col, adj_val, (num_nodes, num_nodes))

    # Build pool matrix as edge list
    batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
    counts = torch.bincount(batch, minlength=num_clauses).float()
    node_idx = torch.arange(num_nodes, device=device)
    pool_val = 1.0 / counts[batch].sqrt()
    pool_matrix = (batch, node_idx, pool_val, (num_clauses, num_nodes))

    result = {
        'x': x,
        'adj': adj,
        'pool_matrix': pool_matrix,
        'batch': batch,
        'num_graphs': num_clauses,
    }

    if all_clause_features:
        cf_np = np.stack(all_clause_features, axis=0)
        result['clause_features'] = torch.tensor(cf_np, dtype=torch.float32, device=device)
        del cf_np

    if all_node_names:
        result['node_names'] = all_node_names  # flat Python list, not tensor

    if all_node_embeddings:
        ne_np = np.concatenate(all_node_embeddings, axis=0)
        result['node_embeddings'] = torch.tensor(ne_np, dtype=torch.float32, device=device)
        del ne_np

    if all_node_sentinel_type:
        st_np = np.concatenate(all_node_sentinel_type, axis=0)
        result['node_sentinel_type'] = torch.tensor(st_np, dtype=torch.long, device=device)
        del st_np

    if labels is not None:
        result['y'] = torch.tensor(labels, dtype=torch.float, device=device)

    return result
