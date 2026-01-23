"""
Utility functions for clause selectors.
"""

import torch


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Normalize adjacency matrix for GCN: D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix [num_nodes, num_nodes] (dense or sparse)
        add_self_loops: Whether to add self-loops before normalizing

    Returns:
        Normalized adjacency matrix (same format as input)
    """
    if adj.is_sparse:
        return normalize_adjacency_sparse(adj, add_self_loops)

    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def normalize_adjacency_sparse(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Normalize sparse adjacency matrix for GCN: D^{-1/2} A D^{-1/2}

    Args:
        adj: Sparse COO adjacency matrix [num_nodes, num_nodes]
        add_self_loops: Whether to add self-loops before normalizing

    Returns:
        Normalized sparse adjacency matrix
    """
    num_nodes = adj.size(0)
    indices = adj.coalesce().indices()
    values = adj.coalesce().values()

    if add_self_loops:
        # Add self-loop indices
        self_loop_idx = torch.arange(num_nodes, device=adj.device)
        self_loop_indices = torch.stack([self_loop_idx, self_loop_idx])
        indices = torch.cat([indices, self_loop_indices], dim=1)
        values = torch.cat([values, torch.ones(num_nodes, device=adj.device)])

    # Compute degree
    row = indices[0]
    deg = torch.zeros(num_nodes, device=adj.device)
    deg.scatter_add_(0, row, values)

    # D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalize edge values: D^{-1/2} A D^{-1/2}
    row, col = indices
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes)).coalesce()


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Convert edge_index [2, num_edges] to dense adjacency matrix.
    For large graphs, use edge_index_to_sparse_adjacency instead.
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0

    if add_self_loops:
        adj = adj + torch.eye(num_nodes, device=adj.device)

    return adj


def edge_index_to_sparse_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Convert edge_index [2, num_edges] to sparse COO adjacency matrix.

    Much more memory efficient for large graphs:
    - Dense: O(num_nodes^2) memory
    - Sparse: O(num_edges) memory

    Args:
        edge_index: [2, num_edges] edge indices
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops

    Returns:
        Sparse COO adjacency matrix [num_nodes, num_nodes]
    """
    num_edges = edge_index.size(1)
    device = edge_index.device

    if add_self_loops:
        # Add self-loop indices
        self_loop_idx = torch.arange(num_nodes, device=device)
        self_loop_indices = torch.stack([self_loop_idx, self_loop_idx])
        indices = torch.cat([edge_index, self_loop_indices], dim=1)
        values = torch.ones(num_edges + num_nodes, device=device)
    else:
        indices = edge_index
        values = torch.ones(num_edges, device=device)

    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def sparse_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiply adjacency (sparse or dense) with node features.

    Args:
        adj: [num_nodes, num_nodes] adjacency matrix (sparse or dense)
        x: [num_nodes, features] node features

    Returns:
        [num_nodes, features] aggregated features
    """
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    else:
        return torch.mm(adj, x)
