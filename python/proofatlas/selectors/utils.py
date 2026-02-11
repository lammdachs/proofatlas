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


def sparse_mm(adj, x: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiply adjacency with node features using scatter-based message passing.

    Accepts either:
    - Edge-list tuple: (row, col, val, shape) — preferred, avoids COO tensor bugs
    - Sparse COO/CSR tensor
    - Dense tensor

    Args:
        adj: Matrix in any of the above formats
        x: [N, features] node features

    Returns:
        [M, features] aggregated features
    """
    if isinstance(adj, (tuple, list)):
        row, col, val, shape = adj
        messages = x[col] * val.unsqueeze(-1)
        out = torch.zeros(shape[0], x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        return out
    elif adj.is_sparse:
        # Fallback for sparse tensors (e.g., from TorchScript export path)
        adj_c = adj.coalesce()
        row = adj_c.indices()[0].clone()
        col = adj_c.indices()[1].clone()
        val = adj_c.values().clone()
        messages = x[col] * val.unsqueeze(-1)
        out = torch.zeros(adj.size(0), x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        return out
    else:
        return torch.mm(adj, x)
