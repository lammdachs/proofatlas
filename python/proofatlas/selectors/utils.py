"""
Utility functions for clause selectors.
"""

import torch


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Normalize adjacency matrix for GCN: D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix [num_nodes, num_nodes]
        add_self_loops: Whether to add self-loops before normalizing

    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Convert edge_index [2, num_edges] to adjacency matrix [num_nodes, num_nodes].
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0

    if add_self_loops:
        adj = adj + torch.eye(num_nodes, device=adj.device)

    return adj
