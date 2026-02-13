"""
Utility functions for clause selectors.
"""

import torch


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
