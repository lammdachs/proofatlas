"""Machine learning utilities for ProofAtlas

This module provides utilities for converting clause graphs to PyTorch tensors
and batching graphs for GNN training.
"""

from .graph_utils import (
    to_torch_tensors,
    to_torch_geometric,
    to_sparse_adjacency,
    batch_graphs,
    batch_graphs_geometric,
    create_dataloader,
    extract_graph_embeddings,
    get_node_type_masks,
    compute_graph_statistics,
)

__all__ = [
    "to_torch_tensors",
    "to_torch_geometric",
    "to_sparse_adjacency",
    "batch_graphs",
    "batch_graphs_geometric",
    "create_dataloader",
    "extract_graph_embeddings",
    "get_node_type_masks",
    "compute_graph_statistics",
]
