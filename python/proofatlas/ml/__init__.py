"""Machine learning utilities for ProofAtlas

This module provides utilities for converting clause graphs to PyTorch tensors,
batching graphs for GNN training, and collecting training data from proofs.
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

from .data_collection import (
    TrainingDataset,
    run_saturation_loop,
    collect_from_problem,
    collect_from_directory,
    load_training_dataset,
)

__all__ = [
    # Graph utilities
    "to_torch_tensors",
    "to_torch_geometric",
    "to_sparse_adjacency",
    "batch_graphs",
    "batch_graphs_geometric",
    "create_dataloader",
    "extract_graph_embeddings",
    "get_node_type_masks",
    "compute_graph_statistics",
    # Data collection
    "TrainingDataset",
    "run_saturation_loop",
    "collect_from_problem",
    "collect_from_directory",
    "load_training_dataset",
]
