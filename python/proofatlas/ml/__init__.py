"""Machine learning utilities for ProofAtlas

This module provides:
- Graph conversion: clause graphs to PyTorch tensors
- Data collection: extract training data from proofs
- Models: GNN architectures for clause scoring
- Training: training loop and utilities
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

from .model import (
    ClauseGNN,
    ClauseGNNWithAttention,
    create_model,
)

from .training import (
    TrainingConfig,
    create_pyg_dataset,
    split_dataset,
    train,
    save_model,
    load_model,
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
    # Models
    "ClauseGNN",
    "ClauseGNNWithAttention",
    "create_model",
    # Training
    "TrainingConfig",
    "create_pyg_dataset",
    "split_dataset",
    "train",
    "save_model",
    "load_model",
]
