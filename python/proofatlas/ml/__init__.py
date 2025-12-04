"""Machine learning utilities for ProofAtlas

This module provides:
- Graph conversion: clause graphs to PyTorch tensors
- Data collection: extract training data from proofs
- Training: training loop and utilities
- Config: configuration loading for data and selectors

For selector models, use proofatlas.selectors directly.
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

from .training import (
    ClauseDataset,
    collate_clause_batch,
    JSONLogger,
    split_dataset,
    train,
    save_model,
    load_model,
)

from .config import (
    # Data config
    DataConfig,
    ProblemFilters,
    SplitConfig,
    SolverConfig,
    TraceCollectionConfig,
    OutputConfig,
    # Selector config
    SelectorConfig,
    ModelConfig,
    TrainingParams,
    OptimizerConfig,
    SchedulerConfig,
    # Utilities
    list_configs,
    merge_configs,
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
    # Training
    "ClauseDataset",
    "collate_clause_batch",
    "JSONLogger",
    "split_dataset",
    "train",
    "save_model",
    "load_model",
    # Config - Data
    "DataConfig",
    "ProblemFilters",
    "SplitConfig",
    "SolverConfig",
    "TraceCollectionConfig",
    "OutputConfig",
    # Config - Selector
    "SelectorConfig",
    "ModelConfig",
    "TrainingParams",
    "OptimizerConfig",
    "SchedulerConfig",
    # Config utilities
    "list_configs",
    "merge_configs",
]
