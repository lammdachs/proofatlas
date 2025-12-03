"""Machine learning utilities for ProofAtlas

This module provides:
- Graph conversion: clause graphs to PyTorch tensors
- Data collection: extract training data from proofs
- Models: GNN architectures for clause scoring (pure PyTorch, no PyG dependency)
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
    # GNN layers
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    # Utility functions
    normalize_adjacency,
    edge_index_to_adjacency,
    # GNN models
    ClauseGCN,
    ClauseGAT,
    ClauseGraphSAGE,
    # Transformer models
    ClauseTransformer,
    ClauseGNNTransformer,
    # Baseline models
    NodeMLP,
    AgeWeightHeuristic,
    # Factory and export
    create_model,
    export_to_onnx,
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
    TraceCollectionConfig,
    OutputConfig,
    # Training config
    TrainingConfig,
    ModelConfig,
    TrainingParams,
    OptimizerConfig,
    SchedulerConfig,
    DistributedConfig,
    EvaluationConfig,
    CheckpointConfig,
    LoggingConfig,
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
    # GNN layers
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    # Utility functions
    "normalize_adjacency",
    "edge_index_to_adjacency",
    # GNN models
    "ClauseGCN",
    "ClauseGAT",
    "ClauseGraphSAGE",
    # Transformer models
    "ClauseTransformer",
    "ClauseGNNTransformer",
    # Baseline models
    "NodeMLP",
    "AgeWeightHeuristic",
    # Factory and export
    "create_model",
    "export_to_onnx",
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
    "TraceCollectionConfig",
    "OutputConfig",
    # Config - Training
    "TrainingConfig",
    "ModelConfig",
    "TrainingParams",
    "OptimizerConfig",
    "SchedulerConfig",
    "DistributedConfig",
    "EvaluationConfig",
    "CheckpointConfig",
    "LoggingConfig",
    # Config utilities
    "list_configs",
    "merge_configs",
]
