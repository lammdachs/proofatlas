"""Machine learning utilities for ProofAtlas

This module provides:
- Graph conversion: clause graphs to PyTorch tensors
- Structured data: JSON trace format with converters
- Training: training loop and utilities
- Config: configuration loading for data and selectors

For selector models, use proofatlas.selectors directly.
"""

from .graph_utils import (
    to_torch_tensors,
    to_sparse_adjacency,
    batch_graphs,
    extract_graph_embeddings,
    get_node_type_masks,
    compute_graph_statistics,
)

from .training import (
    # Loss functions
    info_nce_loss,
    info_nce_loss_per_proof,
    margin_ranking_loss,
    # Datasets
    ProofDataset,
    collate_proof_batch,
    collate_sentence_batch,
    # Training
    run_training,
    load_trace_files,
    save_trace,
    # Utilities
    JSONLogger,
    save_model,
    load_model,
)

from .weights import (
    find_weights,
    is_learned_selector,
    get_model_name,
    get_embedding_type,
    STRING_EMBEDDINGS,
    GRAPH_EMBEDDINGS,
)

from .config import (
    # Data config
    DataConfig,
    ProblemFilters,
    SplitConfig,
    SolverConfig,
    TraceCollectionConfig,
    OutputConfig,
    # Training config
    TrainingConfig,
    ModelConfig,
    TrainingParams,
    OptimizerConfig,
    SchedulerConfig,
    # Utilities
    list_configs,
    merge_configs,
)

from .structured import (
    # Conversion functions
    clause_to_string,
    clause_to_graph,
    clauses_to_strings,
    clauses_to_graphs,
    load_structured_trace,
    batch_graphs as batch_structured_graphs,
)

__all__ = [
    # Graph utilities
    "to_torch_tensors",
    "to_sparse_adjacency",
    "batch_graphs",
    "extract_graph_embeddings",
    "get_node_type_masks",
    "compute_graph_statistics",
    # Loss functions
    "info_nce_loss",
    "info_nce_loss_per_proof",
    "margin_ranking_loss",
    # Datasets
    "ProofDataset",
    "collate_proof_batch",
    "collate_sentence_batch",
    # Training
    "run_training",
    "load_trace_files",
    "save_trace",
    # Utilities
    "JSONLogger",
    "save_model",
    "load_model",
    # Weights
    "find_weights",
    "is_learned_selector",
    "get_model_name",
    "get_embedding_type",
    "STRING_EMBEDDINGS",
    "GRAPH_EMBEDDINGS",
    # Config - Data
    "DataConfig",
    "ProblemFilters",
    "SplitConfig",
    "SolverConfig",
    "TraceCollectionConfig",
    "OutputConfig",
    # Config - Training
    "TrainingConfig",
    "ModelConfig",
    "TrainingParams",
    "OptimizerConfig",
    "SchedulerConfig",
    # Config utilities
    "list_configs",
    "merge_configs",
    # Structured data conversion
    "clause_to_string",
    "clause_to_graph",
    "clauses_to_strings",
    "clauses_to_graphs",
    "load_structured_trace",
    "batch_structured_graphs",
]
