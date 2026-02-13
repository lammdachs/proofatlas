"""Machine learning utilities for ProofAtlas

This module provides:
- Graph conversion: clause graphs to PyTorch tensors
- Structured data: .npz trace format with converters
- Training: training loop and utilities

For selector models, use proofatlas.selectors directly.
"""

from .losses import (
    info_nce_loss,
    info_nce_loss_per_proof,
    margin_ranking_loss,
    compute_loss,
)

from .datasets import (
    ProofBatchDataset,
    collate_proof_batch,
    collate_sentence_batch,
    scan_trace_files,
)

from .logger import JSONLogger

from .export import export_model

from .training import (
    run_training,
)

from .weights import (
    find_weights,
    is_learned_selector,
    get_model_name,
    get_encoder_type,
)

__all__ = [
    # Loss functions
    "info_nce_loss",
    "info_nce_loss_per_proof",
    "margin_ranking_loss",
    "compute_loss",
    # Datasets
    "ProofBatchDataset",
    "collate_proof_batch",
    "collate_sentence_batch",
    "scan_trace_files",
    # Training
    "run_training",
    # Export
    "export_model",
    # Utilities
    "JSONLogger",
    # Weights
    "find_weights",
    "is_learned_selector",
    "get_model_name",
    "get_encoder_type",
]
