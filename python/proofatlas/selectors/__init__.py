"""
Clause selector models for theorem proving.

This module provides modular components for clause selection:

Modular design (recommended):
    - Encoders: GCNEncoder
    - Scorers: MLPScorer, AttentionScorer, TransformerScorer
    - ClauseSelector: Combines encoder + projection + scorer

Convenience wrappers:
    - ClauseGCN

Example:
    # Modular approach
    encoder = GCNEncoder(hidden_dim=64, num_layers=3)
    selector = ClauseSelector(encoder, scorer_type="attention", scorer_dim=64)

    # Or use convenience wrapper
    model = ClauseGCN(hidden_dim=64, num_layers=3, scorer_type="attention")
"""

from .gnn import (
    ClauseGCN,
    GCNLayer,
    NodeFeatureEmbedding,
    ClauseFeatureEmbedding,
)
from .encoders import (
    ClauseEncoder,
    ClauseSelector,
    GCNEncoder,
    create_encoder,
)
from .scorers import (
    MLPScorer,
    AttentionScorer,
    TransformerScorer,
    create_scorer,
)
from .baseline import AgeWeightHeuristic
from .utils import normalize_adjacency, sparse_mm
from .factory import create_model

__all__ = [
    # Modular components (recommended)
    "ClauseEncoder",
    "ClauseSelector",
    "GCNEncoder",
    "create_encoder",
    # Scorers
    "MLPScorer",
    "AttentionScorer",
    "TransformerScorer",
    "create_scorer",
    # GNN models
    "ClauseGCN",
    # GNN layers
    "GCNLayer",
    # Feature embeddings
    "NodeFeatureEmbedding",
    "ClauseFeatureEmbedding",
    # Baselines
    "AgeWeightHeuristic",
    # Utilities
    "normalize_adjacency",
    "sparse_mm",
    # Factory
    "create_model",
]
