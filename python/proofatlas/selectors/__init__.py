"""
Clause selector models for theorem proving.

This module provides components for clause selection:

Models:
    - ClauseGCN: Graph Convolutional Network encoder + scorer
    - SentenceEncoder: Sentence transformer encoder + scorer
    - ClauseFeatures: Clause feature MLP encoder + scorer

Scorers:
    - MLPScorer, AttentionScorer, TransformerScorer

Factory:
    - create_model: Create any model by type string

Example:
    model = ClauseGCN(hidden_dim=64, num_layers=3, scorer_type="attention")
    # Or via factory:
    model = create_model("gcn", hidden_dim=64, num_layers=3, scorer_type="attention")
"""

from .gnn import (
    ClauseGCN,
    GCNLayer,
    NodeFeatureEmbedding,
    ClauseFeatureEmbedding,
)
from .scorers import (
    MLPScorer,
    AttentionScorer,
    TransformerScorer,
    create_scorer,
)
from .utils import sparse_mm
from .factory import create_model

__all__ = [
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
    # Utilities
    "sparse_mm",
    # Factory
    "create_model",
]
