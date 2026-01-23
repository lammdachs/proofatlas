"""
Clause selector models for theorem proving.

This module provides modular components for clause selection:

Modular design (recommended):
    - Encoders: GCNEncoder, GATEncoder, GraphSAGEEncoder
    - Scorers: MLPScorer, AttentionScorer, TransformerScorer, CrossAttentionScorer
    - ClauseSelector: Combines encoder + projection + scorer

Legacy models (convenience wrappers):
    - ClauseGCN, ClauseGAT, ClauseGraphSAGE
    - ClauseTransformer, ClauseGNNTransformer

Example:
    # Modular approach
    encoder = GCNEncoder(hidden_dim=64, num_layers=3)
    selector = ClauseSelector(encoder, scorer_type="attention", scorer_dim=64)

    # Or use convenience wrapper
    model = ClauseGCN(hidden_dim=64, num_layers=3, scorer_type="attention")
"""

from .gnn import (
    ClauseGCN,
    ClauseGAT,
    ClauseGraphSAGE,
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    NodeFeatureEmbedding,
    ClauseFeatureEmbedding,
    FeatureEmbedding,  # Legacy alias
)
from .encoders import (
    ClauseEncoder,
    ClauseSelector,
    GCNEncoder,
    GATEncoder,
    GraphSAGEEncoder,
    create_encoder,
)
from .scorers import (
    MLPScorer,
    AttentionScorer,
    TransformerScorer,
    CrossAttentionScorer,
    create_scorer,
)
from .transformer import ClauseTransformer, ClauseGNNTransformer
from .baseline import NodeMLP, AgeWeightHeuristic
from .utils import normalize_adjacency, edge_index_to_adjacency, edge_index_to_sparse_adjacency, sparse_mm
from .factory import create_model

__all__ = [
    # Modular components (recommended)
    "ClauseEncoder",
    "ClauseSelector",
    "GCNEncoder",
    "GATEncoder",
    "GraphSAGEEncoder",
    "create_encoder",
    # Scorers
    "MLPScorer",
    "AttentionScorer",
    "TransformerScorer",
    "CrossAttentionScorer",
    "create_scorer",
    # Legacy GNN models
    "ClauseGCN",
    "ClauseGAT",
    "ClauseGraphSAGE",
    # GNN layers
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    # Feature embeddings
    "NodeFeatureEmbedding",
    "ClauseFeatureEmbedding",
    "FeatureEmbedding",
    # Transformer models
    "ClauseTransformer",
    "ClauseGNNTransformer",
    # Baselines
    "NodeMLP",
    "AgeWeightHeuristic",
    # Utilities
    "normalize_adjacency",
    "edge_index_to_adjacency",
    "edge_index_to_sparse_adjacency",
    "sparse_mm",
    # Factory
    "create_model",
]
