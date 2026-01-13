"""
Clause selector models for theorem proving.

This module contains neural network models for clause selection:
- GNN-based: ClauseGCN, ClauseGAT, ClauseGraphSAGE
- Transformer-based: ClauseTransformer, ClauseGNNTransformer
- Baselines: NodeMLP, AgeWeightHeuristic

All models output clause scores for selection in the given-clause algorithm.
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
from .transformer import ClauseTransformer, ClauseGNNTransformer
from .baseline import NodeMLP, AgeWeightHeuristic
from .utils import normalize_adjacency, edge_index_to_adjacency
from .factory import create_model, export_to_onnx

__all__ = [
    # GNN models
    "ClauseGCN",
    "ClauseGAT",
    "ClauseGraphSAGE",
    # GNN layers
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    # Feature embeddings (IJCAR26 architecture)
    "NodeFeatureEmbedding",
    "ClauseFeatureEmbedding",
    "FeatureEmbedding",  # Legacy alias
    # Transformer models
    "ClauseTransformer",
    "ClauseGNNTransformer",
    # Baselines
    "NodeMLP",
    "AgeWeightHeuristic",
    # Utilities
    "normalize_adjacency",
    "edge_index_to_adjacency",
    # Factory
    "create_model",
    "export_to_onnx",
]
