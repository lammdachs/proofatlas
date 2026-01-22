"""
Factory functions for creating and exporting clause selector models.
"""

import torch
import torch.nn as nn

from .gnn import ClauseGCN, ClauseGAT, ClauseGraphSAGE
from .transformer import ClauseTransformer, ClauseGNNTransformer
from .baseline import NodeMLP, AgeWeightHeuristic
from .sentence import SentenceEncoder, HAS_TRANSFORMERS


def create_model(
    model_type: str = "gcn",
    node_feature_dim: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a clause scoring model.

    New architecture (IJCAR26 plan):
    - Node features (3d): type, arity, arg_pos (for GCN encoder)
    - Clause features (3d): age, role, size (for scorer, sinusoidal encoded)

    Args:
        model_type: One of:
            - "gcn": Graph Convolutional Network
            - "gat": Graph Attention Network
            - "graphsage": GraphSAGE
            - "transformer": Transformer (no GNN)
            - "gnn_transformer": Hybrid GNN + Transformer
            - "mlp": Simple MLP baseline
            - "age_weight": Age-weight heuristic
            - "sentence": Pretrained sentence encoder (requires transformers)
        node_feature_dim: Input feature dimension (default: 3, legacy: 8 or 13)
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        **kwargs: Model-specific arguments:
            - dropout: Dropout rate (default: 0.1)
            - num_heads: Attention heads for GAT (default: 4)
            - scorer_type: Scoring head type: "mlp", "attention", "transformer", "cross_attention"
            - scorer_num_heads: Attention heads for attention-based scorers (default: 4)
            - scorer_num_layers: Layers for transformer scorer (default: 2)
            - use_clause_features: Use clause-level features in scorer (default: True)
            - sin_dim: Sinusoidal encoding dimension (default: 8)
            - sentence_model: Pretrained model name for "sentence" type
            - freeze_encoder: Freeze pretrained encoder (default: False)

    Returns:
        PyTorch model
    """
    # Common scorer parameters
    scorer_kwargs = {
        'scorer_type': kwargs.get('scorer_type', 'mlp'),
        'scorer_num_heads': kwargs.get('scorer_num_heads', 4),
        'scorer_num_layers': kwargs.get('scorer_num_layers', 2),
    }

    # New architecture parameters
    use_clause_features = kwargs.get('use_clause_features', True)
    sin_dim = kwargs.get('sin_dim', 8)

    if model_type == "gcn":
        return ClauseGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1),
            use_clause_features=use_clause_features,
            sin_dim=sin_dim,
            **scorer_kwargs,
        )
    elif model_type == "gat":
        return ClauseGAT(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
            use_clause_features=use_clause_features,
            sin_dim=sin_dim,
            **scorer_kwargs,
        )
    elif model_type == "graphsage":
        return ClauseGraphSAGE(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1),
            use_clause_features=use_clause_features,
            sin_dim=sin_dim,
            **scorer_kwargs,
        )
    elif model_type == "transformer":
        return ClauseTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == "gnn_transformer":
        return ClauseGNNTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=kwargs.get('num_gnn_layers', 2),
            num_transformer_layers=kwargs.get('num_transformer_layers', 2),
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == "mlp":
        return NodeMLP(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == "age_weight":
        return AgeWeightHeuristic(
            age_probability=kwargs.get('age_probability', 0.5),
        )
    elif model_type == "sentence":
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for sentence model. "
                "Install with: pip install transformers"
            )
        return SentenceEncoder(
            model_name=kwargs.get('sentence_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            hidden_dim=hidden_dim,
            freeze_encoder=kwargs.get('freeze_encoder', False),
            **scorer_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
