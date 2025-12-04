"""
Factory functions for creating and exporting clause selector models.
"""

import torch
import torch.nn as nn

from .gnn import ClauseGCN, ClauseGAT, ClauseGraphSAGE
from .transformer import ClauseTransformer, ClauseGNNTransformer
from .baseline import NodeMLP, AgeWeightHeuristic


def create_model(
    model_type: str = "gcn",
    node_feature_dim: int = 13,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a clause scoring model.

    Args:
        model_type: One of:
            - "gcn": Graph Convolutional Network
            - "gat": Graph Attention Network
            - "graphsage": GraphSAGE
            - "transformer": Transformer (no GNN)
            - "gnn_transformer": Hybrid GNN + Transformer
            - "mlp": Simple MLP baseline
            - "age_weight": Age-weight heuristic
        node_feature_dim: Input feature dimension (default: 13)
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        **kwargs: Model-specific arguments

    Returns:
        PyTorch model
    """
    if model_type == "gcn":
        return ClauseGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == "gat":
        return ClauseGAT(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_type == "graphsage":
        return ClauseGraphSAGE(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.1),
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    num_clauses: int = 10,
    total_nodes: int = 50,
    include_adj: bool = False,
):
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        num_clauses: Example number of clauses
        total_nodes: Example total nodes
        include_adj: Whether model requires adjacency matrix
    """
    model.eval()

    dummy_features = torch.randn(total_nodes, 13)
    dummy_pool = torch.randn(num_clauses, total_nodes)

    if include_adj:
        dummy_adj = torch.randn(total_nodes, total_nodes)
        inputs = (dummy_features, dummy_adj, dummy_pool)
        input_names = ["node_features", "adjacency", "pool_matrix"]
        dynamic_axes = {
            "node_features": {0: "total_nodes"},
            "adjacency": {0: "total_nodes", 1: "total_nodes"},
            "pool_matrix": {0: "num_clauses", 1: "total_nodes"},
            "scores": {0: "num_clauses"},
        }
    else:
        inputs = (dummy_features, dummy_pool)
        input_names = ["node_features", "pool_matrix"]
        dynamic_axes = {
            "node_features": {0: "total_nodes"},
            "pool_matrix": {0: "num_clauses", 1: "total_nodes"},
            "scores": {0: "num_clauses"},
        }

    torch.onnx.export(
        model,
        inputs,
        output_path,
        input_names=input_names,
        output_names=["scores"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print(f"Exported model to {output_path}")
