"""GNN model for clause quality prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from typing import Optional


class ClauseGNN(nn.Module):
    """
    Graph Neural Network for predicting clause quality (proof relevance).

    Takes a clause graph and outputs a score indicating how likely
    the clause is to be useful for finding a proof.

    Architecture:
        Input (node features) -> GCN layers -> Global pooling -> MLP -> Score
    """

    def __init__(
        self,
        node_feature_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features (default: 20)
            hidden_dim: Hidden dimension for GCN layers
            num_layers: Number of GCN layers
            dropout: Dropout rate
            pooling: Graph pooling method ("mean", "sum", "max")
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (for batched graphs)

        Returns:
            Clause scores [batch_size] or [1] for single graph
        """
        # GCN layers with batch norm and dropout
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if batch is None:
            # Single graph - pool all nodes
            if self.pooling == "mean":
                x = x.mean(dim=0, keepdim=True)
            elif self.pooling == "sum":
                x = x.sum(dim=0, keepdim=True)
            elif self.pooling == "max":
                x = x.max(dim=0, keepdim=True)[0]
        else:
            # Batched graphs
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "sum":
                x = global_add_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)

        # MLP to get final score
        x = self.mlp(x)

        return x.squeeze(-1)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get probability that clause is in proof (sigmoid of score)"""
        scores = self.forward(x, edge_index, batch)
        return torch.sigmoid(scores)


class ClauseGNNWithAttention(nn.Module):
    """
    GNN with attention-based pooling for clause scoring.

    Uses attention to weight node contributions to the graph embedding,
    which can help focus on important parts of the clause.
    """

    def __init__(
        self,
        node_feature_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Attention-based pooling
        attn_weights = self.attention(x)  # [num_nodes, 1]

        if batch is None:
            # Single graph
            attn_weights = F.softmax(attn_weights, dim=0)
            x = (x * attn_weights).sum(dim=0, keepdim=True)
        else:
            # Batched graphs - softmax within each graph
            from torch_geometric.utils import softmax
            attn_weights = softmax(attn_weights.squeeze(-1), batch)
            x = x * attn_weights.unsqueeze(-1)
            x = global_add_pool(x, batch)

        # MLP
        x = self.mlp(x)

        return x.squeeze(-1)


def create_model(
    model_type: str = "gcn",
    node_feature_dim: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a clause scoring model.

    Args:
        model_type: "gcn" or "gcn_attention"
        node_feature_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model
    """
    if model_type == "gcn":
        return ClauseGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )
    elif model_type == "gcn_attention":
        return ClauseGNNWithAttention(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
