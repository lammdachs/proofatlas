"""
Transformer-based clause selector models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GCNLayer


class ClauseTransformer(nn.Module):
    """
    Transformer for clause scoring with cross-clause attention.

    Architecture:
        node_features → MLP → pool to clauses → transformer layers → scores

    No GNN - uses only node features pooled to clause level, then
    transformer attention to model clause interactions.
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }))

        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, node_feature_dim]
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        h = self.node_encoder(node_features)
        x = torch.mm(pool_matrix, h)

        x = x.unsqueeze(0)

        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)

        x = x.squeeze(0)
        return self.scorer(x).squeeze(-1)


class ClauseGNNTransformer(nn.Module):
    """
    Hybrid GNN + Transformer for clause scoring.

    Architecture:
        node_features → GCN layers → pool to clauses → transformer layers → scores

    Combines:
    - GCN: Captures within-clause structure
    - Transformer: Captures cross-clause relationships
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNLayer(node_feature_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.gnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }))

        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, node_feature_dim]
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        x = node_features
        for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)

        x = clause_emb.unsqueeze(0)

        for layer in self.transformer_layers:
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)

        x = x.squeeze(0)
        return self.scorer(x).squeeze(-1)
