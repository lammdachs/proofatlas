"""
Clause encoders for modular clause selection.

Encoders produce clause embeddings from graph/text inputs.
They can be combined with any scorer via ClauseSelector.

Architecture:
    Encoder → [num_clauses, encoder_dim]
        ↓
    Projection (optional) → [num_clauses, scorer_dim]
        ↓
    Scorer → [num_clauses]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from .gnn import (
    NodeFeatureEmbedding,
    ClauseFeatureEmbedding,
    GCNLayer,
    GraphNorm,
    _batch_from_pool,
)
from .scorers import create_scorer


class ClauseEncoder(ABC, nn.Module):
    """Base class for clause encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of output clause embeddings."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Encode clauses to embeddings.

        Returns:
            Clause embeddings [num_clauses, output_dim]
        """
        pass


class GCNEncoder(ClauseEncoder):
    """
    GCN-based clause encoder.

    Encodes clause graphs to embeddings using Graph Convolutional Networks.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        sin_dim: int = 8,
        use_clause_features: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_clause_features = use_clause_features

        # Node feature embedding
        self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.node_embedding.output_dim

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])

        # Clause feature embedding (optional)
        if use_clause_features:
            self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)
            # Project concatenated features back to hidden_dim (matching ClauseGCN)
            concat_dim = hidden_dim + self.clause_embedding.output_dim
            self.clause_proj = nn.Linear(concat_dim, hidden_dim)
            self._output_dim = hidden_dim
        else:
            self.clause_embedding = None
            self.clause_proj = None
            self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode clauses to embeddings.

        Args:
            node_features: [total_nodes, 3] raw node features
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes]
            clause_features: [num_clauses, 3] raw clause features (optional)

        Returns:
            Clause embeddings [num_clauses, output_dim]
        """
        # Derive batch tensor (node → graph mapping) from pool matrix
        batch = _batch_from_pool(pool_matrix)

        # Embed node features
        x = self.node_embedding(node_features)

        # GCN message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x, batch)
            x = F.relu(x)

        # Pool to clause level
        clause_emb = torch.mm(pool_matrix, x)

        # Add clause features if available
        if self.use_clause_features and self.clause_embedding is not None:
            if clause_features is not None:
                clause_feat_emb = self.clause_embedding(clause_features)
            else:
                num_clauses = pool_matrix.size(0)
                clause_feat_emb = torch.zeros(
                    num_clauses, self.clause_embedding.output_dim,
                    device=clause_emb.device, dtype=clause_emb.dtype
                )
            clause_emb = self.clause_proj(torch.cat([clause_emb, clause_feat_emb], dim=-1))

        return clause_emb


class ClauseSelector(nn.Module):
    """
    Modular clause selector combining encoder and scorer.

    Architecture:
        encoder → [num_clauses, encoder_dim]
        projection → [num_clauses, scorer_dim]
        scorer → [num_clauses]

    Example:
        encoder = GCNEncoder(hidden_dim=64, num_layers=3)
        selector = ClauseSelector(encoder, scorer_type="attention", scorer_dim=64)
    """

    def __init__(
        self,
        encoder: ClauseEncoder,
        scorer_type: str = "mlp",
        scorer_dim: int = 64,
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.scorer_dim = scorer_dim

        # Projection layer if encoder output dim != scorer dim
        if encoder.output_dim != scorer_dim:
            self.projection = nn.Linear(encoder.output_dim, scorer_dim)
        else:
            self.projection = nn.Identity()

        # Scorer
        self.scorer = create_scorer(
            scorer_type,
            scorer_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through encoder → projection → scorer.

        Args are passed directly to the encoder.

        Returns:
            Scores [num_clauses]
        """
        # Encode
        clause_emb = self.encoder(*args, **kwargs)

        # Project
        clause_emb = self.projection(clause_emb)

        # Score
        return self.scorer(clause_emb).view(-1)

    def export_torchscript(self, path: str):
        """Export model to TorchScript format."""
        self.eval()

        # Create dummy inputs (assumes GNN encoder interface)
        num_nodes, num_clauses = 10, 3

        dummy_node_features = torch.zeros(num_nodes, 3)
        dummy_node_features[:, 0] = torch.randint(0, 6, (num_nodes,)).float()
        dummy_node_features[:, 1] = torch.randint(0, 5, (num_nodes,)).float()
        dummy_node_features[:, 2] = torch.randint(0, 10, (num_nodes,)).float()

        dummy_adj = torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes)
        dummy_adj = dummy_adj / dummy_adj.sum(dim=1, keepdim=True)

        dummy_pool_matrix = torch.ones(num_clauses, num_nodes) / num_nodes

        dummy_clause_features = torch.zeros(num_clauses, 9)
        dummy_clause_features[:, 0] = torch.rand(num_clauses)                       # age
        dummy_clause_features[:, 1] = torch.randint(0, 5, (num_clauses,)).float()   # role
        dummy_clause_features[:, 2] = torch.randint(0, 7, (num_clauses,)).float()   # rule
        dummy_clause_features[:, 3] = torch.randint(1, 20, (num_clauses,)).float()  # size
        dummy_clause_features[:, 4] = torch.randint(0, 10, (num_clauses,)).float()  # depth
        dummy_clause_features[:, 5] = torch.randint(1, 30, (num_clauses,)).float()  # symbol_count
        dummy_clause_features[:, 6] = torch.randint(1, 15, (num_clauses,)).float()  # distinct_symbols
        dummy_clause_features[:, 7] = torch.randint(0, 10, (num_clauses,)).float()  # variable_count
        dummy_clause_features[:, 8] = torch.randint(0, 5, (num_clauses,)).float()   # distinct_vars

        with torch.no_grad():
            traced = torch.jit.trace(
                self,
                (dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
            )

        traced.save(path)
        print(f"Exported TorchScript model to {path}")

        # Verify
        with torch.no_grad():
            original_out = self(dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
            traced_out = traced(dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
            diff = (original_out - traced_out).abs().max().item()
            print(f"Verification: max diff = {diff:.6e}")


def create_encoder(
    encoder_type: str,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> ClauseEncoder:
    """
    Factory function to create an encoder.

    Args:
        encoder_type: "gcn"
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        **kwargs: Additional encoder-specific arguments

    Returns:
        ClauseEncoder instance
    """
    if encoder_type == "gcn":
        return GCNEncoder(hidden_dim=hidden_dim, num_layers=num_layers, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: gcn")
