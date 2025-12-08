"""
GNN-based clause selector models.

Pure PyTorch implementations (no PyTorch Geometric dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    h_i' = σ(W · mean({h_j : j ∈ N(i) ∪ {i}}))

    Uses adjacency matrix for message passing instead of edge_index.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes] (with self-loops, normalized)

        Returns:
            Updated features [num_nodes, out_dim]
        """
        h = torch.mm(adj, x)
        return self.linear(h)


class ScorerHead(nn.Module):
    """
    MLP scoring head for clause scoring.

    Named fields match Burn's ScorerHead structure for weight compatibility.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear2(x)


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Velickovic et al., 2018).

    Computes attention coefficients between connected nodes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes] (binary, with self-loops)

        Returns:
            Updated features [num_nodes, out_dim * num_heads] if concat
                          or [num_nodes, out_dim] if not concat
        """
        num_nodes = x.size(0)

        h = self.W(x).view(num_nodes, self.num_heads, self.out_dim)

        attn_src = (h * self.a_src).sum(dim=-1)
        attn_dst = (h * self.a_dst).sum(dim=-1)

        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn = self.leaky_relu(attn)

        mask = (adj == 0).unsqueeze(-1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        h = h.transpose(0, 1)
        attn = attn.permute(2, 0, 1)
        out = torch.bmm(attn, h)
        out = out.permute(1, 0, 2)

        if self.concat:
            return out.reshape(num_nodes, -1)
        else:
            return out.mean(dim=1)


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer (Hamilton et al., 2017).

    h_i' = σ(W · concat(h_i, mean({h_j : j ∈ N(i)})))
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes] (normalized, without self-loops)

        Returns:
            Updated features [num_nodes, out_dim]
        """
        neighbor_agg = torch.mm(adj, x)
        h = torch.cat([x, neighbor_agg], dim=-1)
        return self.linear(h)


class ClauseGCN(nn.Module):
    """
    Graph Convolutional Network for clause scoring.

    Architecture:
        node_features → GCN layers → pool to clauses → MLP → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.scorer = ScorerHead(hidden_dim, dropout)

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
            pool_matrix: [num_clauses, total_nodes] for pooling nodes to clauses

        Returns:
            Scores [num_clauses]
        """
        x = node_features

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)
        return self.scorer(clause_emb).squeeze(-1)


class ClauseGAT(nn.Module):
    """
    Graph Attention Network for clause scoring.

    Architecture:
        node_features → GAT layers → pool to clauses → MLP → scores
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
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATLayer(node_feature_dim, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=False, dropout=dropout))

        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, node_feature_dim]
            adj: Binary adjacency matrix [total_nodes, total_nodes] (with self-loops)
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        x = node_features

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)
        return self.scorer(clause_emb).squeeze(-1)


class ClauseGraphSAGE(nn.Module):
    """
    GraphSAGE for clause scoring.

    Architecture:
        node_features → GraphSAGE layers → pool to clauses → MLP → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGELayer(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.scorer = ScorerHead(hidden_dim, dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, node_feature_dim]
            adj: Normalized adjacency matrix [total_nodes, total_nodes] (without self-loops for neighbor agg)
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        x = node_features

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)
        return self.scorer(clause_emb).squeeze(-1)
