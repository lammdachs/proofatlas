"""
GNN-based clause selector models.

Pure PyTorch implementations (no PyTorch Geometric dependency).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scorers import create_scorer, MLPScorer


class FeatureEmbedding(nn.Module):
    """
    Embeds raw node features into a richer representation.

    Raw feature layout (8 dims from graph.rs):
        0: Node type (int 0-5: clause, literal, predicate, function, variable, constant)
        1: Arity (raw int)
        2: Arg position (raw int)
        3: Depth (raw int)
        4: Age (normalized 0-1)
        5: Role (int 0-4: axiom, hypothesis, definition, negated_conjecture, derived)
        6: Polarity (binary)
        7: Is equality (binary)

    Output layout:
        - Node type: one-hot (6 dims)
        - Arity: log1p scaled (1 dim)
        - Arg position: sinusoidal (sin_dim dims)
        - Depth: sinusoidal (sin_dim dims)
        - Age: sinusoidal (sin_dim dims)
        - Role: one-hot (5 dims)
        - Polarity: kept (1 dim)
        - Is equality: kept (1 dim)
    """

    def __init__(self, sin_dim: int = 8):
        """
        Args:
            sin_dim: Dimension of sinusoidal encoding for each continuous feature.
                     Must be even. Higher = more expressivity.
        """
        super().__init__()
        assert sin_dim % 2 == 0, "sin_dim must be even"
        self.sin_dim = sin_dim

        # Output dim: 6 (type) + 1 (arity) + 3*sin_dim (pos, depth, age) + 5 (role) + 2 (polarity, eq)
        self.output_dim = 6 + 1 + 3 * sin_dim + 5 + 2

        # Precompute div_term for sinusoidal encoding
        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        div_term = torch.exp(torch.arange(0, sin_dim, 2).float() * (-math.log(10000.0) / sin_dim))
        self.register_buffer('div_term', div_term)

    def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding to values.

        Args:
            values: [N] or [N, 1] tensor of values

        Returns:
            [N, sin_dim] tensor of encoded values
        """
        if values.dim() == 1:
            values = values.unsqueeze(-1)  # [N, 1]

        # Scale values for better frequency coverage
        scaled = values * self.div_term  # [N, sin_dim/2]

        # Interleave sin and cos
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)

        # Stack: [N, sin_dim/2, 2] -> [N, sin_dim]
        return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed raw features.

        Args:
            x: [N, 8] raw node features

        Returns:
            [N, output_dim] embedded features
        """
        # Extract raw features (8-dim compact format)
        node_type = x[:, 0].long()  # int 0-5
        arity = x[:, 1]             # raw int
        arg_pos = x[:, 2]           # raw int
        depth = x[:, 3]             # raw int
        age = x[:, 4]               # normalized 0-1
        role = x[:, 5].long()       # int 0-4
        polarity = x[:, 6:7]        # binary
        is_equality = x[:, 7:8]     # binary

        # Node type to one-hot
        node_type_onehot = F.one_hot(node_type.clamp(0, 5), num_classes=6).float()  # [N, 6]

        # Encode continuous features
        arity_enc = torch.log1p(arity).unsqueeze(-1)  # [N, 1]
        arg_pos_enc = self.sinusoidal_encode(arg_pos)  # [N, sin_dim]
        depth_enc = self.sinusoidal_encode(depth)      # [N, sin_dim]
        age_enc = self.sinusoidal_encode(age * 100)    # [N, sin_dim], scale to 0-100

        # Role to one-hot
        role_onehot = F.one_hot(role.clamp(0, 4), num_classes=5).float()  # [N, 5]

        # Concatenate all
        return torch.cat([
            node_type_onehot,  # 6
            arity_enc,         # 1
            arg_pos_enc,       # sin_dim
            depth_enc,         # sin_dim
            age_enc,           # sin_dim
            role_onehot,       # 5
            polarity,          # 1
            is_equality,       # 1
        ], dim=-1)


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
        raw_features → feature_embedding → GCN layers → pool to clauses → scorer → scores

    The feature embedding applies sinusoidal encoding to continuous features
    (arg_position, depth, age) and one-hot encoding to categorical features (role).
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Feature embedding: raw 13-dim → richer representation
        self.feature_embedding = FeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.feature_embedding.output_dim

        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 13] raw features from graph.rs
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes] for pooling nodes to clauses

        Returns:
            Scores [num_clauses]
        """
        # Embed raw features
        x = self.feature_embedding(node_features)

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
        raw_features → feature_embedding → GAT layers → pool to clauses → scorer → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Feature embedding
        self.feature_embedding = FeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.feature_embedding.output_dim

        self.convs = nn.ModuleList()
        self.convs.append(GATLayer(embed_dim, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=False, dropout=dropout))

        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 13] raw features
            adj: Binary adjacency matrix [total_nodes, total_nodes] (with self-loops)
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        x = self.feature_embedding(node_features)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)
        return self.scorer(clause_emb)


class ClauseGraphSAGE(nn.Module):
    """
    GraphSAGE for clause scoring.

    Architecture:
        raw_features → feature_embedding → GraphSAGE layers → pool to clauses → scorer → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Feature embedding
        self.feature_embedding = FeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.feature_embedding.output_dim

        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGELayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 13] raw features
            adj: Normalized adjacency matrix [total_nodes, total_nodes] (without self-loops for neighbor agg)
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        x = self.feature_embedding(node_features)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)
        return self.scorer(clause_emb).squeeze(-1)
