"""
GNN-based clause selector models.

Pure PyTorch implementations (no PyTorch Geometric dependency).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scorers import create_scorer, MLPScorer


class NodeFeatureEmbedding(nn.Module):
    """
    Embeds node-level features into a richer representation.

    New architecture (IJCAR26 plan):
    Raw feature layout (3 dims):
        0: Node type (int 0-5: clause, literal, predicate, function, variable, constant)
        1: Arity (raw int)
        2: Arg position (raw int)

    Output layout:
        - Node type: one-hot (6 dims)
        - Arity: log1p scaled (1 dim)
        - Arg position: sinusoidal (sin_dim dims)
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

        # Output dim: 6 (type) + 1 (arity) + sin_dim (arg_pos)
        self.output_dim = 6 + 1 + sin_dim

        # Precompute div_term for sinusoidal encoding
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
        Embed raw node features.

        Args:
            x: [N, 3] raw node features (type, arity, arg_pos)

        Returns:
            [N, output_dim] embedded features
        """
        # Extract raw features (3-dim format)
        node_type = x[:, 0].long()  # int 0-5
        arity = x[:, 1]             # raw int
        arg_pos = x[:, 2]           # raw int

        # Node type to one-hot
        node_type_onehot = F.one_hot(node_type.clamp(0, 5), num_classes=6).float()  # [N, 6]

        # Encode continuous features
        arity_enc = torch.log1p(arity).unsqueeze(-1)  # [N, 1]
        arg_pos_enc = self.sinusoidal_encode(arg_pos)  # [N, sin_dim]

        # Concatenate all
        return torch.cat([
            node_type_onehot,  # 6
            arity_enc,         # 1
            arg_pos_enc,       # sin_dim
        ], dim=-1)


class ClauseFeatureEmbedding(nn.Module):
    """
    Embeds clause-level features into a representation for the scorer.

    New architecture (IJCAR26 plan):
    Raw feature layout (3 dims):
        0: Age (normalized 0-1)
        1: Role (int 0-4: axiom, hypothesis, definition, negated_conjecture, derived)
        2: Size (number of literals)

    Output layout:
        - Age: sinusoidal (sin_dim dims)
        - Role: one-hot (5 dims)
        - Size: sinusoidal (sin_dim dims)
    """

    def __init__(self, sin_dim: int = 8):
        """
        Args:
            sin_dim: Dimension of sinusoidal encoding for continuous features.
        """
        super().__init__()
        assert sin_dim % 2 == 0, "sin_dim must be even"
        self.sin_dim = sin_dim

        # Output dim: sin_dim (age) + 5 (role) + sin_dim (size)
        self.output_dim = sin_dim + 5 + sin_dim

        # Precompute div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, sin_dim, 2).float() * (-math.log(10000.0) / sin_dim))
        self.register_buffer('div_term', div_term)

    def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding."""
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        scaled = values * self.div_term
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed raw clause features.

        Args:
            x: [num_clauses, 3] raw clause features (age, role, size)

        Returns:
            [num_clauses, output_dim] embedded features
        """
        age = x[:, 0]           # normalized 0-1
        role = x[:, 1].long()   # int 0-4
        size = x[:, 2]          # number of literals

        # Encode features
        age_enc = self.sinusoidal_encode(age * 100)  # Scale to 0-100
        role_onehot = F.one_hot(role.clamp(0, 4), num_classes=5).float()
        size_enc = self.sinusoidal_encode(size)

        return torch.cat([age_enc, role_onehot, size_enc], dim=-1)


# Legacy alias for backwards compatibility
class FeatureEmbedding(NodeFeatureEmbedding):
    """Legacy alias for NodeFeatureEmbedding.

    For backwards compatibility with 8-dim features, this class accepts
    8-dim input but only uses the first 3 dimensions (type, arity, arg_pos).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 3-dim (new) and 8-dim (legacy) input
        if x.size(-1) == 8:
            x = x[:, :3]  # Use only type, arity, arg_pos
        return super().forward(x)


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

    New architecture (IJCAR26 plan):
        Node features (3d) → node_embedding → GCN layers → pool to clauses
        Clause features (3d) → clause_embedding
        Concatenate(pooled, clause_emb) → scorer → scores

    This separates structural information (encoded by GCN) from
    clause-level metadata (age, role, size) which is sinusoidal encoded.
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
        use_clause_features: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_clause_features = use_clause_features

        # Node feature embedding: raw 3-dim → richer representation
        self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.node_embedding.output_dim

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Clause feature embedding (optional)
        if use_clause_features:
            self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)
            # Project concatenated features back to hidden_dim for scorer
            concat_dim = hidden_dim + self.clause_embedding.output_dim
            self.clause_proj = nn.Linear(concat_dim, hidden_dim)
        else:
            self.clause_embedding = None
            self.clause_proj = None

        # Scorer always receives hidden_dim inputs
        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

        # Keep reference to feature embedding for backwards compatibility
        self.feature_embedding = self.node_embedding

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 3] raw node features (type, arity, arg_pos)
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes] for pooling nodes to clauses
            clause_features: [num_clauses, 3] raw clause features (age, role, size)
                            Optional for backwards compatibility

        Returns:
            Scores [num_clauses]
        """
        # Embed node features
        x = self.node_embedding(node_features)

        # GCN message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to clause level
        clause_emb = torch.mm(pool_matrix, x)

        # Add clause features if available and model expects them
        if self.use_clause_features and self.clause_embedding is not None:
            if clause_features is not None:
                clause_feat_emb = self.clause_embedding(clause_features)
            else:
                # Generate zero clause features if not provided (for backwards compat)
                num_clauses = pool_matrix.size(0)
                clause_feat_emb = torch.zeros(
                    num_clauses, self.clause_embedding.output_dim,
                    device=clause_emb.device, dtype=clause_emb.dtype
                )
            # Concatenate and project back to hidden_dim
            clause_emb = self.clause_proj(torch.cat([clause_emb, clause_feat_emb], dim=-1))

        return self.scorer(clause_emb).squeeze(-1)

    def export_torchscript(self, path: str):
        """
        Export model to TorchScript format for tch-rs inference.

        Args:
            path: Output path for TorchScript model (.pt)
        """
        self.eval()

        # Create dummy inputs for tracing
        # Use realistic feature values (not random, which can have invalid ranges)
        num_nodes = 10
        num_clauses = 3

        # Node features: [type (0-5), arity (>=0), arg_pos (>=0)]
        dummy_node_features = torch.zeros(num_nodes, 3)
        dummy_node_features[:, 0] = torch.randint(0, 6, (num_nodes,)).float()  # node type
        dummy_node_features[:, 1] = torch.randint(0, 5, (num_nodes,)).float()  # arity
        dummy_node_features[:, 2] = torch.randint(0, 10, (num_nodes,)).float()  # arg_pos

        # Adjacency with self-loops, row-normalized
        dummy_adj = torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes)
        dummy_adj = dummy_adj / dummy_adj.sum(dim=1, keepdim=True)

        dummy_pool_matrix = torch.ones(num_clauses, num_nodes) / num_nodes

        # Clause features: [age (0-1), role (0-4), size (>=1)]
        dummy_clause_features = torch.zeros(num_clauses, 3)
        dummy_clause_features[:, 0] = torch.rand(num_clauses)  # age
        dummy_clause_features[:, 1] = torch.randint(0, 5, (num_clauses,)).float()  # role
        dummy_clause_features[:, 2] = torch.randint(1, 10, (num_clauses,)).float()  # size

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


class ClauseGAT(nn.Module):
    """
    Graph Attention Network for clause scoring.

    New architecture (IJCAR26 plan):
        Node features (3d) → node_embedding → GAT layers → pool to clauses
        Clause features (3d) → clause_embedding
        Concatenate(pooled, clause_emb) → scorer → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
        use_clause_features: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_clause_features = use_clause_features

        # Node feature embedding
        self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.node_embedding.output_dim

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

        # Clause feature embedding (optional)
        if use_clause_features:
            self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)
            concat_dim = hidden_dim + self.clause_embedding.output_dim
            self.clause_proj = nn.Linear(concat_dim, hidden_dim)
        else:
            self.clause_embedding = None
            self.clause_proj = None

        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

        # Legacy alias
        self.feature_embedding = self.node_embedding

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 3] raw node features
            adj: Binary adjacency matrix [total_nodes, total_nodes] (with self-loops)
            pool_matrix: [num_clauses, total_nodes]
            clause_features: [num_clauses, 3] raw clause features (optional)

        Returns:
            Scores [num_clauses]
        """
        x = self.node_embedding(node_features)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)

        # Add clause features if available and model expects them
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

        return self.scorer(clause_emb)


class ClauseGraphSAGE(nn.Module):
    """
    GraphSAGE for clause scoring.

    New architecture (IJCAR26 plan):
        Node features (3d) → node_embedding → GraphSAGE layers → pool to clauses
        Clause features (3d) → clause_embedding
        Concatenate(pooled, clause_emb) → scorer → scores
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
        sin_dim: int = 8,
        use_clause_features: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_clause_features = use_clause_features

        # Node feature embedding
        self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.node_embedding.output_dim

        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGELayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Clause feature embedding (optional)
        if use_clause_features:
            self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)
            concat_dim = hidden_dim + self.clause_embedding.output_dim
            self.clause_proj = nn.Linear(concat_dim, hidden_dim)
        else:
            self.clause_embedding = None
            self.clause_proj = None

        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
            dropout=dropout,
        )

        # Legacy alias
        self.feature_embedding = self.node_embedding

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 3] raw node features
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes]
            clause_features: [num_clauses, 3] raw clause features (optional)

        Returns:
            Scores [num_clauses]
        """
        x = self.node_embedding(node_features)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = torch.mm(pool_matrix, x)

        # Add clause features if available and model expects them
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

        return self.scorer(clause_emb).squeeze(-1)
