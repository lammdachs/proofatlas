"""
Pure PyTorch GNN models for clause selection.

No PyTorch Geometric dependency - all GNN layers implemented from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# GNN Layers (Pure PyTorch)
# =============================================================================


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
        # Message passing: aggregate neighbor features
        h = torch.mm(adj, x)  # [num_nodes, in_dim]
        # Transform
        return self.linear(h)


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

        # Linear transformation for each head
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Attention parameters
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

        # Linear transform: [num_nodes, num_heads * out_dim]
        h = self.W(x).view(num_nodes, self.num_heads, self.out_dim)

        # Compute attention scores
        # e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)
        attn_src = (h * self.a_src).sum(dim=-1)  # [num_nodes, num_heads]
        attn_dst = (h * self.a_dst).sum(dim=-1)  # [num_nodes, num_heads]

        # Broadcast to get pairwise scores
        # attn_src[i] + attn_dst[j] for all pairs
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)  # [num_nodes, num_nodes, num_heads]
        attn = self.leaky_relu(attn)

        # Mask non-edges with -inf
        mask = (adj == 0).unsqueeze(-1)  # [num_nodes, num_nodes, 1]
        attn = attn.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=1)  # [num_nodes, num_nodes, num_heads]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention: weighted sum of neighbors
        # [num_nodes, num_nodes, num_heads] @ [num_nodes, num_heads, out_dim]
        h = h.transpose(0, 1)  # [num_heads, num_nodes, out_dim]
        attn = attn.permute(2, 0, 1)  # [num_heads, num_nodes, num_nodes]
        out = torch.bmm(attn, h)  # [num_heads, num_nodes, out_dim]
        out = out.permute(1, 0, 2)  # [num_nodes, num_heads, out_dim]

        if self.concat:
            return out.reshape(num_nodes, -1)  # [num_nodes, num_heads * out_dim]
        else:
            return out.mean(dim=1)  # [num_nodes, out_dim]


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer (Hamilton et al., 2017).

    h_i' = σ(W · concat(h_i, mean({h_j : j ∈ N(i)})))
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        # Transform concatenated [self, neighbors]
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            adj: Adjacency matrix [num_nodes, num_nodes] (normalized, without self-loops)

        Returns:
            Updated features [num_nodes, out_dim]
        """
        # Aggregate neighbors (mean)
        neighbor_agg = torch.mm(adj, x)  # [num_nodes, in_dim]

        # Concatenate self and neighbor embeddings
        h = torch.cat([x, neighbor_agg], dim=-1)  # [num_nodes, 2 * in_dim]

        return self.linear(h)


# =============================================================================
# Utility Functions
# =============================================================================


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Normalize adjacency matrix for GCN: D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix [num_nodes, num_nodes]
        add_self_loops: Whether to add self-loops before normalizing

    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # D^{-1/2} A D^{-1/2}
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def edge_index_to_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Convert edge_index [2, num_edges] to adjacency matrix [num_nodes, num_nodes].
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0

    if add_self_loops:
        adj = adj + torch.eye(num_nodes, device=adj.device)

    return adj


# =============================================================================
# GNN Models for Clause Selection
# =============================================================================


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

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        # Layer normalization (more stable than batch norm for variable-size graphs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output MLP
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
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes] for pooling nodes to clauses

        Returns:
            Scores [num_clauses]
        """
        x = node_features

        # GCN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to clause level
        clause_emb = torch.mm(pool_matrix, x)  # [num_clauses, hidden_dim]

        # Score
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

        # GAT layers
        self.convs = nn.ModuleList()
        # First layer: concat heads
        self.convs.append(GATLayer(node_feature_dim, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout))
        # Last layer: average heads
        if num_layers > 1:
            self.convs.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=False, dropout=dropout))

        # Layer norms
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Output MLP
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

        # Pool to clause level
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

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGELayer(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGELayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

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


# =============================================================================
# Transformer-based Models
# =============================================================================


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

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers for cross-clause attention
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
        # Encode nodes
        h = self.node_encoder(node_features)  # [total_nodes, hidden_dim]

        # Pool to clause level
        x = torch.mm(pool_matrix, h)  # [num_clauses, hidden_dim]

        # Add batch dimension for MultiheadAttention
        x = x.unsqueeze(0)  # [1, num_clauses, hidden_dim]

        # Transformer layers
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)

            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)

        # Remove batch dimension and score
        x = x.squeeze(0)  # [num_clauses, hidden_dim]
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

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNLayer(node_feature_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.gnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])

        # Transformer layers
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
        # GNN: within-clause message passing
        x = node_features
        for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            x = conv(x, adj)
            x = norm(x)
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to clause level
        clause_emb = torch.mm(pool_matrix, x)  # [num_clauses, hidden_dim]

        # Transformer: cross-clause attention
        x = clause_emb.unsqueeze(0)  # [1, num_clauses, hidden_dim]

        for layer in self.transformer_layers:
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)

        x = x.squeeze(0)
        return self.scorer(x).squeeze(-1)


# =============================================================================
# Baseline Models
# =============================================================================


class NodeMLP(nn.Module):
    """
    Simple MLP baseline - no graph structure, just pools node features.

    Architecture:
        node_features → MLP → pool to clauses → score
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = node_feature_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
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
        h = self.encoder(node_features)
        clause_emb = torch.mm(pool_matrix, h)
        return self.scorer(clause_emb).squeeze(-1)


class AgeWeightHeuristic(nn.Module):
    """
    Age-weight heuristic as a neural network (for ONNX export).

    With probability p: prefer oldest clause (highest age)
    With probability 1-p: prefer lightest clause (lowest depth)
    """

    def __init__(self, age_probability: float = 0.5):
        super().__init__()
        self.register_buffer('p', torch.tensor(age_probability))

    def forward(
        self,
        node_features: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 13]
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Logits [num_clauses]
        """
        # Pool to clause features
        clause_features = torch.mm(pool_matrix, node_features)  # [num_clauses, 13]

        # Extract age (index 9) and depth/weight (index 8)
        ages = clause_features[:, 9]      # Higher = older
        weights = clause_features[:, 8]   # Higher = heavier

        num_clauses = clause_features.size(0)

        # Find oldest and lightest
        oldest_idx = torch.argmax(ages)
        lightest_idx = torch.argmin(weights)

        # Build logits using torch.where for ONNX compatibility
        indices = torch.arange(num_clauses, device=clause_features.device)
        oldest_mask = (indices == oldest_idx).float()
        lightest_mask = (indices == lightest_idx).float()

        log_p = torch.log(self.p + 1e-10)
        log_1mp = torch.log(1 - self.p + 1e-10)

        # Logits when oldest != lightest
        logits_diff = oldest_mask * log_p + lightest_mask * log_1mp + (1 - oldest_mask) * (1 - lightest_mask) * (-1e9)

        # Logits when oldest == lightest
        logits_same = oldest_mask * 0.0 + (1 - oldest_mask) * (-1e9)

        # Select based on whether oldest and lightest are the same
        same_clause = (oldest_idx == lightest_idx)
        return torch.where(same_clause, logits_same, logits_diff)


# =============================================================================
# Factory and Export Functions
# =============================================================================


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
