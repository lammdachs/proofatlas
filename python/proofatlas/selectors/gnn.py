"""
GNN-based clause selector models.

Pure PyTorch implementations (no PyTorch Geometric dependency).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scorers import create_scorer, MLPScorer
from .utils import sparse_mm


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


class SymbolEmbedding(nn.Module):
    """
    Precompute symbol embeddings from names using a frozen MiniLM encoder.

    For each unique symbol name in a batch, tokenizes with WordPiece,
    forwards through frozen MiniLM, and mean-pools to get a 384D embedding.
    Sentinel tokens (VAR, CLAUSE, LIT) get learned embeddings.

    Results are cached per batch for efficiency.
    """

    SENTINEL_TOKENS = ["VAR", "CLAUSE", "LIT"]
    MINILM_DIM = 384

    def __init__(self):
        super().__init__()
        # Learned embeddings for sentinel tokens
        self.sentinel_embeddings = nn.Embedding(len(self.SENTINEL_TOKENS), self.MINILM_DIM)
        self._sentinel_map = {name: i for i, name in enumerate(self.SENTINEL_TOKENS)}

        # Lazy-loaded MiniLM (avoids loading at import time)
        self._tokenizer = None
        self._encoder = None

    def _ensure_encoder(self, device):
        """Lazy-load MiniLM encoder on first use."""
        if self._encoder is None:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._encoder = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).to(device)
            # Freeze encoder
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()

    @torch.no_grad()
    def precompute(self, symbol_names: list, device) -> torch.Tensor:
        """
        Compute embeddings for a list of symbol names.

        Args:
            symbol_names: List of symbol name strings (including sentinels)
            device: Target device

        Returns:
            [len(symbol_names), 384] embedding tensor
        """
        self._ensure_encoder(device)

        # Separate sentinels from real symbols
        sentinel_indices = []
        sentinel_ids = []
        real_indices = []
        real_names = []

        for i, name in enumerate(symbol_names):
            if name in self._sentinel_map:
                sentinel_indices.append(i)
                sentinel_ids.append(self._sentinel_map[name])
            else:
                real_indices.append(i)
                real_names.append(name)

        result = torch.zeros(len(symbol_names), self.MINILM_DIM, device=device)

        # Sentinel embeddings (these are learned, not frozen)
        if sentinel_indices:
            ids_tensor = torch.tensor(sentinel_ids, device=device)
            result[sentinel_indices] = self.sentinel_embeddings(ids_tensor)

        # Real symbol embeddings via MiniLM
        if real_names:
            inputs = self._tokenizer(
                real_names, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self._encoder(**inputs)
            # Mean pool over tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            result[real_indices] = embeddings

        return result


class NodeInputProjection(nn.Module):
    """
    Projects node inputs to a common dimension based on the chosen mode.

    Modes:
        "features": Use structural features only (NodeFeatureEmbedding output)
        "names": Use symbol name embeddings only (SymbolEmbedding output, 384D)
        "both": Concatenate structural features and symbol embeddings, project
    """

    def __init__(self, hidden_dim: int, sin_dim: int = 8, mode: str = "features"):
        super().__init__()
        self.mode = mode

        if mode in ("features", "both"):
            self.feature_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
            feat_dim = self.feature_embedding.output_dim
        else:
            feat_dim = 0

        if mode in ("names", "both"):
            name_dim = SymbolEmbedding.MINILM_DIM
        else:
            name_dim = 0

        input_dim = feat_dim + name_dim
        self.output_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        symbol_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, 3] raw structural features
            symbol_embeddings: [N, 384] precomputed symbol embeddings (required for names/both)

        Returns:
            [N, hidden_dim] projected node representations
        """
        parts = []
        if self.mode in ("features", "both"):
            parts.append(self.feature_embedding(node_features))
        if self.mode in ("names", "both"):
            assert symbol_embeddings is not None, "symbol_embeddings required for mode='names' or 'both'"
            parts.append(symbol_embeddings)

        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.proj(x)


class GraphNorm(nn.Module):
    """
    Graph Normalization (Cai et al., 2021).

    Normalizes node features per graph with a learnable shift parameter alpha:
        x_out = (x - alpha * mean_graph) / std_graph * gamma + beta

    Unlike LayerNorm which normalizes per-node, GraphNorm computes statistics
    across all nodes belonging to the same graph in a batch.

    Args:
        hidden_dim: Feature dimension
        eps: Numerical stability constant
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, hidden_dim] node features
            batch: [num_nodes] graph index for each node

        Returns:
            [num_nodes, hidden_dim] normalized features
        """
        # Compute per-graph mean
        num_graphs = batch.max().item() + 1
        # Sum per graph, then divide by count
        count = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1)
        # Scatter sum
        sum_x = torch.zeros(num_graphs, self.hidden_dim, device=x.device, dtype=x.dtype)
        sum_x.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        mean = sum_x / count.unsqueeze(1)

        # Subtract learnable fraction of mean
        x = x - self.alpha * mean[batch]

        # Compute per-graph variance of shifted features
        sum_sq = torch.zeros(num_graphs, self.hidden_dim, device=x.device, dtype=x.dtype)
        sum_sq.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * x)
        var = sum_sq / count.unsqueeze(1)
        std = (var + self.eps).sqrt()

        # Normalize and apply affine
        x = x / std[batch]
        return x * self.gamma + self.beta


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
            adj: Adjacency matrix [num_nodes, num_nodes] (sparse or dense, normalized)

        Returns:
            Updated features [num_nodes, out_dim]
        """
        h = sparse_mm(adj, x)
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


@torch.jit.script
def _batch_from_pool(pool_matrix: torch.Tensor) -> torch.Tensor:
    """Derive node→graph assignment from sparse pool matrix.

    Args:
        pool_matrix: [num_clauses, num_nodes] sparse or dense

    Returns:
        [num_nodes] long tensor of graph indices
    """
    if pool_matrix.is_sparse:
        indices = pool_matrix.coalesce().indices()
        num_nodes = pool_matrix.size(1)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=pool_matrix.device)
        batch[indices[1]] = indices[0]
        return batch
    else:
        return pool_matrix.argmax(dim=0)


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
        node_info: str = "features",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_clause_features = use_clause_features
        self.node_info = node_info

        # Node input projection: determines how node features are computed
        if node_info == "features":
            # Default: structural features only (backward compatible)
            self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
            embed_dim = self.node_embedding.output_dim
            self.node_projection = None
            self.symbol_embedding = None
        else:
            # "names" or "both": use NodeInputProjection
            self.node_projection = NodeInputProjection(hidden_dim, sin_dim=sin_dim, mode=node_info)
            embed_dim = hidden_dim
            self.symbol_embedding = SymbolEmbedding()
            self.node_embedding = None

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])

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
        node_names: list = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [total_nodes, 3] raw node features (type, arity, arg_pos)
            adj: Normalized adjacency matrix [total_nodes, total_nodes]
            pool_matrix: [num_clauses, total_nodes] for pooling nodes to clauses
            clause_features: [num_clauses, 3] raw clause features (age, role, size)
                            Optional for backwards compatibility
            node_names: List of symbol name strings for each node.
                       Required when node_info is "names" or "both".

        Returns:
            Scores [num_clauses]
        """
        # Derive batch tensor (node → graph mapping) from pool matrix
        batch = _batch_from_pool(pool_matrix)

        # Embed node features based on node_info mode
        if self.node_info == "features":
            x = self.node_embedding(node_features)
        else:
            sym_emb = None
            if node_names is not None:
                sym_emb = self.symbol_embedding.precompute(node_names, node_features.device)
            x = self.node_projection(node_features, sym_emb)

        # GCN message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x, batch)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to clause level (handles sparse or dense pool_matrix)
        clause_emb = sparse_mm(pool_matrix, x)

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

        return self.scorer(clause_emb).view(-1)

    def encode(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor = None,
        node_names: list = None,
    ) -> torch.Tensor:
        """
        Encode clauses to embeddings without scoring.

        Same as forward() but returns clause embeddings [num_clauses, hidden_dim]
        instead of scores. Used for cross-attention scoring where U and P
        are encoded separately.
        """
        batch = _batch_from_pool(pool_matrix)

        if self.node_info == "features":
            x = self.node_embedding(node_features)
        else:
            sym_emb = None
            if node_names is not None:
                sym_emb = self.symbol_embedding.precompute(node_names, node_features.device)
            x = self.node_projection(node_features, sym_emb)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, adj)
            x = norm(x, batch)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        clause_emb = sparse_mm(pool_matrix, x)

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


