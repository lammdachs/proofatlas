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
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features (default: 13)
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
        node_feature_dim: int = 13,
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


class ClauseSetScorer(nn.Module):
    """
    Context-aware clause scorer that scores all clauses in a set together.

    The score for each clause depends on all other clauses in the set,
    allowing the model to learn relationships like:
    - Which clauses can resolve together
    - Which clauses are redundant given others
    - Which clauses are most promising in the current context

    Architecture:
    1. Encode each clause independently via node MLP + pooling -> clause embedding
    2. Apply cross-clause interaction via simple attention (ONNX-compatible)
    3. Output one score per clause

    For ONNX export compatibility, uses only basic ops (no dynamic shapes).
    """

    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_node_layers: int = 2,
        num_interaction_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for embeddings
            num_node_layers: Number of layers in node MLP
            num_interaction_layers: Number of cross-clause interaction layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Node feature processor (shared across all nodes in all clauses)
        layers = []
        in_dim = node_feature_dim
        for i in range(num_node_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < num_node_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.node_mlp = nn.Sequential(*layers)

        # Cross-clause interaction layers (simple attention without multi-head complexity)
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_interaction_layers):
            self.interaction_layers.append(nn.ModuleDict({
                'query': nn.Linear(hidden_dim, hidden_dim),
                'key': nn.Linear(hidden_dim, hidden_dim),
                'value': nn.Linear(hidden_dim, hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }))

        # Output projection (one score per clause)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        clause_ids: torch.Tensor,
        num_clauses: int,
    ) -> torch.Tensor:
        """
        Score all clauses in a set together.

        Args:
            node_features: All node features concatenated [total_nodes, node_feature_dim]
            clause_ids: Which clause each node belongs to [total_nodes]
            num_clauses: Number of clauses

        Returns:
            Scores for each clause [num_clauses]
        """
        # Process all nodes through shared MLP
        h = self.node_mlp(node_features)  # [total_nodes, hidden_dim]

        # Pool nodes within each clause to get clause embeddings
        # Use scatter_mean: for each clause_id, average the node embeddings
        clause_embeddings = torch.zeros(num_clauses, self.hidden_dim, device=h.device)
        clause_counts = torch.zeros(num_clauses, device=h.device)

        # Accumulate embeddings per clause
        clause_embeddings.scatter_add_(0, clause_ids.unsqueeze(1).expand(-1, self.hidden_dim), h)
        clause_counts.scatter_add_(0, clause_ids, torch.ones_like(clause_ids, dtype=torch.float))

        # Compute mean (avoid division by zero)
        clause_counts = clause_counts.clamp(min=1)
        clause_embeddings = clause_embeddings / clause_counts.unsqueeze(1)

        # Apply cross-clause interaction layers
        clause_embeddings = self._apply_interaction_layers(clause_embeddings)

        # Output scores
        scores = self.output_proj(clause_embeddings).squeeze(-1)  # [num_clauses]

        return scores

    def _apply_interaction_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-clause attention layers.

        Args:
            x: Clause embeddings [num_clauses, hidden_dim]

        Returns:
            Updated embeddings [num_clauses, hidden_dim]
        """
        for layer in self.interaction_layers:
            # Self-attention
            q = layer['query'](x)  # [num_clauses, hidden_dim]
            k = layer['key'](x)    # [num_clauses, hidden_dim]
            v = layer['value'](x)  # [num_clauses, hidden_dim]

            # Attention scores: [num_clauses, num_clauses]
            scale = self.hidden_dim ** 0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)

            # Apply attention
            attn_out = torch.matmul(attn, v)  # [num_clauses, hidden_dim]

            # Residual + norm
            x = layer['norm1'](x + attn_out)

            # Feed-forward
            ff_out = layer['ff'](x)

            # Residual + norm
            x = layer['norm2'](x + ff_out)

        return x

    def forward_onnx(
        self,
        node_features: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        ONNX-compatible forward pass with pre-computed pooling matrix.

        The pooling matrix is computed on the Rust side to avoid dynamic
        tensor operations that ONNX/tract don't support.

        Args:
            node_features: All node features [total_nodes, node_feature_dim]
            pool_matrix: Pre-computed pooling matrix [num_clauses, total_nodes]
                         Each row sums to 1 and has non-zero entries only for
                         nodes belonging to that clause.

        Returns:
            Scores for each clause [num_clauses]
        """
        # Process all nodes through shared MLP
        h = self.node_mlp(node_features)  # [total_nodes, hidden_dim]

        # Pool: [num_clauses, total_nodes] @ [total_nodes, hidden_dim] -> [num_clauses, hidden_dim]
        clause_embeddings = pool_matrix @ h

        # Apply cross-clause interaction layers
        clause_embeddings = self._apply_interaction_layers(clause_embeddings)

        # Output scores
        scores = self.output_proj(clause_embeddings).squeeze(-1)

        return scores


class AgeWeightHeuristic(nn.Module):
    """
    Age-weight heuristic using only basic operations for ONNX compatibility.

    Uses only: matrix multiply, add, multiply.

    The computation is:
        clause_features = pool_matrix @ node_features  # [num_clauses, 13]
        score = weight_matrix @ clause_features.T      # Extract age/depth and weight them

    Feature layout (13 dimensions):
    - 0-5: Node type one-hot
    - 6: Arity
    - 7: Argument position
    - 8: Depth (proxy for weight/complexity)
    - 9: Age (normalized 0-1)
    - 10: Role
    - 11: Polarity
    - 12: Is equality

    Where weight_vector is [13] with non-zero values only at indices 8 (depth) and 9 (age).
    """

    def __init__(self, age_probability: float = 0.5):
        super().__init__()
        # Create a weight vector that extracts and weights age and depth
        # Index 8 = depth, Index 9 = age
        # Score = p * (1 - age) + (1-p) * (1 - depth/10)
        #       = p - p*age + (1-p) - (1-p)*depth/10
        #       = 1 - p*age - (1-p)*depth/10
        # We want higher score for lower age and lower depth
        weight_vector = torch.zeros(13, dtype=torch.float32)
        weight_vector[8] = -(1.0 - age_probability) / 10.0  # depth weight (negative, normalized)
        weight_vector[9] = -age_probability                  # age weight (negative)

        self.register_buffer('weight_vector', weight_vector)
        self.register_buffer('bias', torch.tensor([1.0], dtype=torch.float32))

    def forward(
        self,
        node_features: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass (same as forward_onnx).

        Args:
            node_features: [total_nodes, 13]
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        return self.forward_onnx(node_features, pool_matrix)

    def forward_onnx(
        self,
        node_features: torch.Tensor,
        pool_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        ONNX-compatible forward pass using only MatMul and Add.

        Args:
            node_features: [total_nodes, 13]
            pool_matrix: [num_clauses, total_nodes]

        Returns:
            Scores [num_clauses]
        """
        # Pool node features to clause level
        # [num_clauses, total_nodes] @ [total_nodes, 13] -> [num_clauses, 13]
        clause_features = torch.mm(pool_matrix, node_features)

        # Apply weight vector to get scores
        # [num_clauses, 13] @ [13] -> [num_clauses]
        scores = torch.mv(clause_features, self.weight_vector) + self.bias

        return scores


def export_age_weight_heuristic(
    output_path: str,
    age_probability: float = 0.5,
    num_clauses: int = 10,
    total_nodes: int = 50,
):
    """
    Export the age-weight heuristic as an ONNX model.

    Args:
        output_path: Path to save the ONNX model
        age_probability: Weight for age component (0.0 to 1.0)
        num_clauses: Example number of clauses (for tracing)
        total_nodes: Example total nodes (for tracing)
    """
    model = AgeWeightHeuristic(age_probability)
    model.eval()

    # Create dummy inputs
    dummy_node_features = torch.randn(total_nodes, 13)
    dummy_pool_matrix = torch.randn(num_clauses, total_nodes)

    # Export
    torch.onnx.export(
        model,
        (dummy_node_features, dummy_pool_matrix),
        output_path,
        input_names=["node_features", "pool_matrix"],
        output_names=["scores"],
        dynamic_axes={
            "node_features": {0: "total_nodes"},
            "pool_matrix": {0: "num_clauses", 1: "total_nodes"},
            "scores": {0: "num_clauses"},
        },
        opset_version=14,
    )
    print(f"Exported age-weight heuristic (p={age_probability}) to {output_path}")


def export_torchscript(
    output_path: str,
    age_probability: float = 0.5,
    num_clauses: int = 10,
    total_nodes: int = 50,
):
    """
    Export the age-weight heuristic as a TorchScript model.

    Args:
        output_path: Path to save the TorchScript model (.pt)
        age_probability: Weight for age component (0.0 to 1.0)
        num_clauses: Example number of clauses (for tracing)
        total_nodes: Example total nodes (for tracing)
    """
    model = AgeWeightHeuristic(age_probability)
    model.eval()

    # Create example inputs for tracing
    example_node_features = torch.randn(total_nodes, 13)
    example_pool_matrix = torch.randn(num_clauses, total_nodes)

    # Use torch.jit.trace to create TorchScript module
    traced_model = torch.jit.trace(model, (example_node_features, example_pool_matrix))

    # Save the traced model
    traced_model.save(output_path)
    print(f"Exported TorchScript model (p={age_probability}) to {output_path}")


def create_model(
    model_type: str = "clause_set",
    node_feature_dim: int = 13,
    hidden_dim: int = 64,
    num_layers: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a clause scoring model.

    Args:
        model_type: Model type, one of:
            - "clause_set" (default): Context-aware learned model
            - "gcn": Graph Convolutional Network
            - "gcn_attention": GCN with attention pooling
            - "age_weight": Age-weight heuristic for ONNX export
        node_feature_dim: Input node feature dimension (default: 13)
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN/MLP/interaction layers
        **kwargs: Additional model-specific arguments
            - age_probability: For age_weight model (default 0.5)

    Returns:
        PyTorch model
    """
    if model_type == "clause_set":
        return ClauseSetScorer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_node_layers=num_layers,
            num_interaction_layers=kwargs.pop('num_interaction_layers', 2),
            **kwargs,
        )
    elif model_type == "gcn":
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
    elif model_type == "age_weight":
        return AgeWeightHeuristic(
            age_probability=kwargs.pop('age_probability', 0.5),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
