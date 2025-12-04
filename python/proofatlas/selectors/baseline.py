"""
Baseline clause selector models.
"""

import torch
import torch.nn as nn


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
        clause_features = torch.mm(pool_matrix, node_features)

        # Extract age (index 9) and depth/weight (index 8)
        ages = clause_features[:, 9]
        weights = clause_features[:, 8]

        num_clauses = clause_features.size(0)

        oldest_idx = torch.argmax(ages)
        lightest_idx = torch.argmin(weights)

        indices = torch.arange(num_clauses, device=clause_features.device)
        oldest_mask = (indices == oldest_idx).float()
        lightest_mask = (indices == lightest_idx).float()

        log_p = torch.log(self.p + 1e-10)
        log_1mp = torch.log(1 - self.p + 1e-10)

        logits_diff = oldest_mask * log_p + lightest_mask * log_1mp + (1 - oldest_mask) * (1 - lightest_mask) * (-1e9)
        logits_same = oldest_mask * 0.0 + (1 - oldest_mask) * (-1e9)

        same_clause = (oldest_idx == lightest_idx)
        return torch.where(same_clause, logits_same, logits_diff)
