"""
Features-only clause encoder.

Uses only clause-level statistics (no graph structure or string representation).
9 raw features → sinusoidal + one-hot → 68D embedding → scorer.

Features:
    Continuous (sinusoidal, d_sin=8 each = 56D total):
        age, size, depth, symbol_count, distinct_symbols, variable_count, distinct_variables
    Categorical:
        role: one-hot 5D (axiom, hypothesis, definition, negated_conjecture, derived)
        rule: one-hot 7D (input, Resolution, Factoring, Superposition,
                          EqualityResolution, EqualityFactoring, Demodulation)
    Total: 56 + 5 + 7 = 68D
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scorers import create_scorer


# Role encoding (must match structured.py ROLE_MAP)
ROLE_CLASSES = 5  # axiom, hypothesis, definition, negated_conjecture, derived

# Rule encoding
RULE_MAP = {
    "input": 0,
    "Resolution": 1,
    "Factoring": 2,
    "Superposition": 3,
    "EqualityResolution": 4,
    "EqualityFactoring": 5,
    "Demodulation": 6,
}
RULE_CLASSES = 7

# Number of continuous features
NUM_CONTINUOUS = 7  # age, size, depth, symbol_count, distinct_symbols, variable_count, distinct_variables


class ClauseFeatures(nn.Module):
    """
    Features-only clause encoder.

    Takes 9 raw clause features and produces a fixed-size embedding
    using sinusoidal encoding for continuous features and one-hot for categorical.

    This serves as a lightweight baseline encoder that captures only
    clause-level statistics without structural information.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        sin_dim: int = 8,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
    ):
        super().__init__()
        self.sin_dim = sin_dim

        # Embedding dimension: 7 * sin_dim + 5 (role) + 7 (rule)
        embed_dim = NUM_CONTINUOUS * sin_dim + ROLE_CLASSES + RULE_CLASSES

        # Precompute div_term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, sin_dim, 2).float() * (-math.log(10000.0) / sin_dim)
        )
        self.register_buffer("div_term", div_term)

        # Project to hidden dim
        self.projection = nn.Linear(embed_dim, hidden_dim)

        # Scorer
        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
        )

    def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
        """Encode a [N] tensor to [N, sin_dim] via sinusoidal positional encoding."""
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        scaled = values * self.div_term
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode clause features to embeddings without scoring.

        Args:
            features: [num_clauses, 9] raw clause features
                [0] age (normalized 0-1)
                [1] size (number of literals)
                [2] depth (max term depth)
                [3] symbol_count
                [4] distinct_symbols
                [5] variable_count
                [6] distinct_variables
                [7] role (int 0-4)
                [8] rule (int 0-6)

        Returns:
            Embeddings [num_clauses, hidden_dim]
        """
        # Continuous features: sinusoidal encode each
        continuous = []
        # age (already 0-1, scale for encoding)
        continuous.append(self.sinusoidal_encode(features[:, 0] * 100))
        # size, depth, counts: encode directly
        for i in range(1, NUM_CONTINUOUS):
            continuous.append(self.sinusoidal_encode(features[:, i]))
        continuous_enc = torch.cat(continuous, dim=-1)  # [N, 7 * sin_dim]

        # Role: one-hot
        role = features[:, 7].long().clamp(0, ROLE_CLASSES - 1)
        role_onehot = F.one_hot(role, num_classes=ROLE_CLASSES).float()

        # Rule: one-hot
        rule = features[:, 8].long().clamp(0, RULE_CLASSES - 1)
        rule_onehot = F.one_hot(rule, num_classes=RULE_CLASSES).float()

        # Concatenate and project
        x = torch.cat([continuous_enc, role_onehot, rule_onehot], dim=-1)
        return self.projection(x)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [num_clauses, 9] raw clause features

        Returns:
            Scores [num_clauses]
        """
        x = self.encode(features)
        return self.scorer(x).view(-1)
