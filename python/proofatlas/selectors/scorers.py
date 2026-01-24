"""
Scoring heads for clause selection.

Given clause embeddings [num_clauses, hidden_dim], compute scores [num_clauses].

Available scorers:
- MLPScorer: Independent MLP per clause (baseline)
- AttentionScorer: Single self-attention layer with pre-norm (clauses attend to each other)
- TransformerScorer: Multi-layer transformer with pre-norm (deeper cross-clause reasoning)
- CrossAttentionScorer: Learnable query attends to clauses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPScorer(nn.Module):
    """
    MLP scoring head - scores each clause independently.

    Architecture: Linear → GELU → Linear → score
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clause embeddings [num_clauses, hidden_dim]

        Returns:
            Scores [num_clauses]
        """
        x = self.linear1(x)
        x = F.gelu(x)
        return self.linear2(x).view(-1)


class AttentionScorer(nn.Module):
    """
    Attention-based scoring - clauses attend to each other before scoring.

    Uses pre-norm architecture (modern transformer style):
        1. LayerNorm → Self-attention → Add (residual)
        2. Linear projection to score

    This allows the model to consider the relative importance of clauses
    in the context of all available clauses.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Pre-norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Multi-head attention projections (QKV combined for efficiency)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Scoring head
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clause embeddings [num_clauses, hidden_dim]

        Returns:
            Scores [num_clauses]
        """
        num_clauses = x.size(0)

        # Pre-norm
        normed = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(normed)
        qkv = qkv.view(num_clauses, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Transpose for attention: [num_heads, num_clauses, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.bmm(attn, v)  # [num_heads, num_clauses, head_dim]
        out = out.transpose(0, 1).contiguous().view(num_clauses, self.hidden_dim)
        out = self.out_proj(out)

        # Residual connection
        x = x + out

        # Score
        return self.scorer(x).view(-1)


class TransformerScorer(nn.Module):
    """
    Transformer-based scoring - multi-layer self-attention for deeper reasoning.

    Uses pre-norm architecture (norm_first=True):
        1. Multiple transformer encoder layers
        2. Each layer: Norm → Self-Attention → Add → Norm → FFN → Add
        3. Final linear projection to score

    This allows for more complex reasoning about clause relationships
    through multiple layers of attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # Transformer encoder layers with pre-norm (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final norm (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Scoring head
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clause embeddings [num_clauses, hidden_dim]

        Returns:
            Scores [num_clauses]
        """
        # Add batch dimension for transformer
        x = x.unsqueeze(0)  # [1, num_clauses, hidden_dim]

        # Apply transformer layers
        x = self.transformer(x)  # [1, num_clauses, hidden_dim]

        # Remove batch dimension
        x = x.squeeze(0)  # [num_clauses, hidden_dim]

        # Final norm and score
        x = self.final_norm(x)
        return self.scorer(x).view(-1)


class CrossAttentionScorer(nn.Module):
    """
    Cross-attention scoring with a learnable query.

    Architecture:
        1. Norm → Learnable query attends to all clause embeddings
        2. Context vector modulates clause scores via dot product

    This is similar to the [CLS] token approach in BERT,
    but the query directly computes attention weights as scores.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learnable query
        self.query = nn.Parameter(torch.randn(1, hidden_dim))

        # Pre-norm for clauses
        self.norm = nn.LayerNorm(hidden_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clause embeddings [num_clauses, hidden_dim]

        Returns:
            Scores [num_clauses]
        """
        num_clauses = x.size(0)

        # Pre-norm
        x_normed = self.norm(x)

        # Project query, keys, values
        q = self.q_proj(self.query)  # [1, hidden_dim]
        k = self.k_proj(x_normed)  # [num_clauses, hidden_dim]
        v = self.v_proj(x_normed)  # [num_clauses, hidden_dim]

        # Reshape for multi-head attention
        q = q.view(1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, 1, head_dim]
        k = k.view(num_clauses, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, num_clauses, head_dim]
        v = v.view(num_clauses, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention: query attends to all clauses
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [num_heads, 1, num_clauses]
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        context = torch.bmm(attn, v)  # [num_heads, 1, head_dim]
        context = context.transpose(0, 1).contiguous().view(1, self.hidden_dim)  # [1, hidden_dim]

        # Clause scores via dot product with context
        scores = torch.mm(x, context.t())  # [num_clauses, 1]

        return scores.view(-1)


def create_scorer(
    scorer_type: str,
    hidden_dim: int,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.0,  # Ignored for modern attention, kept for API compatibility
) -> nn.Module:
    """
    Factory function to create a scorer.

    Args:
        scorer_type: One of "mlp", "attention", "transformer", "cross_attention"
        hidden_dim: Hidden dimension of clause embeddings
        num_heads: Number of attention heads (for attention-based scorers)
        num_layers: Number of transformer layers (for transformer scorer)
        dropout: Ignored (kept for API compatibility)

    Returns:
        Scorer module
    """
    if scorer_type == "mlp":
        return MLPScorer(hidden_dim)
    elif scorer_type == "attention":
        return AttentionScorer(hidden_dim, num_heads=num_heads)
    elif scorer_type == "transformer":
        return TransformerScorer(hidden_dim, num_layers=num_layers, num_heads=num_heads)
    elif scorer_type == "cross_attention":
        return CrossAttentionScorer(hidden_dim, num_heads=num_heads)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}. "
                        f"Available: mlp, attention, transformer, cross_attention")
