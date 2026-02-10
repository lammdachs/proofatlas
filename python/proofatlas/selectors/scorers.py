"""
Scoring heads for clause selection.

Given clause embeddings, compute scores [num_clauses].

All scorers accept forward(u_emb, p_emb=None):
    u_emb: [num_unprocessed, hidden_dim] — clauses to score
    p_emb: [num_processed, hidden_dim] — context (processed set), optional

Available scorers:
- MLPScorer: Independent MLP per clause (ignores p_emb)
- AttentionScorer: Cross-attention (U queries, [s_0; P] keys/values). Falls back to self-attention when p_emb is None.
- TransformerScorer: Multi-layer cross-attention stack. Falls back to self-attention when p_emb is None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPScorer(nn.Module):
    """
    MLP scoring head - scores each clause independently.

    Architecture: Linear → GELU → Linear → score
    Ignores p_emb (no cross-clause reasoning).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, u_emb: torch.Tensor, p_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_emb: Clause embeddings [num_clauses, hidden_dim]
            p_emb: Ignored

        Returns:
            Scores [num_clauses]
        """
        x = self.linear1(u_emb)
        x = F.gelu(x)
        return self.linear2(x).view(-1)


class AttentionScorer(nn.Module):
    """
    Cross-attention scoring — U clauses attend to the processed set P.

    When p_emb is provided:
        Q from u_emb, K/V from [s_0; p_emb]
        s_0 is a learnable sentinel (ensures well-defined attention when P is empty)
        Multi-head cross-attention → residual → score

    When p_emb is None (backward compatible):
        Self-attention over u_emb (Q/K/V all from u_emb)
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

        # Learnable sentinel s_0 — prepended to P keys/values
        self.sentinel = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        # Pre-norms
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        # Separate Q and KV projections for cross-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Scoring head
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, u_emb: torch.Tensor, p_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_emb: Unprocessed clause embeddings [num_u, hidden_dim]
            p_emb: Processed clause embeddings [num_p, hidden_dim] or None

        Returns:
            Scores [num_u]
        """
        num_u = u_emb.size(0)

        # Determine Q source and KV source
        q_input = self.norm_q(u_emb)

        if p_emb is not None:
            # Cross-attention: Q from U, K/V from [s_0; P]
            kv_source = torch.cat([self.sentinel, p_emb], dim=0)
        else:
            # Self-attention fallback: Q/K/V all from U
            kv_source = q_input

        kv_input = self.norm_kv(kv_source)
        num_kv = kv_input.size(0)

        # Project
        q = self.q_proj(q_input).view(num_u, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(kv_input).view(num_kv, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(kv_input).view(num_kv, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.bmm(attn, v)  # [num_heads, num_u, head_dim]
        out = out.transpose(0, 1).contiguous().view(num_u, self.hidden_dim)
        out = self.out_proj(out)

        # Residual connection
        x = u_emb + out

        # Score
        return self.scorer(x).view(-1)


class TransformerScorer(nn.Module):
    """
    Multi-layer cross-attention scoring.

    When p_emb is provided:
        Each layer: cross-attention (Q from U, K/V from [s_0; P]) + FFN
        Multiple layers enable deeper reasoning about U-P relationships.

    When p_emb is None:
        Falls back to self-attention transformer layers.
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
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # Per-layer learnable sentinels s_0^(l)
        self.sentinels = nn.ParameterList([
            nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            for _ in range(num_layers)
        ])

        # Per-layer cross-attention + FFN blocks
        self.q_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.kv_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.q_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.k_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.v_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.out_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

        # FFN blocks
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Final norm and scorer
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, u_emb: torch.Tensor, p_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_emb: Unprocessed clause embeddings [num_u, hidden_dim]
            p_emb: Processed clause embeddings [num_p, hidden_dim] or None

        Returns:
            Scores [num_u]
        """
        x = u_emb

        for i in range(self.num_layers):
            num_u = x.size(0)

            # Attention block
            q_input = self.q_norms[i](x)

            if p_emb is not None:
                kv_source = torch.cat([self.sentinels[i], p_emb], dim=0)
            else:
                kv_source = q_input

            kv_input = self.kv_norms[i](kv_source)
            num_kv = kv_input.size(0)

            q = self.q_projs[i](q_input).view(num_u, self.num_heads, self.head_dim).transpose(0, 1)
            k = self.k_projs[i](kv_input).view(num_kv, self.num_heads, self.head_dim).transpose(0, 1)
            v = self.v_projs[i](kv_input).view(num_kv, self.num_heads, self.head_dim).transpose(0, 1)

            attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(attn, v)
            out = out.transpose(0, 1).contiguous().view(num_u, self.hidden_dim)
            out = self.out_projs[i](out)
            x = x + out

            # FFN block
            ffn_input = self.ffn_norms[i](x)
            x = x + self.ffn_layers[i](ffn_input)

        # Final norm and score
        x = self.final_norm(x)
        return self.scorer(x).view(-1)


def create_scorer(
    scorer_type: str,
    hidden_dim: int,
    num_heads: int = 4,
    num_layers: int = 2,
) -> nn.Module:
    """
    Factory function to create a scorer.

    Args:
        scorer_type: One of "mlp", "attention", "transformer"
        hidden_dim: Hidden dimension of clause embeddings
        num_heads: Number of attention heads (for attention-based scorers)
        num_layers: Number of transformer layers (for transformer scorer)

    Returns:
        Scorer module
    """
    if scorer_type == "mlp":
        return MLPScorer(hidden_dim)
    elif scorer_type == "attention":
        return AttentionScorer(hidden_dim, num_heads=num_heads)
    elif scorer_type == "transformer":
        return TransformerScorer(hidden_dim, num_layers=num_layers, num_heads=num_heads)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}. "
                        f"Available: mlp, attention, transformer")
