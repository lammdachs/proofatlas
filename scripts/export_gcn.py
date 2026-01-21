#!/usr/bin/env python3
"""
Export GCN model to TorchScript format for tch-rs inference.

Loads weights from safetensors and exports as TorchScript.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from safetensors.torch import load_file

project_root = Path(__file__).parent.parent


class NodeFeatureEmbedding(nn.Module):
    """Node embedding for TorchScript (no 8-dim legacy support)."""

    def __init__(self, sin_dim: int = 8):
        super().__init__()
        self.sin_dim = sin_dim
        self.output_dim = 6 + 1 + sin_dim
        div_term = torch.exp(torch.arange(0, sin_dim, 2).float() * (-math.log(10000.0) / sin_dim))
        self.register_buffer('div_term', div_term)

    def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        scaled = values * self.div_term
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_type = x[:, 0].long()
        arity = x[:, 1]
        arg_pos = x[:, 2]
        node_type_onehot = F.one_hot(node_type.clamp(0, 5), num_classes=6).float()
        arity_enc = torch.log1p(arity).unsqueeze(-1)
        arg_pos_enc = self.sinusoidal_encode(arg_pos)
        return torch.cat([node_type_onehot, arity_enc, arg_pos_enc], dim=-1)


class ClauseFeatureEmbedding(nn.Module):
    """Clause embedding for TorchScript."""

    def __init__(self, sin_dim: int = 8):
        super().__init__()
        self.sin_dim = sin_dim
        self.output_dim = sin_dim + 5 + sin_dim
        div_term = torch.exp(torch.arange(0, sin_dim, 2).float() * (-math.log(10000.0) / sin_dim))
        self.register_buffer('div_term', div_term)

    def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        scaled = values * self.div_term
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        age = x[:, 0]
        role = x[:, 1].long()
        size = x[:, 2]
        age_enc = self.sinusoidal_encode(age * 100)
        role_onehot = F.one_hot(role.clamp(0, 4), num_classes=5).float()
        size_enc = self.sinusoidal_encode(size)
        return torch.cat([age_enc, role_onehot, size_enc], dim=-1)


class GCNLayer(nn.Module):
    """GCN layer for TorchScript."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.mm(adj, x)
        return self.linear(h)


class MLPScorer(nn.Module):
    """MLP scorer for TorchScript."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        return self.linear2(x).squeeze(-1)


class ClauseGCNForExport(nn.Module):
    """GCN model optimized for TorchScript export (no conditional branches)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        sin_dim: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.node_embedding = NodeFeatureEmbedding(sin_dim=sin_dim)
        embed_dim = self.node_embedding.output_dim

        self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(embed_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        concat_dim = hidden_dim + self.clause_embedding.output_dim
        self.clause_proj = nn.Linear(concat_dim, hidden_dim)

        self.scorer = MLPScorer(hidden_dim)

        # Legacy alias for weight loading
        self.feature_embedding = self.node_embedding

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        pool_matrix: torch.Tensor,
        clause_features: torch.Tensor,
    ) -> torch.Tensor:
        # Embed node features
        x = self.node_embedding(node_features)

        # GCN message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            x = self.norms[i](x)
            x = F.relu(x)

        # Pool to clause level
        clause_emb = torch.mm(pool_matrix, x)

        # Add clause features
        clause_feat_emb = self.clause_embedding(clause_features)
        clause_emb = self.clause_proj(torch.cat([clause_emb, clause_feat_emb], dim=-1))

        return self.scorer(clause_emb)


def load_gcn_from_safetensors(safetensors_path: Path) -> ClauseGCNForExport:
    """Load GCN model from safetensors file."""
    print(f"Loading safetensors from {safetensors_path}")
    state_dict = load_file(safetensors_path)

    # Print keys to understand structure
    print(f"  Number of keys: {len(state_dict)}")

    # Infer model configuration from weights
    # convs.0.linear.weight: [hidden_dim, embed_dim]
    # convs.X.linear.weight: [hidden_dim, hidden_dim] for X > 0
    num_layers = sum(1 for k in state_dict if k.startswith("convs.") and k.endswith(".linear.weight"))

    # Get hidden_dim from convs.0.linear.weight or clause_proj
    if "convs.0.linear.weight" in state_dict:
        hidden_dim = state_dict["convs.0.linear.weight"].shape[0]
    elif "clause_proj.weight" in state_dict:
        hidden_dim = state_dict["clause_proj.weight"].shape[0]
    else:
        hidden_dim = 256  # default

    # Get sin_dim from node_embedding.div_term
    if "node_embedding.div_term" in state_dict:
        sin_dim = state_dict["node_embedding.div_term"].shape[0] * 2
    else:
        sin_dim = 8  # default

    print(f"  Inferred config: hidden_dim={hidden_dim}, num_layers={num_layers}, sin_dim={sin_dim}")

    # Create model
    model = ClauseGCNForExport(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sin_dim=sin_dim,
    )

    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def export_gcn_to_torchscript(model: ClauseGCNForExport, output_path: Path):
    """Export GCN model to TorchScript."""
    print("Tracing GCN model...")

    num_nodes = 10
    num_clauses = 3

    dummy_node_features = torch.randn(num_nodes, 3)
    dummy_adj = torch.eye(num_nodes)
    dummy_pool_matrix = torch.ones(num_clauses, num_nodes) / num_nodes
    dummy_clause_features = torch.randn(num_clauses, 3)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
        )

    # Save
    traced.save(str(output_path))
    print(f"Saved TorchScript GCN model to {output_path}")

    # Verify
    with torch.no_grad():
        original_out = model(dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
        traced_out = traced(dummy_node_features, dummy_adj, dummy_pool_matrix, dummy_clause_features)
        diff = (original_out - traced_out).abs().max().item()
        print(f"Verification: max diff = {diff:.6e}")

    return traced


def main():
    weights_dir = project_root / ".weights"

    gcn_safetensors = weights_dir / "gcn_mlp.safetensors"

    if gcn_safetensors.exists():
        model = load_gcn_from_safetensors(gcn_safetensors)
        export_gcn_to_torchscript(model, weights_dir / "gcn_model.pt")
    else:
        print(f"No GCN weights found at {gcn_safetensors}")

    print("\nDone!")


if __name__ == "__main__":
    main()
