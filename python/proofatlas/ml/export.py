"""TorchScript model export for Rust inference.

Exports trained PyTorch models to TorchScript format (.pt files) that
can be loaded by the Rust scoring server via tch-rs.

For attention/transformer scorers, additionally exports:
- {name}_encoder.pt — encoder only (returns embeddings)
- {name}_scorer.pt — scorer only (takes u_emb, p_emb → scores)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def export_model(
    model: nn.Module,
    weights_dir: Path,
    model_name: str,
    config: dict,
    is_sentence_model: bool,
    needs_adj: bool,
) -> Path:
    """Export a trained model to TorchScript for Rust inference.

    Args:
        model: Trained PyTorch model (already has best weights loaded)
        weights_dir: Directory to save the .pt file
        model_name: Name for the weights file (without extension)
        config: Training config dict (needs "input_dim" key)
        is_sentence_model: Whether this is a sentence transformer model
        needs_adj: Whether the model needs adjacency matrix input

    Returns:
        Path to the saved weights file
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    model.eval()

    # Trace on CPU — torch.jit.trace bakes in device placement for tensor
    # creation ops. CPU is the default eval backend (bench --gpu-workers 0).
    model.cpu()
    trace_device = torch.device("cpu")

    is_features_model = config.get("embedding", {}).get("type") == "features"

    if is_sentence_model:
        model.export_torchscript(str(weights_path), save_tokenizer=True)
    elif is_features_model:
        # Features-only model: trace with clause features input
        num_clauses = 3
        example_features = torch.randn(num_clauses, 9, device=trace_device)
        traced = torch.jit.trace(model, (example_features,), check_trace=False)
        traced.save(str(weights_path))
    else:
        # GNN models: trace with example inputs (must match Rust call signature)
        # Script GraphNorm modules before tracing — their forward() uses
        # data-dependent shapes (batch.max().item()) that trace bakes as constants
        if hasattr(model, 'norms'):
            for i in range(len(model.norms)):
                model.norms[i] = torch.jit.script(model.norms[i])

        # Rust sends sparse COO tensors for adj and pool_matrix
        num_nodes, num_clauses = 10, 3
        example_x = torch.randn(num_nodes, config.get("input_dim", 13), device=trace_device)
        example_adj = torch.eye(num_nodes, device=trace_device).to_sparse()
        example_pool = (torch.ones(num_clauses, num_nodes, device=trace_device) / num_nodes).to_sparse()
        example_clause_features = torch.randn(num_clauses, 9, device=trace_device)

        if needs_adj:
            traced = torch.jit.trace(model, (example_x, example_adj, example_pool, example_clause_features), check_trace=False)
        else:
            traced = torch.jit.trace(model, (example_x, example_pool), check_trace=False)

        traced.save(str(weights_path))

    # For attention/transformer scorers, also export split encoder + scorer
    scorer_type = config.get("scoring", {}).get("type", "mlp")
    if scorer_type in ("attention", "transformer"):
        _export_split_models(
            model, weights_dir, model_name, config,
            is_sentence_model, is_features_model, needs_adj,
            trace_device,
        )

    return weights_path


def _export_split_models(
    model: nn.Module,
    weights_dir: Path,
    model_name: str,
    config: dict,
    is_sentence_model: bool,
    is_features_model: bool,
    needs_adj: bool,
    trace_device: torch.device,
):
    """Export separate encoder and scorer models for attention/transformer configs.

    The encoder returns embeddings [num_clauses, hidden_dim].
    The scorer takes (u_emb, p_emb) and returns scores [num_u].
    """
    encoder_path = weights_dir / f"{model_name}_encoder.pt"
    scorer_path = weights_dir / f"{model_name}_scorer.pt"

    hidden_dim = config.get("hidden_dim", 64)

    # --- Export encoder ---
    if is_sentence_model:
        _export_sentence_encoder(model, encoder_path, trace_device)
    elif is_features_model:
        _export_features_encoder(model, encoder_path, trace_device)
    else:
        _export_gnn_encoder(model, encoder_path, config, needs_adj, trace_device)

    # --- Export scorer ---
    _export_scorer(model.scorer, scorer_path, hidden_dim, trace_device)

    print(f"Exported split encoder to {encoder_path}")
    print(f"Exported split scorer to {scorer_path}")


def _export_sentence_encoder(model, encoder_path: Path, trace_device: torch.device):
    """Export sentence encoder that returns projected embeddings from tokens."""

    class _SentenceEncoderWrapper(nn.Module):
        def __init__(self, encoder, projection):
            super().__init__()
            self.encoder = encoder
            self.projection = projection

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = (token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            return self.projection(embeddings)

    wrapper = _SentenceEncoderWrapper(model.encoder, model.projection)
    wrapper.eval()

    dummy_ids = torch.zeros((3, 32), dtype=torch.long, device=trace_device)
    dummy_mask = torch.ones((3, 32), dtype=torch.long, device=trace_device)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))
    traced.save(str(encoder_path))

    # Also save tokenizer alongside encoder
    if hasattr(model, 'tokenizer'):
        tokenizer_dir = encoder_path.parent / f"{encoder_path.stem}_tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        model.tokenizer.save_pretrained(str(tokenizer_dir))


def _export_features_encoder(model, encoder_path: Path, trace_device: torch.device):
    """Export features encoder that returns projected embeddings from 9 raw features."""

    class _FeaturesEncoderWrapper(nn.Module):
        def __init__(self, encode_fn_module):
            super().__init__()
            # Copy all relevant parameters from the model
            self.model = encode_fn_module

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.model.encode(features)

    # We can't directly trace model.encode because it may not be a Module.
    # Instead, create a wrapper that calls encode.
    class _FeaturesEncodeModule(nn.Module):
        def __init__(self, parent_model):
            super().__init__()
            # Share parameters by reference
            self.sin_dim = parent_model.sin_dim
            self.register_buffer("div_term", parent_model.div_term)
            self.projection = parent_model.projection

        def sinusoidal_encode(self, values: torch.Tensor) -> torch.Tensor:
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            scaled = values * self.div_term
            sin_enc = torch.sin(scaled)
            cos_enc = torch.cos(scaled)
            return torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F
            # Replicate encode() logic
            continuous = []
            continuous.append(self.sinusoidal_encode(features[:, 0] * 100))
            for i in range(1, 7):
                continuous.append(self.sinusoidal_encode(features[:, i]))
            continuous_enc = torch.cat(continuous, dim=-1)

            role = features[:, 7].long().clamp(0, 4)
            role_onehot = F.one_hot(role, num_classes=5).float()
            rule = features[:, 8].long().clamp(0, 6)
            rule_onehot = F.one_hot(rule, num_classes=7).float()

            x = torch.cat([continuous_enc, role_onehot, rule_onehot], dim=-1)
            return self.projection(x)

    wrapper = _FeaturesEncodeModule(model)
    wrapper.eval()

    num_clauses = 3
    example_features = torch.randn(num_clauses, 9, device=trace_device)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_features,), check_trace=False)
    traced.save(str(encoder_path))


def _export_gnn_encoder(
    model, encoder_path: Path, config: dict, needs_adj: bool, trace_device: torch.device,
):
    """Export GNN encoder that returns clause embeddings."""

    class _GNNEncoderWrapper(nn.Module):
        def __init__(self, parent_model):
            super().__init__()
            # Copy all GNN components (they're already nn.Modules)
            self.node_embedding = parent_model.node_embedding
            self.node_projection = getattr(parent_model, 'node_projection', None)
            self.symbol_embedding = getattr(parent_model, 'symbol_embedding', None)
            self.node_info = parent_model.node_info
            self.convs = parent_model.convs
            self.norms = parent_model.norms
            self.use_clause_features = parent_model.use_clause_features
            self.clause_embedding = parent_model.clause_embedding
            self.clause_proj = parent_model.clause_proj

        def forward(
            self,
            node_features: torch.Tensor,
            adj: torch.Tensor,
            pool_matrix: torch.Tensor,
            clause_features: torch.Tensor,
        ) -> torch.Tensor:
            import torch.nn.functional as F
            from proofatlas.selectors.gnn import _batch_from_pool
            from proofatlas.selectors.utils import sparse_mm

            batch = _batch_from_pool(pool_matrix)

            if self.node_info == "features":
                x = self.node_embedding(node_features)
            else:
                from proofatlas.selectors.gnn import SymbolEmbedding
                sym_emb = torch.zeros(
                    node_features.size(0), SymbolEmbedding.MINILM_DIM,
                    device=node_features.device,
                )
                x = self.node_projection(node_features, sym_emb)

            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, adj)
                x = norm(x, batch)
                x = F.relu(x)

            clause_emb = sparse_mm(pool_matrix, x)

            if self.use_clause_features and self.clause_embedding is not None:
                if clause_features is not None:
                    clause_feat_emb = self.clause_embedding(clause_features)
                else:
                    num_clauses = pool_matrix.size(0)
                    clause_feat_emb = torch.zeros(
                        num_clauses, self.clause_embedding.output_dim,
                        device=clause_emb.device, dtype=clause_emb.dtype,
                    )
                clause_emb = self.clause_proj(torch.cat([clause_emb, clause_feat_emb], dim=-1))

            return clause_emb

    # Script GraphNorm modules
    if hasattr(model, 'norms'):
        for i in range(len(model.norms)):
            if not isinstance(model.norms[i], torch.jit.ScriptModule):
                model.norms[i] = torch.jit.script(model.norms[i])

    wrapper = _GNNEncoderWrapper(model)
    wrapper.eval()

    num_nodes, num_clauses = 10, 3
    example_x = torch.randn(num_nodes, config.get("input_dim", 13), device=trace_device)
    example_adj = torch.eye(num_nodes, device=trace_device).to_sparse()
    example_pool = (torch.ones(num_clauses, num_nodes, device=trace_device) / num_nodes).to_sparse()
    example_clause_features = torch.randn(num_clauses, 9, device=trace_device)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (example_x, example_adj, example_pool, example_clause_features),
            check_trace=False,
        )
    traced.save(str(encoder_path))


def _export_scorer(scorer: nn.Module, scorer_path: Path, hidden_dim: int, trace_device: torch.device):
    """Export scorer model that takes (u_emb, p_emb) → scores."""
    scorer.eval()

    num_u, num_p = 5, 3
    example_u = torch.randn(num_u, hidden_dim, device=trace_device)
    example_p = torch.randn(num_p, hidden_dim, device=trace_device)

    with torch.no_grad():
        traced = torch.jit.trace(scorer, (example_u, example_p), check_trace=False)
    traced.save(str(scorer_path))


# =============================================================================
# Base MiniLM export (for Rust trace embedding)
# =============================================================================


def export_base_minilm(output_dir):
    """Export frozen MiniLM encoder + mean-pool as TorchScript for Rust.

    Produces:
        output_dir/base_minilm.pt — TorchScript model (input_ids, attention_mask) → [B, 384]
        output_dir/base_minilm_tokenizer/ — HuggingFace tokenizer files
    """
    from transformers import AutoTokenizer, AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class MeanPoolWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            return (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    wrapper = MeanPoolWrapper(encoder)
    wrapper.eval()

    dummy_ids = torch.zeros(3, 32, dtype=torch.long)
    dummy_mask = torch.ones(3, 32, dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))
    traced.save(str(output_dir / "base_minilm.pt"))
    tokenizer.save_pretrained(str(output_dir / "base_minilm_tokenizer"))


def ensure_base_minilm(output_dir):
    """Export base MiniLM if not already present.

    Returns:
        (model_path, tokenizer_path) tuple of Path objects
    """
    output_dir = Path(output_dir)
    model_path = output_dir / "base_minilm.pt"
    tokenizer_path = output_dir / "base_minilm_tokenizer"
    if not model_path.exists():
        export_base_minilm(output_dir)
    return model_path, tokenizer_path
