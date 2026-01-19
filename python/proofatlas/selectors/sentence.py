"""
Sentence embedding model for clause encoding.

Uses pretrained transformer models (e.g., MiniLM) to encode clause strings.
Architecture is designed to be mirrored in Burn for inference.
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .scorers import create_scorer


class SentenceEncoder(nn.Module):
    """
    Sentence encoder for clause strings using pretrained transformers.

    Loads a pretrained model (e.g., all-MiniLM-L6-v2) and uses it to encode
    clause strings. The transformer architecture can be reimplemented in Burn
    for inference.

    Architecture:
        - Token embeddings + position embeddings
        - N transformer encoder layers (pre-norm in modern models)
        - Mean pooling over tokens
        - Projection to scorer hidden dim
        - Configurable scorer head
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 64,
        freeze_encoder: bool = False,
        scorer_type: str = "mlp",
        scorer_num_heads: int = 4,
        scorer_num_layers: int = 2,
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            hidden_dim: Output dimension for clause embeddings (scorer input)
            freeze_encoder: Whether to freeze the transformer encoder
            scorer_type: Type of scorer head
            scorer_num_heads: Attention heads for scorer
            scorer_num_layers: Layers for transformer scorer
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for SentenceEncoder. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection to hidden_dim
        self.projection = nn.Linear(self.encoder_dim, hidden_dim)

        # Scorer
        self.scorer = create_scorer(
            scorer_type,
            hidden_dim,
            num_heads=scorer_num_heads,
            num_layers=scorer_num_layers,
        )

    def encode(self, clause_strings: List[str]) -> torch.Tensor:
        """
        Encode clause strings to embeddings.

        Args:
            clause_strings: List of clause string representations

        Returns:
            Embeddings [num_clauses, encoder_dim]
        """
        # Tokenize
        device = self.projection.weight.device
        inputs = self.tokenizer(
            clause_strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Move encoder to same device if needed
        if next(self.encoder.parameters()).device != device:
            self.encoder.to(device)

        # Encode
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(**inputs)

        # Mean pooling over tokens (excluding padding)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        embeddings = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)

        return embeddings

    def forward(self, clause_strings: List[str]) -> torch.Tensor:
        """
        Encode clauses and compute scores.

        Args:
            clause_strings: List of clause string representations

        Returns:
            Scores [num_clauses]
        """
        # Get embeddings
        embeddings = self.encode(clause_strings)

        # Clone if frozen (for gradient through projection)
        if self.freeze_encoder:
            embeddings = embeddings.clone()

        # Project to hidden_dim
        clause_emb = self.projection(embeddings)

        # Score
        return self.scorer(clause_emb)

    def _convert_name_to_burn(self, name: str) -> str:
        """Convert HuggingFace parameter name to Burn naming convention."""
        # HuggingFace: encoder.layer.0.attention.self.query
        # Burn:        encoder.layer.0.attention.self_attn.query
        name = name.replace("attention.self.", "attention.self_attn.")

        # HuggingFace: LayerNorm
        # Burn:        layer_norm
        name = name.replace("LayerNorm", "layer_norm")

        return name

    def export_weights(self, path: str):
        """
        Export weights to safetensors format for Burn loading.

        Exports the transformer encoder weights in a format that can be
        loaded by a Burn implementation of the same architecture.

        Weight names are converted to match Burn's field naming convention:
        - BERT weights → 'encoder.bert.*'
        - Projection → 'encoder.projection.*'
        - Scorer → 'scorer.*'
        - 'attention.self.' → 'attention.self_attn.'
        - 'LayerNorm' → 'layer_norm'

        Args:
            path: Output path for safetensors file
        """
        from safetensors.torch import save_file

        state_dict = {}

        # Export encoder weights with Burn-compatible naming
        # Structure: encoder.bert.* for BERT, encoder.projection.* for projection
        for name, param in self.encoder.named_parameters():
            burn_name = self._convert_name_to_burn(name)
            state_dict[f"encoder.bert.{burn_name}"] = param.data

        # Export projection (part of encoder in Burn)
        for name, param in self.projection.named_parameters():
            state_dict[f"encoder.projection.{name}"] = param.data

        # Export scorer
        for name, param in self.scorer.named_parameters():
            state_dict[f"scorer.{name}"] = param.data

        save_file(state_dict, path)
        print(f"Exported weights to {path}")
        print(f"Encoder: {self.model_name}")
        print(f"Encoder dim: {self.encoder_dim}, Hidden dim: {self.hidden_dim}")

    @property
    def config(self) -> dict:
        """Return model configuration for Burn reimplementation."""
        return {
            "model_name": self.model_name,
            "vocab_size": self.encoder.config.vocab_size,
            "encoder_dim": self.encoder_dim,
            "num_layers": self.encoder.config.num_hidden_layers,
            "num_heads": self.encoder.config.num_attention_heads,
            "intermediate_dim": self.encoder.config.intermediate_size,
            "max_position_embeddings": self.encoder.config.max_position_embeddings,
            "hidden_dim": self.hidden_dim,
        }


# Backwards compatibility aliases
PretrainedClauseEncoder = SentenceEncoder
ClauseSentenceEncoder = SentenceEncoder
HAS_SENTENCE_TRANSFORMERS = HAS_TRANSFORMERS
