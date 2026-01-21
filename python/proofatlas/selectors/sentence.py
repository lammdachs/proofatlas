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

    def export_torchscript(self, path: str):
        """
        Export model to TorchScript format for tch-rs inference.

        Args:
            path: Output path for TorchScript model (.pt)
        """
        self.eval()

        # Create dummy inputs for tracing
        dummy_input_ids = torch.zeros((1, 32), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

        with torch.no_grad():
            traced = torch.jit.trace(self, (dummy_input_ids, dummy_attention_mask))

        traced.save(path)
        print(f"Exported TorchScript model to {path}")
        print(f"Encoder: {self.model_name}")
        print(f"Encoder dim: {self.encoder_dim}, Hidden dim: {self.hidden_dim}")


# Backwards compatibility aliases
PretrainedClauseEncoder = SentenceEncoder
ClauseSentenceEncoder = SentenceEncoder
HAS_SENTENCE_TRANSFORMERS = HAS_TRANSFORMERS
