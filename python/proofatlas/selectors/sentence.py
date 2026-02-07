"""
Sentence embedding model for clause encoding.

Uses pretrained transformer models (e.g., MiniLM) to encode clause strings.
Exports to TorchScript for Rust inference via tch-rs.

Rust Integration:
    The export_torchscript() method saves both the model (.pt) and tokenizer
    (tokenizer.json). Rust loads these via tch::CModule and tokenizers::Tokenizer.
    See crates/proofatlas/src/selectors/sentence.rs for the Rust implementation.

    Tokenization in Python (training) and Rust (inference) produces identical
    token IDs. Model outputs match within floating-point precision (~1e-8).
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
from .gnn import ClauseFeatureEmbedding


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
        use_clause_features: bool = False,
        sin_dim: int = 8,
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            hidden_dim: Output dimension for clause embeddings (scorer input)
            freeze_encoder: Whether to freeze the transformer encoder
            scorer_type: Type of scorer head
            scorer_num_heads: Attention heads for scorer
            scorer_num_layers: Layers for transformer scorer
            use_clause_features: Concatenate clause-level features (age, role, size)
            sin_dim: Sinusoidal encoding dimension for clause features
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
        self.use_clause_features = use_clause_features

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection to hidden_dim
        self.projection = nn.Linear(self.encoder_dim, hidden_dim)

        # Clause feature embedding (optional, matching GCN pattern)
        if use_clause_features:
            self.clause_embedding = ClauseFeatureEmbedding(sin_dim=sin_dim)
            concat_dim = hidden_dim + self.clause_embedding.output_dim
            self.clause_proj = nn.Linear(concat_dim, hidden_dim)
        else:
            self.clause_embedding = None
            self.clause_proj = None

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

    def forward(
        self,
        clause_strings: List[str],
        clause_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode clauses and compute scores.

        Args:
            clause_strings: List of clause string representations
            clause_features: [num_clauses, 3] raw clause features (age, role, size).
                            Used when use_clause_features=True.

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

        # Add clause features if configured
        if self.use_clause_features and self.clause_embedding is not None:
            if clause_features is not None:
                clause_feat_emb = self.clause_embedding(clause_features)
            else:
                num_clauses = len(clause_strings)
                clause_feat_emb = torch.zeros(
                    num_clauses, self.clause_embedding.output_dim,
                    device=clause_emb.device, dtype=clause_emb.dtype
                )
            clause_emb = self.clause_proj(torch.cat([clause_emb, clause_feat_emb], dim=-1))

        # Score
        return self.scorer(clause_emb)

    def forward_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-tokenized inputs (for TorchScript export).

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask (1=real, 0=padding)

        Returns:
            Scores [batch_size]
        """
        # Encode with transformer
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling over tokens (excluding padding)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        embeddings = (token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Project to hidden_dim
        clause_emb = self.projection(embeddings)

        # Score
        return self.scorer(clause_emb)

    def export_torchscript(self, path: str, save_tokenizer: bool = True):
        """
        Export model to TorchScript format for tch-rs inference.

        The exported model takes pre-tokenized inputs (input_ids, attention_mask).
        The tokenizer is saved alongside the model for use in Rust.

        Rust Compatibility:
            The Rust inference code uses the `tokenizers` crate to load the
            exported tokenizer.json. This is equivalent to the Python `tokenizers`
            library (same underlying implementation). Model outputs are identical
            within floating-point precision (~1e-8).

            Compatible models (known to work):
                - sentence-transformers/* (recommended for clause encoding)
                - BERT-family models with fast tokenizers
                - Modern models using the `tokenizers` library internally

            Potentially incompatible:
                - Models without tokenizer.json (legacy vocab.txt format)
                - SentencePiece models (may have edge cases)
                - Models without a padding token defined

        Args:
            path: Output path for TorchScript model (.pt)
            save_tokenizer: If True, save tokenizer to {path_stem}_tokenizer/
        """
        from pathlib import Path

        self.eval()

        # Create a wrapper module that exposes forward_tokens as forward
        class _ExportWrapper(nn.Module):
            def __init__(self, encoder, projection, scorer):
                super().__init__()
                self.encoder = encoder
                self.projection = projection
                self.scorer = scorer

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).float()
                embeddings = (token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                clause_emb = self.projection(embeddings)
                return self.scorer(clause_emb)

        wrapper = _ExportWrapper(self.encoder, self.projection, self.scorer)
        wrapper.eval()

        # Create dummy inputs for tracing
        dummy_input_ids = torch.zeros((1, 32), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (dummy_input_ids, dummy_attention_mask))

        traced.save(path)
        print(f"Exported TorchScript model to {path}")

        # Save tokenizer for Rust inference
        if save_tokenizer:
            path = Path(path)
            tokenizer_dir = path.parent / f"{path.stem}_tokenizer"
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(tokenizer_dir))
            print(f"Saved tokenizer to {tokenizer_dir}/")

        print(f"Encoder: {self.model_name}")
        print(f"Encoder dim: {self.encoder_dim}, Hidden dim: {self.hidden_dim}")
