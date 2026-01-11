"""Tests for sentence embedding model."""

import pytest
import torch

from proofatlas.selectors.sentence import HAS_TRANSFORMERS


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestSentenceEncoder:
    """Tests for the sentence encoder."""

    def test_forward_shape(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=True,
        )

        clause_strings = [
            "p(X, Y) | ~q(Y)",
            "r(a, b)",
            "~p(X, X) | s(X)",
        ]

        scores = model(clause_strings)
        assert scores.shape == (3,)

    def test_encode_shape(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
        )

        clause_strings = ["p(X)", "q(Y, Z)"]
        embeddings = model.encode(clause_strings)

        # all-MiniLM-L6-v2 has 384-dim embeddings
        assert embeddings.shape == (2, 384)

    def test_frozen_encoder(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=True,
        )

        # Check encoder params are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # But projection and scorer should be trainable
        assert model.projection.weight.requires_grad
        for param in model.scorer.parameters():
            assert param.requires_grad

    def test_trainable_encoder(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=False,
        )

        # Check encoder params are trainable
        for param in model.encoder.parameters():
            assert param.requires_grad

    def test_different_scorers(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        for scorer_type in ["mlp", "attention"]:
            model = SentenceEncoder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                hidden_dim=64,
                scorer_type=scorer_type,
                freeze_encoder=True,
            )

            clause_strings = ["p(X)", "q(Y)", "r(Z)"]
            scores = model(clause_strings)
            assert scores.shape == (3,)

    def test_config_property(self):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
        )

        config = model.config
        assert config["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert config["encoder_dim"] == 384
        assert config["hidden_dim"] == 64
        assert config["num_layers"] == 6
        assert config["num_heads"] == 12

    def test_export_weights(self, tmp_path):
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=True,
        )

        path = tmp_path / "model.safetensors"
        model.export_weights(str(path))

        assert path.exists()

        # Check we can load the weights
        from safetensors.torch import load_file
        weights = load_file(str(path))
        # Now uses Burn-compatible naming (bert.* prefix)
        assert "bert.embeddings.word_embeddings.weight" in weights
        assert "projection.weight" in weights

    def test_export_weights_burn_naming(self, tmp_path):
        """Test that exported weight names match Burn's field naming convention."""
        from proofatlas.selectors.sentence import SentenceEncoder
        from safetensors import safe_open

        model = SentenceEncoder(hidden_dim=32, freeze_encoder=True)
        path = tmp_path / "weights.safetensors"
        model.export_weights(str(path))

        with safe_open(str(path), framework="pt") as f:
            names = list(f.keys())

        # Verify Burn-compatible naming - BERT weights use 'bert.' prefix
        bert_names = [n for n in names if n.startswith("bert.")]
        assert len(bert_names) > 0, "Should have bert.* prefixed weights"

        # Check key naming conventions match Burn field names
        assert any("bert.embeddings.word_embeddings" in n for n in names)
        assert any("bert.embeddings.layer_norm" in n for n in names), \
            "Should convert LayerNorm to layer_norm"
        assert any("bert.encoder.layer.0.attention.self_attn" in n for n in names), \
            "Should convert attention.self to attention.self_attn"

        # Check that no HuggingFace naming remains
        assert not any("LayerNorm" in n for n in names), \
            "Should convert LayerNorm to layer_norm"
        assert not any(".self.query" in n for n in names), \
            "Should convert attention.self to attention.self_attn"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestFactoryIntegration:
    """Test factory integration for sentence model."""

    def test_sentence_factory(self):
        from proofatlas.selectors.factory import create_model

        model = create_model(
            model_type="sentence",
            hidden_dim=64,
        )

        clause_strings = ["p(X)", "q(Y)"]
        scores = model(clause_strings)
        assert scores.shape == (2,)

    def test_sentence_factory_with_scorer(self):
        from proofatlas.selectors.factory import create_model

        model = create_model(
            model_type="sentence",
            hidden_dim=64,
            scorer_type="attention",
            freeze_encoder=True,
        )

        clause_strings = ["p(X)", "q(Y)", "r(Z)"]
        scores = model(clause_strings)
        assert scores.shape == (3,)

    def test_sentence_factory_custom_model(self):
        from proofatlas.selectors.factory import create_model

        model = create_model(
            model_type="sentence",
            hidden_dim=32,
            sentence_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        clause_strings = ["p(X)"]
        scores = model(clause_strings)
        assert scores.shape == (1,)
