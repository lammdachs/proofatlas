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

    def test_export_torchscript(self, tmp_path):
        """Test TorchScript export for tch-rs inference."""
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=True,
        )

        path = tmp_path / "model.pt"
        model.export_torchscript(str(path))

        assert path.exists()

        # Check we can load and run the model
        loaded = torch.jit.load(str(path))
        dummy_ids = torch.zeros((2, 16), dtype=torch.long)
        dummy_mask = torch.ones((2, 16), dtype=torch.long)
        scores = loaded(dummy_ids, dummy_mask)
        assert scores.shape == (2,)


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
