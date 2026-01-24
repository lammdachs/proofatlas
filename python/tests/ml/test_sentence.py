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

        # Check tokenizer was saved
        tokenizer_dir = tmp_path / "model_tokenizer"
        assert tokenizer_dir.exists()
        assert (tokenizer_dir / "tokenizer.json").exists()

        # Check we can load and run the model
        loaded = torch.jit.load(str(path))
        dummy_ids = torch.zeros((2, 16), dtype=torch.long)
        dummy_mask = torch.ones((2, 16), dtype=torch.long)
        scores = loaded(dummy_ids, dummy_mask)
        assert scores.shape == (2,)

    def test_export_tokenizer_rust_compatible(self, tmp_path):
        """Test that exported tokenizer works with tokenizers library (same as Rust)."""
        from tokenizers import Tokenizer
        from proofatlas.selectors.sentence import SentenceEncoder

        model = SentenceEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            freeze_encoder=True,
        )

        path = tmp_path / "model.pt"
        model.export_torchscript(str(path))

        # Load tokenizer using tokenizers library (same as Rust crate)
        tokenizer_path = tmp_path / "model_tokenizer" / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Test tokenization
        clauses = ["p(X, Y) | ~q(Y)", "r(a, b)"]
        encodings = tokenizer.encode_batch(clauses)

        assert len(encodings) == 2
        assert len(encodings[0].ids) > 0
        assert len(encodings[0].attention_mask) > 0

        # Verify we can run the model with tokenized inputs
        loaded = torch.jit.load(str(path))

        max_len = max(len(e.ids) for e in encodings)
        input_ids = torch.zeros((2, max_len), dtype=torch.long)
        attention_mask = torch.zeros((2, max_len), dtype=torch.long)

        for i, enc in enumerate(encodings):
            input_ids[i, :len(enc.ids)] = torch.tensor(enc.ids)
            attention_mask[i, :len(enc.attention_mask)] = torch.tensor(enc.attention_mask)

        scores = loaded(input_ids, attention_mask)
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
