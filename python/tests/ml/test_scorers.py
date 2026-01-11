"""Tests for clause scoring heads."""

import pytest
import torch

from proofatlas.selectors.scorers import (
    MLPScorer,
    AttentionScorer,
    TransformerScorer,
    CrossAttentionScorer,
    create_scorer,
)


class TestMLPScorer:
    def test_forward_shape(self):
        scorer = MLPScorer(hidden_dim=64)
        x = torch.randn(10, 64)
        scores = scorer(x)
        assert scores.shape == (10,)

    def test_single_clause(self):
        scorer = MLPScorer(hidden_dim=32)
        x = torch.randn(1, 32)
        scores = scorer(x)
        assert scores.shape == (1,)


class TestAttentionScorer:
    def test_forward_shape(self):
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        scores = scorer(x)
        assert scores.shape == (10,)

    def test_single_clause(self):
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        x = torch.randn(1, 64)
        scores = scorer(x)
        assert scores.shape == (1,)

    def test_residual_connection(self):
        """Verify residual connection preserves gradient flow."""
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        x = torch.randn(5, 64, requires_grad=True)
        scores = scorer(x)
        scores.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_num_heads(self):
        for num_heads in [1, 2, 4, 8]:
            scorer = AttentionScorer(hidden_dim=64, num_heads=num_heads)
            x = torch.randn(8, 64)
            scores = scorer(x)
            assert scores.shape == (8,)


class TestTransformerScorer:
    def test_forward_shape(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        x = torch.randn(10, 64)
        scores = scorer(x)
        assert scores.shape == (10,)

    def test_single_clause(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        x = torch.randn(1, 64)
        scores = scorer(x)
        assert scores.shape == (1,)

    def test_different_num_layers(self):
        for num_layers in [1, 2, 4]:
            scorer = TransformerScorer(hidden_dim=64, num_layers=num_layers, num_heads=4)
            x = torch.randn(8, 64)
            scores = scorer(x)
            assert scores.shape == (8,)

    def test_gradient_flow(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        x = torch.randn(5, 64, requires_grad=True)
        scores = scorer(x)
        scores.sum().backward()
        assert x.grad is not None


class TestCrossAttentionScorer:
    def test_forward_shape(self):
        scorer = CrossAttentionScorer(hidden_dim=64, num_heads=4)
        x = torch.randn(10, 64)
        scores = scorer(x)
        assert scores.shape == (10,)

    def test_single_clause(self):
        scorer = CrossAttentionScorer(hidden_dim=64, num_heads=4)
        x = torch.randn(1, 64)
        scores = scorer(x)
        assert scores.shape == (1,)

    def test_learnable_query(self):
        """Verify the learnable query parameter exists and is trainable."""
        scorer = CrossAttentionScorer(hidden_dim=64, num_heads=4)
        assert hasattr(scorer, 'query')
        assert scorer.query.requires_grad
        assert scorer.query.shape == (1, 64)


class TestCreateScorer:
    def test_mlp(self):
        scorer = create_scorer("mlp", hidden_dim=64)
        assert isinstance(scorer, MLPScorer)

    def test_attention(self):
        scorer = create_scorer("attention", hidden_dim=64, num_heads=4)
        assert isinstance(scorer, AttentionScorer)

    def test_transformer(self):
        scorer = create_scorer("transformer", hidden_dim=64, num_heads=4, num_layers=2)
        assert isinstance(scorer, TransformerScorer)

    def test_cross_attention(self):
        scorer = create_scorer("cross_attention", hidden_dim=64, num_heads=4)
        assert isinstance(scorer, CrossAttentionScorer)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown scorer type"):
            create_scorer("invalid", hidden_dim=64)

    def test_dropout_ignored(self):
        """Verify dropout parameter is accepted but ignored."""
        scorer = create_scorer("attention", hidden_dim=64, dropout=0.5)
        assert isinstance(scorer, AttentionScorer)


class TestScorerTrainEval:
    """Test train/eval mode behavior."""

    def test_mlp_deterministic(self):
        scorer = MLPScorer(hidden_dim=64)
        scorer.eval()
        x = torch.randn(10, 64)
        scores1 = scorer(x)
        scores2 = scorer(x)
        assert torch.allclose(scores1, scores2)

    def test_attention_deterministic(self):
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        scorer.eval()
        x = torch.randn(10, 64)
        scores1 = scorer(x)
        scores2 = scorer(x)
        assert torch.allclose(scores1, scores2)

    def test_transformer_deterministic(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        scorer.eval()
        x = torch.randn(10, 64)
        scores1 = scorer(x)
        scores2 = scorer(x)
        assert torch.allclose(scores1, scores2)


class TestScorerIntegration:
    """Integration tests with GNN models."""

    def test_gcn_with_attention_scorer(self):
        from proofatlas.selectors.gnn import ClauseGCN

        model = ClauseGCN(
            node_feature_dim=13,
            hidden_dim=64,
            num_layers=2,
            scorer_type="attention",
            scorer_num_heads=4,
        )
        node_features = torch.randn(50, 13)
        adj = torch.randn(50, 50)
        pool_matrix = torch.randn(10, 50)

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (10,)

    def test_gcn_with_transformer_scorer(self):
        from proofatlas.selectors.gnn import ClauseGCN

        model = ClauseGCN(
            node_feature_dim=13,
            hidden_dim=64,
            num_layers=2,
            scorer_type="transformer",
            scorer_num_layers=2,
        )
        node_features = torch.randn(50, 13)
        adj = torch.randn(50, 50)
        pool_matrix = torch.randn(10, 50)

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (10,)

    def test_factory_with_scorer(self):
        from proofatlas.selectors.factory import create_model

        model = create_model(
            model_type="gcn",
            hidden_dim=64,
            num_layers=2,
            scorer_type="attention",
        )
        node_features = torch.randn(50, 13)
        adj = torch.randn(50, 50)
        pool_matrix = torch.randn(10, 50)

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (10,)
