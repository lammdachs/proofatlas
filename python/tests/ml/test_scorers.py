"""Tests for clause scoring heads."""

import pytest
import torch

from proofatlas.selectors.scorers import (
    MLPScorer,
    AttentionScorer,
    TransformerScorer,
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

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown scorer type"):
            create_scorer("invalid", hidden_dim=64)

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


def _make_valid_inputs(num_nodes: int, num_clauses: int):
    """Create valid test inputs with correct value ranges."""
    # Node features: [type (0-5), arity (>=0), arg_pos (>=0)]
    node_features = torch.zeros(num_nodes, 3)
    node_features[:, 0] = torch.randint(0, 6, (num_nodes,)).float()
    node_features[:, 1] = torch.randint(0, 5, (num_nodes,)).float()
    node_features[:, 2] = torch.randint(0, 10, (num_nodes,)).float()

    # Normalized adjacency with self-loops
    adj = torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes)
    adj = adj / adj.sum(dim=1, keepdim=True)

    pool_matrix = torch.ones(num_clauses, num_nodes) / num_nodes

    # Clause features: 9D
    clause_features = torch.zeros(num_clauses, 9)
    clause_features[:, 0] = torch.rand(num_clauses)                       # age
    clause_features[:, 1] = torch.randint(0, 5, (num_clauses,)).float()   # role
    clause_features[:, 2] = torch.randint(0, 7, (num_clauses,)).float()   # rule
    clause_features[:, 3] = torch.randint(1, 20, (num_clauses,)).float()  # size
    clause_features[:, 4] = torch.randint(0, 10, (num_clauses,)).float()  # depth
    clause_features[:, 5] = torch.randint(1, 30, (num_clauses,)).float()  # symbol_count
    clause_features[:, 6] = torch.randint(1, 15, (num_clauses,)).float()  # distinct_symbols
    clause_features[:, 7] = torch.randint(0, 10, (num_clauses,)).float()  # variable_count
    clause_features[:, 8] = torch.randint(0, 5, (num_clauses,)).float()   # distinct_vars

    return node_features, adj, pool_matrix, clause_features


class TestAttentionScorerCrossAttention:
    """Tests for AttentionScorer cross-attention (p_emb) interface."""

    def test_cross_attention_shape(self):
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        u_emb = torch.randn(8, 64)
        p_emb = torch.randn(5, 64)
        scores = scorer(u_emb, p_emb)
        assert scores.shape == (8,)

    def test_cross_attention_empty_p(self):
        """With sentinel only (0 processed), should still produce valid output."""
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        u_emb = torch.randn(8, 64)
        # Empty P: KV source is just the sentinel [1, hidden_dim]
        p_emb = torch.zeros(0, 64)
        # sentinel is prepended, so KV has 1 element
        scores = scorer(u_emb, p_emb)
        assert scores.shape == (8,)
        assert not torch.isnan(scores).any()

    def test_self_attention_fallback(self):
        """p_emb=None should behave same as self-attention (backward compat)."""
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        scorer.eval()
        u_emb = torch.randn(8, 64)
        with torch.no_grad():
            scores_none = scorer(u_emb, None)
            scores_no_arg = scorer(u_emb)
        assert torch.allclose(scores_none, scores_no_arg)

    def test_cross_attention_gradient_through_both(self):
        scorer = AttentionScorer(hidden_dim=64, num_heads=4)
        u_emb = torch.randn(8, 64, requires_grad=True)
        p_emb = torch.randn(5, 64, requires_grad=True)
        scores = scorer(u_emb, p_emb)
        scores.sum().backward()
        assert u_emb.grad is not None
        assert p_emb.grad is not None


class TestTransformerScorerCrossAttention:
    """Tests for TransformerScorer cross-attention interface."""

    def test_cross_attention_shape(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        u_emb = torch.randn(8, 64)
        p_emb = torch.randn(5, 64)
        scores = scorer(u_emb, p_emb)
        assert scores.shape == (8,)

    def test_cross_attention_empty_p(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        u_emb = torch.randn(8, 64)
        p_emb = torch.zeros(0, 64)
        scores = scorer(u_emb, p_emb)
        assert scores.shape == (8,)
        assert not torch.isnan(scores).any()

    def test_self_attention_fallback(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        scorer.eval()
        u_emb = torch.randn(8, 64)
        with torch.no_grad():
            scores_none = scorer(u_emb, None)
            scores_no_arg = scorer(u_emb)
        assert torch.allclose(scores_none, scores_no_arg)

    def test_cross_attention_gradient_through_both(self):
        scorer = TransformerScorer(hidden_dim=64, num_layers=2, num_heads=4)
        u_emb = torch.randn(8, 64, requires_grad=True)
        p_emb = torch.randn(5, 64, requires_grad=True)
        scores = scorer(u_emb, p_emb)
        scores.sum().backward()
        assert u_emb.grad is not None
        assert p_emb.grad is not None


class TestMLPScorerIgnoresPEmb:
    """Tests that MLPScorer ignores p_emb."""

    def test_ignores_p_emb(self):
        scorer = MLPScorer(hidden_dim=64)
        scorer.eval()
        u_emb = torch.randn(10, 64)
        p_emb = torch.randn(5, 64)
        with torch.no_grad():
            scores_with = scorer(u_emb, p_emb)
            scores_without = scorer(u_emb)
        assert torch.allclose(scores_with, scores_without)


class TestScorerIntegration:
    """Integration tests with GNN models."""

    def test_gcn_with_attention_scorer(self):
        from proofatlas.selectors.gnn import ClauseGCN

        model = ClauseGCN(
            hidden_dim=64,
            num_layers=2,
            scorer_type="attention",
            scorer_num_heads=4,
        )
        node_features, adj, pool_matrix, clause_features = _make_valid_inputs(50, 10)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()

    def test_gcn_with_transformer_scorer(self):
        from proofatlas.selectors.gnn import ClauseGCN

        model = ClauseGCN(
            hidden_dim=64,
            num_layers=2,
            scorer_type="transformer",
            scorer_num_layers=2,
        )
        node_features, adj, pool_matrix, clause_features = _make_valid_inputs(50, 10)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()

    def test_factory_with_scorer(self):
        from proofatlas.selectors.factory import create_model

        model = create_model(
            model_type="gcn",
            hidden_dim=64,
            num_layers=2,
            scorer_type="attention",
        )
        node_features, adj, pool_matrix, clause_features = _make_valid_inputs(50, 10)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()
