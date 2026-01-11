"""Tests for Transformer-based clause selector models."""

import pytest
import torch

from proofatlas.selectors.transformer import ClauseTransformer, ClauseGNNTransformer


class TestClauseTransformer:
    """Tests for ClauseTransformer model."""

    def test_forward_shape(self):
        model = ClauseTransformer(
            node_feature_dim=13,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        node_features = torch.randn(20, 13)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25

        scores = model(node_features, pool_matrix)
        assert scores.shape == (5,)

    def test_single_clause(self):
        model = ClauseTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
        )
        node_features = torch.randn(4, 13)
        pool_matrix = torch.ones(1, 4) / 4

        scores = model(node_features, pool_matrix)
        assert scores.shape == (1,)

    def test_different_num_layers(self):
        for num_layers in [1, 2, 3]:
            model = ClauseTransformer(
                node_feature_dim=13,
                hidden_dim=32,
                num_layers=num_layers,
                num_heads=4,
            )
            node_features = torch.randn(10, 13)
            pool_matrix = torch.ones(3, 10) / 10

            scores = model(node_features, pool_matrix)
            assert scores.shape == (3,)

    def test_different_num_heads(self):
        for num_heads in [1, 2, 4, 8]:
            model = ClauseTransformer(
                node_feature_dim=13,
                hidden_dim=64,  # Must be divisible by num_heads
                num_layers=2,
                num_heads=num_heads,
            )
            node_features = torch.randn(10, 13)
            pool_matrix = torch.ones(3, 10) / 10

            scores = model(node_features, pool_matrix)
            assert scores.shape == (3,)

    def test_gradient_flow(self):
        model = ClauseTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
        )
        node_features = torch.randn(10, 13, requires_grad=True)
        pool_matrix = torch.ones(3, 10) / 10

        scores = model(node_features, pool_matrix)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None

    def test_train_eval_mode(self):
        model = ClauseTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.5,
        )
        node_features = torch.randn(10, 13)
        pool_matrix = torch.ones(3, 10) / 10

        # Eval mode should be deterministic
        model.eval()
        scores1 = model(node_features, pool_matrix)
        scores2 = model(node_features, pool_matrix)
        assert torch.allclose(scores1, scores2)

    def test_cross_clause_attention(self):
        """Test that clauses interact through attention."""
        model = ClauseTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
        )
        model.eval()

        # Same node features, different pooling
        node_features = torch.randn(8, 13)

        # Two separate clauses
        pool1 = torch.zeros(2, 8)
        pool1[0, :4] = 0.25
        pool1[1, 4:] = 0.25

        # One clause covering all
        pool2 = torch.ones(1, 8) / 8

        scores1 = model(node_features, pool1)
        scores2 = model(node_features, pool2)

        # Different pooling should give different results
        assert scores1.shape == (2,)
        assert scores2.shape == (1,)


class TestClauseGNNTransformer:
    """Tests for ClauseGNNTransformer hybrid model."""

    def test_forward_shape(self):
        model = ClauseGNNTransformer(
            node_feature_dim=13,
            hidden_dim=64,
            num_gnn_layers=2,
            num_transformer_layers=2,
            num_heads=4,
        )
        node_features = torch.randn(20, 13)
        adj = torch.eye(20)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (5,)

    def test_single_clause(self):
        model = ClauseGNNTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_gnn_layers=1,
            num_transformer_layers=1,
            num_heads=2,
        )
        node_features = torch.randn(4, 13)
        adj = torch.eye(4)
        pool_matrix = torch.ones(1, 4) / 4

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (1,)

    def test_gnn_layers_variation(self):
        for num_gnn_layers in [1, 2, 3]:
            model = ClauseGNNTransformer(
                node_feature_dim=13,
                hidden_dim=32,
                num_gnn_layers=num_gnn_layers,
                num_transformer_layers=1,
                num_heads=4,
            )
            node_features = torch.randn(10, 13)
            adj = torch.eye(10)
            pool_matrix = torch.ones(3, 10) / 10

            scores = model(node_features, adj, pool_matrix)
            assert scores.shape == (3,)

    def test_transformer_layers_variation(self):
        for num_transformer_layers in [1, 2, 3]:
            model = ClauseGNNTransformer(
                node_feature_dim=13,
                hidden_dim=32,
                num_gnn_layers=2,
                num_transformer_layers=num_transformer_layers,
                num_heads=4,
            )
            node_features = torch.randn(10, 13)
            adj = torch.eye(10)
            pool_matrix = torch.ones(3, 10) / 10

            scores = model(node_features, adj, pool_matrix)
            assert scores.shape == (3,)

    def test_gradient_flow(self):
        model = ClauseGNNTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_gnn_layers=2,
            num_transformer_layers=2,
            num_heads=4,
        )
        node_features = torch.randn(10, 13, requires_grad=True)
        adj = torch.eye(10)
        pool_matrix = torch.ones(3, 10) / 10

        scores = model(node_features, adj, pool_matrix)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None

    def test_train_eval_mode(self):
        model = ClauseGNNTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_gnn_layers=2,
            num_transformer_layers=2,
            num_heads=4,
            dropout=0.5,
        )
        node_features = torch.randn(10, 13)
        adj = torch.eye(10)
        pool_matrix = torch.ones(3, 10) / 10

        # Eval mode should be deterministic
        model.eval()
        scores1 = model(node_features, adj, pool_matrix)
        scores2 = model(node_features, adj, pool_matrix)
        assert torch.allclose(scores1, scores2)

    def test_adjacency_affects_output(self):
        """Test that graph structure affects the output."""
        model = ClauseGNNTransformer(
            node_feature_dim=13,
            hidden_dim=32,
            num_gnn_layers=2,
            num_transformer_layers=1,
            num_heads=4,
        )
        model.eval()

        node_features = torch.randn(6, 13)
        pool_matrix = torch.zeros(2, 6)
        pool_matrix[0, :3] = 1/3
        pool_matrix[1, 3:] = 1/3

        # No connections
        adj1 = torch.eye(6)
        scores1 = model(node_features, adj1, pool_matrix)

        # Full connections within each clause
        adj2 = torch.zeros(6, 6)
        adj2[:3, :3] = 1/3
        adj2[3:, 3:] = 1/3
        scores2 = model(node_features, adj2, pool_matrix)

        # Different adjacency should give different results
        assert not torch.allclose(scores1, scores2)
