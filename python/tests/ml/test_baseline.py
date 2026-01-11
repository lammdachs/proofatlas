"""Tests for baseline clause selector models."""

import pytest
import torch

from proofatlas.selectors.baseline import NodeMLP, AgeWeightHeuristic


class TestNodeMLP:
    """Tests for NodeMLP baseline model."""

    def test_forward_shape(self):
        model = NodeMLP(node_feature_dim=13, hidden_dim=64, num_layers=2)
        node_features = torch.randn(20, 13)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25

        scores = model(node_features, pool_matrix)
        assert scores.shape == (5,)

    def test_single_clause(self):
        model = NodeMLP(node_feature_dim=13, hidden_dim=32, num_layers=2)
        node_features = torch.randn(4, 13)
        pool_matrix = torch.ones(1, 4) / 4

        scores = model(node_features, pool_matrix)
        assert scores.shape == (1,)

    def test_different_num_layers(self):
        for num_layers in [1, 2, 3, 4]:
            model = NodeMLP(node_feature_dim=13, hidden_dim=32, num_layers=num_layers)
            node_features = torch.randn(10, 13)
            pool_matrix = torch.ones(3, 10) / 10

            scores = model(node_features, pool_matrix)
            assert scores.shape == (3,)

    def test_gradient_flow(self):
        model = NodeMLP(node_feature_dim=13, hidden_dim=32, num_layers=2)
        node_features = torch.randn(10, 13, requires_grad=True)
        pool_matrix = torch.ones(3, 10) / 10

        scores = model(node_features, pool_matrix)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None

    def test_train_eval_mode(self):
        model = NodeMLP(node_feature_dim=13, hidden_dim=32, num_layers=2, dropout=0.5)
        node_features = torch.randn(10, 13)
        pool_matrix = torch.ones(3, 10) / 10

        # Eval mode should be deterministic
        model.eval()
        scores1 = model(node_features, pool_matrix)
        scores2 = model(node_features, pool_matrix)
        assert torch.allclose(scores1, scores2)

    def test_no_graph_structure(self):
        """NodeMLP ignores adjacency - only uses pooled features."""
        model = NodeMLP(node_feature_dim=13, hidden_dim=32, num_layers=2)
        model.eval()

        node_features = torch.randn(8, 13)
        pool_matrix = torch.ones(2, 8) / 8

        # Same features, same pooling - should give same results
        scores1 = model(node_features, pool_matrix)
        scores2 = model(node_features, pool_matrix)
        assert torch.allclose(scores1, scores2)


class TestAgeWeightHeuristic:
    """Tests for AgeWeightHeuristic model."""

    def test_forward_shape(self):
        model = AgeWeightHeuristic(age_probability=0.5)
        node_features = torch.randn(20, 13)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25

        logits = model(node_features, pool_matrix)
        assert logits.shape == (5,)

    def test_single_clause(self):
        model = AgeWeightHeuristic(age_probability=0.5)
        node_features = torch.randn(4, 13)
        pool_matrix = torch.ones(1, 4) / 4

        logits = model(node_features, pool_matrix)
        assert logits.shape == (1,)

    def test_age_probability_extremes(self):
        """Test with extreme age probabilities."""
        for p in [0.0, 0.5, 1.0]:
            model = AgeWeightHeuristic(age_probability=p)
            node_features = torch.randn(8, 13)
            pool_matrix = torch.ones(2, 8) / 8

            logits = model(node_features, pool_matrix)
            assert logits.shape == (2,)
            assert not torch.isnan(logits).any()

    def test_oldest_clause_preferred_with_p1(self):
        """With p=1, oldest clause should have highest logit."""
        model = AgeWeightHeuristic(age_probability=1.0)

        # Create features where clause 1 is oldest (higher age at index 9)
        node_features = torch.zeros(6, 13)
        # Clause 0 nodes (indices 0-2): age = 1
        node_features[0:3, 9] = 1.0
        # Clause 1 nodes (indices 3-5): age = 10 (older)
        node_features[3:6, 9] = 10.0

        pool_matrix = torch.zeros(2, 6)
        pool_matrix[0, :3] = 1/3
        pool_matrix[1, 3:] = 1/3

        logits = model(node_features, pool_matrix)
        # Clause 1 should have higher logit (it's older)
        assert logits[1] > logits[0]

    def test_lightest_clause_preferred_with_p0(self):
        """With p=0, lightest clause should have highest logit."""
        model = AgeWeightHeuristic(age_probability=0.0)

        # Create features where clause 0 is lighter (lower depth at index 8)
        node_features = torch.zeros(6, 13)
        # Clause 0 nodes: depth/weight = 1 (lighter)
        node_features[0:3, 8] = 1.0
        # Clause 1 nodes: depth/weight = 10 (heavier)
        node_features[3:6, 8] = 10.0
        # Different ages to avoid same clause selection
        node_features[0:3, 9] = 1.0
        node_features[3:6, 9] = 10.0

        pool_matrix = torch.zeros(2, 6)
        pool_matrix[0, :3] = 1/3
        pool_matrix[1, 3:] = 1/3

        logits = model(node_features, pool_matrix)
        # Clause 0 should have higher logit (it's lighter)
        assert logits[0] > logits[1]

    def test_deterministic(self):
        """Heuristic should be deterministic."""
        model = AgeWeightHeuristic(age_probability=0.5)
        node_features = torch.randn(10, 13)
        pool_matrix = torch.ones(3, 10) / 10

        logits1 = model(node_features, pool_matrix)
        logits2 = model(node_features, pool_matrix)
        assert torch.allclose(logits1, logits2)

    def test_buffer_registered(self):
        """Test that age_probability is registered as a buffer."""
        model = AgeWeightHeuristic(age_probability=0.7)
        assert 'p' in dict(model.named_buffers())
        assert model.p.item() == pytest.approx(0.7)
