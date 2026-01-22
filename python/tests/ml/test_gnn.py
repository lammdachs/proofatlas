"""Tests for GNN-based clause selector models."""

import pytest
import torch

from proofatlas.selectors.gnn import (
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    ScorerHead,
    ClauseGCN,
    ClauseGAT,
    ClauseGraphSAGE,
)


def make_node_features(num_nodes: int) -> torch.Tensor:
    """Create valid node features: [type (0-5), arity (>=0), arg_pos (>=0)]."""
    features = torch.zeros(num_nodes, 3)
    features[:, 0] = torch.randint(0, 6, (num_nodes,)).float()
    features[:, 1] = torch.randint(0, 5, (num_nodes,)).float()
    features[:, 2] = torch.randint(0, 10, (num_nodes,)).float()
    return features


def make_clause_features(num_clauses: int) -> torch.Tensor:
    """Create valid clause features: [age (0-1), role (0-4), size (>=1)]."""
    features = torch.zeros(num_clauses, 3)
    features[:, 0] = torch.rand(num_clauses)
    features[:, 1] = torch.randint(0, 5, (num_clauses,)).float()
    features[:, 2] = torch.randint(1, 10, (num_clauses,)).float()
    return features


def make_adj(num_nodes: int) -> torch.Tensor:
    """Create normalized adjacency matrix with self-loops."""
    adj = torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes)
    return adj / adj.sum(dim=1, keepdim=True)


class TestGCNLayer:
    """Tests for GCNLayer."""

    def test_forward_shape(self):
        layer = GCNLayer(in_dim=15, out_dim=64)
        x = torch.randn(10, 15)  # Embedded features (after NodeFeatureEmbedding)
        adj = torch.eye(10)
        out = layer(x, adj)
        assert out.shape == (10, 64)

    def test_message_passing(self):
        """Test that adjacency matrix affects output."""
        layer = GCNLayer(in_dim=4, out_dim=4)
        x = torch.randn(3, 4)

        adj_id = torch.eye(3)
        out_id = layer(x, adj_id)

        adj_full = torch.ones(3, 3) / 3
        out_full = layer(x, adj_full)

        assert not torch.allclose(out_id, out_full)

    def test_gradient_flow(self):
        layer = GCNLayer(in_dim=8, out_dim=8)
        x = torch.randn(5, 8, requires_grad=True)
        adj = torch.eye(5)
        out = layer(x, adj)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGATLayer:
    """Tests for GATLayer."""

    def test_forward_shape_concat(self):
        layer = GATLayer(in_dim=15, out_dim=16, num_heads=4, concat=True)
        x = torch.randn(10, 15)
        adj = torch.eye(10)
        out = layer(x, adj)
        assert out.shape == (10, 16 * 4)

    def test_forward_shape_mean(self):
        layer = GATLayer(in_dim=15, out_dim=16, num_heads=4, concat=False)
        x = torch.randn(10, 15)
        adj = torch.eye(10)
        out = layer(x, adj)
        assert out.shape == (10, 16)

    def test_attention_mask(self):
        """Test that disconnected nodes don't attend to each other."""
        layer = GATLayer(in_dim=4, out_dim=4, num_heads=2, concat=False)
        x = torch.randn(3, 4)
        adj = torch.eye(3)
        out = layer(x, adj)
        assert out.shape == (3, 4)

    def test_gradient_flow(self):
        layer = GATLayer(in_dim=8, out_dim=8, num_heads=2)
        x = torch.randn(5, 8, requires_grad=True)
        adj = torch.eye(5)
        out = layer(x, adj)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGraphSAGELayer:
    """Tests for GraphSAGELayer."""

    def test_forward_shape(self):
        layer = GraphSAGELayer(in_dim=15, out_dim=64)
        x = torch.randn(10, 15)
        adj = torch.eye(10)
        out = layer(x, adj)
        assert out.shape == (10, 64)

    def test_neighbor_aggregation(self):
        """Test that neighbors affect output."""
        layer = GraphSAGELayer(in_dim=4, out_dim=4)
        x = torch.randn(3, 4)

        adj_empty = torch.zeros(3, 3)
        out_empty = layer(x, adj_empty)

        adj_conn = torch.tensor([
            [0., 0.5, 0.5],
            [0.5, 0., 0.5],
            [0.5, 0.5, 0.],
        ])
        out_conn = layer(x, adj_conn)

        assert not torch.allclose(out_empty, out_conn)


class TestScorerHead:
    """Tests for ScorerHead."""

    def test_forward_shape(self):
        scorer = ScorerHead(hidden_dim=64)
        x = torch.randn(10, 64)
        out = scorer(x)
        assert out.shape == (10, 1)

    def test_train_eval_difference(self):
        """Test that dropout affects training mode."""
        scorer = ScorerHead(hidden_dim=64, dropout=0.5)
        x = torch.randn(10, 64)

        scorer.train()
        out1 = scorer(x)
        out2 = scorer(x)

        scorer.eval()
        out3 = scorer(x)
        out4 = scorer(x)
        assert torch.allclose(out3, out4)


class TestClauseGCN:
    """Tests for ClauseGCN model."""

    def test_forward_shape(self):
        model = ClauseGCN(hidden_dim=64, num_layers=3)
        node_features = make_node_features(20)
        adj = make_adj(20)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25
        clause_features = make_clause_features(5)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

    def test_single_clause(self):
        model = ClauseGCN(hidden_dim=32, num_layers=2)
        node_features = make_node_features(4)
        adj = make_adj(4)
        pool_matrix = torch.ones(1, 4) / 4
        clause_features = make_clause_features(1)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.numel() == 1
        assert not torch.isnan(scores).any()

    def test_different_num_layers(self):
        for num_layers in [1, 2, 3, 4]:
            model = ClauseGCN(hidden_dim=32, num_layers=num_layers)
            node_features = make_node_features(10)
            adj = make_adj(10)
            pool_matrix = torch.ones(2, 10) / 10
            clause_features = make_clause_features(2)

            scores = model(node_features, adj, pool_matrix, clause_features)
            assert scores.shape == (2,)
            assert not torch.isnan(scores).any()

    def test_with_different_scorers(self):
        for scorer_type in ["mlp", "attention", "transformer", "cross_attention"]:
            model = ClauseGCN(
                hidden_dim=64,
                num_layers=2,
                scorer_type=scorer_type,
            )
            node_features = make_node_features(10)
            adj = make_adj(10)
            pool_matrix = torch.ones(3, 10) / 10
            clause_features = make_clause_features(3)

            scores = model(node_features, adj, pool_matrix, clause_features)
            assert scores.shape == (3,)
            assert not torch.isnan(scores).any()

    def test_gradient_flow(self):
        model = ClauseGCN(hidden_dim=32, num_layers=2)
        node_features = make_node_features(10)
        node_features.requires_grad_(True)
        adj = make_adj(10)
        pool_matrix = torch.ones(2, 10) / 10
        clause_features = make_clause_features(2)

        scores = model(node_features, adj, pool_matrix, clause_features)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None


class TestClauseGAT:
    """Tests for ClauseGAT model."""

    def test_forward_shape(self):
        model = ClauseGAT(hidden_dim=64, num_layers=2, num_heads=4)
        node_features = make_node_features(20)
        adj = make_adj(20)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25
        clause_features = make_clause_features(5)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.numel() == 5
        assert not torch.isnan(scores).any()

    def test_multi_layer(self):
        model = ClauseGAT(hidden_dim=32, num_layers=2, num_heads=2)
        node_features = make_node_features(8)
        adj = make_adj(8)
        pool_matrix = torch.ones(2, 8) / 8
        clause_features = make_clause_features(2)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.numel() == 2
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self):
        model = ClauseGAT(hidden_dim=32, num_layers=2, num_heads=2)
        node_features = make_node_features(10)
        node_features.requires_grad_(True)
        adj = make_adj(10)
        pool_matrix = torch.ones(2, 10) / 10
        clause_features = make_clause_features(2)

        scores = model(node_features, adj, pool_matrix, clause_features)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None


class TestClauseGraphSAGE:
    """Tests for ClauseGraphSAGE model."""

    def test_forward_shape(self):
        model = ClauseGraphSAGE(hidden_dim=64, num_layers=3)
        node_features = make_node_features(20)
        adj = make_adj(20)
        pool_matrix = torch.zeros(5, 20)
        for i in range(5):
            pool_matrix[i, i*4:(i+1)*4] = 0.25
        clause_features = make_clause_features(5)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

    def test_single_clause(self):
        model = ClauseGraphSAGE(hidden_dim=32, num_layers=2)
        node_features = make_node_features(4)
        adj = torch.zeros(4, 4)  # No neighbors for GraphSAGE
        pool_matrix = torch.ones(1, 4) / 4
        clause_features = make_clause_features(1)

        scores = model(node_features, adj, pool_matrix, clause_features)
        assert scores.numel() == 1
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self):
        model = ClauseGraphSAGE(hidden_dim=32, num_layers=2)
        node_features = make_node_features(10)
        node_features.requires_grad_(True)
        adj = make_adj(10)
        pool_matrix = torch.ones(2, 10) / 10
        clause_features = make_clause_features(2)

        scores = model(node_features, adj, pool_matrix, clause_features)
        loss = scores.sum()
        loss.backward()
        assert node_features.grad is not None


class TestTorchScriptExport:
    """Tests for TorchScript export functionality."""

    @pytest.fixture
    def valid_inputs(self):
        """Create valid inputs with correct value ranges."""
        num_nodes, num_clauses = 20, 5
        node_features = make_node_features(num_nodes)
        adj = make_adj(num_nodes)
        pool_matrix = torch.ones(num_clauses, num_nodes) / num_nodes
        clause_features = make_clause_features(num_clauses)
        return node_features, adj, pool_matrix, clause_features

    def test_gcn_export(self, valid_inputs, tmp_path):
        """Test ClauseGCN TorchScript export."""
        model = ClauseGCN(hidden_dim=64, num_layers=3)
        path = tmp_path / "gcn.pt"
        model.export_torchscript(str(path))

        loaded = torch.jit.load(str(path))
        node_features, adj, pool_matrix, clause_features = valid_inputs

        with torch.no_grad():
            original = model(node_features, adj, pool_matrix, clause_features)
            exported = loaded(node_features, adj, pool_matrix, clause_features)

        assert torch.allclose(original, exported)

    @pytest.mark.parametrize("scorer_type", ["mlp", "attention", "transformer", "cross_attention"])
    def test_gcn_export_all_scorers(self, valid_inputs, tmp_path, scorer_type):
        """Test TorchScript export with all scorer types."""
        model = ClauseGCN(hidden_dim=64, num_layers=2, scorer_type=scorer_type)
        path = tmp_path / f"gcn_{scorer_type}.pt"
        model.export_torchscript(str(path))

        loaded = torch.jit.load(str(path))
        node_features, adj, pool_matrix, clause_features = valid_inputs

        with torch.no_grad():
            original = model(node_features, adj, pool_matrix, clause_features)
            exported = loaded(node_features, adj, pool_matrix, clause_features)

        assert not torch.isnan(original).any()
        assert torch.allclose(original, exported)
