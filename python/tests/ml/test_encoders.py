"""Tests for modular encoder/selector architecture."""

import pytest
import torch

from proofatlas.selectors.encoders import (
    ClauseEncoder,
    ClauseSelector,
    GCNEncoder,
    create_encoder,
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


@pytest.fixture
def valid_inputs():
    """Create valid inputs for testing."""
    num_nodes, num_clauses = 20, 5
    return (
        make_node_features(num_nodes),
        make_adj(num_nodes),
        torch.ones(num_clauses, num_nodes) / num_nodes,
        make_clause_features(num_clauses),
    )


class TestGCNEncoder:
    """Tests for GCNEncoder."""

    def test_output_shape(self, valid_inputs):
        encoder = GCNEncoder(hidden_dim=64, num_layers=3)
        node_features, adj, pool_matrix, clause_features = valid_inputs

        out = encoder(node_features, adj, pool_matrix, clause_features)
        assert out.shape == (5, encoder.output_dim)
        assert not torch.isnan(out).any()

    def test_output_dim_with_clause_features(self):
        encoder = GCNEncoder(hidden_dim=64, use_clause_features=True)
        # output_dim = hidden_dim + clause_feature_dim (21 with sin_dim=8)
        assert encoder.output_dim == 64 + 21

    def test_output_dim_without_clause_features(self):
        encoder = GCNEncoder(hidden_dim=64, use_clause_features=False)
        assert encoder.output_dim == 64

    def test_gradient_flow(self, valid_inputs):
        encoder = GCNEncoder(hidden_dim=32, num_layers=2)
        node_features, adj, pool_matrix, clause_features = valid_inputs
        node_features.requires_grad_(True)

        out = encoder(node_features, adj, pool_matrix, clause_features)
        out.sum().backward()
        assert node_features.grad is not None


class TestCreateEncoder:
    """Tests for create_encoder factory."""

    def test_gcn(self):
        encoder = create_encoder("gcn", hidden_dim=64, num_layers=3)
        assert isinstance(encoder, GCNEncoder)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_encoder("invalid", hidden_dim=64)


class TestClauseSelector:
    """Tests for ClauseSelector modular composition."""

    def test_basic_composition(self, valid_inputs):
        encoder = GCNEncoder(hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type="mlp", scorer_dim=64)

        node_features, adj, pool_matrix, clause_features = valid_inputs
        scores = selector(node_features, adj, pool_matrix, clause_features)

        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

    @pytest.mark.parametrize("scorer_type", ["mlp", "attention", "transformer"])
    def test_all_scorer_combinations(self, valid_inputs, scorer_type):
        """Test that GCN encoder works with any scorer."""
        encoder = create_encoder("gcn", hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer_type, scorer_dim=64)

        node_features, adj, pool_matrix, clause_features = valid_inputs
        scores = selector(node_features, adj, pool_matrix, clause_features)

        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

    def test_projection_layer_created(self):
        """Test that projection layer is created when dims differ."""
        encoder = GCNEncoder(hidden_dim=64, use_clause_features=True)
        # encoder.output_dim = 64 + 21 = 85, but scorer_dim = 32
        selector = ClauseSelector(encoder, scorer_type="mlp", scorer_dim=32)

        assert isinstance(selector.projection, torch.nn.Linear)
        assert selector.projection.in_features == encoder.output_dim
        assert selector.projection.out_features == 32

    def test_no_projection_when_dims_match(self):
        """Test that no projection is created when dims match."""
        encoder = GCNEncoder(hidden_dim=64, use_clause_features=False)
        # encoder.output_dim = 64 = scorer_dim
        selector = ClauseSelector(encoder, scorer_type="mlp", scorer_dim=64)

        assert isinstance(selector.projection, torch.nn.Identity)

    def test_gradient_flow(self, valid_inputs):
        encoder = GCNEncoder(hidden_dim=32, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type="attention", scorer_dim=32)

        node_features, adj, pool_matrix, clause_features = valid_inputs
        node_features.requires_grad_(True)

        scores = selector(node_features, adj, pool_matrix, clause_features)
        scores.sum().backward()
        assert node_features.grad is not None

    def test_torchscript_export(self, valid_inputs, tmp_path):
        """Test TorchScript export of ClauseSelector."""
        encoder = GCNEncoder(hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type="mlp", scorer_dim=64)

        path = tmp_path / "selector.pt"
        selector.export_torchscript(str(path))

        # Load and verify
        loaded = torch.jit.load(str(path))
        node_features, adj, pool_matrix, clause_features = valid_inputs

        with torch.no_grad():
            original = selector(node_features, adj, pool_matrix, clause_features)
            exported = loaded(node_features, adj, pool_matrix, clause_features)

        assert torch.allclose(original, exported)

    @pytest.mark.parametrize("scorer_type", ["mlp", "attention", "transformer"])
    def test_torchscript_all_scorers(self, valid_inputs, tmp_path, scorer_type):
        """Test TorchScript export works with all scorer types."""
        encoder = GCNEncoder(hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer_type, scorer_dim=64)

        path = tmp_path / f"selector_{scorer_type}.pt"
        selector.export_torchscript(str(path))

        loaded = torch.jit.load(str(path))
        node_features, adj, pool_matrix, clause_features = valid_inputs

        with torch.no_grad():
            original = selector(node_features, adj, pool_matrix, clause_features)
            exported = loaded(node_features, adj, pool_matrix, clause_features)

        assert not torch.isnan(original).any()
        assert torch.allclose(original, exported)
