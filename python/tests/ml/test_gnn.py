"""Tests for GNN-based clause selector models."""

import pytest
import torch

from proofatlas.selectors.gnn import (
    GCNLayer,
    ScorerHead,
    ClauseGCN,
    GraphNorm,
    NodeInputProjection,
)

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


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


class TestScorerHead:
    """Tests for ScorerHead."""

    def test_forward_shape(self):
        scorer = ScorerHead(hidden_dim=64)
        x = torch.randn(10, 64)
        out = scorer(x)
        assert out.shape == (10, 1)

class TestGraphNorm:
    """Tests for GraphNorm."""

    def test_output_shape(self):
        norm = GraphNorm(hidden_dim=64)
        x = torch.randn(10, 64)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = norm(x, batch)
        assert out.shape == (10, 64)

    def test_single_graph(self):
        norm = GraphNorm(hidden_dim=32)
        x = torch.randn(8, 32)
        batch = torch.zeros(8, dtype=torch.long)
        out = norm(x, batch)
        assert out.shape == (8, 32)

    def test_learnable_alpha(self):
        norm = GraphNorm(hidden_dim=64)
        assert hasattr(norm, 'alpha')
        assert norm.alpha.requires_grad

    def test_per_graph_normalization(self):
        """Two graphs with different means should produce different stats than global LayerNorm."""
        norm = GraphNorm(hidden_dim=16)
        # Graph 0: mean ~10, Graph 1: mean ~0
        x = torch.randn(6, 16)
        x[:3] += 10.0
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        out = norm(x, batch)
        # Per-graph normalization should reduce the gap between graphs
        # compared to the raw input
        raw_gap = x[:3].mean() - x[3:].mean()
        out_gap = out[:3].mean() - out[3:].mean()
        assert abs(out_gap.item()) < abs(raw_gap.item())


class TestNodeInputProjection:
    """Tests for NodeInputProjection."""

    def test_features_mode(self):
        proj = NodeInputProjection(hidden_dim=64, mode="features")
        x = make_node_features(10)
        out = proj(x)
        assert out.shape == (10, 64)

    def test_names_mode(self):
        proj = NodeInputProjection(hidden_dim=64, mode="names")
        x = make_node_features(10)
        sym_emb = torch.randn(10, 384)
        out = proj(x, symbol_embeddings=sym_emb)
        assert out.shape == (10, 64)

    def test_both_mode(self):
        proj = NodeInputProjection(hidden_dim=64, mode="both")
        x = make_node_features(10)
        sym_emb = torch.randn(10, 384)
        out = proj(x, symbol_embeddings=sym_emb)
        assert out.shape == (10, 64)

    def test_names_mode_asserts_without_embeddings(self):
        proj = NodeInputProjection(hidden_dim=64, mode="names")
        x = make_node_features(10)
        with pytest.raises(AssertionError):
            proj(x)


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
        for scorer_type in ["mlp", "attention", "transformer"]:
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

    def test_encode_shape(self):
        model = ClauseGCN(hidden_dim=64, num_layers=2)
        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(3, 10) / 10
        clause_features = make_clause_features(3)

        emb = model.encode(node_features, adj, pool_matrix, clause_features)
        assert emb.shape == (3, 64)

    def test_encode_vs_forward_consistency(self):
        """encode() embeddings fed to scorer should match forward() scores."""
        model = ClauseGCN(hidden_dim=64, num_layers=2)
        model.eval()
        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(3, 10) / 10
        clause_features = make_clause_features(3)

        with torch.no_grad():
            scores_forward = model(node_features, adj, pool_matrix, clause_features)
            emb = model.encode(node_features, adj, pool_matrix, clause_features)
            scores_encode = model.scorer(emb).view(-1)

        assert torch.allclose(scores_forward, scores_encode, atol=1e-5)

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_node_info_names(self):
        """node_info='names' with node_names should produce valid scores."""
        model = ClauseGCN(hidden_dim=64, num_layers=2, node_info="names")
        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(3, 10) / 10
        clause_features = make_clause_features(3)
        # node_names required for 'names' mode - provide synthetic names
        node_names = ["p", "q", "f", "X", "a", "CLAUSE", "LIT", "VAR", "g", "b"]

        scores = model(node_features, adj, pool_matrix, clause_features, node_names=node_names)
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_node_info_both(self):
        """node_info='both' should produce valid scores."""
        model = ClauseGCN(hidden_dim=64, num_layers=2, node_info="both")
        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(3, 10) / 10
        clause_features = make_clause_features(3)
        node_names = ["p", "q", "f", "X", "a", "CLAUSE", "LIT", "VAR", "g", "b"]

        scores = model(node_features, adj, pool_matrix, clause_features, node_names=node_names)
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()

    def test_no_clause_features(self):
        """use_clause_features=False should still produce scores."""
        model = ClauseGCN(hidden_dim=64, num_layers=2, use_clause_features=False)
        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(3, 10) / 10

        scores = model(node_features, adj, pool_matrix)
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()


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

    @pytest.mark.parametrize("scorer_type", ["mlp", "attention", "transformer"])
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
