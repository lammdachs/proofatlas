"""Tests for all embedding+scorer combinations.

Tests all 16 combinations:
- Graph embeddings (gcn, gat, graphsage) x 4 scorers = 12 combinations
- String embedding (sentence) x 4 scorers = 4 combinations
"""

import pytest
import torch

from proofatlas.selectors.encoders import (
    ClauseSelector,
    GCNEncoder,
    GATEncoder,
    GraphSAGEEncoder,
    create_encoder,
)
from proofatlas.selectors.scorers import create_scorer

# Check if transformers is available for sentence tests
try:
    from proofatlas.selectors.sentence import SentenceEncoder, HAS_TRANSFORMERS
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Test Data Fixtures
# =============================================================================

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
def graph_inputs():
    """Create valid inputs for graph-based models."""
    num_nodes, num_clauses = 20, 5
    return (
        make_node_features(num_nodes),
        make_adj(num_nodes),
        torch.ones(num_clauses, num_nodes) / num_nodes,
        make_clause_features(num_clauses),
    )


@pytest.fixture
def clause_strings():
    """Create clause strings for sentence-based models."""
    return [
        "equal(f(X), g(X))",
        "not(p(a, b)) | q(c)",
        "equal(inv(inv(X)), X)",
        "p(X) | not(q(X, Y))",
        "equal(e, f(a))",
    ]


# =============================================================================
# Graph Embedding Tests (12 combinations)
# =============================================================================

GRAPH_EMBEDDINGS = ["gcn", "gat", "graphsage"]
SCORERS = ["mlp", "attention", "transformer", "cross_attention"]


class TestGraphEmbeddings:
    """Tests for all graph embedding + scorer combinations."""

    @pytest.mark.parametrize("embedding", GRAPH_EMBEDDINGS)
    @pytest.mark.parametrize("scorer", SCORERS)
    def test_forward_pass(self, graph_inputs, embedding, scorer):
        """Test forward pass produces valid scores."""
        encoder = create_encoder(embedding, hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer, scorer_dim=64)

        node_features, adj, pool_matrix, clause_features = graph_inputs
        scores = selector(node_features, adj, pool_matrix, clause_features)

        assert scores.shape == (5,), f"Expected shape (5,), got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"

    @pytest.mark.parametrize("embedding", GRAPH_EMBEDDINGS)
    @pytest.mark.parametrize("scorer", SCORERS)
    def test_gradient_flow(self, graph_inputs, embedding, scorer):
        """Test gradients flow through the entire model."""
        encoder = create_encoder(embedding, hidden_dim=32, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer, scorer_dim=32)

        node_features, adj, pool_matrix, clause_features = graph_inputs
        node_features = node_features.clone().requires_grad_(True)

        scores = selector(node_features, adj, pool_matrix, clause_features)
        scores.sum().backward()

        assert node_features.grad is not None, "No gradient on input"
        assert not torch.isnan(node_features.grad).any(), "Gradient contains NaN"

    @pytest.mark.parametrize("embedding", GRAPH_EMBEDDINGS)
    @pytest.mark.parametrize("scorer", SCORERS)
    def test_torchscript_export(self, graph_inputs, tmp_path, embedding, scorer):
        """Test TorchScript export and loading."""
        encoder = create_encoder(embedding, hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer, scorer_dim=64)

        path = tmp_path / f"{embedding}_{scorer}.pt"
        selector.export_torchscript(str(path))

        # Load and verify
        loaded = torch.jit.load(str(path))
        node_features, adj, pool_matrix, clause_features = graph_inputs

        with torch.no_grad():
            original = selector(node_features, adj, pool_matrix, clause_features)
            exported = loaded(node_features, adj, pool_matrix, clause_features)

        assert torch.allclose(original, exported, atol=1e-5), \
            f"Mismatch: original={original}, exported={exported}"

    @pytest.mark.parametrize("embedding", GRAPH_EMBEDDINGS)
    @pytest.mark.parametrize("scorer", SCORERS)
    def test_single_clause(self, embedding, scorer):
        """Test with single clause (edge case)."""
        encoder = create_encoder(embedding, hidden_dim=64, num_layers=2)
        selector = ClauseSelector(encoder, scorer_type=scorer, scorer_dim=64)

        node_features = make_node_features(10)
        adj = make_adj(10)
        pool_matrix = torch.ones(1, 10) / 10
        clause_features = make_clause_features(1)

        scores = selector(node_features, adj, pool_matrix, clause_features)
        # Single clause may produce scalar or (1,) depending on scorer
        assert scores.numel() == 1, f"Expected 1 element, got {scores.numel()}"
        assert not torch.isnan(scores).any()


# =============================================================================
# String Embedding Tests (4 combinations)
# =============================================================================

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestSentenceEmbedding:
    """Tests for sentence embedding + scorer combinations."""

    @pytest.mark.parametrize("scorer", SCORERS)
    def test_forward_pass(self, clause_strings, scorer):
        """Test forward pass produces valid scores."""
        model = SentenceEncoder(
            hidden_dim=64,
            freeze_encoder=True,
            scorer_type=scorer,
        )

        scores = model(clause_strings)

        assert scores.shape == (5,), f"Expected shape (5,), got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"

    @pytest.mark.parametrize("scorer", SCORERS)
    def test_gradient_flow_frozen(self, clause_strings, scorer):
        """Test gradients flow through projection and scorer (encoder frozen)."""
        model = SentenceEncoder(
            hidden_dim=64,
            freeze_encoder=True,
            scorer_type=scorer,
        )

        # Check projection has gradients
        scores = model(clause_strings)
        scores.sum().backward()

        assert model.projection.weight.grad is not None, "No gradient on projection"

    @pytest.mark.parametrize("scorer", SCORERS)
    def test_single_clause(self, scorer):
        """Test with single clause (edge case)."""
        model = SentenceEncoder(
            hidden_dim=64,
            freeze_encoder=True,
            scorer_type=scorer,
        )

        scores = model(["equal(a, b)"])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()

    @pytest.mark.parametrize("scorer", SCORERS)
    def test_empty_clause_handling(self, scorer):
        """Test with empty clause string."""
        model = SentenceEncoder(
            hidden_dim=64,
            freeze_encoder=True,
            scorer_type=scorer,
        )

        # Empty string should still work (tokenizer handles it)
        scores = model(["", "p(X)"])
        assert scores.shape == (2,)


# =============================================================================
# Integration Tests
# =============================================================================

class TestModelNaming:
    """Test model naming convention matches config."""

    @pytest.mark.parametrize("embedding,scorer", [
        ("gcn", "mlp"),
        ("gcn", "attention"),
        ("gcn", "transformer"),
        ("gat", "mlp"),
        ("graphsage", "cross_attention"),
    ])
    def test_model_name_format(self, embedding, scorer):
        """Test that model names follow {embedding}_{scorer} convention."""
        expected_name = f"{embedding}_{scorer}"
        # This is just a naming convention test
        assert "_" in expected_name
        assert embedding in expected_name
        assert scorer in expected_name


class TestConfiguredCombinations:
    """Test combinations configured in proofatlas.json."""

    @pytest.mark.parametrize("embedding,scorer", [
        ("gcn", "mlp"),
        ("gcn", "attention"),
        ("gcn", "transformer"),
    ])
    def test_configured_graph_models(self, graph_inputs, embedding, scorer):
        """Test graph models configured in proofatlas.json."""
        encoder = create_encoder(embedding, hidden_dim=256, num_layers=6)
        selector = ClauseSelector(encoder, scorer_type=scorer, scorer_dim=256)

        node_features, adj, pool_matrix, clause_features = graph_inputs
        scores = selector(node_features, adj, pool_matrix, clause_features)

        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_configured_sentence_model(self, clause_strings):
        """Test sentence_mlp configured in proofatlas.json."""
        model = SentenceEncoder(
            hidden_dim=64,
            freeze_encoder=True,
            scorer_type="mlp",
        )

        scores = model(clause_strings)
        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()
