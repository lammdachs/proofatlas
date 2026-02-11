"""Integration tests for cross-attention end-to-end."""

import pytest
import torch

from proofatlas.selectors.gnn import ClauseGCN
from proofatlas.selectors.scorers import AttentionScorer, TransformerScorer
from proofatlas.selectors.features import ClauseFeatures


def make_node_features(num_nodes: int) -> torch.Tensor:
    features = torch.zeros(num_nodes, 3)
    features[:, 0] = torch.randint(0, 6, (num_nodes,)).float()
    features[:, 1] = torch.randint(0, 5, (num_nodes,)).float()
    features[:, 2] = torch.randint(0, 10, (num_nodes,)).float()
    return features


def make_clause_features(num_clauses: int) -> torch.Tensor:
    features = torch.zeros(num_clauses, 3)
    features[:, 0] = torch.rand(num_clauses)
    features[:, 1] = torch.randint(0, 5, (num_clauses,)).float()
    features[:, 2] = torch.randint(1, 10, (num_clauses,)).float()
    return features


def make_adj(num_nodes: int) -> torch.Tensor:
    adj = torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes)
    return adj / adj.sum(dim=1, keepdim=True)


def make_clause_features_9d(num_clauses: int) -> torch.Tensor:
    features = torch.zeros(num_clauses, 9)
    features[:, 0] = torch.rand(num_clauses)
    features[:, 1] = torch.randint(1, 10, (num_clauses,)).float()
    features[:, 2] = torch.randint(0, 8, (num_clauses,)).float()
    features[:, 3] = torch.randint(1, 20, (num_clauses,)).float()
    features[:, 4] = torch.randint(1, 10, (num_clauses,)).float()
    features[:, 5] = torch.randint(0, 10, (num_clauses,)).float()
    features[:, 6] = torch.randint(0, 5, (num_clauses,)).float()
    features[:, 7] = torch.randint(0, 5, (num_clauses,)).float()
    features[:, 8] = torch.randint(0, 7, (num_clauses,)).float()
    return features


class TestGCNEncodeWithScorers:
    """Test ClauseGCN.encode() + scorer cross-attention pipeline."""

    def test_gcn_encode_plus_attention_scorer(self):
        model = ClauseGCN(hidden_dim=64, num_layers=2, scorer_type="attention")
        u_nf = make_node_features(20)
        u_adj = make_adj(20)
        u_pool = torch.ones(8, 20) / 20
        u_cf = make_clause_features(8)

        p_nf = make_node_features(15)
        p_adj = make_adj(15)
        p_pool = torch.ones(5, 15) / 15
        p_cf = make_clause_features(5)

        u_emb = model.encode(u_nf, u_adj, u_pool, u_cf)
        p_emb = model.encode(p_nf, p_adj, p_pool, p_cf)

        scores = model.scorer(u_emb, p_emb)
        assert scores.shape == (8,)
        assert not torch.isnan(scores).any()

    def test_gcn_encode_plus_transformer_scorer(self):
        model = ClauseGCN(hidden_dim=64, num_layers=2, scorer_type="transformer")
        u_nf = make_node_features(20)
        u_adj = make_adj(20)
        u_pool = torch.ones(8, 20) / 20
        u_cf = make_clause_features(8)

        p_nf = make_node_features(15)
        p_adj = make_adj(15)
        p_pool = torch.ones(5, 15) / 15
        p_cf = make_clause_features(5)

        u_emb = model.encode(u_nf, u_adj, u_pool, u_cf)
        p_emb = model.encode(p_nf, p_adj, p_pool, p_cf)

        scores = model.scorer(u_emb, p_emb)
        assert scores.shape == (8,)
        assert not torch.isnan(scores).any()


class TestStateBatchThroughModel:
    """Test collate_proof_batch results through model pipeline."""

    def test_state_batch_through_model(self):
        from proofatlas.ml.structured import clause_to_graph, batch_graphs
        from proofatlas.ml.datasets import collate_proof_batch

        # Build synthetic pre-sampled batch item
        clauses = []
        for i in range(6):
            clauses.append({
                "literals": [{"polarity": True, "atom": {"predicate": f"p{i}", "args": [{"type": "Variable", "name": f"X{i}"}]}}],
                "label": 1 if i < 2 else 0,
                "age": i,
                "role": "axiom" if i < 2 else "derived",
            })

        graphs = [clause_to_graph(c) for c in clauses]
        labels = [c["label"] for c in clauses]

        u_indices = [0, 1, 2]
        p_indices = [3, 4, 5]
        item = {
            "u_graphs": [graphs[i] for i in u_indices],
            "p_graphs": [graphs[i] for i in p_indices],
            "u_labels": [labels[i] for i in u_indices],
            "problem": "test",
        }

        batch_result = collate_proof_batch([item])
        assert batch_result is not None

        # Create model and run encode + score pipeline
        model = ClauseGCN(hidden_dim=64, num_layers=2, scorer_type="attention")

        u_emb = model.encode(
            batch_result["u_node_features"],
            batch_result["u_adj"],
            batch_result["u_pool_matrix"],
            batch_result.get("u_clause_features"),
        )

        p_emb = None
        if "p_node_features" in batch_result:
            p_emb = model.encode(
                batch_result["p_node_features"],
                batch_result["p_adj"],
                batch_result["p_pool_matrix"],
                batch_result.get("p_clause_features"),
            )

        scores = model.scorer(u_emb, p_emb)
        assert scores.shape[0] == u_emb.shape[0]
        assert not torch.isnan(scores).any()


class TestFeaturesWithCrossAttention:
    """Test ClauseFeatures with attention scorer."""

    def test_features_with_cross_attention(self):
        """ClauseFeatures with attention scorer should work end-to-end."""
        model = ClauseFeatures(hidden_dim=64, scorer_type="attention")
        features = make_clause_features_9d(10)
        scores = model(features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()
