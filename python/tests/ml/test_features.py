"""Tests for ClauseFeatures encoder."""

import pytest
import torch

from proofatlas.selectors.features import ClauseFeatures


def make_clause_features_9d(num_clauses: int) -> torch.Tensor:
    """Create valid 9D clause features tensor.

    [0] age (0-1), [1] size, [2] depth, [3] symbol_count,
    [4] distinct_symbols, [5] variable_count, [6] distinct_variables,
    [7] role (0-4), [8] rule (0-6)
    """
    features = torch.zeros(num_clauses, 9)
    features[:, 0] = torch.rand(num_clauses)  # age
    features[:, 1] = torch.randint(1, 10, (num_clauses,)).float()  # size
    features[:, 2] = torch.randint(0, 8, (num_clauses,)).float()  # depth
    features[:, 3] = torch.randint(1, 20, (num_clauses,)).float()  # symbol_count
    features[:, 4] = torch.randint(1, 10, (num_clauses,)).float()  # distinct_symbols
    features[:, 5] = torch.randint(0, 10, (num_clauses,)).float()  # variable_count
    features[:, 6] = torch.randint(0, 5, (num_clauses,)).float()  # distinct_variables
    features[:, 7] = torch.randint(0, 5, (num_clauses,)).float()  # role
    features[:, 8] = torch.randint(0, 7, (num_clauses,)).float()  # rule
    return features


class TestClauseFeatures:
    """Tests for ClauseFeatures model."""

    def test_forward_shape(self):
        model = ClauseFeatures(hidden_dim=64)
        features = make_clause_features_9d(10)
        scores = model(features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()

    def test_sinusoidal_encode_shape(self):
        model = ClauseFeatures(hidden_dim=64, sin_dim=8)
        values = torch.rand(5)
        enc = model.sinusoidal_encode(values)
        assert enc.shape == (5, 8)

    def test_role_one_hot(self):
        """Roles 0-4 should produce valid one-hot vectors."""
        model = ClauseFeatures(hidden_dim=64)
        for role_val in range(5):
            features = make_clause_features_9d(1)
            features[0, 7] = role_val
            scores = model(features)
            assert not torch.isnan(scores).any()

    def test_rule_one_hot(self):
        """Rules 0-6 should produce valid one-hot vectors."""
        model = ClauseFeatures(hidden_dim=64)
        for rule_val in range(7):
            features = make_clause_features_9d(1)
            features[0, 8] = rule_val
            scores = model(features)
            assert not torch.isnan(scores).any()

    def test_with_attention_scorer(self):
        model = ClauseFeatures(hidden_dim=64, scorer_type="attention")
        features = make_clause_features_9d(10)
        scores = model(features)
        assert scores.shape == (10,)
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self):
        model = ClauseFeatures(hidden_dim=64)
        features = make_clause_features_9d(10)
        features.requires_grad_(True)
        scores = model(features)
        scores.sum().backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_single_clause(self):
        model = ClauseFeatures(hidden_dim=64)
        features = make_clause_features_9d(1)
        scores = model(features)
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()
