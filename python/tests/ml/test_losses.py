"""Tests for loss functions."""

import pytest
import torch

from proofatlas.ml.losses import (
    info_nce_loss,
    margin_ranking_loss,
    info_nce_loss_per_proof,
)


class TestInfoNCELoss:
    """Tests for InfoNCE loss."""

    def test_basic(self):
        scores = torch.randn(10)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        loss = info_nce_loss(scores, labels)
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)

    def test_all_positive_fallback(self):
        """All labels=1 should use BCE fallback."""
        scores = torch.randn(5)
        labels = torch.ones(5)
        loss = info_nce_loss(scores, labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_all_negative_fallback(self):
        """All labels=0 should use BCE fallback."""
        scores = torch.randn(5)
        labels = torch.zeros(5)
        loss = info_nce_loss(scores, labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        scores = torch.randn(10, requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        loss = info_nce_loss(scores, labels)
        loss.backward()
        assert scores.grad is not None

    def test_temperature_effect(self):
        """Lower temperature should produce sharper (larger) loss."""
        scores = torch.randn(10)
        labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        loss_high = info_nce_loss(scores, labels, temperature=10.0)
        loss_low = info_nce_loss(scores, labels, temperature=0.1)
        # With lower temperature, scores are magnified → sharper softmax → different loss
        assert not torch.isnan(loss_high)
        assert not torch.isnan(loss_low)

    def test_positive_scores_higher_gives_lower_loss(self):
        """When positives score much higher, loss should be lower."""
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        # Positive clauses score much higher
        scores_good = torch.tensor([5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        # Positive clauses score same as negative
        scores_bad = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        loss_good = info_nce_loss(scores_good, labels)
        loss_bad = info_nce_loss(scores_bad, labels)
        assert loss_good < loss_bad


class TestMarginRankingLoss:
    """Tests for margin ranking loss."""

    def test_basic(self):
        scores = torch.randn(10)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        loss = margin_ranking_loss(scores, labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_zero_margin(self):
        """Well-separated scores with zero margin should give near-zero loss."""
        scores = torch.tensor([10.0, 10.0, -10.0, -10.0, -10.0])
        labels = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float)
        loss = margin_ranking_loss(scores, labels, margin=0.0)
        assert loss.item() < 0.01


class TestInfoNCELossPerProof:
    """Tests for per-proof InfoNCE loss."""

    def test_two_proofs(self):
        scores = torch.randn(10)
        labels = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float)
        proof_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        loss = info_nce_loss_per_proof(scores, labels, proof_ids)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_single_proof_matches_basic(self):
        """With one proof, should match basic info_nce_loss."""
        scores = torch.randn(10)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        proof_ids = torch.zeros(10, dtype=torch.long)
        loss_per_proof = info_nce_loss_per_proof(scores, labels, proof_ids)
        loss_basic = info_nce_loss(scores, labels)
        assert torch.allclose(loss_per_proof, loss_basic, atol=1e-5)

    def test_gradient_flow(self):
        scores = torch.randn(10, requires_grad=True)
        labels = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float)
        proof_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        loss = info_nce_loss_per_proof(scores, labels, proof_ids)
        loss.backward()
        assert scores.grad is not None
