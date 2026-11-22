"""
Tests for evaluation metrics.
"""
import numpy as np
import pytest
import torch

from src.utils.metrics import (
    compute_auc,
    compute_mpjpe,
    compute_mpvpe,
    compute_pa_mpjpe,
)


class TestMetrics:
    """Test implemented metrics."""

    def test_mpjpe_perfect(self):
        """MPJPE = 0 for perfect prediction."""
        joints = torch.randn(21, 3)
        mpjpe = compute_mpjpe(joints, joints)
        assert mpjpe == 0.0

    def test_mpjpe_batch(self):
        """MPJPE on batch."""
        B = 4
        pred = torch.randn(B, 21, 3)
        gt = torch.randn(B, 21, 3)
        mpjpe = compute_mpjpe(pred, gt)
        assert mpjpe >= 0.0

    @pytest.mark.skip(reason="PA-MPJPE not implemented yet")
    def test_pa_mpjpe_perfect(self):
        """PA-MPJPE = 0 for perfect prediction."""
        joints = torch.randn(21, 3)
        pa_mpjpe = compute_pa_mpjpe(joints, joints)
        assert pa_mpjpe == 0.0

    def test_mpvpe_perfect(self):
        """MPVPE = 0 for perfect prediction."""
        verts = torch.randn(778, 3)
        mpvpe = compute_mpvpe(verts, verts)
        assert mpvpe == 0.0

    def test_auc(self):
        """AUC calculation."""
        errors = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        auc = compute_auc(errors, max_threshold=50.0)
        assert 0.0 <= auc <= 1.0
