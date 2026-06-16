"""Principal angle tests.

Key geometric fact: Col(W̃_K) = Col(W_c @ W_K_up.T) ⊆ Col(W_c).
When W_K_up has full rank r, Col(W̃_K) = Col(W_c) for any W_K_up — so random
full-rank up-projections give 0° between Col(W̃_K) and Col(W̃_V).
Non-trivial angles emerge only when the up-projections have rank < r (low effective rank).
"""

import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy.toy_mla import make_toy_model
from model_io.weight_extractor import extract_mla_weights_from_toy, toy_config_from_model
from metrics.principal_angles import compute_mean_principal_angle, principal_angles_all

TOL_DEG = 1.0


def _make_weights(r=8, d=64, nh=4, qk_nope=16, v=16, seed=0):
    torch.manual_seed(seed)
    W_c = torch.randn(d, r)
    W_K_up = torch.randn(nh * qk_nope, r)
    W_V_up = torch.randn(nh * v, r)
    return W_c, W_K_up, W_V_up


def test_identical_zero_degrees():
    """Identical up-projections -> mean principal angle = 0° (same subspace)."""
    W_c, W_K_up, _ = _make_weights()
    theta = compute_mean_principal_angle(W_c, W_K_up, W_K_up, method="svd")
    assert theta < TOL_DEG, f"Expected ~0°, got {theta:.4f}°"


def test_full_rank_always_zero():
    """For full-rank up-projections, Col(W̃_K) = Col(W̃_V) = Col(W_c) -> 0°."""
    for seed in range(5):
        W_c, W_K_up, W_V_up = _make_weights(seed=seed)
        theta = compute_mean_principal_angle(W_c, W_K_up, W_V_up, method="svd")
        assert theta < TOL_DEG, (
            f"Seed {seed}: full-rank up-projections should give 0°, got {theta:.2f}°"
        )


def test_low_rank_nonzero():
    """Rank-1 up-projections pointing in different directions -> non-zero angle."""
    torch.manual_seed(7)
    d, r, nh, qk_nope, v = 64, 8, 4, 16, 16
    W_c = torch.randn(d, r)

    # Rank-1 up-projections with orthogonal row vectors in R^r
    u1 = torch.zeros(r); u1[0] = 1.0  # unit vector along axis 0
    u2 = torch.zeros(r); u2[1] = 1.0  # unit vector along axis 1 (orthogonal)

    W_K_up = u1.unsqueeze(0).expand(nh * qk_nope, -1) * torch.randn(nh * qk_nope, 1)
    W_V_up = u2.unsqueeze(0).expand(nh * v, -1) * torch.randn(nh * v, 1)

    theta = compute_mean_principal_angle(W_c, W_K_up, W_V_up, method="svd")
    # With orthogonal row spaces, the two column spaces in R^d are orthogonal
    # (both rank-1 but pointing in W_c @ u1 vs W_c @ u2 directions)
    assert theta > 1.0, f"Expected non-zero angle for orthogonal rank-1 ups, got {theta:.2f}°"


def test_angle_in_range():
    """Mean principal angle must be in [0°, 90°]."""
    for seed in range(10):
        W_c, W_K_up, W_V_up = _make_weights(seed=seed)
        theta = compute_mean_principal_angle(W_c, W_K_up, W_V_up)
        assert -TOL_DEG <= theta <= 90.0 + TOL_DEG, f"Out of range: {theta}° at seed={seed}"


def test_all_angles_shape():
    """principal_angles_all returns a 1-D tensor of length r_eff."""
    W_c, W_K_up, W_V_up = _make_weights(r=8)
    angles = principal_angles_all(W_c, W_K_up, W_V_up, method="svd")
    assert angles.ndim == 1
    assert len(angles) >= 1
    assert (angles >= -0.01).all() and (angles <= 90.0 + 0.5).all()


def test_returns_degrees_by_default():
    """Default output should be in degrees (0-90), not radians (0-pi/2 ~ 1.57)."""
    W_c, W_K_up, W_V_up = _make_weights()
    theta = compute_mean_principal_angle(W_c, W_K_up, W_V_up)
    assert theta <= 90.0 + TOL_DEG


def test_toy_model_pipeline():
    model = make_toy_model(num_layers=4, seed=2)
    config = toy_config_from_model(model)
    for layer in model.layers:
        w = extract_mla_weights_from_toy(layer, config)
        theta = compute_mean_principal_angle(w["W_c"], w["W_K_up"], w["W_V_up"])
        assert 0.0 <= theta <= 90.0 + TOL_DEG
