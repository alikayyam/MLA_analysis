import torch
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy.toy_mla import make_toy_model
from model_io.weight_extractor import extract_mla_weights_from_toy, toy_config_from_model
from metrics.gram_distance import compute_gram_distance

TOL = 1e-4


def _make_weights(r=8, d=64, nh=4, qk_nope=16, v=16, seed=0):
    torch.manual_seed(seed)
    W_c = torch.randn(d, r)
    W_K_up = torch.randn(nh * qk_nope, r)
    W_V_up = torch.randn(nh * v, r)
    return W_c, W_K_up, W_V_up


def test_identical_zero():
    """When K and V up-projections are identical, D_Gram must be 0."""
    W_c, W_K_up, _ = _make_weights()
    d = compute_gram_distance(W_c, W_K_up, W_K_up, method="latent")
    assert d < TOL, f"Expected ~0, got {d}"

    d_full = compute_gram_distance(W_c, W_K_up, W_K_up, method="full_gram")
    assert d_full < TOL, f"full_gram: Expected ~0, got {d_full}"


def test_latent_vs_full_gram_consistency():
    """latent and full_gram methods must agree to within tolerance on small tensors."""
    W_c, W_K_up, W_V_up = _make_weights()
    d_latent = compute_gram_distance(W_c, W_K_up, W_V_up, method="latent")
    d_full = compute_gram_distance(W_c, W_K_up, W_V_up, method="full_gram")
    assert abs(d_latent - d_full) < TOL, (
        f"Methods disagree: latent={d_latent:.6f}, full={d_full:.6f}"
    )


def test_non_negative():
    """D_Gram must be >= 0 for arbitrary random matrices."""
    for seed in range(10):
        W_c, W_K_up, W_V_up = _make_weights(seed=seed)
        d = compute_gram_distance(W_c, W_K_up, W_V_up)
        assert d >= 0.0, f"Got negative D_Gram={d} at seed={seed}"


def test_symmetry():
    """D_Gram(K, V) == D_Gram(V, K) (swap the two pathways)."""
    W_c, W_K_up, W_V_up = _make_weights()
    d_kv = compute_gram_distance(W_c, W_K_up, W_V_up, method="latent")
    d_vk = compute_gram_distance(W_c, W_V_up, W_K_up, method="latent")
    assert abs(d_kv - d_vk) < TOL, f"Not symmetric: d_kv={d_kv}, d_vk={d_vk}"


def test_bfloat16_stability():
    """Result in bfloat16 should be close to float32 result (within 2%)."""
    W_c, W_K_up, W_V_up = _make_weights()
    d_f32 = compute_gram_distance(W_c, W_K_up, W_V_up)
    d_bf16 = compute_gram_distance(
        W_c.bfloat16(), W_K_up.bfloat16(), W_V_up.bfloat16()
    )
    rel_err = abs(d_f32 - d_bf16) / (d_f32 + 1e-8)
    assert rel_err < 0.02, f"bfloat16 instability: f32={d_f32:.4f}, bf16={d_bf16:.4f}"


def test_toy_model_pipeline():
    """End-to-end: extract weights from toy model, compute D_Gram on all layers."""
    model = make_toy_model(num_layers=4, seed=1)
    config = toy_config_from_model(model)
    for layer in model.layers:
        w = extract_mla_weights_from_toy(layer, config)
        d = compute_gram_distance(w["W_c"], w["W_K_up"], w["W_V_up"])
        assert 0.0 <= d < 2.0, f"Unexpected D_Gram value: {d}"
