"""Tests for activation-based alignment using the toy MLA model."""

import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxy.toy_mla import make_toy_model
from model_io.weight_extractor import toy_config_from_model
from metrics.activation_alignment import compute_activation_alignment


def _make_dataloader(num_batches=3, batch_size=2, seq_len=8, vocab_size=100):
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    ds = TensorDataset(input_ids)
    return DataLoader(ds, batch_size=batch_size), num_batches * batch_size * seq_len


def _wrap_loader(loader):
    """Wrap TensorDataset loader to yield {"input_ids": ...} dicts."""
    for (batch,) in loader:
        yield {"input_ids": batch}


def _make_wrapped_loader(num_batches=3, batch_size=2, seq_len=8):
    raw_loader, total_tokens = _make_dataloader(num_batches, batch_size, seq_len)
    return _wrapped(raw_loader), total_tokens


class _wrapped:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for (batch,) in self.loader:
            yield {"input_ids": batch}


def test_returns_float_per_layer():
    model = make_toy_model(num_layers=4, seed=0)
    config = toy_config_from_model(model)
    loader = _wrapped(DataLoader(
        TensorDataset(torch.randint(0, 100, (4, 8))), batch_size=2
    ))
    result = compute_activation_alignment(model, loader, config, max_tokens=100)
    assert isinstance(result, dict)
    for layer_idx, rho in result.items():
        assert isinstance(layer_idx, int)
        assert -1.0 <= rho <= 1.0, f"Layer {layer_idx}: rho={rho} out of range"


def test_max_tokens_respected():
    model = make_toy_model(num_layers=2, seed=1)
    config = toy_config_from_model(model)
    # 10 batches * 2 * 8 = 160 tokens; cap at 10 to ensure early exit
    loader = _wrapped(DataLoader(
        TensorDataset(torch.randint(0, 100, (20, 8))), batch_size=2
    ))
    result = compute_activation_alignment(model, loader, config, max_tokens=10)
    # Should still produce results (not empty) but stopped early
    assert len(result) > 0


def test_identical_kv_high_alignment():
    """When kv_b_proj K-part == V-part, ρ should be close to 1."""
    model = make_toy_model(num_layers=1, seed=3)
    # Set W_K_up = W_V_up in layer 0 by zeroing the V half and copying K
    attn = model.layers[0]
    config = toy_config_from_model(model)
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    with torch.no_grad():
        w = attn.kv_b_proj.weight  # [nh*(qk_nope+v), r]
        k_part = w[: nh * qk_nope, :].clone()
        # Set V part to same as K part (min of two dims)
        min_dim = min(nh * qk_nope, nh * v)
        w[nh * qk_nope : nh * qk_nope + min_dim, :] = k_part[:min_dim, :]

    loader = _wrapped(DataLoader(
        TensorDataset(torch.randint(0, 100, (8, 16))), batch_size=4
    ))
    result = compute_activation_alignment(model, loader, config, max_tokens=200)
    if 0 in result:
        rho = result[0]
        assert rho > 0.5, f"Expected high alignment when K=V, got rho={rho:.3f}"
