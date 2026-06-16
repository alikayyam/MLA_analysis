"""Surgical K=V parameter clamping for Experiment 3.

Sets W_K_up ← W_V_up in specified layers. This is equivalent to W̃_K ← W̃_V
when W_c has full column rank (the generic case), since:
  W_c A = W_c B  ⟹  A = B  (when W_c has full column rank)

The clamp eliminates r × qk_nope parameters per layer (the K-up projection),
as the attention block can reuse the V-up projection for both pathways.

Constraint: requires qk_nope_head_dim == v_head_dim. This holds for
DeepSeek-V2/V2-Lite/V3 (both = 128). Asserted before any modification.
"""

import torch
import torch.nn as nn
from typing import Iterator


def clamp_layers_inplace(
    model: nn.Module,
    layer_indices: list[int],
    config: dict,
) -> None:
    """Set W_K_up ← W_V_up in the specified layers (modifies model in-place).

    Args:
        model:         Loaded nn.Module (model.model.layers[i].self_attn.kv_b_proj).
        layer_indices: Layer indices to clamp.
        config:        YAML config dict (needs num_attention_heads, qk_nope/v_head_dims).
    """
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    assert qk_nope == v, (
        f"Clamping requires qk_nope_head_dim == v_head_dim, got {qk_nope} vs {v}. "
        "For mismatched dims, clamp only the overlapping portion."
    )

    layers = getattr(model, "model", model).layers
    for idx in layer_indices:
        layer = layers[idx]
        attn = _get_attn(layer)
        if attn is None or not hasattr(attn, "kv_b_proj"):
            print(f"  Warning: layer {idx} has no kv_b_proj, skipping.")
            continue
        weight = attn.kv_b_proj.weight  # [nh*(qk_nope+v), r]
        with torch.no_grad():
            # K-up: rows [0 : nh*qk_nope], V-up: rows [nh*qk_nope : nh*(qk_nope+v)]
            k_start, k_end = 0, nh * qk_nope
            v_start, v_end = nh * qk_nope, nh * (qk_nope + v)
            weight[k_start:k_end, :] = weight[v_start:v_end, :].clone()


def clamp_layers_in_state_dict(
    state_dict: dict[str, torch.Tensor],
    layer_indices: list[int],
    config: dict,
) -> dict[str, torch.Tensor]:
    """Return a modified copy of the state dict with K-up ← V-up in specified layers.

    Does not modify the input dict in-place. Suitable for checkpoint surgery
    without loading a live model.
    """
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    assert qk_nope == v, f"Clamping requires qk_nope_head_dim == v_head_dim"

    prefix_tmpl = config["layer_prefix"]
    key_up = config["weight_key_up"]

    modified = dict(state_dict)  # shallow copy; tensors shared until modified
    for idx in layer_indices:
        prefix = prefix_tmpl.format(i=idx)
        full_key = f"{prefix}.{key_up}"
        if full_key not in modified:
            continue
        w = modified[full_key].clone()
        k_start, k_end = 0, nh * qk_nope
        v_start, v_end = nh * qk_nope, nh * (qk_nope + v)
        w[k_start:k_end, :] = w[v_start:v_end, :].clone()
        modified[full_key] = w

    return modified


def _get_attn(layer: nn.Module):
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None
