"""Single source of truth for mapping HuggingFace MLA weights to paper notation.

HuggingFace key -> paper notation:
  kv_a_proj_with_mqa.weight  shape [r+qk_rope, d]  -> W_c = weight[:r, :].T  shape [d, r]
  kv_b_proj.weight           shape [nh*(qk_nope+v), r] -> W_K_up = weight[:nh*qk_nope, :]
                                                           W_V_up = weight[nh*qk_nope:, :]
"""

import torch
import torch.nn as nn
from typing import Union
import yaml
from pathlib import Path


def load_config(config_path: Union[str, Path]) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_mla_weights(
    source: Union[dict, nn.Module],
    layer_idx: int,
    config: dict,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Extract W_c, W_K_up, W_V_up for a single layer.

    Args:
        source: Either a state dict (for shard loading) or a live nn.Module.
        layer_idx: Which transformer layer to extract.
        config: YAML config dict with keys kv_lora_rank, num_attention_heads,
                qk_nope_head_dim, v_head_dim, layer_prefix, weight_key_down, weight_key_up.
        dtype: Cast all returned tensors to this dtype.

    Returns:
        {"W_c": [d, r], "W_K_up": [nh*qk_nope, r], "W_V_up": [nh*v, r]}
    """
    r = config["kv_lora_rank"]
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    prefix = config["layer_prefix"].format(i=layer_idx)
    key_down = config["weight_key_down"]
    key_up = config["weight_key_up"]

    if isinstance(source, dict):
        weight_down = source[f"{prefix}.{key_down}"]
        weight_up = source[f"{prefix}.{key_up}"]
    else:
        # Live model: navigate the module tree
        attn = _get_attn_module(source, layer_idx)
        weight_down = getattr(attn, key_down.replace(".weight", "")).weight
        weight_up = getattr(attn, key_up.replace(".weight", "")).weight

    # weight_down: [r + qk_rope, d] -> W_c = first r rows transposed -> [d, r]
    W_c = weight_down[:r, :].T.to(dtype=dtype)

    # weight_up: [nh*(qk_nope + v), r]
    W_K_up = weight_up[: nh * qk_nope, :].to(dtype=dtype)
    W_V_up = weight_up[nh * qk_nope : nh * (qk_nope + v), :].to(dtype=dtype)

    return {"W_c": W_c, "W_K_up": W_K_up, "W_V_up": W_V_up}


def extract_mla_weights_from_toy(
    layer: nn.Module,
    config: dict,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convenience wrapper for ToyMLAAttention layers (no layer_idx needed)."""
    r = config["kv_lora_rank"]
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]

    weight_down = layer.kv_a_proj_with_mqa.weight  # [r + qk_rope, d]
    weight_up = layer.kv_b_proj.weight              # [nh*(qk_nope+v), r]

    W_c = weight_down[:r, :].T.to(dtype=dtype)
    W_K_up = weight_up[: nh * qk_nope, :].to(dtype=dtype)
    W_V_up = weight_up[nh * qk_nope : nh * (qk_nope + v), :].to(dtype=dtype)

    return {"W_c": W_c, "W_K_up": W_K_up, "W_V_up": W_V_up}


def _get_attn_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """Navigate model.layers[layer_idx].self_attn (or .attention)."""
    layers = getattr(model, "model", model).layers
    layer = layers[layer_idx]
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError(f"Cannot find attention module in layer {layer_idx}")


def toy_config_from_model(model) -> dict:
    """Build a minimal config dict from a ToyMLAModel instance."""
    cfg = model.cfg
    return {
        "kv_lora_rank": cfg.kv_lora_rank,
        "num_attention_heads": cfg.num_attention_heads,
        "qk_nope_head_dim": cfg.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "layer_prefix": "layers.{i}",
        "weight_key_down": "kv_a_proj_with_mqa.weight",
        "weight_key_up": "kv_b_proj.weight",
    }
