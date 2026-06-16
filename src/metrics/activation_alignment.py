"""Activation-based K-V alignment (Experiment 2).

Captures key and value activations at runtime (before RoPE) and computes
the Expected Empirical Activation Alignment:

    ρ = (1/n) Σ_t cosine_sim(x_t W̃_K, x_t W̃_V)

Uses Welford online averaging — at most one batch lives in memory at a time.
Hooks on kv_b_proj output, splitting into K-part and V-part.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Generator


class _WelfordAccumulator:
    """Online mean accumulator (Welford's algorithm)."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Update with a flat tensor of new observations."""
        flat = values.detach().float().flatten()
        n = flat.numel()
        if n == 0:
            return
        batch_mean = flat.mean().item()
        # Welford update for the combined mean
        delta = batch_mean - self.mean
        self.count += n
        self.mean += delta * n / self.count

    def result(self) -> float:
        return self.mean if self.count > 0 else float("nan")


def compute_activation_alignment(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    max_tokens: int = 50_000,
    device: str = "cpu",
) -> dict[int, float]:
    """Compute per-layer activation alignment ρ on the dataloader.

    Hooks are registered on kv_b_proj in each attention layer, capturing
    the concatenated [K-part | V-part] output before RoPE. The K-part and
    V-part are split and per-token cosine similarity is accumulated online.

    Args:
        model:      Loaded nn.Module (model.eval() recommended).
        dataloader: DataLoader yielding {"input_ids": Tensor} batches.
        config:     YAML config dict (needs num_hidden_layers, qk_nope_head_dim, etc.).
        max_tokens: Stop after processing this many tokens.
        device:     "cpu" or "cuda".

    Returns:
        dict mapping layer_idx -> float ρ value.
    """
    nh = config["num_attention_heads"]
    qk_nope = config["qk_nope_head_dim"]
    v = config["v_head_dim"]
    num_layers = config["num_hidden_layers"]

    accumulators: dict[int, _WelfordAccumulator] = {
        i: _WelfordAccumulator() for i in range(num_layers)
    }
    total_tokens = [0]

    def _make_hook(layer_idx: int):
        def hook(module, input, output):
            if total_tokens[0] >= max_tokens:
                return
            # output shape: [B, T, nh*(qk_nope + v)]
            out = output.detach().float()
            k_acts = out[..., : nh * qk_nope]   # [B, T, nh*qk_nope]
            v_acts = out[..., nh * qk_nope :]    # [B, T, nh*v]

            # Flatten heads: [B*T, nh*qk_nope] and [B*T, nh*v]
            B, T, _ = k_acts.shape
            k_flat = k_acts.reshape(B * T, -1)
            v_flat = v_acts.reshape(B * T, -1)

            # Align dimensions for cosine similarity (take the min)
            min_dim = min(k_flat.shape[1], v_flat.shape[1])
            cos_sim = torch.nn.functional.cosine_similarity(
                k_flat[:, :min_dim], v_flat[:, :min_dim], dim=1
            )  # [B*T]

            accumulators[layer_idx].update(cos_sim)
            total_tokens[0] += B * T
        return hook

    # Register hooks on kv_b_proj in each attention layer
    hooks = []
    layers = getattr(model, "model", model).layers
    for i, layer in enumerate(layers):
        attn = _get_attn(layer)
        if attn is not None and hasattr(attn, "kv_b_proj"):
            h = attn.kv_b_proj.register_forward_hook(_make_hook(i))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if total_tokens[0] >= max_tokens:
                break
            input_ids = batch["input_ids"].to(device)
            model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    return {i: acc.result() for i, acc in accumulators.items() if acc.count > 0}


def _get_attn(layer: nn.Module):
    for attr in ("self_attn", "attention", "attn"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    # The layer itself may be the attention module (e.g., ToyMLAAttention)
    if hasattr(layer, "kv_b_proj"):
        return layer
    return None
