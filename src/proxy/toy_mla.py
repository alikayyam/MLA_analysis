"""Synthetic MLA attention module for unit testing.

Mirrors the exact weight structure of DeepSeek's MLA attention (kv_a_proj_with_mqa
and kv_b_proj) with configurable tiny dimensions. No real model download needed.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToyMLAConfig:
    hidden_size: int = 64
    kv_lora_rank: int = 8
    num_attention_heads: int = 4
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    num_hidden_layers: int = 4


class ToyMLAAttention(nn.Module):
    """Single MLA attention layer with DeepSeek-style weight names."""

    def __init__(self, cfg: ToyMLAConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        r = cfg.kv_lora_rank
        nh = cfg.num_attention_heads
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v = cfg.v_head_dim

        # Down-projection: [r + qk_rope, d] (mirrors kv_a_proj_with_mqa)
        self.kv_a_proj_with_mqa = nn.Linear(d, r + qk_rope, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(r)

        # Up-projection: [nh*(qk_nope + v), r] (mirrors kv_b_proj)
        self.kv_b_proj = nn.Linear(r, nh * (qk_nope + v), bias=False)

        # Query projection (independent of KV pathway)
        self.q_proj = nn.Linear(d, nh * (qk_nope + qk_rope), bias=False)

    def get_mla_weights(self) -> dict[str, torch.Tensor]:
        """Return W_c, W_K_up, W_V_up in paper notation as float32 tensors."""
        r = self.cfg.kv_lora_rank
        nh = self.cfg.num_attention_heads
        qk_nope = self.cfg.qk_nope_head_dim

        # kv_a_proj_with_mqa.weight shape: [r + qk_rope, d]
        # W_c is the first r rows, transposed: [d, r]
        W_c = self.kv_a_proj_with_mqa.weight[:r, :].T.float()

        # kv_b_proj.weight shape: [nh*(qk_nope + v), r]
        # K-up: first nh*qk_nope rows; V-up: next nh*v rows
        W_K_up = self.kv_b_proj.weight[: nh * qk_nope, :].float()
        W_V_up = self.kv_b_proj.weight[nh * qk_nope :, :].float()

        return {"W_c": W_c, "W_K_up": W_K_up, "W_V_up": W_V_up}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Minimal forward pass (attention output, ignoring RoPE for simplicity)."""
        B, T, d = x.shape
        r = self.cfg.kv_lora_rank
        nh = self.cfg.num_attention_heads
        qk_nope = self.cfg.qk_nope_head_dim
        v = self.cfg.v_head_dim

        # Down-project and layer-norm the latent
        kv_a = self.kv_a_proj_with_mqa(x)  # [B, T, r + qk_rope]
        latent = self.kv_a_layernorm(kv_a[..., :r])  # [B, T, r]

        # Up-project to K and V
        kv_b = self.kv_b_proj(latent)  # [B, T, nh*(qk_nope+v)]
        k = kv_b[..., : nh * qk_nope].view(B, T, nh, qk_nope)
        v_val = kv_b[..., nh * qk_nope :].view(B, T, nh, v)

        # Query
        q_full = self.q_proj(x)  # [B, T, nh*(qk_nope+qk_rope)]
        q = q_full[..., : nh * qk_nope].view(B, T, nh, qk_nope)

        # Attention (simplified, no masking)
        scale = qk_nope ** -0.5
        attn = torch.einsum("bthd,bshd->bhts", q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v_val)
        return out.reshape(B, T, nh * v)


class ToyMLAModel(nn.Module):
    """Stack of ToyMLAAttention layers for multi-layer testing.

    Accepts either:
    - float tensor [B, T, d]:  passed directly as embeddings
    - int tensor   [B, T]:     converted via an embedding table (vocab_size=1000)
    """

    VOCAB_SIZE = 1000

    def __init__(self, cfg: ToyMLAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(self.VOCAB_SIZE, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [ToyMLAAttention(cfg) for _ in range(cfg.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            x = self.embed(x)  # [B, T] -> [B, T, d]
        for layer in self.layers:
            x = x + layer(x)
        return x

    def get_all_mla_weights(self) -> list[dict[str, torch.Tensor]]:
        return [layer.get_mla_weights() for layer in self.layers]


def make_toy_model(
    num_layers: int = 4,
    hidden_size: int = 64,
    kv_lora_rank: int = 8,
    num_attention_heads: int = 4,
    qk_nope_head_dim: int = 16,
    qk_rope_head_dim: int = 8,
    v_head_dim: int = 16,
    seed: Optional[int] = 42,
) -> ToyMLAModel:
    if seed is not None:
        torch.manual_seed(seed)
    cfg = ToyMLAConfig(
        hidden_size=hidden_size,
        kv_lora_rank=kv_lora_rank,
        num_attention_heads=num_attention_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        num_hidden_layers=num_layers,
    )
    return ToyMLAModel(cfg)
