"""Shard-by-shard safetensors loader for large MLA models (DeepSeek-V2/V3).

Reads model.safetensors.index.json, groups the 3 attention weight tensors per
layer by shard file, then opens each shard exactly once — extracting only the
needed tensors before closing it. Peak memory at any moment is bounded by one
shard's subset (typically a few hundred MB).

FP8 support (DeepSeek-V3): weights stored as FP8 with companion _scale_inv
tensors are automatically dequantized using block-wise scaling (128×128 tiles).
The scale_inv tensor is always co-located in the same shard as its weight.

Usage:
    for layer_idx, weight_dict in iter_layers_from_shards(model_dir, all_layers, config):
        W_c, W_K_up, W_V_up = extract_mla_weights(weight_dict, layer_idx, config)
        ...
"""

import json
from pathlib import Path
from typing import Generator
import torch
from safetensors import safe_open

_FP8_BLOCK_SIZE = 128


def _dequant_fp8_blockwise(
    weight: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    """Dequantize a block-wise FP8 weight to float32.

    weight:    [out, in] in FP8 (element_size() == 1)
    scale_inv: [ceil(out/128), ceil(in/128)] in float32
    Each 128×128 tile of weight is multiplied by the corresponding scalar in scale_inv.
    """
    out, in_ = weight.shape
    B = _FP8_BLOCK_SIZE
    s = scale_inv.float()
    # Expand scale to [out, in_] via repeat_interleave then trim
    s_exp = s.repeat_interleave(B, dim=0)[:out]        # [out, n_in_blocks]
    s_exp = s_exp.repeat_interleave(B, dim=1)[:, :in_]  # [out, in_]
    return weight.float() * s_exp


def list_available_layers(model_dir: str) -> list[int]:
    """Discover all attention layer indices from the safetensors index."""
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard model (e.g., small toy or V2-Lite single file)
        shard = next(Path(model_dir).glob("model.safetensors"), None)
        if shard is None:
            raise FileNotFoundError(f"No safetensors files found in {model_dir}")
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    else:
        with open(index_path) as fp:
            index = json.load(fp)
        keys = list(index["weight_map"].keys())

    # Extract layer indices from keys like "model.layers.7.self_attn.kv_b_proj.weight"
    layers = set()
    for key in keys:
        parts = key.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            try:
                layers.add(int(parts[idx + 1]))
            except (IndexError, ValueError):
                pass
    return sorted(layers)


def iter_layers_from_shards(
    model_dir: str,
    layer_indices: list[int],
    config: dict,
    dtype: torch.dtype = torch.bfloat16,
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield (layer_idx, raw_weight_dict) for each requested layer.

    raw_weight_dict contains the HuggingFace-keyed tensors for that layer,
    already cast to `dtype`. FP8 weights (DeepSeek-V3) are dequantized
    automatically before casting.

    Pass directly to weight_extractor.extract_mla_weights(source=..., layer_idx=...).

    Memory: opens one shard at a time, reads only the 2-3 keys needed per layer.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as fp:
            weight_map = json.load(fp)["weight_map"]
        _iter = _iter_from_index(model_dir, layer_indices, config, dtype, weight_map)
    else:
        # Single shard — load the whole file once and yield from it
        shard_path = next(model_dir.glob("model.safetensors"))
        _iter = _iter_from_single_shard(shard_path, layer_indices, config, dtype)

    yield from _iter


def _layer_keys(layer_idx: int, config: dict, weight_map: dict | None = None) -> list[str]:
    """Return the HF weight keys to read for this layer.

    Includes _scale_inv keys for FP8 models (DeepSeek-V3) when they appear in
    the weight_map.  For non-FP8 models the list is just the two base keys.
    """
    prefix = config["layer_prefix"].format(i=layer_idx)
    base_keys = [
        f"{prefix}.{config['weight_key_down']}",
        f"{prefix}.{config['weight_key_up']}",
    ]
    if weight_map is None:
        return base_keys
    scale_keys = [
        f"{k}_scale_inv" for k in base_keys if f"{k}_scale_inv" in weight_map
    ]
    return base_keys + scale_keys


def _apply_dequant(
    raw: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Dequantize FP8 weights in raw and cast everything to dtype.

    Removes _scale_inv entries from the result dict.
    """
    result = {}
    for key, tensor in raw.items():
        if key.endswith("_scale_inv"):
            continue
        scale_key = f"{key}_scale_inv"
        if scale_key in raw and tensor.element_size() == 1:
            tensor = _dequant_fp8_blockwise(tensor, raw[scale_key])
        result[key] = tensor.to(dtype=dtype)
    return result


def _iter_from_index(
    model_dir: Path,
    layer_indices: list[int],
    config: dict,
    dtype: torch.dtype,
    weight_map: dict,
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    # Build: shard_file -> {layer_idx -> [key1, key2, ...]}
    shard_to_layers: dict[str, dict[int, list[str]]] = {}
    for layer_idx in layer_indices:
        for key in _layer_keys(layer_idx, config, weight_map):
            shard = weight_map.get(key)
            if shard is None:
                continue
            shard_to_layers.setdefault(shard, {}).setdefault(layer_idx, []).append(key)

    # Open each shard once, extract all needed tensors (including scale_inv), then close
    for shard_file, layers_in_shard in shard_to_layers.items():
        shard_path = str(model_dir / shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for layer_idx, keys in layers_in_shard.items():
                raw = {k: f.get_tensor(k) for k in keys}
                yield layer_idx, _apply_dequant(raw, dtype)


def _iter_from_single_shard(
    shard_path: Path,
    layer_indices: list[int],
    config: dict,
    dtype: torch.dtype,
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        for layer_idx in layer_indices:
            keys = _layer_keys(layer_idx, config)
            # Include scale_inv keys present in this shard
            scale_keys = [f"{k}_scale_inv" for k in keys if f"{k}_scale_inv" in all_keys]
            keys_to_read = [k for k in keys + scale_keys if k in all_keys]
            if not keys_to_read:
                continue
            raw = {k: f.get_tensor(k) for k in keys_to_read}
            yield layer_idx, _apply_dequant(raw, dtype)
