"""Load a full DeepSeek model into RAM (for Exp 2 and Exp 3 with the Lite model).

For DeepSeek-V2-Lite (~31 GB bfloat16), the model fits in the 62 GB RAM of
this machine. For V2 (236B) and V3 (671B), Exp 2 requires a cluster; Exp 1
uses shard_loader instead.
"""

from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache


def _patch_dynamic_cache():
    """Patch DynamicCache for transformers 5.x compatibility with DeepSeek's remote code."""
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (k, v) in enumerate(past_key_values):
                    cache.update(k, v, layer_idx)
            return cache
        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            return tuple(
                (layer.keys, layer.values)
                for layer in self.layers
            )
        DynamicCache.to_legacy_cache = to_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())


def load_model_and_tokenizer(
    model_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
    trust_remote_code: bool = True,
) -> tuple:
    """Load model and tokenizer from a local directory."""
    _patch_dynamic_cache()
    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    print(f"Loading model from {model_dir} (dtype={dtype}, device_map={device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    _reload_all_weights(model, model_dir, dtype)
    print("Model loaded.")
    return model, tokenizer


def _reload_all_weights(model, model_dir: str, dtype: torch.dtype) -> None:
    """Force-reload every parameter from the safetensors checkpoint.

    transformers 5.x fails to load most DeepSeek V2 weights because
    _tied_weights_keys is declared as a list (old 4.x format) and the 5.x
    loader calls .keys() on it, causing silent failures. Rather than guessing
    which weights are broken via std comparison (unreliable: many projection
    weights have trained std ≈ init std ≈ 0.02), we unconditionally overwrite
    every named parameter that appears in the safetensors index.

    Peak extra memory = one open shard (~7-8 GB); the four shards are processed
    sequentially and never held in memory simultaneously.
    """
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        return

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    model_params = dict(model.named_parameters())

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_file in weight_map.items():
        if key in model_params:
            shard_to_keys.setdefault(shard_file, []).append(key)

    total = sum(len(v) for v in shard_to_keys.values())
    loaded = 0

    for shard_file, keys in sorted(shard_to_keys.items()):
        shard_path = Path(model_dir) / shard_file
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                model_params[key].data.copy_(f.get_tensor(key).to(dtype))
                loaded += 1

    print(f"  Reloaded {loaded}/{total} weights from checkpoint.")
