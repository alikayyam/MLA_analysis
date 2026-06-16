"""Experiment 1: Gauge-invariant layer-wise weight audit.

For each attention layer, extracts W_c, W_K_up, W_V_up and computes:
  - D_Gram: Frobenius-normalised Gram distance (latent method, O(r^2))
  - θ̄: Mean principal angle between effective subspaces (SVD method, O(d*r))

Shard-by-shard loading keeps peak memory bounded to one shard's subset.
Results saved as .npz for downstream use by Exp 2 and Exp 3.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

from model_io.shard_loader import iter_layers_from_shards, list_available_layers
from model_io.weight_extractor import extract_mla_weights
from metrics.gram_distance import compute_gram_distance
from metrics.principal_angles import compute_mean_principal_angle


def run_weight_audit(
    model_dir: str,
    config: dict,
    output_path: str,
    layer_indices: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
    gram_method: str = "latent",
    angle_method: str = "svd",
) -> dict:
    """Run the full weight audit and save results.

    Args:
        model_dir:     Path to local safetensors checkpoint directory.
        config:        YAML config dict (from load_config).
        output_path:   Path to save .npz results.
        layer_indices: Subset of layers to audit (None = all layers).
        dtype:         Dtype for weight loading (bfloat16 saves memory).
        gram_method:   "latent" (memory-safe) or "full_gram" (small models only).
        angle_method:  "svd" (correct) or "latent" (approximation).

    Returns:
        dict with keys: layer_idx, gram_dist, mean_angle
    """
    if layer_indices is None:
        layer_indices = list_available_layers(model_dir)
    layer_indices = sorted(layer_indices)

    results = {
        "layer_idx": [],
        "gram_dist": [],
        "mean_angle": [],
    }

    for layer_idx, raw_weights in tqdm(
        iter_layers_from_shards(model_dir, layer_indices, config, dtype=dtype),
        total=len(layer_indices),
        desc=f"Auditing {config.get('model_name', 'model')}",
    ):
        w = extract_mla_weights(raw_weights, layer_idx, config, dtype=torch.float32)

        d_gram = compute_gram_distance(
            w["W_c"], w["W_K_up"], w["W_V_up"], method=gram_method
        )
        theta_bar = compute_mean_principal_angle(
            w["W_c"], w["W_K_up"], w["W_V_up"], method=angle_method, degrees=True
        )

        results["layer_idx"].append(layer_idx)
        results["gram_dist"].append(d_gram)
        results["mean_angle"].append(theta_bar)

    # Sort by layer index (shard iterator may not yield in order)
    order = np.argsort(results["layer_idx"])
    for key in results:
        results[key] = np.array(results[key])[order]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **results)
    print(f"Saved results to {output_path}")

    return results


def load_audit_results(npz_path: str) -> dict:
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}
