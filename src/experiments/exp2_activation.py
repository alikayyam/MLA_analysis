"""Experiment 2: Activation-based K-V alignment on WikiText evaluation data.

Loads the model, hooks into kv_b_proj, and computes ρ per layer on WikiText-103
(or WikiText-2 for faster CPU runs). Also correlates ρ with static Exp 1 metrics.

Note: feasible on this machine only with DeepSeek-V2-Lite (~31 GB bfloat16 in RAM).
Full V2/V3 requires a multi-GPU cluster with device_map="auto".
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader, Dataset

from model_io.model_loader import load_model_and_tokenizer
from metrics.activation_alignment import compute_activation_alignment
from experiments.exp1_weight_audit import load_audit_results


class _WikitextDataset(Dataset):
    def __init__(self, encodings, seq_len: int = 512):
        self.input_ids = encodings["input_ids"]
        self.seq_len = seq_len

    def __len__(self):
        return max(1, self.input_ids.shape[1] // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return {"input_ids": self.input_ids[0, start:end]}


def _build_dataloader(
    tokenizer,
    dataset_name: str = "wikitext-2-raw-v1",
    seq_len: int = 512,
    batch_size: int = 1,
) -> DataLoader:
    from datasets import load_dataset
    raw = load_dataset("Salesforce/wikitext", dataset_name, split="validation")
    text = "\n\n".join(raw["text"])
    encodings = tokenizer(text, return_tensors="pt")
    ds = _WikitextDataset(encodings, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_activation_experiment(
    model_dir: str,
    config: dict,
    exp1_results_path: str,
    output_path: str,
    dataset_name: str = "wikitext-2-raw-v1",
    max_tokens: int = 50_000,
    seq_len: int = 512,
    device: str = "cpu",
) -> dict:
    """Run activation alignment experiment and compute correlations with Exp 1.

    Args:
        model_dir:         Local model directory.
        config:            YAML config dict.
        exp1_results_path: Path to Exp 1 .npz results.
        output_path:       Path to save .npz results.
        dataset_name:      HuggingFace wikitext variant.
        max_tokens:        Token budget for activation capture.
        seq_len:           Sequence length per batch.
        device:            "cpu" or "cuda".

    Returns:
        dict with layer_idx, rho, corr_gram_pearson, corr_angle_pearson, etc.
    """
    model, tokenizer = load_model_and_tokenizer(model_dir, device_map=device)

    print(f"Building dataloader ({dataset_name}, seq_len={seq_len}) ...")
    loader = _build_dataloader(tokenizer, dataset_name=dataset_name, seq_len=seq_len)

    print(f"Running activation hooks (max_tokens={max_tokens}) ...")
    rho_by_layer = compute_activation_alignment(
        model, loader, config, max_tokens=max_tokens, device=device
    )

    layer_indices = sorted(rho_by_layer.keys())
    rho_arr = np.array([rho_by_layer[i] for i in layer_indices])

    # Correlate with Exp 1 static metrics
    exp1 = load_audit_results(exp1_results_path + ".npz")
    # Align on common layer indices
    exp1_layers = list(exp1["layer_idx"])
    common_layers = [i for i in layer_indices if i in exp1_layers]
    idx_in_exp1 = [exp1_layers.index(i) for i in common_layers]
    idx_in_rho = [layer_indices.index(i) for i in common_layers]

    gram_dist = exp1["gram_dist"][idx_in_exp1]
    mean_angle = exp1["mean_angle"][idx_in_exp1]
    rho_common = rho_arr[idx_in_rho]

    corr_gram_p, _ = pearsonr(gram_dist, rho_common) if len(rho_common) > 2 else (float("nan"), None)
    corr_gram_s, _ = spearmanr(gram_dist, rho_common) if len(rho_common) > 2 else (float("nan"), None)
    corr_angle_p, _ = pearsonr(mean_angle, rho_common) if len(rho_common) > 2 else (float("nan"), None)
    corr_angle_s, _ = spearmanr(mean_angle, rho_common) if len(rho_common) > 2 else (float("nan"), None)

    print(f"Correlations (D_Gram vs ρ): Pearson={corr_gram_p:.3f}, Spearman={corr_gram_s:.3f}")
    print(f"Correlations (θ̄ vs ρ):     Pearson={corr_angle_p:.3f}, Spearman={corr_angle_s:.3f}")

    results = {
        "layer_idx": np.array(layer_indices),
        "rho": rho_arr,
        "common_layers": np.array(common_layers),
        "gram_dist_common": gram_dist,
        "mean_angle_common": mean_angle,
        "rho_common": rho_common,
        "corr_gram_pearson": np.array(corr_gram_p),
        "corr_gram_spearman": np.array(corr_gram_s),
        "corr_angle_pearson": np.array(corr_angle_p),
        "corr_angle_spearman": np.array(corr_angle_s),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **results)
    print(f"Saved Exp 2 results to {output_path}.npz")

    return results
