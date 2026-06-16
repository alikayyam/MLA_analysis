"""Sliding-window perplexity computation on WikiText-103 and C4.

Uses the standard stride-based approach to avoid edge effects at sequence boundaries.
All computation is CPU-only on this machine.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
from tqdm import tqdm


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext-2-raw-v1",
    dataset_split: str = "validation",
    stride: int = 512,
    max_length: int = 2048,
    max_samples: int | None = None,
    device: str = "cpu",
) -> float:
    """Compute sliding-window perplexity on a HuggingFace text dataset.

    Args:
        model:       Loaded nn.Module in eval mode.
        tokenizer:   Corresponding tokenizer.
        dataset_name: e.g. "wikitext-2-raw-v1" or "wikitext-103-raw-v1".
        dataset_split: "validation" or "test".
        stride:      Sliding window stride (smaller = more accurate, slower).
        max_length:  Context window size.
        max_samples: Limit number of text chunks (None = all).
        device:      "cpu" or "cuda".

    Returns:
        Perplexity (exp of mean negative log-likelihood).
    """
    from datasets import load_dataset

    raw = load_dataset(
        "Salesforce/wikitext" if "wikitext" in dataset_name else "allenai/c4",
        dataset_name,
        split=dataset_split,
    )
    text = "\n\n".join(raw["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]

    seq_len = input_ids.shape[1]
    nlls = []

    for begin in tqdm(range(0, seq_len, stride), desc=f"PPL ({dataset_name})"):
        end = min(begin + max_length, seq_len)
        target_len = end - (begin + stride - stride)  # tokens we're actually scoring
        if target_len <= 0:
            break

        chunk = input_ids[:, begin:end].to(device)
        target_len = min(stride, end - begin)

        with torch.no_grad():
            outputs = model(chunk, labels=chunk, use_cache=False)
            # outputs.loss is mean NLL over the full chunk; we need the last target_len tokens
            # Re-compute NLL manually for the last target_len tokens
            logits = outputs.logits  # [1, T, vocab]
            T = chunk.shape[1]

            # Score only the last target_len tokens
            score_start = T - target_len
            logits_slice = logits[0, score_start:-1, :]    # [target_len-1, vocab]
            labels_slice = chunk[0, score_start + 1:]       # [target_len-1]

            nll = torch.nn.functional.cross_entropy(logits_slice, labels_slice)
            nlls.append(nll.item())

        if max_samples is not None and len(nlls) >= max_samples:
            break

        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / max(1, len(nlls)))
    return ppl


def compute_delta_ppl(
    baseline_ppl: float,
    intervened_ppl: float,
) -> float:
    """ΔPPL = PPL_intervened - PPL_baseline. Positive = degradation."""
    return intervened_ppl - baseline_ppl
