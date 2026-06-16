"""Scheduling strategies for the surgical clamping experiment (Exp 3)."""

import numpy as np


def middle_block_schedule(num_layers: int) -> list[int]:
    """Return layer indices in [0.25L, 0.75L] (middle 50% of network depth)."""
    lo = int(0.25 * num_layers)
    hi = int(0.75 * num_layers)
    return list(range(lo, hi))


def progressive_outward_schedule(
    gram_distances: np.ndarray,
    layer_indices: np.ndarray | None = None,
) -> list[list[int]]:
    """Return a list of cumulative layer sets sorted by ascending D_Gram.

    The first entry contains just the single most-aligned (lowest D_Gram) layer.
    Each subsequent entry adds the next most-aligned layer. This produces a
    compression-quality tradeoff curve when PPL is evaluated at each step.

    Args:
        gram_distances: 1-D array of D_Gram values, one per layer.
        layer_indices:  Optional 1-D array of layer indices (defaults to 0..N-1).

    Returns:
        List of length N, where entry i is the sorted set of the i+1 most aligned layers.
    """
    if layer_indices is None:
        layer_indices = np.arange(len(gram_distances))
    order = np.argsort(gram_distances)  # ascending: most aligned first
    schedules = []
    cumulative = []
    for idx in order:
        cumulative.append(int(layer_indices[idx]))
        schedules.append(sorted(cumulative))
    return schedules


def global_schedule(num_layers: int) -> list[int]:
    """Return all layer indices (global baseline)."""
    return list(range(num_layers))
