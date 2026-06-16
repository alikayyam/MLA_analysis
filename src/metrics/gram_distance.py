"""Frobenius-normalised Gram distance between effective K and V operators.

D_Gram(W̃_K, W̃_V) = || G_K - G_V ||_F
where G_X = W̃_X W̃_X^T / ||W̃_X||_F^2

The "latent" method avoids forming any [d, output_dim] matrix by working
entirely in [r, r] space via trace identities:

  W̃_X = W_c @ W_X_up.T   (conceptually, never materialised for large models)
  G_X = W̃_X W̃_X^T / ||W̃_X||_F^2

Let C = W_c.T @ W_c   [r, r]
    M_KK = W_K_up @ W_K_up.T  [r, r]
    M_VV = W_V_up @ W_V_up.T  [r, r]
    M_KV = W_K_up @ W_V_up.T  [r, r]

Then:
  ||W̃_K||_F^2 = tr(W̃_K W̃_K^T) = tr(W_c M_KK W_c^T) = tr(C M_KK)
  tr(G_K G_V) = tr(G_K^T G_V) = tr((W̃_K W̃_K^T)(W̃_V W̃_V^T)) / (nk2 * nv2)
              = tr(W_c M_KK W_c^T W_c M_VV W_c^T) / (nk2 * nv2)
              = tr(C M_KK C M_VV) / (nk2 * nv2)
  ||G_K - G_V||_F^2 = tr(G_K^2) + tr(G_V^2) - 2 tr(G_K G_V)
                    = tr(C M_KK C M_KK)/nk2^2 + tr(C M_VV C M_VV)/nv2^2
                      - 2 tr(C M_KK C M_VV)/(nk2*nv2)
"""

import torch
import math


def compute_gram_distance(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
    method: str = "latent",
) -> float:
    """Frobenius-normalised Gram distance between effective K and V operators.

    Args:
        W_c:    [d, r] down-projection (paper notation, transposed from HF weight).
        W_K_up: [nh*qk_nope, r] key up-projection.
        W_V_up: [nh*v, r] value up-projection.
        method: "latent" (memory-safe, O(r^2)) or "full_gram" (explicit, for small models).

    Returns:
        Scalar D_Gram >= 0. Lower means more aligned subspaces.
    """
    if method == "full_gram":
        return _gram_distance_full(W_c, W_K_up, W_V_up)
    return _gram_distance_latent(W_c, W_K_up, W_V_up)


def _gram_distance_latent(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
) -> float:
    """Memory-safe computation entirely in [r, r] space.

    All intermediate matrices are [r, r] (r = kv_lora_rank).

    Key: M_KK = W_K_up.T @ W_K_up  (NOT W_K_up @ W_K_up.T which is [nh*qk_nope, nh*qk_nope])
    because tr(W̃_K W̃_K^T) = tr(W_c W_K_up^T W_K_up W_c^T) = tr(C M_KK) via trace cyclic property,
    where C = W_c^T W_c [r,r] and M_KK = W_K_up^T W_K_up [r,r].
    """
    W_c = W_c.float()
    W_K_up = W_K_up.float()
    W_V_up = W_V_up.float()

    # All [r, r]
    C = W_c.T @ W_c                      # W_c^T W_c
    M_KK = W_K_up.T @ W_K_up             # W_K_up^T W_K_up
    M_VV = W_V_up.T @ W_V_up             # W_V_up^T W_V_up

    nk2 = _trace(C @ M_KK)
    nv2 = _trace(C @ M_VV)

    if nk2 < 1e-30 or nv2 < 1e-30:
        return 0.0

    CM_KK = C @ M_KK
    CM_VV = C @ M_VV

    tr_GK2 = _trace(CM_KK @ CM_KK) / (nk2 ** 2)
    tr_GV2 = _trace(CM_VV @ CM_VV) / (nv2 ** 2)
    tr_GKGV = _trace(CM_KK @ CM_VV) / (nk2 * nv2)

    dist_sq = tr_GK2 + tr_GV2 - 2.0 * tr_GKGV
    # Numerical noise can push this slightly below zero
    return math.sqrt(max(dist_sq.item(), 0.0))


def _gram_distance_full(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
) -> float:
    """Explicit computation — only use for small toy models."""
    W_c = W_c.float()
    W_K_up = W_K_up.float()
    W_V_up = W_V_up.float()

    W_tilde_K = W_c @ W_K_up.T  # [d, output_K]
    W_tilde_V = W_c @ W_V_up.T  # [d, output_V]

    nk2 = (W_tilde_K ** 2).sum()
    nv2 = (W_tilde_V ** 2).sum()

    if nk2 < 1e-30 or nv2 < 1e-30:
        return 0.0

    G_K = (W_tilde_K @ W_tilde_K.T) / nk2  # [d, d]
    G_V = (W_tilde_V @ W_tilde_V.T) / nv2  # [d, d]

    diff = G_K - G_V
    return math.sqrt((_trace(diff.T @ diff)).item())


def _trace(M: torch.Tensor) -> torch.Tensor:
    return M.diagonal().sum()
