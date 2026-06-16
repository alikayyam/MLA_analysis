"""Mean principal angle between effective K and V subspaces.

Both effective operators W̃_K = W_c @ W_K_up.T and W̃_V = W_c @ W_V_up.T have
column spaces that are subsets of Col(W_c) in R^d. When W_K_up and W_V_up have
full rank r, both column spaces equal Col(W_c) and the angle is 0°. In practice,
trained models have lower-effective-rank up-projections, so the dominant directions
of W̃_K and W̃_V differ and produce non-trivial angles.

Algorithm (memory-efficient, O(d*r) peak memory):
  1. SVD of W_K_up [nh*qk_nope, r] → V_K [r, r_K] (right singular vectors)
  2. W_c @ V_K [d, r] — projects W_c onto Row(W_K_up) basis
  3. QR of (W_c @ V_K) → Q_K [d, r] (orthonormal basis for Col(W̃_K))
  4. Repeat for V
  5. SVD of Q_K.T @ Q_V [r, r] → singular values = cosines of principal angles

This never forms the [d, nh*qk_nope] effective operator. Peak memory: O(d*r).

"Latent" method (approximation for large models):
  Uses only the right singular vectors of W_K_up / W_V_up, skipping the W_c step.
  Valid when W_c is approximately orthogonal or isotropic. Documented approximation.
"""

import torch
import math


def compute_mean_principal_angle(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
    method: str = "svd",
    degrees: bool = True,
) -> float:
    """Mean principal angle between Col(W̃_K) and Col(W̃_V).

    Args:
        W_c:     [d, r] down-projection.
        W_K_up:  [nh*qk_nope, r] key up-projection.
        W_V_up:  [nh*v, r] value up-projection.
        method:  "svd" (memory-efficient, exact) or "latent" (approx, skips W_c).
        degrees: If True return degrees; else radians.

    Returns:
        Mean principal angle θ̄ in [0, 90]° (or [0, π/2] rad).
    """
    if method == "latent":
        theta = _mean_angle_latent(W_K_up, W_V_up)
    else:
        theta = _mean_angle_svd(W_c, W_K_up, W_V_up)
    return math.degrees(theta) if degrees else theta


def principal_angles_all(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
    method: str = "svd",
) -> torch.Tensor:
    """Return all principal angles as a 1-D tensor (degrees)."""
    if method == "latent":
        sigma = _principal_cosines_latent(W_K_up, W_V_up)
    else:
        sigma = _principal_cosines_svd(W_c, W_K_up, W_V_up)
    return torch.rad2deg(torch.arccos(sigma.clamp(-1.0, 1.0)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _principal_cosines_svd(
    W_c: torch.Tensor,
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
) -> torch.Tensor:
    """Cosines of principal angles, computed without forming [d, nh*...] matrices.

    Col(W̃_K) = Col(W_c @ W_K_up.T) = W_c @ Row(W_K_up) = W_c @ Col(W_K_up.T)
    An orthonormal basis for Row(W_K_up) is V_K from SVD of W_K_up.
    So Q_K = orth(W_c @ V_K), shape [d, r].
    """
    W_c = W_c.float()
    W_K_up = W_K_up.float()
    W_V_up = W_V_up.float()

    Q_K = _column_space_basis(W_c, W_K_up)
    Q_V = _column_space_basis(W_c, W_V_up)

    # Align column counts (may differ if effective ranks differ)
    r = min(Q_K.shape[1], Q_V.shape[1])
    Q_K, Q_V = Q_K[:, :r], Q_V[:, :r]

    sigma = torch.linalg.svdvals(Q_K.T @ Q_V)
    return sigma.clamp(-1.0, 1.0)


def _column_space_basis(W_c: torch.Tensor, W_up: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for Col(W_c @ W_up.T), shape [d, r_eff].

    W_c:  [d, r]
    W_up: [out_dim, r]  (up-projection in HF format)
    """
    # SVD of W_up [out_dim, r] -> V [r, r_eff] (right singular vectors)
    # Row(W_up) = Col(W_up.T) = span of V columns
    _, S, Vh = torch.linalg.svd(W_up, full_matrices=False)  # Vh: [r, r]

    # Keep only directions with significant singular values (numerical rank)
    thresh = S.max() * W_up.shape[0] * torch.finfo(S.dtype).eps * 10
    r_eff = (S > thresh).sum().item()
    r_eff = max(r_eff, 1)
    V = Vh[:r_eff, :].T  # [r, r_eff]

    # Project W_c onto these directions: [d, r] @ [r, r_eff] = [d, r_eff]
    basis = W_c @ V

    # Orthonormalise via QR
    Q, _ = torch.linalg.qr(basis, mode="reduced")
    return Q  # [d, r_eff]


def _principal_cosines_latent(
    W_K_up: torch.Tensor,
    W_V_up: torch.Tensor,
) -> torch.Tensor:
    """Approximation: cosines computed in up-projection space, ignoring W_c.

    Valid when W_c is approximately orthogonal. Returns cosines in [0,1].
    """
    W_K_up = W_K_up.float()
    W_V_up = W_V_up.float()

    # Right singular vectors of each up-projection
    _, _, Vh_K = torch.linalg.svd(W_K_up, full_matrices=False)  # Vh_K: [r, r]
    _, _, Vh_V = torch.linalg.svd(W_V_up, full_matrices=False)  # Vh_V: [r, r]

    r = min(Vh_K.shape[0], Vh_V.shape[0])
    sigma = torch.linalg.svdvals(Vh_K[:r, :] @ Vh_V[:r, :].T)
    return sigma.clamp(-1.0, 1.0)


def _mean_angle_svd(W_c, W_K_up, W_V_up) -> float:
    sigma = _principal_cosines_svd(W_c, W_K_up, W_V_up)
    return torch.mean(torch.arccos(sigma)).item()


def _mean_angle_latent(W_K_up, W_V_up) -> float:
    sigma = _principal_cosines_latent(W_K_up, W_V_up)
    return torch.mean(torch.arccos(sigma)).item()
