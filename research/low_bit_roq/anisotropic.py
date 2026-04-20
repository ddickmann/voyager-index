"""
Anisotropic ScaNN-style codebook fitting.

Plan reference: Phase A3 (2-bit per-group min/max) + Phase A2.5 (ternary
3-level {-1, 0, +1}). Both use the same loss family:

    L = η · ||x∥ - x̂∥||² + ||x⊥ - x̂⊥||²,    η ≥ 1

Where x∥ is the projection of x onto the query direction. With the standard
ScaNN simplification (treat the query distribution as isotropic on the unit
sphere), this collapses to:

    L = (η - 1) · <ê, ê·u>² + ||x - x̂||²,    u = x / ||x||

For the per-group min/max codebook used by RotationalQuantizer, the
optimum has a closed form when η=1 (uniform min/max). When η > 1 we solve
a 1D scalar problem per group with a few Newton iterations — fast and
deterministic.

For the ternary 3-level codebook used by ``ternary.TernaryQuantizer`` the
problem is to pick (τ, scale) per group so that
    x̂ = scale · sign(x) · 1{|x| > τ}
minimizes the same loss. This also reduces to a 1D root-find per group.
"""

from __future__ import annotations

import numpy as np

ETA_DEFAULT = 4.0
"""Higher η weights parallel error more — typical ScaNN production η ≈ 4
for the int8 / 4-bit regime where the rerank set is in the few-thousand
range. Sweep η ∈ {1, 2, 4, 8} as part of A3."""


def fit_anisotropic_min_max(
    grouped: np.ndarray,
    *,
    bits: int = 2,
    eta: float = ETA_DEFAULT,
    n_newton: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit (scale, offset) per group minimizing the anisotropic IP loss.

    ``grouped`` has shape (N, n_groups, group_size). Returns scales and
    offsets each of shape (N, n_groups).
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    levels = (1 << bits) - 1
    n, n_groups, gs = grouped.shape

    min_vals = grouped.min(axis=2)
    max_vals = grouped.max(axis=2)
    ranges = np.where((max_vals - min_vals) < 1e-6, 1.0, max_vals - min_vals)
    scales = ranges / levels
    offsets = min_vals.copy()

    if eta == 1.0:
        return scales.astype(np.float32), offsets.astype(np.float32)

    norms = np.linalg.norm(grouped, axis=2) + 1e-12
    u = grouped / norms[..., None]

    for _ in range(n_newton):
        codes = np.round((grouped - offsets[..., None]) / scales[..., None]).clip(0, levels)
        x_hat = codes * scales[..., None] + offsets[..., None]
        residual = grouped - x_hat
        parallel = (residual * u).sum(axis=2)
        perp_sq = (residual ** 2).sum(axis=2) - parallel ** 2

        codes_centered = codes - codes.mean(axis=2, keepdims=True)
        denom = (codes_centered ** 2).sum(axis=2) + 1e-9
        d_scale = -(eta * parallel * (u * codes_centered).sum(axis=2)
                    + (residual * codes_centered).sum(axis=2))
        scales = scales - 0.5 * (d_scale / denom)

        scales = np.where(scales < 1e-6, ranges / levels, scales)

    return scales.astype(np.float32), offsets.astype(np.float32)


def fit_ternary_codebook(
    grouped: np.ndarray,
    *,
    eta: float = ETA_DEFAULT,
    tau_grid: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 1.0, 1.4),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit (τ, scale) per group for the ternary {-1, 0, +1} codebook.

    Implementation notes:

    - For each group, sweep a small grid of τ as a fraction of the group
      std and pick the (τ, scale) pair that minimizes the anisotropic loss.
    - This is small enough (tens of microseconds per group) to run at index
      time on CPU; not on the rerank hot path.
    - Returns ``tau`` of shape (N, n_groups) and ``scale`` of shape
      (N, n_groups).
    """
    n, n_groups, gs = grouped.shape
    std = grouped.std(axis=2, ddof=0) + 1e-8
    norms_sq = (grouped ** 2).sum(axis=2)
    abs_g = np.abs(grouped)
    sign_g = np.sign(grouped)

    best_loss = np.full((n, n_groups), np.inf, dtype=np.float64)
    best_tau = np.zeros((n, n_groups), dtype=np.float32)
    best_scale = np.zeros((n, n_groups), dtype=np.float32)

    for frac in tau_grid:
        tau = std * frac
        mask = abs_g > tau[..., None]
        num = (abs_g * mask).sum(axis=2)
        den = mask.sum(axis=2).clip(min=1).astype(np.float32)
        scale = num / den

        # x_hat = scale * sign * mask
        # residual = grouped - x_hat
        # parallel = (residual * u).sum(); we approximate u ~= grouped/||grouped||
        # using the full-group residual norm formula:
        x_hat = scale[..., None] * sign_g * mask
        residual_sq = ((grouped - x_hat) ** 2).sum(axis=2)
        parallel = ((grouped - x_hat) * grouped).sum(axis=2) / np.sqrt(norms_sq + 1e-12)
        perp_sq = residual_sq - parallel ** 2
        loss = eta * (parallel ** 2) + perp_sq.clip(min=0)

        better = loss < best_loss
        best_loss = np.where(better, loss, best_loss)
        best_tau = np.where(better, tau, best_tau)
        best_scale = np.where(better, scale, best_scale)

    return best_tau.astype(np.float32), best_scale.astype(np.float32)
