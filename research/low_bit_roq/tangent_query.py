"""
Tangent-aware query: B0.

The cheapest geometry probe: no doc rebuild, no kernel change. At query
time, for each touched centroid ``c``, replace the inner product
``<q, c>`` with a geodesic-aware bonus that uses the magnitude of
``log_c(q)`` (the tangent-space residual).

Plan reference: Phase B0. Reuses
``voyager_zero/src/voyager_zero/geometry/riemannian.py`` ``batched_log_map``
when available; otherwise falls back to a pure-numpy implementation that
matches the same formula on the unit sphere:

    log_c(q) = (q - <q, c> · c) · θ / sin(θ),  θ = arccos(<q, c>)

Score:

    s(q, c) = <q, c> - λ · ||log_c(q)||²
            = <q, c> - λ · θ²

The squared-magnitude form is identical to the squared geodesic distance
on S^{d-1}. λ is the only hyperparameter; sweep ∈ {0, 0.05, 0.1, 0.25, 0.5}
in B0.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _try_voyager_zero_log_map():
    try:
        from voyager_zero.geometry.riemannian import batched_log_map  # type: ignore

        return batched_log_map
    except ImportError:
        return None


def log_map_unit_sphere(c: np.ndarray, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """log_c(q) on S^{d-1}.

    ``c`` and ``q`` are both shape (..., d), assumed unit-normalized. The
    formula is:

        cos_theta = <c, q>
        theta     = arccos(cos_theta)
        log_c(q)  = (q - cos_theta · c) · theta / sin(theta)

    With theta ≈ 0 the tangent vanishes; we return zeros and avoid a NaN
    from sin(theta) = 0.
    """
    cos_theta = np.clip((c * q).sum(axis=-1, keepdims=True), -1.0 + eps, 1.0 - eps)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    direction = q - cos_theta * c
    safe = np.where(sin_theta > eps, theta / sin_theta, 0.0)
    return direction * safe


def tangent_geodesic_score(
    q: np.ndarray,
    centroids: np.ndarray,
    *,
    lam: float = 0.1,
) -> np.ndarray:
    """Score every centroid with ``s(q, c) = <q, c> - λ · θ²``.

    Returns ``(n_centroids,)``. ``q`` is (d,). ``centroids`` is (n, d).
    Both are assumed unit-normalized; if not, the caller should normalize
    them first (the plan's A1 normalize-on axis).
    """
    cos_theta = np.clip(centroids @ q, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    return cos_theta - lam * (theta ** 2)


def patch_centroid_router_route(centroid_router, lam: float = 0.1) -> None:
    """Monkey-patch a CentroidRouter instance's ``route`` method to use the
    geodesic-aware score. Reversible by re-importing the original method.

    Used by ``run_a1.py`` / ``run_b0.py`` to A/B the change without forking
    the main router code.
    """
    import torch
    from voyager_index._internal.inference.shard_engine import (
        centroid_router as cr_module,
    )

    original_route = cr_module.CentroidRouter.route

    def route_with_tangent(self, q_tokens, *args, **kwargs):
        q_norm = torch.nn.functional.normalize(q_tokens, dim=-1)
        centroid_table = self.centroid_table
        centroid_table_n = torch.nn.functional.normalize(centroid_table, dim=-1)
        cos_theta = (q_norm @ centroid_table_n.T).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.arccos(cos_theta)
        adjusted = cos_theta - lam * (theta ** 2)
        kwargs["scores_override"] = adjusted
        return original_route(self, q_tokens, *args, **kwargs)

    centroid_router.route = route_with_tangent.__get__(centroid_router, type(centroid_router))
    centroid_router._tangent_lam = lam
    log.info("patched CentroidRouter.route with tangent geodesic score (lam=%.3f)", lam)
