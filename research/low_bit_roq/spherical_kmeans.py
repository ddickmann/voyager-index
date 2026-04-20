"""
Spherical Lloyd k-means.

Plan reference: Phase B1 (and the A1 ``pre-cluster L2-normalize`` axis).

For already-L2-normalized inputs, spherical Lloyd is mathematically
Lloyd-update-equivalent to Euclidean k-means; the practical difference is
that we (a) re-normalize centroids back to the unit sphere after every
update, and (b) use cosine distance for assignment. The plan calls this
out explicitly: most of the "spherical" win in the routing branch comes
from ``normalize first``, not from the spherical update — B1 is the
geometry-only delta after subtracting A1's normalize-on contribution.

This module is intentionally a small numpy implementation to avoid
pulling in faiss for the simple unit-sphere case. For production, swap
``cluster`` for the existing FAISS-backed
``_lemur/ann.py`` / ``centroid_router.py`` paths once the routing axis is
proven to win.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SphericalKMeansConfig:
    n_clusters: int
    n_iter: int = 25
    tol: float = 1e-4
    seed: int = 0
    init: str = "kpp"  # "kpp" (k-means++) | "random"
    min_cluster_size: int = 1


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _kpp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    idx = [int(rng.integers(0, n))]
    closest_sq = np.full(n, np.inf, dtype=np.float32)
    for _ in range(1, k):
        last = x[idx[-1]]
        d_sq = ((x - last) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, d_sq)
        probs = closest_sq / closest_sq.sum()
        idx.append(int(rng.choice(n, p=probs)))
    return x[np.asarray(idx)]


def cluster(
    x: np.ndarray,
    cfg: SphericalKMeansConfig,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroids of shape (k, d), assignments of shape (n,)).

    ``x`` is L2-normalized internally.
    """
    rng = np.random.default_rng(cfg.seed)
    x = l2_normalize(x.astype(np.float32))
    n, d = x.shape
    k = cfg.n_clusters
    if k > n:
        raise ValueError(f"n_clusters={k} > n={n}")
    if cfg.init == "kpp":
        centroids = _kpp_init(x, k, rng)
    else:
        centroids = x[rng.choice(n, size=k, replace=False)]
    centroids = l2_normalize(centroids)
    assign = np.zeros(n, dtype=np.int64)

    prev_inertia = np.inf
    for it in range(cfg.n_iter):
        sims = x @ centroids.T
        assign_new = sims.argmax(axis=1)
        if it > 0 and (assign_new != assign).sum() < cfg.tol * n:
            log.debug("spherical_kmeans converged at iter %d", it)
            assign = assign_new
            break
        assign = assign_new

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = assign == c
            count = mask.sum()
            if count < cfg.min_cluster_size:
                far = (1.0 - sims[np.arange(n), assign]).argmax()
                new_centroids[c] = x[far]
                assign[far] = c
            else:
                if sample_weight is not None:
                    new_centroids[c] = (sample_weight[mask, None] * x[mask]).sum(axis=0)
                else:
                    new_centroids[c] = x[mask].sum(axis=0)
        centroids = l2_normalize(new_centroids)
        inertia = float((1.0 - sims[np.arange(n), assign]).sum())
        if abs(prev_inertia - inertia) < cfg.tol:
            break
        prev_inertia = inertia

    return centroids, assign
