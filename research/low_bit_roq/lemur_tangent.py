"""
LEMUR-tangent proxy features.

Plan reference: Phase B2 / variant B4. Replaces LEMUR's per-token proxy
weights with ``[centroid_id_one_hot · stop_grad, log_c(token).pca_first_k]``
so the IndexFlatIP shortlist becomes geometry-aware.

The integration point is ``shard_engine/_lemur/backends.py`` lines 73-88
(see plan A5 for the file map). This module produces the per-token feature
matrix that the LEMUR builder consumes; nothing else in the LEMUR training
pipeline needs to change.

Design knobs (must be pinned before training):

- ``k_pca``: dimensionality of the per-cluster tangent PCA basis used to
  project ``log_c(x)``. Default 32 — enough to recover most of the
  in-cluster variance while keeping the per-token feature small (≤ 4× the
  baseline LEMUR feature width).
- ``norm_bucket``: optional bucket id derived from ``||x||`` (B3 variant).
  Adds a 4-dim one-hot to the feature vector. Cheap and orthogonal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .spherical_kmeans import l2_normalize
from .tangent_query import log_map_unit_sphere

log = logging.getLogger(__name__)


@dataclass
class LemurTangentConfig:
    n_clusters: int
    k_pca: int = 32
    norm_buckets: Optional[tuple[float, ...]] = None
    """Edges for ``np.digitize(||x||, edges)``. e.g. (0.5, 0.8, 1.1)."""
    eps: float = 1e-7


class LemurTangentFeaturizer:
    """Fits a per-cluster PCA basis once at index time, then emits per-token
    features the LEMUR builder feeds into IndexFlatIP."""

    def __init__(self, cfg: LemurTangentConfig):
        self.cfg = cfg
        self.centroids: np.ndarray | None = None
        self.bases: np.ndarray | None = None  # (n_clusters, d, k_pca)

    # ------------------------------------------------------------------
    def fit(self, tokens: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> None:
        if assignments.shape[0] != tokens.shape[0]:
            raise ValueError("assignments / tokens shape mismatch")
        d = tokens.shape[1]
        k = self.cfg.n_clusters
        kp = min(self.cfg.k_pca, d - 1)
        self.centroids = l2_normalize(centroids.astype(np.float32))
        self.bases = np.zeros((k, d, kp), dtype=np.float32)
        x_norm = l2_normalize(tokens.astype(np.float32))

        for c in range(k):
            mask = assignments == c
            if mask.sum() < kp + 2:
                self.bases[c, :, : min(kp, d)] = np.eye(d, dtype=np.float32)[:, : min(kp, d)]
                continue
            tangents = log_map_unit_sphere(self.centroids[c][None, :], x_norm[mask])
            tangents -= tangents.mean(axis=0, keepdims=True)
            cov = tangents.T @ tangents / max(1, tangents.shape[0] - 1)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                eigvecs = np.eye(d, dtype=np.float32)
            self.bases[c] = eigvecs[:, ::-1][:, :kp].astype(np.float32)

        log.info("fit per-cluster PCA basis: K=%d, k_pca=%d, d=%d", k, kp, d)

    # ------------------------------------------------------------------
    def featurize(self, tokens: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        if self.centroids is None or self.bases is None:
            raise RuntimeError("call fit() first")
        cfg = self.cfg
        d = tokens.shape[1]
        x_norm = l2_normalize(tokens.astype(np.float32))
        norms = np.linalg.norm(tokens, axis=1).astype(np.float32)
        kp = self.bases.shape[2]

        feature_dim = cfg.n_clusters + kp
        if cfg.norm_buckets:
            feature_dim += len(cfg.norm_buckets) + 1

        out = np.zeros((tokens.shape[0], feature_dim), dtype=np.float32)
        out[np.arange(tokens.shape[0]), assignments] = 1.0  # cluster-id one-hot
        for c in range(cfg.n_clusters):
            mask = assignments == c
            if not mask.any():
                continue
            tangents = log_map_unit_sphere(
                self.centroids[c][None, :], x_norm[mask]
            )
            out[mask, cfg.n_clusters : cfg.n_clusters + kp] = tangents @ self.bases[c]
        if cfg.norm_buckets:
            buckets = np.digitize(norms, cfg.norm_buckets)
            base = cfg.n_clusters + kp
            out[np.arange(tokens.shape[0]), base + buckets] = 1.0
        return out
