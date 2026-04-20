"""
Per-cluster PCA basis (OPQ-style).

Plan reference: Phase B5. Run as a standalone ablation against B3 with
identity basis to quantify the basis-only contribution to rroq's win.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .spherical_kmeans import l2_normalize
from .tangent_query import log_map_unit_sphere

log = logging.getLogger(__name__)


@dataclass
class PerClusterPcaBasis:
    K: int
    k_components: int
    bases: np.ndarray | None = None  # (K, d, k_components)
    centroids: np.ndarray | None = None  # (K, d)

    def fit(self, tokens: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> None:
        d = tokens.shape[1]
        kp = min(self.k_components, d)
        self.k_components = kp
        self.centroids = l2_normalize(centroids.astype(np.float32))
        self.bases = np.zeros((self.K, d, kp), dtype=np.float32)
        x_norm = l2_normalize(tokens.astype(np.float32))

        for c in range(self.K):
            mask = assignments == c
            if mask.sum() < kp + 2:
                self.bases[c] = np.eye(d, dtype=np.float32)[:, :kp]
                continue
            tangents = log_map_unit_sphere(self.centroids[c][None, :], x_norm[mask])
            tangents -= tangents.mean(axis=0, keepdims=True)
            cov = tangents.T @ tangents / max(1, tangents.shape[0] - 1)
            try:
                _, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                eigvecs = np.eye(d, dtype=np.float32)
            self.bases[c] = eigvecs[:, ::-1][:, :kp].astype(np.float32)

    def project_per_token(self, tangents: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        if self.bases is None:
            raise RuntimeError("call fit() first")
        out = np.zeros((tangents.shape[0], self.k_components), dtype=np.float32)
        for c in range(self.K):
            mask = assignments == c
            if not mask.any():
                continue
            out[mask] = tangents[mask] @ self.bases[c]
        return out

    def unproject_per_token(self, projected: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        if self.bases is None:
            raise RuntimeError("call fit() first")
        d = self.bases.shape[1]
        out = np.zeros((projected.shape[0], d), dtype=np.float32)
        for c in range(self.K):
            mask = assignments == c
            if not mask.any():
                continue
            out[mask] = projected[mask] @ self.bases[c].T
        return out
