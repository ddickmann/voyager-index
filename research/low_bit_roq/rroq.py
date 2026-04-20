"""
rroq: Riemannian-aware low-bit ROQ (Phase B3).

Pinned design knobs (from the plan):

- Centroid codebook size K ∈ {256, 1024, 4096, 16384}. Disk overhead =
  ``ceil(log2(K)) / dim`` bits per coord, additive to the body bits.
- Tangent basis ∈ {identity, FWHT, per-centroid PCA}. Per-centroid PCA
  (B5) is exposed as a dedicated `PerClusterPcaBasis`; identity / FWHT are
  the cheap baselines.
- Residual quantization: per-group min/max with anisotropic loss + asym
  query (composes with A2 / A3).
- Approximate scoring (kernel C2):
    <q, exp_c(r̂)> ≈ cos(||r̂||)·<q, c> + sinc(||r̂||)·<q, r̂_ambient>

This module hosts the offline encoder (B3 / 2A) and a numpy-only reference
scorer (B3 / 2B). The fast Triton kernel (B3 / 2C) reuses the
asymmetric-2-bit kernel from ``kernels/triton_roq_2bit_asym.py`` over the
projected residual codes — only the "compose <q,c> + tangent term"
post-processing is rroq-specific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .anisotropic import fit_anisotropic_min_max
from .per_cluster_pca import PerClusterPcaBasis
from .spherical_kmeans import (
    SphericalKMeansConfig,
    cluster as spherical_cluster,
    l2_normalize,
)
from .tangent_query import log_map_unit_sphere

log = logging.getLogger(__name__)


@dataclass
class RroqConfig:
    K: int = 4096
    """Centroid codebook size."""
    bits: float = 2
    """Bits per residual coord. 2 = standard 2-bit; 1.58 (passed as
    'ternary') routes through ``ternary.TernaryQuantizer`` for the
    residual."""
    group_size: int = 16
    tangent_basis: str = "fwht"  # "identity" | "fwht" | "pca"
    spherical_kmeans_iter: int = 25
    seed: int = 42
    eta: float = 4.0


@dataclass
class RroqEncoded:
    centroid_id: np.ndarray            # (N,) int32 (or smaller)
    code_payload: dict[str, np.ndarray]  # output of the underlying quantizer
    norms: np.ndarray                  # (N,) float32, ||x||
    centroids: np.ndarray              # (K, dim) float32
    config: RroqConfig
    pca_basis: Optional[PerClusterPcaBasis] = None
    extras: dict = field(default_factory=dict)


class RroqEncoder:
    def __init__(self, cfg: RroqConfig):
        self.cfg = cfg
        self._pca: Optional[PerClusterPcaBasis] = None
        self._centroids: Optional[np.ndarray] = None
        self._assignments: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, tokens: np.ndarray) -> None:
        cfg = self.cfg
        x_norm = l2_normalize(tokens.astype(np.float32))
        sk_cfg = SphericalKMeansConfig(
            n_clusters=cfg.K, n_iter=cfg.spherical_kmeans_iter, seed=cfg.seed
        )
        self._centroids, self._assignments = spherical_cluster(x_norm, sk_cfg)
        if cfg.tangent_basis == "pca":
            self._pca = PerClusterPcaBasis(K=cfg.K, k_components=tokens.shape[1])
            self._pca.fit(tokens, self._assignments, self._centroids)

    # ------------------------------------------------------------------
    def encode(self, tokens: np.ndarray) -> RroqEncoded:
        if self._centroids is None or self._assignments is None:
            raise RuntimeError("call fit() first")
        cfg = self.cfg
        x = tokens.astype(np.float32)
        x_norm = l2_normalize(x)
        sims = x_norm @ self._centroids.T
        assign = sims.argmax(axis=1).astype(np.int32)

        c_per_token = self._centroids[assign]
        tangent = log_map_unit_sphere(c_per_token, x_norm)
        if cfg.tangent_basis == "pca":
            assert self._pca is not None
            projected = self._pca.project_per_token(tangent, assign)
        elif cfg.tangent_basis == "fwht":
            from voyager_index._internal.inference.quantization.rotational import (
                FastWalshHadamard,
            )
            import torch

            rot = FastWalshHadamard(dim=tangent.shape[1], num_rounds=3, block_size=tangent.shape[1], seed=cfg.seed)
            projected = rot.forward(torch.from_numpy(tangent).float()).cpu().numpy()
        else:
            projected = tangent

        if cfg.bits == 2:
            grouped = projected.reshape(projected.shape[0], -1, cfg.group_size)
            scales, offsets = fit_anisotropic_min_max(grouped, bits=2, eta=cfg.eta)
            codes = np.round(
                (grouped - offsets[..., None]) / np.where(scales[..., None] < 1e-6, 1.0, scales[..., None])
            ).clip(0, 3).astype(np.uint8)
            payload = {
                "codes": codes,
                "scales": scales,
                "offsets": offsets,
                "code_sums": codes.astype(np.float32).sum(axis=2),
            }
        elif int(cfg.bits * 100) == 158:
            from .ternary import TernaryConfig, TernaryQuantizer

            tq = TernaryQuantizer(
                TernaryConfig(
                    dim=projected.shape[1],
                    group_size=cfg.group_size,
                    rotate=False,
                    seed=cfg.seed,
                    fit_method="anisotropic",
                )
            )
            payload = tq.quantize(projected)
        else:
            raise ValueError(f"unsupported rroq bits={cfg.bits}")

        return RroqEncoded(
            centroid_id=assign,
            code_payload=payload,
            norms=np.linalg.norm(x, axis=1).astype(np.float32),
            centroids=self._centroids,
            config=cfg,
            pca_basis=self._pca,
        )

    # ------------------------------------------------------------------
    def reference_score(
        self, queries: np.ndarray, encoded: RroqEncoded
    ) -> np.ndarray:
        """Slow numpy scorer used by B3 / 2B — dequantizes and applies
        the formula ``<q, exp_c(r̂)>`` exactly. Used in distortion bench
        and as a parity oracle for the fast kernel."""
        cfg = self.cfg
        q = queries.astype(np.float32)
        q_norm = l2_normalize(q)
        c = encoded.centroids[encoded.centroid_id]

        if cfg.bits == 2:
            payload = encoded.code_payload
            grouped_codes = payload["codes"].astype(np.float32)
            scales = payload["scales"]
            offsets = payload["offsets"]
            r_proj = grouped_codes * scales[..., None] + offsets[..., None]
            r_proj = r_proj.reshape(r_proj.shape[0], -1)
        else:
            from .ternary import TernaryConfig, TernaryQuantizer

            tq = TernaryQuantizer(
                TernaryConfig(
                    dim=encoded.centroids.shape[1],
                    group_size=cfg.group_size,
                    rotate=False,
                    fit_method="anisotropic",
                )
            )
            r_proj = tq.decode(encoded.code_payload)

        if cfg.tangent_basis == "pca":
            r_amb = encoded.pca_basis.unproject_per_token(r_proj, encoded.centroid_id)  # type: ignore[union-attr]
        elif cfg.tangent_basis == "fwht":
            from voyager_index._internal.inference.quantization.rotational import (
                FastWalshHadamard,
            )
            import torch

            rot = FastWalshHadamard(
                dim=r_proj.shape[1], num_rounds=3, block_size=r_proj.shape[1], seed=cfg.seed
            )
            r_amb = rot.forward(torch.from_numpy(r_proj).float()).cpu().numpy()
        else:
            r_amb = r_proj

        r_norm = np.linalg.norm(r_amb, axis=1, keepdims=True) + 1e-12
        sin_t = np.sinc(r_norm / np.pi)
        cos_t = np.cos(r_norm)

        qc = q_norm @ c.T
        qr = q_norm @ r_amb.T
        approx = (cos_t.T * qc) + (sin_t.T * qr)
        return approx
