"""
Ternary (1.58-bit) quantizer.

Plan reference: Phase A2.5. The Python side here:

- fits a per-group threshold τ from a fraction-of-std heuristic (or, when
  ``anisotropic.fit_ternary_codebook`` is available, the closed-form
  anisotropic minimizer).
- emits two bit-planes per token: ``sign_plane`` (1 if ``x > 0``) and
  ``nonzero_plane`` (1 if ``|x| > τ``). Reconstruction is
  ``decoded[i] = (sign_plane[i] ? +1 : -1) * (nonzero_plane[i] ? scale : 0)``.
- packs the two planes into the layout the Triton kernel consumes
  (``(N, dim/8 + dim/8)`` bytes per token, identical word-aligned packing
  to the existing 1-bit asymmetric path).

Storage cost per token of dim D:
    ``2 * D / 8`` payload bytes + ``ceil(D / group_size) * 4`` bytes meta
    (one float32 scale per group). For D=128, group_size=16:
    32 payload + 32 meta = 64 bytes/token.

For the same D and config, current grouped 2-bit RoQ uses
    ``(D * 2 / 8) + (D / group_size) * 8`` = 32 + 64 = 96 bytes/token.

So ternary is ~33% smaller than 2-bit on disk while running on the same
popcount-only kernel architecture as 1-bit (see
``kernels/triton_roq_ternary.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TernaryConfig:
    dim: int
    group_size: int | None = 16
    rotate: bool = True
    rotation_rounds: int = 3
    rotation_block_size: int | None = None
    tau_frac: float = 0.5
    """Threshold τ as a fraction of per-group std. Sweep ∈ {0.3, 0.5, 0.7, 1.0}.

    Too-small τ collapses the encoder to pure 1-bit (almost everything
    nonzero → sign-only). Too-large τ wastes the 1.58 bits because most
    coords land in the zero bucket — the kernel still has to fetch them but
    they contribute zero to the dot product. The 0.5 default is the
    closed-form anisotropic optimum for an iid-Gaussian-after-FWHT
    distribution and reproduces well empirically.
    """
    seed: int = 42
    fit_method: str = "tau_frac"  # "tau_frac" | "anisotropic"


def _fwht_padded_dim(dim: int, block_size: int | None) -> int:
    if block_size is None:
        block_size = 1
        while block_size < dim:
            block_size *= 2
    return ((dim + block_size - 1) // block_size) * block_size, block_size


def _fwht_rotator(dim: int, rounds: int, block_size: int, seed: int):
    """Match the FWHT rotation used by RotationalQuantizer so that ternary
    is comparable cell-by-cell with the existing 1-bit / 2-bit paths."""
    from voyager_index._internal.inference.quantization.rotational import (
        FastWalshHadamard,
    )

    return FastWalshHadamard(dim=dim, num_rounds=rounds, block_size=block_size, seed=seed)


@dataclass
class TernaryQuantizer:
    """Ternary (1.58-bit) encoder.

    The encoder is intentionally numpy-only (cpu) for index-time use; the
    rerank-time decode is run by the Triton kernel in
    ``kernels/triton_roq_ternary.py``. The numpy ``decode`` in this class
    is a reference implementation used by ``distortion_bench`` and tests.
    """

    config: TernaryConfig
    _rotator: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config.rotate:
            padded_dim, block_size = _fwht_padded_dim(
                self.config.dim, self.config.rotation_block_size
            )
            self._padded_dim = padded_dim
            self._rotator = _fwht_rotator(
                dim=self.config.dim,
                rounds=self.config.rotation_rounds,
                block_size=block_size,
                seed=self.config.seed,
            )
        else:
            self._padded_dim = self.config.dim
            self._rotator = None
        gs = self.config.group_size or self._padded_dim
        if gs % 32 != 0:
            raise ValueError(
                f"group_size must be a multiple of 32 (one 32-bit popcount word per group); "
                f"got group_size={gs}. Use 32, 64, 128, ..."
            )
        if self._padded_dim % gs != 0:
            raise ValueError(
                f"padded_dim={self._padded_dim} must be divisible by group_size={gs}"
            )

    @property
    def padded_dim(self) -> int:
        return self._padded_dim

    @property
    def n_groups(self) -> int:
        gs = self.config.group_size or self._padded_dim
        return self._padded_dim // gs

    @property
    def group_size(self) -> int:
        return self.config.group_size or self._padded_dim

    # ------------------------------------------------------------------
    def quantize(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if x.ndim != 2:
            raise ValueError(f"expected (N, D), got {x.shape}")
        if x.shape[1] != self.config.dim:
            raise ValueError(f"dim mismatch: cfg={self.config.dim} got={x.shape[1]}")
        rotated = self._rotate(x)
        gs = self.group_size
        n_groups = self.n_groups
        n = rotated.shape[0]
        grouped = rotated.reshape(n, n_groups, gs)

        if self.config.fit_method == "anisotropic":
            from .anisotropic import fit_ternary_codebook

            tau, scales = fit_ternary_codebook(grouped)
        else:
            std_per_group = grouped.std(axis=2, ddof=0) + 1e-8
            tau = std_per_group * self.config.tau_frac
            mask_init = np.abs(grouped) > tau[..., None]
            num = (np.abs(grouped) * mask_init).sum(axis=2)
            den = mask_init.sum(axis=2).clip(min=1)
            scales = (num / den).astype(np.float32)

        sign = (rotated > 0).astype(np.uint8)
        nonzero = (np.abs(rotated) > np.repeat(tau, gs, axis=1)).astype(np.uint8)

        sign_packed = np.packbits(sign, axis=1, bitorder="little")
        nonzero_packed = np.packbits(nonzero, axis=1, bitorder="little")

        return {
            "sign_plane": sign_packed,
            "nonzero_plane": nonzero_packed,
            "scales": scales,
            "tau": tau.astype(np.float32),
            "rotated": rotated,
            "norms_sq": (rotated ** 2).sum(axis=1).astype(np.float32),
        }

    # ------------------------------------------------------------------
    def decode(self, encoded: dict[str, np.ndarray]) -> np.ndarray:
        sign_packed = encoded["sign_plane"]
        nonzero_packed = encoded["nonzero_plane"]
        scales = encoded["scales"]
        n = sign_packed.shape[0]
        sign = np.unpackbits(sign_packed, axis=1, bitorder="little")[:, : self._padded_dim]
        nonzero = np.unpackbits(nonzero_packed, axis=1, bitorder="little")[:, : self._padded_dim]
        signed = (2.0 * sign.astype(np.float32) - 1.0) * nonzero.astype(np.float32)
        gs = self.group_size
        scales_per_dim = np.repeat(scales, gs, axis=1)
        return (signed * scales_per_dim).astype(np.float32)

    # ------------------------------------------------------------------
    def _rotate(self, x: np.ndarray) -> np.ndarray:
        if self._rotator is None:
            return x.astype(np.float32, copy=False)
        import torch

        return self._rotator.forward(torch.from_numpy(x).float()).cpu().numpy()

    # ------------------------------------------------------------------
    def encode_query_bit_planes(
        self, queries: np.ndarray, query_bits: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asymmetric query encoding: rotate, scalar-quantize to ``query_bits``
        levels, return (planes, scales, offsets) suitable for the Triton
        ternary kernel.

        Mirrors the existing 1-bit asymmetric query construction in
        ``RotationalQuantizer.build_1bit_query_triton_inputs``.
        """
        if query_bits not in (4, 6, 8):
            raise ValueError("query_bits ∈ {4, 6, 8}")
        rotated = self._rotate(queries)
        levels = float((1 << query_bits) - 1)
        min_vals = rotated.min(axis=1)
        max_vals = rotated.max(axis=1)
        ranges = np.where((max_vals - min_vals) < 1e-6, 1.0, max_vals - min_vals)
        scales = ranges / levels
        quant = np.round(
            (rotated - min_vals[:, None]) / scales[:, None]
        ).clip(0, levels).astype(np.uint8)

        planes = []
        for bit in range(query_bits):
            plane = ((quant >> bit) & 0x01).astype(np.uint8)
            planes.append(np.packbits(plane, axis=1))
        return np.stack(planes, axis=1), scales.astype(np.float32), min_vals.astype(np.float32)
