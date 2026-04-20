"""
Mixed-precision rroq + token-salience interaction.

Plan reference: Phase B4. Folds A4 token-salience into the rroq promotion
criterion: high-salience tokens get 4-bit residual, low-salience get 2-bit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    promote_fraction: float = 0.10
    promote_signal: str = "salience"  # "salience" | "residual_norm" | "centroid_spread"
    high_bits: int = 4
    low_bits: int = 2


def select_promoted_tokens(
    salience: np.ndarray | None = None,
    residual_norms: np.ndarray | None = None,
    centroid_spread: np.ndarray | None = None,
    *,
    cfg: MixedPrecisionConfig,
) -> np.ndarray:
    """Returns boolean mask of length N — True for tokens promoted to high
    bits. The signal is selected by ``cfg.promote_signal``."""
    if cfg.promote_signal == "salience":
        if salience is None:
            raise ValueError("salience signal requires salience array")
        signal = salience
    elif cfg.promote_signal == "residual_norm":
        if residual_norms is None:
            raise ValueError("residual_norm signal requires residual_norms array")
        signal = residual_norms
    elif cfg.promote_signal == "centroid_spread":
        if centroid_spread is None:
            raise ValueError("centroid_spread signal requires centroid_spread array")
        signal = centroid_spread
    else:
        raise ValueError(f"unknown promote_signal={cfg.promote_signal}")
    threshold = float(np.quantile(signal, 1.0 - cfg.promote_fraction))
    mask = signal >= threshold
    log.info(
        "mixed precision: promoted %d/%d tokens (%.1f%%) to %d-bit",
        int(mask.sum()),
        mask.size,
        100 * mask.mean(),
        cfg.high_bits,
    )
    return mask
