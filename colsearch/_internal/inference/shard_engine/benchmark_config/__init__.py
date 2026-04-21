"""Compatibility package for benchmark sweep configuration imports."""
from __future__ import annotations

from ..sweep_config import (
    BenchmarkConfig,
    SWEEP_COMPRESSION,
    SWEEP_CORPUS_SIZES,
    SWEEP_K_CANDIDATES,
    SWEEP_LAYOUT,
    SWEEP_MAX_DOCS_EXACT,
    SWEEP_TOP_SHARDS,
    SWEEP_TRANSFER,
)

__all__ = [
    "BenchmarkConfig",
    "SWEEP_COMPRESSION",
    "SWEEP_CORPUS_SIZES",
    "SWEEP_K_CANDIDATES",
    "SWEEP_LAYOUT",
    "SWEEP_MAX_DOCS_EXACT",
    "SWEEP_TOP_SHARDS",
    "SWEEP_TRANSFER",
]
