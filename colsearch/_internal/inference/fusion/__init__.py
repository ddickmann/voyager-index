"""
Fusion Module

Provides strategies for combining results from multiple search engines.

Supported Strategies:
- Reciprocal Rank Fusion (RRF): Robust, no normalization needed
- Weighted Sum: Flexible with configurable weights
- Max/Min: Simple aggregation
"""

from ..fusion.strategies import (
    fuse_results,
    max_fusion,
    min_fusion,
    normalize_min_max,
    normalize_softmax,
    normalize_z_score,
    reciprocal_rank_fusion,
    weighted_sum_fusion,
)

__all__ = [
    'fuse_results',
    'reciprocal_rank_fusion',
    'weighted_sum_fusion',
    'max_fusion',
    'min_fusion',
    'normalize_min_max',
    'normalize_z_score',
    'normalize_softmax'
]



