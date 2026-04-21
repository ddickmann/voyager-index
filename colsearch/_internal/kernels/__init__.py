"""
Kernel entrypoints for colsearch internals.

This package is internal. Use `colsearch.kernels` for the supported public
API.
"""

from .maxsim import TRITON_AVAILABLE, compute_maxsim_token_coverage_matrix, fast_colbert_scores
from .roq import (
    roq_maxsim_1bit,
    roq_maxsim_2bit,
    roq_maxsim_4bit,
    roq_maxsim_8bit,
)

__all__ = [
    "TRITON_AVAILABLE",
    "compute_maxsim_token_coverage_matrix",
    "fast_colbert_scores",
    "roq_maxsim_1bit",
    "roq_maxsim_2bit",
    "roq_maxsim_4bit",
    "roq_maxsim_8bit",
]
