"""
Pre-autotune Triton kernels via dummy end-to-end queries at init time.

Runs through the exact same code paths the real rerank uses (maxsim.py wrapper,
triton_maxsim.py bucketing, Triton kernel) so that every first-time cost —
JIT compilation, autotuning, PyTorch lazy init — is paid upfront.

Triton caches compiled kernels on disk, so subsequent process starts are instant.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def _next_power_of_2(n: int, minimum: int = 32) -> int:
    """Round up to the next power of 2, with a floor of `minimum`."""
    v = max(n, minimum)
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


_ROQ_SHAPES_TOKENS = [32, 64, 128]
_ROQ_SHAPES_DOCS = [32, 64, 128]

_MAXSIM_SHAPES_Q_TOKENS = [8, 16, 32]
_MAXSIM_SHAPES_D_TOKENS = [32, 64, 128]


def warmup_triton_kernels(
    device: str = "cuda",
    dim: int = 128,
    include_roq: bool = True,
    include_maxsim: bool = True,
) -> None:
    """Pre-compile Triton kernels for all bucketed shape combinations.

    Calls go through the full public API (maxsim.fast_colbert_scores,
    roq_maxsim_4bit) so every wrapper/bucketing/kernel path is warmed.
    """
    try:
        import torch
    except ImportError:
        return

    if device == "cpu" or not torch.cuda.is_available():
        return

    t0 = time.monotonic()
    compiled = 0

    if include_roq:
        try:
            from .roq import roq_maxsim_4bit, ROQ_TRITON_AVAILABLE
            if ROQ_TRITON_AVAILABLE:
                nb = dim // 2
                for n_d_tokens in _ROQ_SHAPES_TOKENS:
                    for n_docs in _ROQ_SHAPES_DOCS:
                        q_codes = torch.zeros(1, 8, nb, dtype=torch.uint8, device=device)
                        q_meta = torch.zeros(1, 8, 4, dtype=torch.float32, device=device)
                        d_codes = torch.zeros(n_docs, n_d_tokens, nb, dtype=torch.uint8, device=device)
                        d_meta = torch.zeros(n_docs, n_d_tokens, 4, dtype=torch.float32, device=device)
                        d_mask = torch.ones(n_docs, n_d_tokens, dtype=torch.float32, device=device)
                        roq_maxsim_4bit(q_codes, q_meta, d_codes, d_meta, documents_mask=d_mask)
                        compiled += 1
                torch.cuda.synchronize()
                logger.debug("Warmed up %d ROQ 4-bit kernel shapes", compiled)
        except Exception as exc:
            logger.debug("ROQ warmup skipped: %s", exc)

    maxsim_compiled = 0
    if include_maxsim:
        try:
            from .maxsim import fast_colbert_scores as _maxsim_fcs
            n_warmup_docs = 32
            for s in _MAXSIM_SHAPES_Q_TOKENS:
                for t in _MAXSIM_SHAPES_D_TOKENS:
                    q = torch.zeros(1, s, dim, dtype=torch.float32, device=device)
                    d = torch.zeros(n_warmup_docs, t, dim, dtype=torch.float32, device=device)
                    d_mask = torch.ones(n_warmup_docs, t, dtype=torch.float32, device=device)
                    _maxsim_fcs(q, d, documents_mask=d_mask)
                    maxsim_compiled += 1
            torch.cuda.synchronize()
            compiled += maxsim_compiled
            logger.debug("Warmed up %d MaxSim kernel shapes", maxsim_compiled)
        except Exception as exc:
            logger.debug("MaxSim warmup skipped: %s", exc)

    elapsed = time.monotonic() - t0
    if compiled > 0:
        logger.info(
            "Triton kernel warmup: %d shapes compiled in %.1fs",
            compiled, elapsed,
        )
