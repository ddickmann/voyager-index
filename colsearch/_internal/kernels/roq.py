"""
Lazy RoQ exports that only require Triton when the kernels are actually used.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from .triton_roq import roq_maxsim_1bit, roq_maxsim_2bit, roq_maxsim_4bit, roq_maxsim_8bit
    ROQ_TRITON_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local Triton install
    ROQ_TRITON_AVAILABLE = False
    logger.debug("Triton RoQ unavailable: %s", exc)

    def _missing_roq(*args, **kwargs):
        raise ImportError(
            "RoQ Triton kernels are unavailable. Install colsearch[gpu] to use them."
        )

    roq_maxsim_1bit = _missing_roq
    roq_maxsim_2bit = _missing_roq
    roq_maxsim_4bit = _missing_roq
    roq_maxsim_8bit = _missing_roq


__all__ = [
    "ROQ_TRITON_AVAILABLE",
    "roq_maxsim_1bit",
    "roq_maxsim_2bit",
    "roq_maxsim_4bit",
    "roq_maxsim_8bit",
]
