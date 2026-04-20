"""
Triton kernels for the low-bit ROQ research module.

Lazy imports so the module loads on hosts without Triton (CI, dev laptops):

    >>> from research.low_bit_roq.kernels import roq_maxsim_ternary, roq_maxsim_2bit_asym

The Python-side encoders (``ternary.TernaryQuantizer``, the
2-bit-asym builders living in ``ternary.py`` and ``rroq.py``) work
without Triton; only the kernel calls require GPU + Triton.
"""

from __future__ import annotations

from typing import Any

__all__ = ["roq_maxsim_ternary", "roq_maxsim_2bit_asym"]


def __getattr__(name: str) -> Any:
    if name == "roq_maxsim_ternary":
        from .triton_roq_ternary import roq_maxsim_ternary

        return roq_maxsim_ternary
    if name == "roq_maxsim_2bit_asym":
        from .triton_roq_2bit_asym import roq_maxsim_2bit_asym

        return roq_maxsim_2bit_asym
    raise AttributeError(name)
