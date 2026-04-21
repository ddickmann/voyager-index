"""Runtime capability probes and summaries for the shard engine."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict

import torch

from .lemur_router import FAISS_AVAILABLE, OFFICIAL_LEMUR_AVAILABLE


def _can_allocate_pinned_memory() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.empty((1, 1), dtype=torch.float16, pin_memory=True)
    except RuntimeError:
        return False
    return bool(getattr(probe, "is_pinned", lambda: False)())


def detect_runtime_capabilities() -> Dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "faiss_available": FAISS_AVAILABLE,
        "official_lemur_available": OFFICIAL_LEMUR_AVAILABLE,
        "native_shard_engine_importable": importlib.util.find_spec("latence_shard_engine") is not None,
        "pinned_memory_allocatable": _can_allocate_pinned_memory(),
    }


__all__ = ["detect_runtime_capabilities"]
