"""Public compatibility facade for the shard-engine LEMUR router."""
from __future__ import annotations

from ._lemur.common import FAISS_AVAILABLE, OFFICIAL_LEMUR_AVAILABLE
from ._lemur.core import LemurRouter
from ._lemur.state import CandidatePlan, RouterState

__all__ = [
    "CandidatePlan",
    "FAISS_AVAILABLE",
    "LemurRouter",
    "OFFICIAL_LEMUR_AVAILABLE",
    "RouterState",
]
