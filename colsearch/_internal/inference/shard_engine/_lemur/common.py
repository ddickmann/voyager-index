"""Shared imports and optional backend flags for LEMUR routing internals."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    FAISS_AVAILABLE = False

try:
    from .._lemur_vendor import Lemur as _OfficialLemur
    OFFICIAL_LEMUR_AVAILABLE = True
except Exception:
    try:
        from lemur import Lemur as _OfficialLemur  # type: ignore
        OFFICIAL_LEMUR_AVAILABLE = True
    except Exception:  # pragma: no cover - optional dependency
        _OfficialLemur = None
        OFFICIAL_LEMUR_AVAILABLE = False
