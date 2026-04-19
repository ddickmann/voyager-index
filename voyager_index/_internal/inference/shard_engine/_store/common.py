"""Shared imports and helpers for shard storage internals."""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

try:
    from voyager_index._internal.inference.index_core.io_utils import atomic_json_write
except ImportError:
    def atomic_json_write(path, data):
        """Fallback: plain json.dump when io_utils is unavailable."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

try:
    from safetensors.numpy import save_file as st_save_np
    from safetensors.torch import load_file as st_load
    SAFETENSORS_AVAILABLE = True
except Exception:
    SAFETENSORS_AVAILABLE = False
    st_save_np = None
    st_load = None

try:
    from safetensors import safe_open as _st_safe_open
    _MMAP_AVAILABLE = True
except Exception:
    _st_safe_open = None
    _MMAP_AVAILABLE = False

from ..config import Compression

logger = logging.getLogger(__name__)
