"""Typed WAL operations and replay records."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

class WalOp(IntEnum):
    INSERT = 0
    DELETE = 1
    UPSERT = 2
    UPDATE_PAYLOAD = 3

@dataclass
class WalEntry:
    op: WalOp
    doc_id: int
    vectors: Optional[np.ndarray] = None
    payload: Optional[dict] = None

