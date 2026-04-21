"""Shared constants for shard-engine WAL internals."""
from __future__ import annotations

import json
import logging
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WAL_MAGIC = b"SWAL"
WAL_VERSION = 1

HEADER_FMT = "<4sBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CRC_SIZE = 4

_BATCH_FLUSH_THRESHOLD = 1000
_REPLAY_CHUNK_SIZE = 1 << 20
