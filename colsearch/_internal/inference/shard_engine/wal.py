"""WAL (Write-Ahead Log) for the shard engine.

Binary format per entry:
  [HEADER: magic(4) + version(1) + entry_len(4)] [PAYLOAD] [CRC32(4)]

The shard WAL uses its own CRC scope (header + payload) which differs from
gem_wal.py (payload-only CRC). The two formats are not wire-compatible.
"""
from __future__ import annotations

from ._wal import WalEntry, WalOp, WalReader, WalWriter
from ._wal.common import CRC_SIZE, HEADER_FMT, HEADER_SIZE, WAL_MAGIC, WAL_VERSION

__all__ = [
    "CRC_SIZE",
    "HEADER_FMT",
    "HEADER_SIZE",
    "WAL_MAGIC",
    "WAL_VERSION",
    "WalEntry",
    "WalOp",
    "WalReader",
    "WalWriter",
]
