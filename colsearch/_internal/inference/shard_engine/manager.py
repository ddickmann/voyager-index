"""ShardSegmentManager: production manager for the LEMUR-routed shard engine.

Supports WAL, retrieve, scroll, flush, memtable CRUD, payload filters,
ROQ-4 bit scoring, and optional BM25-hybrid re-ranking.
"""
from __future__ import annotations

from ._manager.common import ShardEngineConfig, atomic_json_write
from ._manager.core import ShardSegmentManager

__all__ = ["ShardEngineConfig", "ShardSegmentManager", "atomic_json_write"]
