"""ShardStore: safetensors-backed shard storage with doc-selective fetch."""
from __future__ import annotations

from ._store.common import SAFETENSORS_AVAILABLE
from ._store.core import ShardStore
from ._store.manifest import DocMeta, ShardMeta, StoreManifest

__all__ = ["DocMeta", "SAFETENSORS_AVAILABLE", "ShardMeta", "ShardStore", "StoreManifest"]
