"""Manifest and metadata types for shard storage."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .common import atomic_json_write

@dataclass
class ShardMeta:
    shard_id: int
    num_docs: int
    total_tokens: int
    centroid_ids: List[int]
    byte_size: int
    file_name: str
    compression: str
    p50_tokens: float = 0.0
    p95_tokens: float = 0.0
    shard_max_tokens: int = 0

@dataclass
class StoreManifest:
    num_shards: int
    num_docs: int
    dim: int
    total_tokens: int
    avg_tokens_per_chunk: float
    p50_tokens: float
    p95_tokens: float
    compression: str
    shards: List[ShardMeta]
    global_target_len: int = 0
    version: int = 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "StoreManifest":
        with open(path) as f:
            d = json.load(f)
        d["shards"] = [ShardMeta(**s) for s in d["shards"]]
        d.setdefault("global_target_len", 0)
        d.setdefault("version", 1)
        return cls(**d)

@dataclass
class DocMeta:
    doc_id: int
    shard_id: int
    local_offset_start: int
    local_offset_end: int
    row_index: int

