"""Persisted router state and candidate-plan data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass(slots=True)
class CandidatePlan:
    doc_ids: List[int]
    shard_ids: List[int]
    by_shard: Dict[int, List[int]]
    generation: int
    post_tombstone_count: int

@dataclass(slots=True)
class RouterState:
    generation: int = 0
    ann_backend: str = "faiss_flat_ip"
    feature_dim: int = 0
    backend_name: str = "official_lemur"
    dirty_ops_count: int = 0
    dirty_doc_ratio: float = 0.0
    dirty_shard_ratio: float = 0.0
    live_docs: int = 0
    total_docs: int = 0
    max_age_hours: float = 0.0

