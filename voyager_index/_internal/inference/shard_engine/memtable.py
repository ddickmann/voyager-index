"""In-memory write buffer (MemTable) for the shard engine.

Holds recently added/upserted documents that haven't been flushed to
sealed shards yet. Supports brute-force MaxSim search so that new
documents are immediately searchable.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MemTable:
    """Thread-safe in-memory document buffer with brute-force search."""

    def __init__(self, dim: int, device: str = "cuda"):
        self._dim = dim
        self._device = device
        self._lock = threading.RLock()
        self._docs: Dict[int, np.ndarray] = {}
        self._payloads: Dict[int, dict] = {}
        self._tombstones: Set[int] = set()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._docs)

    @property
    def tombstone_count(self) -> int:
        with self._lock:
            return len(self._tombstones)

    def insert(self, doc_id: int, vectors: np.ndarray,
               payload: Optional[dict] = None) -> None:
        with self._lock:
            self._docs[doc_id] = vectors.astype(np.float32)
            if payload:
                self._payloads[doc_id] = payload
            self._tombstones.discard(doc_id)

    def delete(self, doc_id: int) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._payloads.pop(doc_id, None)
            self._tombstones.add(doc_id)

    def upsert(self, doc_id: int, vectors: np.ndarray,
               payload: Optional[dict] = None) -> None:
        self.insert(doc_id, vectors, payload)

    def is_tombstoned(self, doc_id: int) -> bool:
        with self._lock:
            return doc_id in self._tombstones

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Brute-force MaxSim over memtable documents."""
        with self._lock:
            if not self._docs:
                return []
            doc_ids = list(self._docs.keys())
            doc_vecs = list(self._docs.values())

        from .scorer import brute_force_maxsim
        result_ids, result_scores = brute_force_maxsim(
            query, doc_vecs, doc_ids, self._dim, k=k, device=self._device,
        )
        return list(zip(result_ids, result_scores))

    def drain(self) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Set[int]]:
        """Remove all data from the memtable and return it."""
        with self._lock:
            docs = dict(self._docs)
            payloads = dict(self._payloads)
            tombstones = set(self._tombstones)
            self._docs.clear()
            self._payloads.clear()
            self._tombstones.clear()
            return docs, payloads, tombstones

    def snapshot(self) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Set[int]]:
        """Return a copy of current state without draining."""
        with self._lock:
            return dict(self._docs), dict(self._payloads), set(self._tombstones)
