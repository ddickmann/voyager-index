"""Thread-safe cache helpers for decoded shard data."""
from __future__ import annotations

import threading
from collections import OrderedDict

import torch

class _ShardLRU:
    """Thread-safe LRU cache for decoded shard data."""

    def __init__(self, max_shards: int = 512):
        self._max = max_shards
        self._data: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, shard_id: int):
        with self._lock:
            if shard_id in self._data:
                self._data.move_to_end(shard_id)
                return self._data[shard_id]
            return None

    def put(self, shard_id: int, emb: torch.Tensor, offsets_t: torch.Tensor, ids_t: torch.Tensor):
        with self._lock:
            if shard_id in self._data:
                self._data.move_to_end(shard_id)
                return
            if len(self._data) >= self._max:
                self._data.popitem(last=False)
            self._data[shard_id] = (emb, offsets_t, ids_t)

