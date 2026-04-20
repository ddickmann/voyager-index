"""
Filter-aware probing.

Plan reference: cross-cut 3 (also Innovation 1 from
``voyager_index_inno.md``). Per-cluster filter sketches let the router
skip clusters where the filter eliminates >90% of docs. Orthogonal to
all of A and B; major p95 win when applicable.

Sketch: a Bloom-style summary of {cluster_id -> set of valid filter
values} computed once at index time. Per-cluster query: ``frac_passing(c)``
returned in O(1) with a small false-positive rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FilterSketchConfig:
    n_buckets: int = 1024
    n_hashes: int = 3
    skip_cluster_threshold: float = 0.9
    """Skip a cluster if fraction-eliminated > threshold."""


class PerClusterFilterSketch:
    """Holds one Bloom-like per cluster.

    For each (cluster_id, filter_value), set ``buckets[cluster_id, hash_i(value)] += 1``.
    On query: for each (cluster_id, query_filter), look up
    ``buckets[cluster_id, hash_i(query_filter)]`` for all hashes, take min,
    divide by cluster_size for an estimate of fraction-passing.
    """

    def __init__(self, n_clusters: int, cfg: FilterSketchConfig):
        self.n_clusters = n_clusters
        self.cfg = cfg
        self.buckets = np.zeros((n_clusters, cfg.n_buckets), dtype=np.int32)
        self.cluster_size = np.zeros(n_clusters, dtype=np.int32)

    def _hash(self, value: int, idx: int) -> int:
        return int((value * (1315423911 + idx * 2654435761)) % self.cfg.n_buckets)

    def fit(self, doc_to_cluster: np.ndarray, doc_to_filter: np.ndarray) -> None:
        for doc, c in enumerate(doc_to_cluster):
            self.cluster_size[c] += 1
            value = int(doc_to_filter[doc])
            for h in range(self.cfg.n_hashes):
                self.buckets[c, self._hash(value, h)] += 1

    def fraction_passing(self, cluster_id: int, query_filter: int) -> float:
        counts = [
            self.buckets[cluster_id, self._hash(query_filter, h)]
            for h in range(self.cfg.n_hashes)
        ]
        size = max(1, int(self.cluster_size[cluster_id]))
        return float(min(counts)) / size

    def select_clusters(self, candidate_clusters: Iterable[int], query_filter: int) -> list[int]:
        kept: list[int] = []
        for c in candidate_clusters:
            if self.fraction_passing(c, query_filter) <= 1.0 - self.cfg.skip_cluster_threshold:
                continue
            kept.append(int(c))
        return kept
