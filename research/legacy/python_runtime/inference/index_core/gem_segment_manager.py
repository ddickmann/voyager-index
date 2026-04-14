"""
GEM Segment Manager — archived sealed GEM graph segment wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from latence_gem_index import GemSegment
except ImportError:
    GemSegment = None
    logger.warning("latence_gem_index not available — GemSegmentManager will not work.")


class GemSegmentManager:
    """
    Manages a single sealed GEM graph segment for multi-vector retrieval.
    """

    def __init__(
        self,
        shard_path: str,
        dim: int,
        *,
        n_fine: int = 256,
        n_coarse: int = 32,
        max_degree: int = 32,
        ef_construction: int = 200,
        max_kmeans_iter: int = 30,
        ctop_r: int = 3,
        n_probes: int = 4,
    ):
        self._shard_path = Path(shard_path)
        self._dim = dim
        self._n_fine = n_fine
        self._n_coarse = n_coarse
        self._max_degree = max_degree
        self._ef_construction = ef_construction
        self._max_kmeans_iter = max_kmeans_iter
        self._ctop_r = ctop_r
        self._n_probes = n_probes
        self._segment: Optional[GemSegment] = None

    def is_ready(self) -> bool:
        return self._segment is not None and self._segment.is_ready()

    def build(self, all_vectors: np.ndarray, doc_ids: List[int], doc_offsets: List[Tuple[int, int]]):
        if GemSegment is None:
            raise RuntimeError("latence_gem_index not installed")
        self._segment = GemSegment()
        self._segment.build(
            all_vectors.astype(np.float32, copy=False),
            doc_ids,
            doc_offsets,
            n_fine=self._n_fine,
            n_coarse=self._n_coarse,
            max_degree=self._max_degree,
            ef_construction=self._ef_construction,
            max_kmeans_iter=self._max_kmeans_iter,
            ctop_r=self._ctop_r,
        )

    def build_from_embeddings(self, doc_embeddings: List[np.ndarray], doc_ids: List[int]):
        all_vectors = np.vstack(doc_embeddings).astype(np.float32)
        offsets = []
        pos = 0
        for emb in doc_embeddings:
            n = emb.shape[0]
            offsets.append((pos, pos + n))
            pos += n
        self.build(all_vectors, doc_ids, offsets)

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: int = 100,
        n_probes: Optional[int] = None,
        enable_shortcuts: bool = False,
    ) -> List[Tuple[int, float]]:
        if not self.is_ready():
            return []
        return self._segment.search(
            query_vectors.astype(np.float32, copy=False),
            k=k,
            ef=ef,
            n_probes=n_probes or self._n_probes,
            enable_shortcuts=enable_shortcuts,
        )

    def inject_shortcuts(self, training_pairs: List[Tuple[np.ndarray, int]], max_shortcuts_per_node: int = 4):
        if not self.is_ready():
            raise RuntimeError("segment not ready")
        pairs = [(q.astype(np.float32).ravel().tolist(), int(t)) for q, t in training_pairs]
        self._segment.inject_shortcuts(pairs, max_shortcuts_per_node)

    def save(self, path: Optional[str] = None):
        if not self.is_ready():
            raise RuntimeError("segment not ready")
        save_path = path or str(self._shard_path / "sealed.gem")
        self._segment.save(save_path)

    def _load(self):
        gem_path = self._shard_path / "sealed.gem"
        if gem_path.exists():
            self.load(str(gem_path))

    def load(self, path: str):
        if GemSegment is None:
            raise RuntimeError("latence_gem_index not installed")
        self._segment = GemSegment()
        self._segment.load(path)

    @property
    def n_docs(self) -> int:
        return self._segment.n_docs() if self.is_ready() else 0

    @property
    def n_nodes(self) -> int:
        return self._segment.n_nodes() if self.is_ready() else 0

    @property
    def n_edges(self) -> int:
        return self._segment.n_edges() if self.is_ready() else 0

    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_ready():
            return {"ready": False}
        return {
            "ready": True,
            "n_docs": self.n_docs,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "dim": self._dim,
            "total_shortcuts": self._segment.total_shortcuts(),
        }
