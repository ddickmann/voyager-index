"""
HNSW Graph Search Engine

Hierarchical Navigable Small World graph index for approximate
nearest neighbor search.

Features:
- Fast approximate search with tunable recall/speed trade-off
- Memory-efficient graph structure
- Support for online insertions
- Compatible with ontology service graph data

Reference: search-index-innovations/search_enhancements.md

Author: Latence Team
License: Apache-2.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import GraphSearchEngine, SearchResult

logger = logging.getLogger(__name__)

# Try to import hnswlib
_HNSWLIB_AVAILABLE = False
try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    logger.debug("hnswlib not installed, HNSW engine will be unavailable")


class HNSWConfig:
    """
    Configuration for HNSW index.

    Attributes:
        space: Distance metric ('cosine', 'l2', 'ip')
        M: Number of bi-directional links per layer (higher = more accurate, slower)
        ef_construction: Size of dynamic list during construction (higher = better quality)
        ef_search: Size of dynamic list during search (higher = better recall)
        max_elements: Maximum number of elements in index
    """

    def __init__(
        self,
        space: str = 'cosine',
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_elements: int = 1_000_000,
    ):
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements


class HNSWEngine(GraphSearchEngine):
    """
    HNSW (Hierarchical Navigable Small World) graph search engine.

    Uses hnswlib for efficient approximate nearest neighbor search.
    Ideal for large-scale vector databases with tunable accuracy/speed.

    Key Parameters:
    - M (16-64): More connections = better accuracy, more memory
    - ef_construction (200-400): Higher = better index quality, slower build
    - ef_search (50-200): Higher = better recall, slower search

    Example:
        >>> engine = HNSWEngine(config=HNSWConfig(M=32, ef_search=100))
        >>>
        >>> # Index vectors
        >>> vectors = np.random.rand(10000, 128).astype(np.float32)
        >>> engine.index(vectors, doc_ids=list(range(10000)))
        >>>
        >>> # Search
        >>> query = np.random.rand(128).astype(np.float32)
        >>> results = engine.search(query, top_k=10)
    """

    def __init__(
        self,
        config: Optional[HNSWConfig] = None,
        dim: Optional[int] = None,
        index_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize HNSW engine.

        Args:
            config: HNSW configuration
            dim: Embedding dimension (auto-detected if None)
            index_path: Path to save/load index

        Raises:
            ImportError: If hnswlib is not installed
        """
        super().__init__(engine_name='hnsw')

        if not _HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is required for HNSWEngine. "
                "Install with: pip install hnswlib"
            )

        self.config = config or HNSWConfig()
        self.dim = dim
        self.index_path = Path(index_path) if index_path else None

        self._index: Optional[hnswlib.Index] = None
        self._doc_ids: List[Any] = []
        self._id_to_internal: Dict[Any, int] = {}

        # Load existing index if path provided
        if self.index_path and self.index_path.exists():
            self._load_index()

        logger.info(f"Initialized HNSW engine (M={self.config.M}, ef={self.config.ef_search})")

    def _init_index(self, dim: int):
        """Initialize the hnswlib index."""
        self._index = hnswlib.Index(
            space=self.config.space,
            dim=dim,
        )
        self._index.init_index(
            max_elements=self.config.max_elements,
            M=self.config.M,
            ef_construction=self.config.ef_construction,
        )
        self._index.set_ef(self.config.ef_search)
        self.dim = dim

    def _load_index(self):
        """Load index from disk."""
        if self.index_path is None:
            return

        try:
            index_file = self.index_path / "hnsw.bin"
            meta_file = self.index_path / "meta.json"

            if not index_file.exists():
                return

            # Load metadata
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self.dim = meta.get('dim')
                self._doc_ids = meta.get('doc_ids', [])
                self._id_to_internal = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}

            # Initialize and load index
            if self.dim:
                self._index = hnswlib.Index(space=self.config.space, dim=self.dim)
                self._index.load_index(str(index_file), max_elements=self.config.max_elements)
                self._index.set_ef(self.config.ef_search)
                self._indexed = True
                logger.info(f"Loaded HNSW index from {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to load HNSW index: {e}")

    def _save_index(self):
        """Save index to disk."""
        if self.index_path is None or self._index is None:
            return

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)

            index_file = self.index_path / "hnsw.bin"
            meta_file = self.index_path / "meta.json"

            # Save index
            self._index.save_index(str(index_file))

            # Save metadata
            meta = {
                'dim': self.dim,
                'doc_ids': self._doc_ids,
            }
            meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

            logger.info(f"Saved HNSW index to {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to save HNSW index: {e}")

    def index(
        self,
        embeddings: np.ndarray,
        doc_ids: Optional[List[Any]] = None,
        **kwargs
    ) -> None:
        """
        Index vectors for search.

        Args:
            embeddings: Vectors to index (N, D)
            doc_ids: Document IDs (auto-generated if None)
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")

        num_vectors, dim = embeddings.shape

        # Initialize index if needed
        if self._index is None:
            self._init_index(dim)
        elif self.dim != dim:
            raise ValueError(f"Dimension mismatch: index={self.dim}, input={dim}")

        # Generate doc_ids if not provided
        if doc_ids is None:
            start_id = len(self._doc_ids)
            doc_ids = list(range(start_id, start_id + num_vectors))

        if len(doc_ids) != num_vectors:
            raise ValueError(f"doc_ids length ({len(doc_ids)}) != embeddings count ({num_vectors})")

        # Add to index
        internal_ids = np.arange(len(self._doc_ids), len(self._doc_ids) + num_vectors)
        self._index.add_items(embeddings, internal_ids)

        # Update mappings
        for doc_id, internal_id in zip(doc_ids, internal_ids):
            self._doc_ids.append(doc_id)
            self._id_to_internal[doc_id] = internal_id

        self._indexed = True

        # Save if path configured
        if self.index_path:
            self._save_index()

        logger.info(f"Indexed {num_vectors} vectors, total: {len(self._doc_ids)}")

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        ef: Optional[int] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query vector (D,) or (1, D)
            top_k: Number of results to return
            ef: Override ef_search for this query

        Returns:
            List of SearchResult objects
        """
        if not self._indexed or self._index is None:
            raise RuntimeError("Index must be built before searching")

        self.validate_top_k(top_k)

        # Handle query shape
        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dim:
            raise ValueError(f"Query dimension ({query.shape[1]}) != index dimension ({self.dim})")

        # Set ef for this search
        if ef is not None:
            self._index.set_ef(ef)
        else:
            self._index.set_ef(self.config.ef_search)

        # Search
        internal_ids, distances = self._index.knn_query(query, k=min(top_k, len(self._doc_ids)))

        # Convert to SearchResult format
        results = []
        for rank, (internal_id, distance) in enumerate(zip(internal_ids[0], distances[0])):
            doc_id = self._doc_ids[internal_id]

            # Convert distance to similarity score
            if self.config.space == 'cosine':
                score = 1.0 - distance
            elif self.config.space == 'l2':
                score = 1.0 / (1.0 + distance)
            else:  # inner product
                score = distance

            results.append(SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank + 1,
                source='hnsw',
                metadata={'distance': float(distance)}
            ))

        return results

    def add_documents(
        self,
        embeddings: np.ndarray,
        doc_ids: Optional[List[Any]] = None,
    ) -> None:
        """Add documents to existing index."""
        self.index(embeddings, doc_ids)

    def delete_documents(self, doc_ids: List[Any]) -> None:
        """
        Mark documents as deleted.

        Note: hnswlib doesn't support true deletion, so we mark them.
        Use rebuild() to physically remove deleted documents.
        """
        for doc_id in doc_ids:
            if doc_id in self._id_to_internal:
                internal_id = self._id_to_internal[doc_id]
                # hnswlib supports marking as deleted
                self._index.mark_deleted(internal_id)
                logger.debug(f"Marked document {doc_id} as deleted")

    def rebuild(self, embeddings: np.ndarray, doc_ids: List[Any]) -> None:
        """
        Rebuild index from scratch.

        Use this to physically remove deleted documents.
        """
        # Clear existing
        self._index = None
        self._doc_ids = []
        self._id_to_internal = {}
        self._indexed = False

        # Rebuild
        self.index(embeddings, doc_ids)

    def set_ef(self, ef: int) -> None:
        """
        Set ef parameter for search.

        Higher ef = better recall, slower search.
        """
        if self._index:
            self._index.set_ef(ef)
            self.config.ef_search = ef

    def get_statistics(self) -> dict:
        """Get engine statistics."""
        return {
            'engine': 'hnsw',
            'num_elements': len(self._doc_ids),
            'dim': self.dim,
            'M': self.config.M,
            'ef_construction': self.config.ef_construction,
            'ef_search': self.config.ef_search,
            'space': self.config.space,
            'max_elements': self.config.max_elements,
        }

    def cleanup(self) -> None:
        """Clean up engine resources."""
        self._index = None
        self._doc_ids = []
        self._id_to_internal = {}
        self._indexed = False
        logger.info("HNSW engine cleanup complete")

    def __repr__(self) -> str:
        num_elements = len(self._doc_ids) if self._indexed else 0
        return f"HNSWEngine(elements={num_elements}, dim={self.dim}, M={self.config.M})"


__all__ = ['HNSWEngine', 'HNSWConfig']

