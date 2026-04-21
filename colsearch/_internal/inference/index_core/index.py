"""
ColBERT Index

High-level index interface with automatic strategy selection.
Coordinates storage, search strategies, and PLAID integration.

Features:
- Automatic scaling (Triton → Triton+mmap → PLAID+Triton)
- Full CRUD operations via storage layer
- Collection management
- Statistics and monitoring

Strategies:
- Small (<5K docs): Pure Triton kernel (cached in VRAM)
- Medium (5K-50K docs): Triton + mmap streaming
- Large (>50K docs): PLAID candidates → Triton reranking

Author: ColBERT Team
License: CC-BY-NC-4.0
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple

import torch

from colsearch._internal.kernels.maxsim import fast_colbert_scores

from .async_storage import AsyncPipelineStorage
from .storage import IndexStatistics, Storage

logger = logging.getLogger(__name__)

try:
    from latence_gem_router import PyGemRouter
    _GEM_ROUTER_AVAILABLE = True
except ImportError:
    PyGemRouter = None
    _GEM_ROUTER_AVAILABLE = False

StorageMode = Literal['sync', 'async']


class ColbertIndex:
    """
    Production-grade ColBERT index with automatic scaling.

    This is the main interface for ColBERT indexing and search.
    It automatically selects the optimal search strategy based on corpus size.

    Strategies:
    - **Triton (small)**: Entire index cached in VRAM for instant search
    - **Triton+mmap (medium)**: Stream from disk using mmap, GPU scoring
    - **PLAID+Triton (large)**: PLAID for candidate generation, Triton for reranking

    Attributes:
        index_path: Path to index directory
        config: Index configuration
        storage: Storage layer instance
        cached_embeddings: Cached embeddings for small indexes
        use_cache: Whether to use VRAM cache
        plaid_index: PLAID index for large-scale search
        plaid_path: Path to PLAID index

    Example:
        >>> # Create new index
        >>> index = ColbertIndex(
        ...     index_path="/data/colbert",
        ...     config=IndexConfig(small_threshold=5000)
        ... )
        >>>
        >>> # Build from batches
        >>> def batch_gen():
        ...     for batch in batches:
        ...         yield embeddings, metadata
        >>> index.build_from_batches(batch_gen(), max_tokens=512)
        >>>
        >>> # Search
        >>> scores, indices = index.search(queries, top_k=10)
        >>>
        >>> # Load existing index
        >>> index = ColbertIndex.load("/data/colbert")
    """

    def __init__(
        self,
        index_path: Path,
        config,
        create_if_missing: bool = True,
        storage_mode: StorageMode = 'sync'
    ):
        """
        Initialize enterprise ColBERT index.

        Args:
            index_path: Path to index directory
            config: IndexConfig instance
            create_if_missing: Create directory if it doesn't exist
            storage_mode: 'sync' (default, reliable) or 'async' (2x faster, GPU-optimized)

        Notes:
            - 'sync': Simple, reliable, good for <10K docs, CPU-bound during indexing
            - 'async': 2x faster, GPU-optimized, best for >10K docs, uses async pipeline
            - Both modes are production-ready
        """
        self.index_path = Path(index_path)
        self.config = config
        self.storage_mode = storage_mode

        # Create directory
        if create_if_missing:
            self.index_path.mkdir(parents=True, exist_ok=True)

        # Storage layer (sync or async)
        if storage_mode == 'sync':
            self.storage = Storage(self.index_path, self.config)
        elif storage_mode == 'async':
            self.storage = AsyncPipelineStorage(
                self.index_path,
                self.config,
                max_batch_size=self.config.batch_size,
                num_workers=2
            )
        else:
            raise ValueError(f"Invalid storage_mode: {storage_mode}. Use 'sync' or 'async'")

        # Load existing metadata
        if (self.index_path / "metadata.json").exists():
            self.storage.load_metadata()
            logger.info(f"Loaded existing index from {index_path}")

        # Cache (for small indexes)
        self.cached_embeddings: Optional[torch.Tensor] = None
        self.cached_doc_ids: List[int] = []
        self.cached_doc_id_to_pos: dict[int, int] = {}
        self.use_cache = False
        self.last_search_profile: dict[str, object] = {}
        self.last_write_profile: dict[str, object] = {}

        # PLAID index (for large scale)
        self.plaid_index = None
        self.plaid_path = self.index_path / "plaid"

        # GEM router (optional alternative to PLAID for balanced mode)
        self.gem_router = None
        self.gem_router_path = self.index_path / "gem_router.gemr"
        if _GEM_ROUTER_AVAILABLE and self.gem_router_path.exists():
            try:
                self.gem_router = PyGemRouter(dim=self.config.embed_dim)
                self.gem_router.load(str(self.gem_router_path))
                logger.info(
                    "Loaded GEM router for ColBERT: %d docs, %d fine centroids",
                    self.gem_router.n_docs(),
                    self.gem_router.n_fine(),
                )
            except Exception:
                logger.debug("Could not load GEM router; will be built on demand")
                self.gem_router = None

        # Save config
        self._save_config()

        logger.info(f"Initialized ColbertIndex at {index_path}")

    def build_from_batches(
        self,
        batch_generator: Generator[Tuple[torch.Tensor, Optional[List[dict]]], None, None],
        collection_name: str = "default",
        max_tokens: Optional[int] = None
    ) -> List[int]:
        """
        Build index from batch generator (TRUE STREAMING - no RAM spikes!).

        Args:
            batch_generator: Generator yielding (embeddings, metadata)
                embeddings: [batch_size, num_tokens, embed_dim]
                metadata: Optional list of metadata dicts
            collection_name: Collection name
            max_tokens: Maximum token length (RECOMMENDED for true streaming!)

        Returns:
            List of assigned document IDs

        Example:
            >>> def batch_gen():
            ...     for i in range(0, len(docs), batch_size):
            ...         yield doc_embeddings[i:i+batch_size], None
            >>>
            >>> doc_ids = index.build_from_batches(
            ...     batch_gen(),
            ...     collection_name="main",
            ...     max_tokens=512  # Recommended!
            ... )
        """
        logger.info(f"Building index from batches (collection: {collection_name})")
        start_time = time.time()

        # Stream to disk via storage layer (sync or async)
        storage_start = time.perf_counter()
        if self.storage_mode == 'sync':
            doc_ids = self.storage.create_from_batches(
                batch_generator,
                collection_name=collection_name,
                show_progress=True,
                max_tokens=max_tokens
            )
        elif self.storage_mode == 'async':
            doc_ids = self.storage.create_from_batches_async(
                batch_generator,
                collection_name=collection_name,
                show_progress=True,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Invalid storage_mode: {self.storage_mode}")
        storage_elapsed_ms = (time.perf_counter() - storage_start) * 1000.0

        # Determine strategy and build secondary indexes
        strategy_start = time.perf_counter()
        self._update_strategy()
        strategy_elapsed_ms = (time.perf_counter() - strategy_start) * 1000.0

        build_time = time.time() - start_time
        self.last_write_profile = {
            "mode": "build_from_batches",
            "doc_count": len(doc_ids),
            "storage_ms": storage_elapsed_ms,
            "update_strategy_ms": strategy_elapsed_ms,
            "total_ms": build_time * 1000.0,
        }
        logger.info(f"Index built in {build_time:.2f}s ({len(doc_ids)} documents)")

        return doc_ids

    def add_documents(
        self,
        embeddings: torch.Tensor,
        collection_name: str = "default",
        metadata: Optional[List[dict]] = None
    ) -> List[int]:
        """
        Add documents to index.

        Args:
            embeddings: Document embeddings [num_docs, num_tokens, embed_dim]
            collection_name: Collection name
            metadata: Optional metadata per document

        Returns:
            List of assigned document IDs
        """
        storage_start = time.perf_counter()
        doc_ids = self.storage.add_documents(embeddings, collection_name, metadata)
        storage_elapsed_ms = (time.perf_counter() - storage_start) * 1000.0
        strategy_start = time.perf_counter()
        self._update_strategy()
        strategy_elapsed_ms = (time.perf_counter() - strategy_start) * 1000.0
        self.last_write_profile = {
            "mode": "add_documents",
            "doc_count": len(doc_ids),
            "storage_ms": storage_elapsed_ms,
            "update_strategy_ms": strategy_elapsed_ms,
            "total_ms": storage_elapsed_ms + strategy_elapsed_ms,
        }
        return doc_ids

    def delete_documents(self, doc_ids: List[int]) -> None:
        """
        Delete documents from index.

        Args:
            doc_ids: Document IDs to delete
        """
        self.storage.delete_documents(doc_ids)
        self._update_strategy()

    def update_documents(
        self,
        doc_ids: List[int],
        embeddings: torch.Tensor
    ) -> None:
        """
        Update documents in index.

        Args:
            doc_ids: Document IDs to update
            embeddings: New embeddings [len(doc_ids), num_tokens, embed_dim]
        """
        self.storage.update_documents(doc_ids, embeddings)
        self._update_strategy()

    def create_collection(self, name: str, doc_ids: List[int]) -> None:
        """
        Create a collection.

        Args:
            name: Collection name
            doc_ids: Document IDs in collection
        """
        self.storage.create_collection(name, doc_ids)

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name
        """
        self.storage.delete_collection(name)

    def compact(self) -> None:
        """Compact storage (physically remove deleted documents)."""
        self.storage.compact()
        self._update_strategy()

    def search(
        self,
        queries: torch.Tensor,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        doc_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search with automatic mode selection.

        Automatically routes to the optimal search mode based on corpus size:
        - 🚀 REAL-TIME MODE: Pure Triton (in-memory, 50+ QPS)
        - 💎 HIGH QUALITY MODE: Triton + mmap (exact search, no compromises)
        - ⚖️ BALANCED MODE: PLAID + Triton (optimized speed-quality ratio)

        Args:
            queries: Query embeddings [num_queries, num_tokens, embed_dim]
            top_k: Number of results to return
            collection_name: Optional collection to search

        Returns:
            scores: [num_queries, top_k]
            indices: [num_queries, top_k]

        Example:
            >>> scores, indices = index.search(query_embeddings, top_k=10)
            >>> for i in range(queries.shape[0]):
            ...     print(f"Query {i} top result: doc {indices[i, 0]} (score: {scores[i, 0]:.4f})")
        """
        queries = queries.to(self.config.device)
        explicit_doc_ids = doc_ids is not None

        # Get active document IDs
        if doc_ids is not None:
            doc_ids = [int(doc_id) for doc_id in doc_ids if int(doc_id) not in self.storage.deleted_ids]
        elif collection_name:
            doc_ids = self.storage.collections.get(collection_name, [])
            doc_ids = [i for i in doc_ids if i not in self.storage.deleted_ids]
        else:
            doc_ids = [
                i for i in range(self.storage.num_docs)
                if i not in self.storage.deleted_ids
            ]

        if not doc_ids:
            empty_scores = torch.empty((queries.shape[0], 0), device=self.config.device, dtype=torch.float32)
            empty_indices = torch.empty((queries.shape[0], 0), device=self.config.device, dtype=torch.long)
            return empty_scores, empty_indices

        if explicit_doc_ids:
            if len(doc_ids) < self.config.realtime_threshold:
                return self._search_triton(queries, top_k, doc_ids)
            return self._search_triton_mmap(queries, top_k, doc_ids)

        # Route to appropriate mode
        stats = self.storage.get_statistics(
            self.config.realtime_threshold,
            self.config.balanced_threshold
        )
        strategy = stats.strategy

        if strategy == "realtime":
            return self._search_triton(queries, top_k, doc_ids)
        elif strategy == "high_quality":
            return self._search_triton_mmap(queries, top_k, doc_ids)
        else:  # balanced
            if self.gem_router is not None and self.gem_router.is_ready():
                return self._search_gem_triton(queries, top_k, doc_ids)
            return self._search_plaid_triton(queries, top_k, doc_ids)

    def _map_local_indices(
        self,
        topk_indices: torch.Tensor,
        doc_ids: Optional[List[int]],
    ) -> torch.Tensor:
        if doc_ids is None:
            return topk_indices
        mapped = [
            [doc_ids[index] for index in row.tolist()]
            for row in topk_indices.detach().cpu()
        ]
        return torch.tensor(mapped, device=topk_indices.device, dtype=torch.long)

    def _search_triton(
        self,
        queries: torch.Tensor,
        top_k: int,
        doc_ids: Optional[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Small scale: Pure Triton (cached in VRAM).

        Fastest strategy for small corpora. Entire index is cached in VRAM.
        """
        load_start = time.perf_counter()
        used_cache = False
        if doc_ids is None:
            if self.cached_embeddings is None:
                self.cached_doc_ids = [i for i in range(self.storage.num_docs) if i not in self.storage.deleted_ids]
                self.cached_doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(self.cached_doc_ids)}
                self.cached_embeddings = self.storage.load_documents(
                    doc_ids=self.cached_doc_ids,
                    device=self.config.device
                )
            documents = self.cached_embeddings
            used_cache = True
        else:
            documents = self._load_cached_subset(doc_ids, device=self.config.device)
            if documents is not None:
                used_cache = True
            else:
                documents = self.storage.load_documents(
                    doc_ids=doc_ids,
                    device=self.config.device
                )
        load_elapsed_ms = (time.perf_counter() - load_start) * 1000.0
        score_start = time.perf_counter()
        scores = fast_colbert_scores(queries, documents)
        score_elapsed_ms = (time.perf_counter() - score_start) * 1000.0
        topk_start = time.perf_counter()
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        topk_elapsed_ms = (time.perf_counter() - topk_start) * 1000.0
        self.last_search_profile = {
            "mode": "triton_cache" if used_cache else "triton_load",
            "doc_count": int(scores.shape[1]),
            "load_ms": load_elapsed_ms,
            "score_ms": score_elapsed_ms,
            "topk_ms": topk_elapsed_ms,
            "used_cache": used_cache,
        }
        return topk_scores, self._map_local_indices(topk_indices, doc_ids)

    def _search_triton_mmap(
        self,
        queries: torch.Tensor,
        top_k: int,
        doc_ids: Optional[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Medium scale: Triton + mmap streaming.

        Streams documents from disk in chunks, scores with Triton kernel.
        Memory-efficient for medium-sized corpora.
        """
        all_scores = []
        score_ms = 0.0
        for chunk in self.storage.load_documents_streaming(doc_ids, self.config.device):
            score_start = time.perf_counter()
            chunk_scores = fast_colbert_scores(queries, chunk)
            score_ms += (time.perf_counter() - score_start) * 1000.0
            all_scores.append(chunk_scores)
            del chunk

        scores = torch.cat(all_scores, dim=1)
        topk_start = time.perf_counter()
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        topk_elapsed_ms = (time.perf_counter() - topk_start) * 1000.0
        self.last_search_profile = {
            "mode": "triton_mmap",
            "doc_count": int(scores.shape[1]),
            "load_ms": 0.0,
            "score_ms": score_ms,
            "topk_ms": topk_elapsed_ms,
            "chunk_size": self.config.chunk_size,
        }
        return topk_scores, self._map_local_indices(topk_indices, doc_ids)

    def _search_plaid_triton(
        self,
        queries: torch.Tensor,
        top_k: int,
        doc_ids: Optional[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Large scale: PLAID candidates → Triton reranking.

        Two-stage search:
        1. PLAID generates candidates (fast approximate search)
        2. Triton reranks candidates (exact scoring on subset)
        """
        if self.plaid_index is None:
            self._build_plaid_index()

        # Get candidates from PLAID (use reasonable number, not entire small_threshold!)
        num_candidates = min(500, self.storage.num_docs)  # 500 candidates is plenty
        Q_list = [queries[i] for i in range(queries.shape[0])]

        plaid_results = self.plaid_index.search(
            queries_embeddings=Q_list,
            top_k=num_candidates,
            n_ivf_probe=self.config.plaid_n_ivf_probe,
            n_full_scores=min(100, num_candidates)  # Also limit full scoring
        )

        # Rerank with Triton
        all_scores = []
        all_indices = []

        for q_idx in range(queries.shape[0]):
            candidate_ids = [doc_id for doc_id, _ in plaid_results[q_idx]]

            if not candidate_ids:
                all_scores.append(torch.zeros(top_k, device=self.config.device))
                all_indices.append(torch.zeros(top_k, dtype=torch.long, device=self.config.device))
                continue

            # Sort candidate IDs (HDF5 requires sorted indices)
            candidate_ids_sorted = sorted(candidate_ids)

            # Load candidates
            candidates = self.storage.load_documents(
                doc_ids=candidate_ids_sorted,
                device=self.config.device
            )

            # Score with Triton
            candidate_scores = fast_colbert_scores(queries[q_idx:q_idx+1], candidates).squeeze(0)

            # Top-k
            k = min(top_k, len(candidate_ids))
            topk_scores, topk_local = torch.topk(candidate_scores, k=k)
            topk_global = torch.tensor(
                [candidate_ids_sorted[i] for i in topk_local.cpu().numpy()],
                device=self.config.device
            )

            all_scores.append(topk_scores)
            all_indices.append(topk_global)

        scores = torch.stack(all_scores, dim=0)
        indices = torch.stack(all_indices, dim=0)

        return scores, indices

    def _search_gem_triton(
        self,
        queries: torch.Tensor,
        top_k: int,
        doc_ids: Optional[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GEM router candidates -> Triton reranking.

        Two-stage search using the Rust GEM router for fast cluster-based
        candidate generation, then exact MaxSim reranking with Triton.
        Falls back to PLAID or mmap if the router is not available.
        """
        if self.gem_router is None or not self.gem_router.is_ready():
            return self._search_plaid_triton(queries, top_k, doc_ids)

        import numpy as np

        search_start = time.perf_counter()
        all_scores = []
        all_indices = []
        total_route_ms = 0.0
        total_candidates = 0

        for q_idx in range(queries.shape[0]):
            q_np = queries[q_idx].detach().cpu().numpy().astype(np.float32)
            num_candidates = min(500, self.storage.num_docs)

            route_start = time.perf_counter()
            candidates = self.gem_router.route_query(
                q_np,
                n_probes=4,
                max_candidates=num_candidates,
            )
            total_route_ms += (time.perf_counter() - route_start) * 1000.0

            candidate_ids = [int(doc_id) for doc_id, _ in candidates]
            if doc_ids is not None:
                allowed = set(doc_ids)
                candidate_ids = [c for c in candidate_ids if c in allowed]

            total_candidates += len(candidate_ids)

            if not candidate_ids:
                all_scores.append(torch.full((top_k,), float('-inf'), device=self.config.device))
                all_indices.append(torch.zeros(top_k, dtype=torch.long, device=self.config.device))
                continue

            candidate_ids_sorted = sorted(candidate_ids)

            docs = self.storage.load_documents(
                doc_ids=candidate_ids_sorted,
                device=self.config.device,
            )

            candidate_scores = fast_colbert_scores(
                queries[q_idx:q_idx + 1], docs
            ).squeeze(0)

            k = min(top_k, len(candidate_ids_sorted))
            topk_scores, topk_local = torch.topk(candidate_scores, k=k)
            topk_global = torch.tensor(
                [candidate_ids_sorted[i] for i in topk_local.cpu().numpy()],
                device=self.config.device,
            )

            if k < top_k:
                pad_n = top_k - k
                topk_scores = torch.cat([topk_scores, torch.full((pad_n,), float('-inf'), device=self.config.device)])
                topk_global = torch.cat([topk_global, torch.zeros(pad_n, dtype=torch.long, device=self.config.device)])

            all_scores.append(topk_scores)
            all_indices.append(topk_global)

        scores = torch.stack(all_scores, dim=0)
        indices = torch.stack(all_indices, dim=0)

        self.last_search_profile = {
            "mode": "gem_triton",
            "routing_ms": total_route_ms,
            "total_ms": (time.perf_counter() - search_start) * 1000.0,
            "avg_candidates": total_candidates / max(1, queries.shape[0]),
        }
        return scores, indices

    def build_gem_router(self) -> None:
        """Build or rebuild the GEM router for large-scale candidate generation."""
        if not _GEM_ROUTER_AVAILABLE:
            logger.warning("latence_gem_router not available; skipping GEM router build")
            return

        import numpy as np

        active_ids = [
            i for i in range(self.storage.num_docs)
            if i not in self.storage.deleted_ids
        ]
        if len(active_ids) < 8:
            logger.info("Too few documents for GEM router (%d); skipping", len(active_ids))
            return

        logger.info("Building GEM router for %d documents...", len(active_ids))
        all_vecs = []
        offsets = []
        offset = 0
        for chunk in self.storage.load_documents_streaming(active_ids, 'cpu'):
            for i in range(chunk.shape[0]):
                doc = chunk[i].numpy().astype(np.float32)
                norms = np.linalg.norm(doc, axis=1)
                n_real = int(np.sum(norms > 1e-6))
                n_real = max(1, n_real)
                doc = doc[:n_real]
                all_vecs.append(doc)
                offsets.append((offset, offset + n_real))
                offset += n_real

        stacked = np.concatenate(all_vecs, axis=0).astype(np.float32)
        n_fine = min(4096, max(4, int(stacked.shape[0] * 0.1)))
        n_coarse = min(64, n_fine // 2)

        router = PyGemRouter(dim=self.config.embed_dim)
        router.build(
            stacked,
            active_ids,
            offsets,
            n_fine=n_fine,
            n_coarse=n_coarse,
            max_kmeans_iter=30,
            ctop_r=3,
        )
        router.save(str(self.gem_router_path))
        self.gem_router = router
        logger.info(
            "GEM router built: %d docs, %d fine centroids, %d coarse clusters",
            router.n_docs(), router.n_fine(), router.n_coarse(),
        )

    def _build_plaid_index(self) -> None:
        """Build PLAID index for large-scale search."""
        try:
            from fast_plaid import search as fast_plaid_search
        except ImportError:
            logger.warning("fast-plaid not installed, falling back to triton_mmap")
            return

        logger.info("Building PLAID index...")

        # Load documents
        doc_list = []
        for chunk in self.storage.load_documents_streaming(device='cpu'):
            doc_list.extend([chunk[i] for i in range(chunk.shape[0])])

        self.plaid_path.mkdir(parents=True, exist_ok=True)

        self.plaid_index = fast_plaid_search.FastPlaid(
            index=str(self.plaid_path),
            preload_index=True
        )

        self.plaid_index.create(
            documents_embeddings=doc_list,
            nbits=self.config.plaid_nbits
        )

        logger.info("PLAID index built")

    def _update_strategy(self) -> None:
        """
        Update strategy based on corpus size.

        Automatically selects optimal mode:
        - REAL-TIME MODE: < realtime_threshold (default 1000 docs)
        - HIGH QUALITY MODE: realtime_threshold to balanced_threshold
        - BALANCED MODE: > balanced_threshold (default 50K docs)
        """
        stats = self.storage.get_statistics(
            self.config.realtime_threshold,
            self.config.balanced_threshold
        )

        if stats.strategy == "realtime" and self.config.cache_realtime_index:
            # REAL-TIME MODE: Pure Triton, cached in VRAM
            self.cached_doc_ids = [i for i in range(self.storage.num_docs) if i not in self.storage.deleted_ids]
            self.cached_doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(self.cached_doc_ids)}
            self.cached_embeddings = self.storage.load_documents(doc_ids=self.cached_doc_ids, device=self.config.device)
            self.use_cache = True
            logger.info("🚀 REAL-TIME MODE: Pure Triton (in-memory, 50+ QPS)")
        elif stats.strategy == "high_quality":
            # HIGH QUALITY MODE: Triton + mmap streaming (exact search)
            self.cached_embeddings = None
            self.cached_doc_ids = []
            self.cached_doc_id_to_pos = {}
            self.use_cache = False
            logger.info("💎 HIGH QUALITY MODE: Triton + mmap (exact search, no compromises)")
        elif stats.strategy == "balanced":
            # BALANCED MODE: PLAID + Triton hybrid
            self.cached_embeddings = None
            self.cached_doc_ids = []
            self.cached_doc_id_to_pos = {}
            self.use_cache = False
            if self.plaid_index is None:
                self._build_plaid_index()
            logger.info("⚖️ BALANCED MODE: PLAID + Triton (optimized speed-quality ratio)")

    def get_statistics(self) -> IndexStatistics:
        """
        Get index statistics.

        Returns:
            IndexStatistics object with all metrics and selected mode
        """
        return self.storage.get_statistics(
            self.config.realtime_threshold,
            self.config.balanced_threshold
        )

    def _save_config(self) -> None:
        """Save configuration to JSON."""
        config_path = self.index_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, index_path: Path, device: str = 'cuda') -> 'ColbertIndex':
        """
        Load existing index from disk.

        Args:
            index_path: Path to index directory
            device: Compute device ('cuda' or 'cpu')

        Returns:
            Loaded index instance

        Example:
            >>> index = ColbertIndex.load("/data/colbert", device='cuda')
            >>> scores, indices = index.search(queries, top_k=10)
        """
        index_path = Path(index_path)

        # Load config
        config_path = index_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Import IndexConfig from config module
            from ..config import IndexConfig
            config = IndexConfig.from_dict(config_dict)
            config.device = device
        else:
            from ..config import IndexConfig
            config = IndexConfig(device=device)

        return cls(index_path, config, create_if_missing=False)

    def cleanup(self) -> None:
        """
        Clean up resources and delete index from disk.

        Warning: This permanently deletes the index!
        """
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        logger.info(f"Cleaned up index at {self.index_path}")

    def _load_cached_subset(self, doc_ids: List[int], device: str) -> Optional[torch.Tensor]:
        if not self.use_cache or self.cached_embeddings is None or not doc_ids:
            return None
        normalized_ids = [int(doc_id) for doc_id in doc_ids]
        if any(doc_id not in self.cached_doc_id_to_pos for doc_id in normalized_ids):
            return None
        positions = [self.cached_doc_id_to_pos[doc_id] for doc_id in normalized_ids]
        index_tensor = torch.as_tensor(positions, device=self.cached_embeddings.device, dtype=torch.long)
        return self.cached_embeddings.index_select(0, index_tensor).to(device)

    def get_document_embeddings(self, doc_ids: List[int], device: str = "cpu") -> torch.Tensor:
        normalized_ids = [int(doc_id) for doc_id in doc_ids if int(doc_id) not in self.storage.deleted_ids]
        if not normalized_ids:
            return torch.empty((0, self.storage.max_tokens, self.storage.embed_dim), dtype=torch.float32, device=device)
        cached = self._load_cached_subset(normalized_ids, device=device)
        if cached is not None:
            return cached
        return self.storage.load_documents(doc_ids=normalized_ids, device=device)


__all__ = ['ColbertIndex']

