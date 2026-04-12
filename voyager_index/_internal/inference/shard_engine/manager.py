"""ShardSegmentManager: production manager for the LEMUR-routed shard engine."""
from __future__ import annotations

import gc
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import (
    AnnBackend,
    BuildConfig,
    Compression,
    LemurConfig,
    RouterType,
    SearchConfig,
    StorageLayout,
    TransferMode,
)
from .fetch_pipeline import FetchPipeline, PinnedBufferPool
from .lemur_router import LemurRouter
from .profiler import QueryProfile, Timer
from .scorer import (
    PreloadedGpuCorpus,
    brute_force_maxsim,
    score_all_docs_topk,
    warmup_maxsim,
)
from .shard_store import ShardStore

logger = logging.getLogger(__name__)


class ShardEngineConfig:
    """Production configuration for the shard engine.

    Wraps BuildConfig + SearchConfig with sensible defaults
    for the LEMUR-routed path.
    """

    def __init__(
        self,
        *,
        n_shards: int = 256,
        dim: int = 128,
        compression: Compression = Compression.FP16,
        layout: StorageLayout = StorageLayout.PROXY_GROUPED,
        router_type: RouterType = RouterType.LEMUR,
        ann_backend: AnnBackend = AnnBackend.FAISS_HNSW_IP,
        lemur_epochs: int = 10,
        k_candidates: int = 2000,
        transfer_mode: TransferMode = TransferMode.PINNED,
        pinned_pool_buffers: int = 3,
        pinned_buffer_max_tokens: int = 50_000,
        use_colbandit: bool = False,
        uniform_shard_tokens: bool = True,
        seed: int = 42,
    ):
        self.n_shards = n_shards
        self.dim = dim
        self.compression = compression
        self.layout = layout
        self.router_type = router_type
        self.ann_backend = ann_backend
        self.lemur_epochs = lemur_epochs
        self.k_candidates = k_candidates
        self.transfer_mode = transfer_mode
        self.pinned_pool_buffers = pinned_pool_buffers
        self.pinned_buffer_max_tokens = pinned_buffer_max_tokens
        self.use_colbandit = use_colbandit
        self.uniform_shard_tokens = uniform_shard_tokens
        self.seed = seed

    def to_build_config(self, corpus_size: int) -> BuildConfig:
        cfg = BuildConfig(
            corpus_size=corpus_size,
            n_shards=self.n_shards,
            dim=self.dim,
            compression=self.compression,
            layout=self.layout,
            router_type=self.router_type,
            uniform_shard_tokens=self.uniform_shard_tokens,
            seed=self.seed,
        )
        cfg.lemur = LemurConfig(
            enabled=self.router_type == RouterType.LEMUR,
            ann_backend=self.ann_backend,
            epochs=self.lemur_epochs,
            k_candidates=self.k_candidates,
        )
        return cfg

    def to_search_config(self) -> SearchConfig:
        return SearchConfig(
            k_candidates=self.k_candidates,
            transfer_mode=self.transfer_mode,
            pinned_pool_buffers=self.pinned_pool_buffers,
            pinned_buffer_max_tokens=self.pinned_buffer_max_tokens,
            use_colbandit=self.use_colbandit,
        )


class ShardSegmentManager:
    """Production segment manager for the LEMUR-routed shard engine.

    Provides the same interface contract as GemNativeSegmentManager:
    build, search_multivector, add_multidense, delete, get_statistics, close.
    """

    def __init__(
        self,
        path: Path | str,
        config: ShardEngineConfig | None = None,
        device: str = "cuda",
    ):
        self._path = Path(path)
        self._config = config or ShardEngineConfig()
        self._device = device
        self._lock = threading.RLock()

        self._store: Optional[ShardStore] = None
        self._router: Optional[LemurRouter] = None
        self._pipeline: Optional[FetchPipeline] = None
        self._gpu_corpus: Optional[PreloadedGpuCorpus] = None
        self._doc_vecs: Optional[list] = None
        self._doc_ids: Optional[List[int]] = None
        self._dim: int = self._config.dim
        self._is_built = False
        self._warmup_done = False

        if (self._path / "manifest.json").exists():
            self.load()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        """Build the shard index from a corpus of multi-vector documents.

        Args:
            vectors: Per-document token embeddings (list of [n_tokens, dim] arrays).
            ids: Document IDs (same length as vectors).
            payloads: Optional per-document metadata (stored but not indexed).
        """
        with self._lock:
            self._path.mkdir(parents=True, exist_ok=True)
            n_docs = len(vectors)
            if n_docs == 0:
                raise ValueError("Cannot build index from empty corpus")
            if len(ids) != n_docs:
                raise ValueError(f"vectors ({n_docs}) and ids ({len(ids)}) length mismatch")

            self._dim = vectors[0].shape[1]
            cfg = self._config

            all_vecs = np.concatenate(vectors, axis=0).astype(np.float16)
            doc_offsets: List[Tuple[int, int]] = []
            offset = 0
            for v in vectors:
                doc_offsets.append((offset, offset + v.shape[0]))
                offset += v.shape[0]
            doc_counts = np.array([e - s for s, e in doc_offsets], dtype=np.int32)

            logger.info("Building shard index: %d docs, dim=%d, %d total tokens",
                        n_docs, self._dim, all_vecs.shape[0])

            from .builder import assign_storage_shards

            lemur_dir = self._path / "lemur"
            router = LemurRouter(
                index_dir=lemur_dir,
                ann_backend=cfg.ann_backend.value,
                device=self._device,
            )
            doc_vecs_f16 = torch.from_numpy(all_vecs).to(torch.float16)
            doc_counts_t = torch.from_numpy(doc_counts)

            shard_assignments = assign_storage_shards(
                doc_offsets, cfg.n_shards, cfg.seed, StorageLayout.TOKEN_BALANCED,
            )
            doc_id_to_shard = {did: int(shard_assignments[i]) for i, did in enumerate(ids)}
            router.fit_initial(
                pooled_doc_vectors=doc_vecs_f16,
                pooled_doc_counts=doc_counts_t,
                doc_ids=list(ids),
                doc_id_to_shard=doc_id_to_shard,
                epochs=cfg.lemur_epochs,
            )

            proxy_weights = router._weights.clone()
            shard_assignments = assign_storage_shards(
                doc_offsets, cfg.n_shards, cfg.seed, cfg.layout, proxy_weights=proxy_weights,
            )
            doc_id_to_shard = {did: int(shard_assignments[i]) for i, did in enumerate(ids)}
            router.fit_initial(
                pooled_doc_vectors=doc_vecs_f16,
                pooled_doc_counts=doc_counts_t,
                doc_ids=list(ids),
                doc_id_to_shard=doc_id_to_shard,
                epochs=cfg.lemur_epochs,
            )
            del doc_vecs_f16, doc_counts_t
            gc.collect()

            store = ShardStore(self._path)
            store.build(
                all_vectors=all_vecs,
                doc_offsets=doc_offsets,
                doc_ids=list(ids),
                shard_assignments=shard_assignments,
                n_shards=cfg.n_shards,
                dim=self._dim,
                compression=cfg.compression,
                uniform_shard_tokens=cfg.uniform_shard_tokens,
            )

            meta = {
                "dim": self._dim,
                "n_docs": n_docs,
                "n_shards": cfg.n_shards,
                "compression": cfg.compression.value,
                "layout": cfg.layout.value,
                "router_type": cfg.router_type.value,
            }
            with open(self._path / "engine_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            if payloads:
                with open(self._path / "payloads.json", "w") as f:
                    json.dump({str(did): p for did, p in zip(ids, payloads)}, f)

            self._store = store
            self._router = router
            self._doc_vecs = vectors
            self._doc_ids = list(ids)
            self._is_built = True
            self._init_pipeline()
            self._try_gpu_preload()

            logger.info("Shard index built: %d docs at %s", n_docs, self._path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load a previously built index from disk."""
        with self._lock:
            meta_path = self._path / "engine_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._dim = meta.get("dim", self._config.dim)

            self._store = ShardStore(self._path)
            if not self._store.manifest:
                raise FileNotFoundError(f"No manifest at {self._path}")

            lemur_dir = self._path / "lemur"
            self._router = LemurRouter(
                index_dir=lemur_dir,
                ann_backend=self._config.ann_backend.value,
                device=self._device,
            )
            self._router.load()

            self._doc_ids = self._store.all_doc_ids()
            self._is_built = True
            self._init_pipeline()
            logger.info("Shard index loaded: %d docs from %s",
                        len(self._doc_ids), self._path)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_multivector(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: int = 100,
        n_probes: int = 4,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """Search with a multi-vector query (ColBERT-style token embeddings).

        Args:
            query_vectors: Query token embeddings, shape [n_tokens, dim].
            k: Number of results to return.
            filters: Optional payload filters (not yet implemented).

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if not self._is_built:
            raise RuntimeError("Index not built or loaded")

        self._ensure_warmup()

        q = torch.from_numpy(query_vectors).float() if isinstance(query_vectors, np.ndarray) else query_vectors.float()
        dev = torch.device(self._device if torch.cuda.is_available() else "cpu")

        scfg = self._config.to_search_config()

        with Timer(sync_cuda=True) as t_route:
            routed = self._router.route(
                q,
                k_candidates=scfg.k_candidates,
                prefetch_doc_cap=scfg.max_docs_exact,
            )

        if self._gpu_corpus is not None:
            candidate_ids = routed.doc_ids[:scfg.max_docs_exact]
            with Timer(sync_cuda=True) as t_score:
                ids, scores = self._gpu_corpus.score_candidates(q, candidate_ids, k=k)
            return list(zip(ids, scores))

        shard_chunks, _stats = self._pipeline.fetch_candidate_docs(
            routed.by_shard, max_docs=scfg.max_docs_exact,
        )
        if not shard_chunks:
            return []

        ids, scores = score_all_docs_topk(q, shard_chunks, k=k, device=dev)
        return list(zip(ids, scores))

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """Alias for search_multivector."""
        return self.search_multivector(query, k=k, filters=filters)

    # ------------------------------------------------------------------
    # CRUD stubs (implemented in Chunk 3)
    # ------------------------------------------------------------------

    def add_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        if not self._is_built:
            self.build(vectors, ids, payloads)
            return
        raise NotImplementedError("Incremental add will be implemented in Chunk 3")

    def delete(self, ids: List[int]) -> None:
        raise NotImplementedError("CRUD operations will be added in Chunk 3")

    def upsert_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        raise NotImplementedError("CRUD operations will be added in Chunk 3")

    # ------------------------------------------------------------------
    # Statistics / lifecycle
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "engine": "shard",
            "path": str(self._path),
            "dim": self._dim,
            "is_built": self._is_built,
            "device": self._device,
            "gpu_corpus_loaded": self._gpu_corpus is not None,
        }
        if self._store and self._store.manifest:
            m = self._store.manifest
            stats.update({
                "total_vectors": m.num_docs,
                "num_docs": m.num_docs,
                "num_shards": m.num_shards,
                "total_tokens": m.total_tokens,
                "compression": m.compression,
                "avg_tokens_per_doc": m.avg_tokens_per_chunk,
            })
        if self._router:
            stats["router_type"] = "lemur"
        return stats

    def total_vectors(self) -> int:
        if self._store and self._store.manifest:
            return self._store.manifest.num_docs
        return 0

    def allocate_ids(self, n: int) -> List[int]:
        """Allocate n new sequential document IDs."""
        with self._lock:
            existing = set(self._doc_ids) if self._doc_ids else set()
            start = max(existing) + 1 if existing else 0
            return list(range(start, start + n))

    def upsert_payload(self, doc_id: int, payload: Dict[str, Any]) -> None:
        raise NotImplementedError("Payload upsert will be added in Chunk 3")

    def flush(self) -> None:
        """Flush pending writes. No-op until CRUD is implemented in Chunk 3."""
        pass

    def save(self) -> None:
        """Persist current state (router already auto-saves during fit)."""
        if self._router:
            self._router.save()

    def close(self) -> None:
        """Release resources."""
        with self._lock:
            self._gpu_corpus = None
            self._pipeline = None
            self._store = None
            self._router = None
            self._doc_vecs = None
            self._is_built = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        if self._store is None:
            return
        cfg = self._config
        pool = None
        if cfg.transfer_mode in (TransferMode.PINNED, TransferMode.DOUBLE_BUFFERED):
            pool = PinnedBufferPool(
                max_tokens=cfg.pinned_buffer_max_tokens,
                dim=self._dim,
                n_buffers=cfg.pinned_pool_buffers,
            )
        self._pipeline = FetchPipeline(
            store=self._store,
            mode=cfg.transfer_mode,
            pinned_pool=pool,
            device=self._device,
        )

    def _try_gpu_preload(self) -> None:
        """Attempt to pre-load corpus to GPU (GEM pattern) if it fits."""
        if self._doc_vecs is None or not self._doc_ids:
            return
        max_tok = max((v.shape[0] for v in self._doc_vecs), default=1)
        if PreloadedGpuCorpus.fits_on_gpu(len(self._doc_vecs), max_tok, self._dim):
            self._gpu_corpus = PreloadedGpuCorpus(
                self._doc_vecs, self._doc_ids, self._dim, device=self._device,
            )
        else:
            logger.info("Corpus too large for GPU preload (%d docs) — using shard fetch",
                        len(self._doc_ids))

    def _ensure_warmup(self) -> None:
        if self._warmup_done:
            return
        token_counts = [128, 256]
        if self._store and self._store.manifest:
            token_counts = [
                int(s.p95_tokens) for s in self._store.manifest.shards if s.p95_tokens > 0
            ] or token_counts
        warmup_maxsim(dim=self._dim, doc_token_counts=token_counts, device=self._device)
        self._warmup_done = True
