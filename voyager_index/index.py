"""
voyager_index.Index — the primary public API for voyager-index.

Provides a simple, DX-optimized interface for creating, querying,
updating, and managing multi-vector indexes.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with document ID, score, and optional payload."""

    doc_id: int
    score: float
    payload: Optional[Dict[str, Any]] = None
    token_scores: Optional[List[float]] = None
    matched_tokens: Optional[List[int]] = None

    def __repr__(self) -> str:
        pay = f", payload={self.payload}" if self.payload is not None else ""
        tok = f", token_scores=[{len(self.token_scores)} tokens]" if self.token_scores is not None else ""
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}{pay}{tok})"


@dataclass
class ScrollPage:
    """A page of results from scroll iteration."""

    results: List[SearchResult]
    next_offset: Optional[int] = None


@dataclass
class IndexStats:
    """Summary statistics for an Index."""

    total_documents: int = 0
    sealed_segments: int = 0
    active_documents: int = 0
    dim: int = 0
    engine: str = ""


def _check_gem_available() -> bool:
    try:
        from voyager_index._internal.inference.index_core.gem_manager import (
            GemNativeSegmentManager,
        )

        if GemNativeSegmentManager is None:
            return False
        from latence_gem_index import GemSegment, PyMutableGemSegment

        return GemSegment is not None and PyMutableGemSegment is not None
    except ImportError:
        return False


def _check_hnsw_available() -> bool:
    try:
        from voyager_index._internal.inference.index_core.hnsw_manager import (
            HnswSegmentManager,
        )

        return HnswSegmentManager is not None
    except ImportError:
        return False


def _check_shard_available() -> bool:
    try:
        from voyager_index._internal.inference.shard_engine.manager import (
            ShardSegmentManager,
        )

        return ShardSegmentManager is not None
    except ImportError:
        return False


class Index:
    """
    Primary interface for voyager-index.

    Supports shard (mainline), GEM, and HNSW backends.
    Provides CRUD operations, search, scroll, snapshot, and lifecycle management.

    Example::

        idx = Index("my_index", dim=128, engine="shard")
        idx.add(embeddings, ids=[1, 2, 3])
        results = idx.search(query, k=10)
        idx.close()
    """

    def __init__(
        self,
        path: str,
        dim: int,
        *,
        engine: str = "auto",
        mode: Optional[str] = None,
        embedding_fn: Optional[Any] = None,
        n_fine: int = 256,
        n_coarse: int = 32,
        max_degree: int = 32,
        ef_construction: int = 200,
        n_probes: int = 4,
        enable_wal: bool = True,
        **kwargs,
    ):
        """
        Create or open an index.

        Args:
            path: Directory to store the index.
            dim: Vector dimensionality.
            engine: 'shard', 'gem', 'hnsw', or 'auto' (default: compatibility auto-detect).
            mode: Optional mode hint: 'colbert', 'colpali', or None.
                  When set, auto-configures engine and parameters for
                  the specified model family.
            embedding_fn: Optional callable for auto-embedding.
                  Must implement embed_documents(texts) -> List[np.ndarray]
                  and embed_query(text) -> np.ndarray.
            n_fine: Number of fine centroids for quantization (0 = auto).
            n_coarse: Number of coarse clusters for routing.
            max_degree: Maximum graph node degree.
            ef_construction: Beam width during graph construction.
            n_probes: Number of coarse clusters to probe at search time.
            enable_wal: Enable write-ahead log for crash recovery.

        Keyword Args (passed through to GemNativeSegmentManager):
            rerank_device: Device for MaxSim reranking ('cuda', 'cpu', or None
                to disable reranking). When set, GEM proxy results are reranked
                with exact late-interaction scoring on GPU.
            roq_rerank: Enable ROQ quantized MaxSim reranking with a fused
                Triton kernel (~8x memory reduction vs FP32). Requires CUDA.
                When enabled and ``rerank_device`` is None, auto-selects CUDA.
            roq_bits: Bit width for ROQ quantization (default 4). The fused
                Triton rerank kernel uses 4-bit regardless; this controls the
                quantizer's compression level.
            use_emd: Use qEMD (Sinkhorn OT) for graph construction (default
                True in the manager). qEMD provides metric-decoupled
                construction for stable navigation; qCH is faster but
                non-metric. Set False to use qCH for both build and search.
            dual_graph: Build per-cluster local graphs with cross-cluster
                bridge enforcement (default True). Algorithms 1-3 from the
                GEM paper.
            warmup_kernels: Pre-compile Triton kernels at init for zero
                cold-start latency (default True). Only runs when
                ``rerank_device`` is set and not 'cpu'.
            seed_batch_size: Minimum documents to buffer before training the
                codebook and building the first mutable graph (default 256).
            max_kmeans_iter: Maximum K-means iterations for codebook training
                (default 15).
            ctop_r: Number of top coarse clusters per document for routing
                (default 3).
            enable_shortcuts: Enable semantic shortcut edges from training
                pairs (default False).
            store_raw_vectors: Keep raw float32 vectors in sealed segments
                for in-segment MaxSim reranking (default True). Set False
                to save memory when using external GPU reranking.
        """
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._dim = dim
        self._mode = mode
        self._embedding_fn = embedding_fn
        self._closed = False
        self._lock = threading.RLock()
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._metrics_hook = None

        # Mode-based engine selection and parameter tuning
        resolved_engine = engine
        if mode in ("colbert", "colpali"):
            resolved_engine = "gem"
            kwargs.setdefault("enable_shortcuts", mode == "colpali")
        if engine == "auto":
            if _check_gem_available():
                resolved_engine = "gem"
            elif _check_shard_available():
                resolved_engine = "shard"
            else:
                resolved_engine = "hnsw"

        self._engine = resolved_engine

        if resolved_engine == "gem":
            from voyager_index._internal.inference.index_core.gem_manager import (
                GemNativeSegmentManager,
            )

            mgr_kwargs = dict(kwargs)
            mgr_kwargs["enable_wal"] = enable_wal
            self._manager = GemNativeSegmentManager(
                shard_path=str(self._path / "shard"),
                dim=dim,
                n_fine=n_fine,
                n_coarse=n_coarse,
                max_degree=max_degree,
                gem_ef_construction=ef_construction,
                n_probes=n_probes,
                **mgr_kwargs,
            )
        elif resolved_engine == "shard":
            from voyager_index._internal.inference.shard_engine.manager import (
                ShardEngineConfig,
                ShardSegmentManager,
            )

            shard_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "n_shards",
                    "compression",
                    "layout",
                    "router_type",
                    "ann_backend",
                    "lemur_epochs",
                    "k_candidates",
                    "max_docs_exact",
                    "lemur_search_k_cap",
                    "n_full_scores",
                    "transfer_mode",
                    "pinned_pool_buffers",
                    "pinned_buffer_max_tokens",
                    "use_colbandit",
                    "quantization_mode",
                    "variable_length_strategy",
                    "gpu_corpus_rerank_topn",
                    "n_centroid_approx",
                    "router_device",
                    "uniform_shard_tokens",
                    "seed",
                }
            }
            shard_config = kwargs.get("shard_config") or ShardEngineConfig(
                dim=dim,
                **shard_kwargs,
            )
            device = kwargs.get("device", "cuda")
            self._manager = ShardSegmentManager(
                path=self._path / "shard",
                config=shard_config,
                device=device,
            )
        elif resolved_engine == "hnsw":
            from voyager_index._internal.inference.index_core.hnsw_manager import (
                HnswSegmentManager,
            )

            self._manager = HnswSegmentManager(
                shard_path=str(self._path / "shard"),
                dim=dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown engine: {engine!r}. Use 'gem', 'hnsw', 'shard', or 'auto'.")

        if hasattr(self._manager, "_payloads"):
            self._payloads = dict(self._manager._payloads)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Index is closed")

    def add_texts(
        self,
        texts: List[str],
        *,
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add documents by text, using the configured embedding_fn."""
        if self._embedding_fn is None:
            raise RuntimeError(
                "No embedding_fn configured. Pass embedding_fn= to Index() or use add() with pre-computed vectors."
            )
        vecs = self._embedding_fn.embed_documents(texts)
        self.add(vecs, ids=ids, payloads=payloads)

    def search_text(
        self,
        text: str,
        k: int = 10,
        *,
        ef: int = 100,
        filters: Optional[Dict] = None,
        explain: bool = False,
    ) -> List[SearchResult]:
        """Search by text, using the configured embedding_fn."""
        if self._embedding_fn is None:
            raise RuntimeError("No embedding_fn configured.")
        query = self._embedding_fn.embed_query(text)
        return self.search(query, k=k, ef=ef, filters=filters, explain=explain)

    def add(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        *,
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add documents to the index.

        Args:
            vectors: (n_docs, n_tokens, dim) array or list of (n_tokens, dim) arrays.
            ids: Optional external document IDs. Auto-assigned if omitted.
            payloads: Optional per-document metadata dicts.
        """
        self._check_open()

        with self._lock:
            if isinstance(vectors, list):
                vecs_list = [np.asarray(v, dtype=np.float32) for v in vectors]
            elif isinstance(vectors, np.ndarray) and vectors.ndim == 3:
                vecs_list = [vectors[i] for i in range(vectors.shape[0])]
            elif isinstance(vectors, np.ndarray) and vectors.ndim == 2:
                vecs_list = [vectors]
            else:
                raise ValueError("vectors must be 2D, 3D array or list of 2D arrays")

            n_docs = len(vecs_list)

            for v in vecs_list:
                if v.shape[-1] != self._dim:
                    raise ValueError(f"dimension mismatch: expected {self._dim}, got {v.shape[-1]}")

            if ids is not None and len(ids) != n_docs:
                raise ValueError(f"ids length ({len(ids)}) != vectors count ({n_docs})")
            if payloads is not None and len(payloads) != n_docs:
                raise ValueError(f"payloads length ({len(payloads)}) != vectors count ({n_docs})")

            if ids is None:
                if hasattr(self._manager, "allocate_ids"):
                    ids = self._manager.allocate_ids(n_docs)
                else:
                    assigned = []
                    for _ in range(n_docs):
                        doc_id = self._manager._next_doc_id
                        assigned.append(doc_id)
                        self._manager._next_doc_id += 1
                    ids = assigned

            if payloads is None:
                payloads = [{} for _ in range(n_docs)]

            self._manager.add_multidense(vecs_list, ids, payloads)

            for i, doc_id in enumerate(ids):
                self._payloads[doc_id] = payloads[i]

    def add_batch(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        *,
        ids: Optional[List[int]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Alias for add() — same semantics."""
        self.add(vectors, ids=ids, payloads=payloads)

    def upsert(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        *,
        ids: List[int],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Insert or update documents by ID.

        If a document with a given ID already exists, it is replaced.
        Unlike ``add()``, ``ids`` is required.

        Args:
            vectors: (n_docs, n_tokens, dim) array or list of (n_tokens, dim) arrays.
            ids: External document IDs (required).
            payloads: Optional per-document metadata dicts.
        """
        self._check_open()

        with self._lock:
            if isinstance(vectors, list):
                vecs_list = [np.asarray(v, dtype=np.float32) for v in vectors]
            elif isinstance(vectors, np.ndarray) and vectors.ndim == 3:
                vecs_list = [vectors[i] for i in range(vectors.shape[0])]
            elif isinstance(vectors, np.ndarray) and vectors.ndim == 2:
                vecs_list = [vectors]
            else:
                raise ValueError("vectors must be 2D, 3D array or list of 2D arrays")

            n_docs = len(vecs_list)

            for v in vecs_list:
                if v.shape[-1] != self._dim:
                    raise ValueError(f"dimension mismatch: expected {self._dim}, got {v.shape[-1]}")

            if len(ids) != n_docs:
                raise ValueError(f"ids length ({len(ids)}) != vectors count ({n_docs})")
            if payloads is not None and len(payloads) != n_docs:
                raise ValueError(f"payloads length ({len(payloads)}) != vectors count ({n_docs})")

            if payloads is None:
                payloads = [{} for _ in range(n_docs)]

            self._manager.upsert_multidense(vecs_list, ids, payloads)

            for i, doc_id in enumerate(ids):
                self._payloads[doc_id] = payloads[i]

    def delete(self, ids: List[int]):
        """Delete documents by their IDs."""
        self._check_open()
        with self._lock:
            self._manager.delete(ids)
            for doc_id in ids:
                self._payloads.pop(doc_id, None)

    def update_payload(self, doc_id: int, payload: Dict[str, Any]):
        """
        Update the payload for a specific document.

        Args:
            doc_id: The document ID whose payload to update.
            payload: New metadata dict (replaces the old payload entirely).
        """
        self._check_open()
        with self._lock:
            self._payloads[doc_id] = payload
            self._manager.upsert_payload(doc_id, payload)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        ef: int = 100,
        n_probes: int = 4,
        filters: Optional[Dict] = None,
        explain: bool = False,
    ) -> List[SearchResult]:
        """
        Search the index for the k nearest documents.

        Args:
            query: ``(n_tokens, dim)`` float32 query vectors.
            k: Number of results to return.
            ef: Search beam width — higher values improve recall at the
                cost of latency.
            n_probes: Number of coarse clusters to probe (GEM only).
            filters: Qdrant-compatible payload filters (``$eq``, ``$in``,
                ``$contains``, etc.). Applied natively during graph traversal
                via cluster-level bitmap pruning.
            explain: If True, include per-query-token score attribution
                in each :class:`SearchResult` (``token_scores``,
                ``matched_tokens``).

        Returns:
            List of :class:`SearchResult` ordered by descending score.
        """
        self._check_open()

        search_kwargs_inner = {"k": k, "ef": ef, "filters": filters}
        if hasattr(self._manager, "search_multivector"):
            qv = query.astype(np.float32, copy=False)
            qv = qv if qv.ndim == 2 else qv.reshape(1, -1)
            raw = self._manager.search_multivector(
                qv,
                k=k,
                ef=ef,
                n_probes=n_probes,
                filters=filters,
            )
        else:
            raw = self._manager.search(
                query.astype(np.float32, copy=False),
                **search_kwargs_inner,
            )

        with self._lock:
            payloads_snap = dict(self._payloads)

        results = []
        for doc_id, score in raw:
            payload = payloads_snap.get(doc_id)
            tok_scores = None
            matched = None
            if explain and hasattr(self._manager, "explain_score"):
                tok_scores, matched = self._manager.explain_score(
                    query.astype(np.float32, copy=False),
                    doc_id,
                )
            results.append(
                SearchResult(
                    doc_id=int(doc_id),
                    score=float(score),
                    payload=payload,
                    token_scores=tok_scores,
                    matched_tokens=matched,
                )
            )

        return results

    def search_batch(
        self,
        queries: List[np.ndarray],
        k: int = 10,
        *,
        ef: int = 100,
        n_probes: int = 4,
        filters: Optional[Dict] = None,
    ) -> List[List[SearchResult]]:
        """
        Batch search: process multiple queries in parallel (fused SGEMM).

        Args:
            queries: list of (n_tokens, dim) float32 query arrays.
            k: number of results per query.
            ef: search beam width.
            n_probes: coarse clusters to probe.
            filters: payload filter applied post-hoc to all queries.

        Returns:
            List of result lists, one per query.
        """
        self._check_open()

        raw_batched = self._manager.search_batch(
            [q.astype(np.float32, copy=False) for q in queries],
            k=k,
            ef=ef,
            n_probes=n_probes,
        )

        with self._lock:
            payloads_snap = dict(self._payloads)

        all_results = []
        for raw in raw_batched:
            results = []
            for doc_id, score in raw:
                payload = payloads_snap.get(doc_id)
                if filters and hasattr(self._manager, "_evaluate_filter"):
                    if not self._manager._evaluate_filter(payload or {}, filters):
                        continue
                results.append(
                    SearchResult(
                        doc_id=int(doc_id),
                        score=float(score),
                        payload=payload,
                    )
                )
            all_results.append(results[:k])

        return all_results

    def get(self, ids: List[int]) -> List[Optional[Dict[str, Any]]]:
        """Retrieve payloads for the given document IDs."""
        self._check_open()
        with self._lock:
            return [self._payloads.get(doc_id) for doc_id in ids]

    def scroll(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        filters: Optional[Dict] = None,
    ) -> ScrollPage:
        """
        Paginated iteration over all documents.

        Returns a ScrollPage with results and next_offset (None if done).
        """
        self._check_open()

        with self._lock:
            if filters and hasattr(self._manager, "_match_filter"):
                all_ids = sorted(doc_id for doc_id in self._payloads if self._manager._match_filter(doc_id, filters))
                page_ids = all_ids[offset : offset + limit]
            else:
                ids_list = sorted(self._payloads)
                page_ids = ids_list[offset : offset + limit]
                all_ids = ids_list
            results = [
                SearchResult(doc_id=doc_id, score=0.0, payload=self._payloads.get(doc_id)) for doc_id in page_ids
            ]

            next_off = offset + limit if offset + limit < len(all_ids) else None
            return ScrollPage(results=results, next_offset=next_off)

    def snapshot(self, output_path: str):
        """Create a tarball snapshot of the index."""
        self._check_open()
        import shutil
        import tarfile
        import tempfile

        self.flush()

        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
            os.close(fd)
            tmp_path = tmp_name
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(str(self._path), arcname=os.path.basename(str(self._path)))
            shutil.move(tmp_path, output_path)
            tmp_path = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def stats(self) -> IndexStats:
        """Return summary statistics. Thread-safe via manager's internal RWLock."""
        self._check_open()
        raw = self._manager.get_statistics()
        total = raw.get("total_vectors", 0)
        sealed = raw.get("sealed_segments", 0)
        active_raw = raw.get("active", 0)
        if isinstance(active_raw, dict):
            active = active_raw.get("n_live", 0)
        else:
            active = int(active_raw) if active_raw else 0
        return IndexStats(
            total_documents=total,
            sealed_segments=sealed,
            active_documents=active,
            dim=self._dim,
            engine=self._engine,
        )

    def set_metrics_hook(self, hook: Callable[[str, float], None]) -> None:
        """
        Register a callback for internal search metrics (e.g. Prometheus).

        The hook is called with ``(metric_name, value)`` after each search
        operation.  Metric names include ``"search_latency_us"``,
        ``"candidates_scored"``, and ``"nodes_visited"``.

        Args:
            hook: Callable accepting ``(name: str, value: float)``.
        """
        self._metrics_hook = hook
        if hasattr(self._manager, "set_metrics_hook"):
            self._manager.set_metrics_hook(hook)

    def flush(self):
        """Flush all pending writes to disk."""
        self._check_open()
        self._manager.flush()

    def close(self):
        """Close the index, releasing all resources."""
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._manager.close()
                except Exception as e:
                    logger.warning("Error closing manager: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"Index(path={self._path!r}, dim={self._dim}, engine={self._engine!r}, {status})"

    def __del__(self):
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def engine(self) -> str:
        return self._engine


class IndexBuilder:
    """
    Fluent builder for creating an :class:`Index` with custom configuration.

    Every ``with_*`` method returns ``self`` so calls can be chained::

        idx = (IndexBuilder("my_index", dim=128)
               .with_shard(n_shards=64, k_candidates=512)
               .with_wal(enabled=True)
               .build())

    Args:
        path: Directory to store the index.
        dim: Vector dimensionality (must match the embedding model).
    """

    def __init__(self, path: str, dim: int) -> None:
        self._path = path
        self._dim = dim
        self._engine = "auto"
        self._kwargs: Dict[str, Any] = {}

    def with_gem(self, **kwargs: Any) -> "IndexBuilder":
        """Select the GEM engine with optional keyword overrides.

        Common kwargs: ``seed_batch_size``, ``n_fine``, ``n_coarse``,
        ``max_degree``, ``ef_construction``, ``dual_graph``, ``use_emd``.
        """
        self._engine = "gem"
        self._kwargs.update(kwargs)
        return self

    def with_hnsw(self, **kwargs: Any) -> "IndexBuilder":
        """Select the HNSW engine (legacy single-vector backend)."""
        self._engine = "hnsw"
        self._kwargs.update(kwargs)
        return self

    def with_shard(self, **kwargs: Any) -> "IndexBuilder":
        """Select the LEMUR-routed shard engine for scalable late-interaction retrieval.

        Common kwargs: ``n_shards``, ``k_candidates``, ``lemur_epochs``,
        ``compression``, ``device``.
        """
        self._engine = "shard"
        self._kwargs.update(kwargs)
        return self

    def with_wal(self, enabled: bool = True) -> "IndexBuilder":
        """Enable or disable the write-ahead log for crash recovery.

        Args:
            enabled: ``True`` (default) enables WAL + CRC32 checkpointing.
        """
        self._kwargs["enable_wal"] = enabled
        return self

    def with_quantization(self, n_fine: int = 256, n_coarse: int = 32) -> "IndexBuilder":
        """Configure the two-stage codebook for qCH proxy scoring.

        Args:
            n_fine: Number of fine centroids (higher = better proxy accuracy,
                more build time). Recommended: 128–2048 depending on corpus.
            n_coarse: Number of coarse routing clusters.
        """
        self._kwargs["n_fine"] = n_fine
        self._kwargs["n_coarse"] = n_coarse
        return self

    def with_gpu_rerank(self, device: str = "cuda") -> IndexBuilder:
        """Enable GPU-accelerated MaxSim reranking.

        GEM proxy search produces candidates ranked by quantized Chamfer
        distance.  This option adds a second-stage exact late-interaction
        rerank on the specified device for higher recall.

        Args:
            device: PyTorch device string (default ``'cuda'``).
        """
        self._kwargs["rerank_device"] = device
        return self

    def with_roq(self, bits: int = 4, device: str = "cuda") -> IndexBuilder:
        """Enable ROQ (Rotational Quantization) compressed reranking.

        Stores document vectors in ``bits``-bit quantized form and runs
        MaxSim reranking with a fused Triton kernel.  Reduces memory by
        ~8x (at 4-bit) compared to FP32 with minimal recall loss.

        Args:
            bits: Quantization bit width (default 4).
            device: PyTorch device string (default ``'cuda'``).
        """
        self._kwargs["roq_rerank"] = True
        self._kwargs["roq_bits"] = bits
        self._kwargs["rerank_device"] = device
        return self

    def build(self) -> Index:
        return Index(
            self._path,
            self._dim,
            engine=self._engine,
            **self._kwargs,
        )
