"""ShardSegmentManager: production manager for the LEMUR-routed shard engine.

Supports WAL, retrieve, scroll, flush, memtable CRUD, payload filters,
ROQ-4 bit scoring, and optional BM25-hybrid re-ranking.
"""
from __future__ import annotations

import gc
import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .checkpoint import ShardCheckpointManager
try:
    from voyager_index._internal.inference.index_core.io_utils import (
        FileLock as _FileLock,
        atomic_json_write as _atomic_json_write,
    )
except ImportError:
    _FileLock = None
    _atomic_json_write = None


def atomic_json_write(path: Path, data: Any) -> None:
    """Atomic JSON write with graceful fallback to plain json.dump."""
    if _atomic_json_write is not None:
        _atomic_json_write(path, data)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
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
from .memtable import MemTable
from .profiler import QueryProfile, Timer
from .scorer import (
    PreloadedGpuCorpus,
    brute_force_maxsim,
    proxy_score_candidates,
    score_all_docs_topk,
    warmup_maxsim,
)
from .shard_store import ShardStore
from .wal import WalEntry, WalOp, WalReader, WalWriter

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
        ann_backend: AnnBackend = AnnBackend.FAISS_FLAT_IP,
        lemur_epochs: int = 10,
        k_candidates: int = 2000,
        n_full_scores: int = 4096,
        transfer_mode: TransferMode = TransferMode.PINNED,
        pinned_pool_buffers: int = 3,
        pinned_buffer_max_tokens: int = 50_000,
        use_colbandit: bool = False,
        uniform_shard_tokens: bool = True,
        quantization_mode: str = "",
        n_centroids: int = 1024,
        n_centroid_approx: int = 512,
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
        self.n_full_scores = n_full_scores
        self.transfer_mode = transfer_mode
        self.pinned_pool_buffers = pinned_pool_buffers
        self.pinned_buffer_max_tokens = pinned_buffer_max_tokens
        self.use_colbandit = use_colbandit
        self.uniform_shard_tokens = uniform_shard_tokens
        self.quantization_mode = quantization_mode
        self.n_centroids = n_centroids
        self.n_centroid_approx = n_centroid_approx
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
            n_full_scores=self.n_full_scores,
            n_centroid_approx=self.n_centroid_approx,
            transfer_mode=self.transfer_mode,
            pinned_pool_buffers=self.pinned_pool_buffers,
            pinned_buffer_max_tokens=self.pinned_buffer_max_tokens,
            use_colbandit=self.use_colbandit,
            quantization_mode=self.quantization_mode,
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
        self._memtable: Optional[MemTable] = None
        self._wal_writer: Optional[WalWriter] = None
        self._next_doc_id: int = 0
        self._metrics_hook: Optional[Callable[[str, float], None]] = None
        self._page_cache_last: Optional[Tuple[float, float]] = None  # (timestamp, hit_rate)
        self._doc_means: Optional[torch.Tensor] = None
        self._doc_mean_id_to_idx: Optional[Dict[int, int]] = None
        self._rust_index = None
        self._rust_tmpdir: Optional[str] = None
        self._checkpoint_mgr = ShardCheckpointManager(self._path)
        self._file_lock: Optional[Any] = None
        if _FileLock is not None:
            try:
                self._file_lock = _FileLock(self._path / ".lock")
                self._file_lock.acquire()
            except Exception as exc:
                logger.warning("FileLock not acquired (non-fatal): %s", exc)
                self._file_lock = None

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

            roq_doc_codes = None
            roq_doc_meta = None
            effective_compression = cfg.compression
            if cfg.compression == Compression.ROQ4:
                try:
                    from voyager_index._internal.inference.quantization.rotational import (
                        RotationalQuantizer, RoQConfig,
                    )
                    import pickle
                    logger.info("Training ROQ 4-bit quantizer ...")
                    roq_q = RotationalQuantizer(RoQConfig(dim=self._dim, num_bits=4, seed=cfg.seed))
                    roq_doc_codes = []
                    roq_doc_meta = []
                    for i, (s, e) in enumerate(doc_offsets):
                        vecs_slice = np.asarray(all_vecs[s:e], dtype=np.float32)
                        q = roq_q.quantize(vecs_slice, store=False)
                        roq_doc_codes.append(np.asarray(q["codes"], dtype=np.uint8))
                        roq_doc_meta.append(roq_q.build_triton_meta(q, include_norm_sq=True))
                    with open(self._path / "roq_quantizer.pkl", "wb") as f:
                        pickle.dump(roq_q, f)
                    logger.info("ROQ 4-bit encoding done for %d docs", n_docs)
                except (ImportError, Exception) as exc:
                    logger.warning("ROQ quantizer unavailable (%s), falling back to FP16", exc)
                    effective_compression = Compression.FP16

            store = ShardStore(self._path)
            store.build(
                all_vectors=all_vecs,
                doc_offsets=doc_offsets,
                doc_ids=list(ids),
                shard_assignments=shard_assignments,
                n_shards=cfg.n_shards,
                dim=self._dim,
                compression=effective_compression,
                uniform_shard_tokens=cfg.uniform_shard_tokens,
                roq_doc_codes=roq_doc_codes,
                roq_doc_meta=roq_doc_meta,
            )

            meta = {
                "dim": self._dim,
                "n_docs": n_docs,
                "n_shards": cfg.n_shards,
                "compression": effective_compression.value,
                "layout": cfg.layout.value,
                "router_type": cfg.router_type.value,
            }
            atomic_json_write(self._path / "engine_meta.json", meta)

            payload_dict = {}
            if payloads:
                payload_dict = {str(did): p for did, p in zip(ids, payloads)}
            atomic_json_write(self._path / "payloads.json", payload_dict)

            self._store = store
            self._router = router
            self._doc_vecs = vectors
            self._doc_ids = list(ids)
            self._sealed_payloads: Dict[int, dict] = {
                did: payload_dict.get(str(did), {}) for did in ids
            }
            self._is_built = True
            self._next_doc_id = max(ids) + 1 if ids else 0
            self._init_pipeline()
            self._try_gpu_preload()
            self._init_wal_and_memtable()
            self._build_and_save_doc_means(all_vecs, doc_offsets, list(ids))

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
            self._next_doc_id = max(self._doc_ids) + 1 if self._doc_ids else 0

            self._sealed_payloads = self._load_sealed_payloads()

            self._init_wal_and_memtable()
            self._replay_wal()
            self._load_doc_means()
            self._init_rust_index()

            if self._rust_index is not None and self._config.n_centroid_approx > 0:
                logger.info(
                    "Centroid-approx active (n=%d) — skipping sealed vector preload",
                    self._config.n_centroid_approx,
                )
                self._doc_vecs = None
            else:
                self._doc_vecs = self._load_sealed_vectors()
                self._try_gpu_preload()
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
        ef: Optional[int] = None,
        n_probes: Optional[int] = None,
        filters: Optional[Dict] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """Search with a multi-vector query (ColBERT-style token embeddings).

        Args:
            query_vectors: Query token embeddings, shape [n_tokens, dim].
            k: Number of results to return.
            ef: Accepted for API compatibility with HNSW; not used by shard routing.
            n_probes: Optional nprobe override for IVF-PQ routing indices.
            filters: Optional payload filters (post-filter with widened k).

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if not self._is_built:
            raise RuntimeError("Index not built or loaded")

        t_start = time.perf_counter()
        self._ensure_warmup()

        q = torch.from_numpy(query_vectors).float() if isinstance(query_vectors, np.ndarray) else query_vectors.float()
        dev = torch.device(self._device if torch.cuda.is_available() else "cpu")

        scfg = self._config.to_search_config()
        n_sealed = len(self._doc_ids) if self._doc_ids else 0

        internal_k = k * 4 if filters else k
        effective_k = min(scfg.k_candidates, max(n_sealed, 1))

        with self._lock:
            if self._memtable:
                mt_docs_snap, mt_payloads_snap, tombstones = self._memtable.snapshot()
            else:
                mt_docs_snap, mt_payloads_snap, tombstones = {}, {}, set()
            sealed_payloads_snap = dict(self._sealed_payloads) if hasattr(self, '_sealed_payloads') and self._sealed_payloads else {}

        with Timer(sync_cuda=True) as t_route:
            routed = self._router.route(
                q,
                k_candidates=effective_k,
                prefetch_doc_cap=scfg.max_docs_exact,
                nprobe_override=n_probes,
            )

        # --- Stage 1: Centroid-code approximate MaxSim (Rust + Rayon) ---
        if (
            self._rust_index is not None
            and scfg.n_centroid_approx > 0
            and len(routed.doc_ids) > scfg.n_centroid_approx
        ):
            q_np = q.cpu().numpy().astype(np.float32)
            approx_ids, _approx_scores = self._rust_index.score_candidates_approx(
                q_np, routed.doc_ids, scfg.n_centroid_approx,
            )
            pruned_ids = approx_ids.tolist()
            pruned_set = set(pruned_ids)
            routed_by_shard = {
                sid: [d for d in dids if d in pruned_set]
                for sid, dids in routed.by_shard.items()
            }
            routed_by_shard = {sid: dids for sid, dids in routed_by_shard.items() if dids}
        elif self._doc_means is not None and self._doc_mean_id_to_idx is not None:
            pruned_ids = proxy_score_candidates(
                query=q,
                doc_means=self._doc_means,
                candidate_doc_ids=routed.doc_ids,
                doc_id_to_idx=self._doc_mean_id_to_idx,
                n_full_scores=scfg.n_full_scores,
                device=dev,
            )
            pruned_set = set(pruned_ids)
            routed_by_shard = {
                sid: [d for d in dids if d in pruned_set]
                for sid, dids in routed.by_shard.items()
            }
            routed_by_shard = {sid: dids for sid, dids in routed_by_shard.items() if dids}
        else:
            pruned_ids = routed.doc_ids
            routed_by_shard = routed.by_shard

        sealed_results: List[Tuple[int, float]] = []

        if self._gpu_corpus is not None:
            candidate_ids = pruned_ids[:scfg.max_docs_exact]
            ids, scores = self._gpu_corpus.score_candidates(q, candidate_ids, k=internal_k)
            sealed_results = list(zip(ids, scores))
        else:
            shard_chunks, _stats = self._pipeline.fetch_candidate_docs(
                routed_by_shard, max_docs=scfg.max_docs_exact,
            )
            if shard_chunks:
                ids, scores = score_all_docs_topk(
                    q, shard_chunks, k=internal_k, device=dev,
                    quantization_mode=scfg.quantization_mode,
                )
                sealed_results = list(zip(ids, scores))

        memtable_results: List[Tuple[int, float]] = []
        if self._memtable and self._memtable.size > 0:
            memtable_results = self._memtable.search(
                query_vectors if isinstance(query_vectors, np.ndarray) else query_vectors.numpy(),
                k=internal_k,
            )

        merged: Dict[int, float] = {}
        for did, sc in sealed_results:
            if did not in tombstones:
                merged[did] = max(merged.get(did, float("-inf")), sc)
        for did, sc in memtable_results:
            merged[did] = max(merged.get(did, float("-inf")), sc)

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        if filters:
            filtered = []
            for did, sc in ranked:
                payload = dict(sealed_payloads_snap.get(did, {}))
                if did in mt_payloads_snap:
                    payload.update(mt_payloads_snap[did])
                if self._evaluate_filter(payload, filters):
                    filtered.append((did, sc))
            ranked = filtered

        result = ranked[:k]

        elapsed_us = (time.perf_counter() - t_start) * 1_000_000
        self._emit_metric("search_latency_us", elapsed_us)
        self._emit_metric("candidates_routed", len(routed.doc_ids))
        self._emit_metric("candidates_scored", len(pruned_ids))
        self._emit_metric("route_ms", t_route.elapsed_ms)

        now = time.perf_counter()
        if (
            self._metrics_hook
            and self._store
            and (self._page_cache_last is None or now - self._page_cache_last[0] > 5.0)
        ):
            try:
                pc = self._store.page_cache_residency()
                if pc is not None:
                    self._page_cache_last = (now, pc["hit_rate"])
                    self._emit_metric("page_cache_hit_rate", pc["hit_rate"])
            except Exception:
                pass

        return result

    def search_batch(
        self,
        queries: List[np.ndarray],
        k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search over multiple queries."""
        return [self.search_multivector(q, k=k, filters=filters) for q in queries]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """Alias for search_multivector."""
        return self.search_multivector(query, k=k, filters=filters)

    def scroll(self, limit: int = 100, offset: int = 0,
               filters: Optional[Dict] = None) -> Tuple[List[int], Optional[int]]:
        """Paginate through all document IDs.

        Returns:
            (page_ids, next_offset) — next_offset is None when done.
        """
        with self._lock:
            all_ids = sorted(set(self._doc_ids or []))
            if self._memtable:
                docs, _, tombstones = self._memtable.snapshot()
                all_ids = sorted((set(all_ids) | set(docs.keys())) - tombstones)

            if filters:
                all_ids = [did for did in all_ids if self._match_filter(did, filters)]

        page = all_ids[offset:offset + limit]
        next_off = offset + limit if offset + limit < len(all_ids) else None
        return page, next_off

    def explain_score(
        self, query: np.ndarray, doc_id: int,
    ) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """Per-query-token score attribution for a single document.

        Returns (token_scores, matched_doc_tokens) or (None, None) if unavailable.
        """
        doc_vec = self._get_doc_vectors(doc_id)
        if doc_vec is None:
            return None, None
        q = torch.from_numpy(query).float()
        d = torch.from_numpy(doc_vec).float()
        sim = q @ d.T
        max_sim, matched = sim.max(dim=1)
        return max_sim.tolist(), matched.tolist()

    _explain_score = explain_score

    @staticmethod
    def _evaluate_filter(payload: dict, filters: Dict) -> bool:
        """Evaluate Qdrant-style payload filters with $and/$or support."""
        if not filters:
            return True
        if "$and" in filters:
            return all(
                ShardSegmentManager._evaluate_filter(payload, sub)
                for sub in filters["$and"]
            )
        if "$or" in filters:
            return any(
                ShardSegmentManager._evaluate_filter(payload, sub)
                for sub in filters["$or"]
            )
        for key, condition in filters.items():
            val = payload.get(key)
            if isinstance(condition, dict):
                if "$eq" in condition and val != condition["$eq"]:
                    return False
                if "$in" in condition and val not in condition["$in"]:
                    return False
                if "$contains" in condition:
                    if not isinstance(val, (list, str)) or condition["$contains"] not in val:
                        return False
                if "$gt" in condition:
                    try:
                        if val is None or val <= condition["$gt"]:
                            return False
                    except TypeError:
                        return False
                if "$lt" in condition:
                    try:
                        if val is None or val >= condition["$lt"]:
                            return False
                    except TypeError:
                        return False
            elif val != condition:
                return False
        return True

    def _match_filter(self, doc_id: int, filters: Dict) -> bool:
        """Check if a document matches the given filters (sealed + memtable)."""
        payload = self._get_payload(doc_id)
        return self._evaluate_filter(payload, filters)

    def _get_payload(self, doc_id: int, tombstones: Optional[Set[int]] = None) -> Optional[dict]:
        """Retrieve merged payload for a doc from sealed store and memtable."""
        if tombstones and doc_id in tombstones:
            return None
        payload = {}
        if hasattr(self, '_sealed_payloads') and self._sealed_payloads:
            payload = dict(self._sealed_payloads.get(doc_id, {}))
        if self._memtable:
            _, mt_payloads, _ = self._memtable.snapshot()
            if doc_id in mt_payloads:
                payload.update(mt_payloads[doc_id])
        return payload

    def _get_doc_vectors(self, doc_id: int) -> Optional[np.ndarray]:
        """Retrieve vectors for a doc from memtable or sealed store."""
        if self._memtable:
            docs, _, _ = self._memtable.snapshot()
            if doc_id in docs:
                return docs[doc_id]
        if self._doc_vecs and self._doc_ids:
            try:
                idx = self._doc_ids.index(doc_id)
                v = self._doc_vecs[idx]
                return np.asarray(v, dtype=np.float32)
            except (ValueError, IndexError):
                pass
        if self._store:
            try:
                fetched = self._store.fetch_docs([doc_id])
                if fetched:
                    return list(fetched.values())[0]
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # CRUD
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
        with self._lock:
            plds = payloads or [None] * len(vectors)
            for v, did, p in zip(vectors, ids, plds):
                arr = np.asarray(v, dtype=np.float32)
                if self._wal_writer:
                    self._wal_writer.log_insert(did, arr, p)
                self._memtable.insert(did, arr, p)
                self._next_doc_id = max(self._next_doc_id, did + 1)

    def delete(self, ids: List[int]) -> None:
        with self._lock:
            for did in ids:
                if self._wal_writer:
                    self._wal_writer.log_delete(did)
                if self._memtable:
                    self._memtable.delete(did)
            if self._router:
                self._router.delete_docs(ids)

    def upsert_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        if not self._is_built:
            self.build(vectors, ids, payloads)
            return
        with self._lock:
            plds = payloads or [None] * len(vectors)
            for v, did, p in zip(vectors, ids, plds):
                arr = np.asarray(v, dtype=np.float32)
                if self._wal_writer:
                    self._wal_writer.log_upsert(did, arr, p)
                if self._memtable:
                    self._memtable.upsert(did, arr, p)
                self._next_doc_id = max(self._next_doc_id, did + 1)

    # ------------------------------------------------------------------
    # Statistics / lifecycle
    # ------------------------------------------------------------------

    def _compute_live_count(self) -> int:
        """Compute distinct live doc count (sealed + memtable - tombstones), no double-counting."""
        sealed_ids = set(self._doc_ids) if self._doc_ids else set()
        if self._memtable:
            docs, _, tombstones = self._memtable.snapshot()
            unique_ids = (sealed_ids | set(docs.keys())) - tombstones
        else:
            unique_ids = sealed_ids
        return len(unique_ids)

    def set_metrics_hook(self, hook: Callable[[str, float], None]) -> None:
        """Register a callback for internal search metrics (e.g. Prometheus)."""
        self._metrics_hook = hook

    def _emit_metric(self, name: str, value: Any) -> None:
        if self._metrics_hook:
            try:
                self._metrics_hook(name, float(value))
            except Exception as exc:
                logger.debug("Metrics hook error: %s", exc)

    def get_statistics(self) -> Dict[str, Any]:
        n_memtable = self._memtable.size if self._memtable else 0
        n_tombstones = self._memtable.tombstone_count if self._memtable else 0
        n_live = self._compute_live_count()

        stats: Dict[str, Any] = {
            "engine": "shard",
            "path": str(self._path),
            "dim": self._dim,
            "is_built": self._is_built,
            "device": self._device,
            "gpu_corpus_loaded": self._gpu_corpus is not None,
            "total_vectors": n_live,
            "active": n_memtable,
            "n_live": n_live,
            "n_tombstones": n_tombstones,
        }
        if self._store and self._store.manifest:
            m = self._store.manifest
            stats.update({
                "num_docs": m.num_docs,
                "sealed_docs": m.num_docs,
                "num_shards": m.num_shards,
                "total_tokens": m.total_tokens,
                "compression": m.compression,
                "avg_tokens_per_doc": m.avg_tokens_per_chunk,
            })
        if self._router:
            stats["router_type"] = "lemur"
        if self._store:
            pc = self._store.page_cache_residency()
            if pc is not None:
                stats["page_cache"] = pc
        return stats

    def total_vectors(self) -> int:
        return self._compute_live_count()

    def allocate_ids(self, n: int) -> List[int]:
        """Allocate n new sequential document IDs using monotonic counter."""
        with self._lock:
            start = self._next_doc_id
            self._next_doc_id += n
            return list(range(start, start + n))

    def upsert_payload(self, doc_id: int, payload: Dict[str, Any]) -> None:
        """Update or insert payload for a document."""
        with self._lock:
            if hasattr(self, '_sealed_payloads') and doc_id in (self._sealed_payloads or {}):
                self._sealed_payloads[doc_id].update(payload)
            if self._memtable:
                _, mt_payloads, _ = self._memtable.snapshot()
                merged = mt_payloads.get(doc_id, {})
                merged.update(payload)
                self._memtable.upsert_payload(doc_id, merged)
            if self._wal_writer:
                self._wal_writer.log_update_payload(doc_id, payload)

    def retrieve(self, ids: List[int], with_vector: bool = False,
                 with_payload: bool = True) -> list:
        """Retrieve documents by ID from sealed store and memtable.

        Tombstoned documents are skipped (not included in results).
        """
        tombstones = self._memtable.tombstones_snapshot() if self._memtable else set()
        results = []
        for did in ids:
            if did in tombstones:
                continue
            entry: Dict[str, Any] = {"id": did}
            if with_payload:
                entry["payload"] = self._get_payload(did, tombstones) or {}
            else:
                entry["payload"] = {}
            if with_vector:
                entry["vector"] = self._get_doc_vectors(did)
            results.append(entry)
        return results

    def flush(self) -> None:
        """Checkpoint: persist WAL but do NOT drain memtable.

        Real L0-to-sealed merge is not yet implemented; draining memtable
        without persisting to ShardStore would cause data loss. The WAL
        ensures crash recovery of memtable state.
        """
        with self._lock:
            if self._wal_writer:
                self._wal_writer.sync()
            if self._memtable:
                mt_docs, mt_payloads, mt_tombstones = self._memtable.snapshot()
                wal_offset = self._wal_writer.n_entries if self._wal_writer else 0
                sealed_snap = dict(self._sealed_payloads) if hasattr(self, '_sealed_payloads') and self._sealed_payloads else {}
                try:
                    self._checkpoint_mgr.save(
                        memtable_docs=mt_docs,
                        payloads=mt_payloads,
                        tombstones=mt_tombstones,
                        next_doc_id=self._next_doc_id,
                        wal_offset=wal_offset,
                        sealed_payloads=sealed_snap,
                    )
                except Exception as exc:
                    logger.warning("Checkpoint save failed (WAL still safe): %s", exc)
            logger.debug("flush(): WAL synced + checkpoint saved")

    def refresh_gpu_corpus(self) -> None:
        """Rebuild GPU corpus tensors from current sealed snapshot (call after compaction)."""
        if self._gpu_corpus and self._doc_vecs and self._doc_ids:
            self._gpu_corpus.refresh(self._doc_vecs, self._doc_ids)

    def save(self) -> None:
        """Persist current state (router already auto-saves during fit)."""
        with self._lock:
            if self._router:
                self._router.save()

    def close(self) -> None:
        """Release resources."""
        with self._lock:
            if self._wal_writer:
                self._wal_writer.close()
                self._wal_writer = None
            self._memtable = None
            self._gpu_corpus = None
            self._pipeline = None
            self._store = None
            self._router = None
            self._doc_vecs = None
            self._doc_means = None
            self._doc_mean_id_to_idx = None
            if self._rust_index is not None:
                try:
                    self._rust_index.close()
                except Exception:
                    pass
                self._rust_index = None
            if self._rust_tmpdir:
                import shutil
                shutil.rmtree(self._rust_tmpdir, ignore_errors=True)
                self._rust_tmpdir = None
            self._sealed_payloads = {}
            self._is_built = False
            if self._file_lock is not None:
                try:
                    self._file_lock.release()
                except Exception:
                    pass
                self._file_lock = None
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

    def _init_wal_and_memtable(self) -> None:
        self._memtable = MemTable(dim=self._dim, device=self._device)
        wal_path = self._path / "wal.bin"
        self._wal_writer = WalWriter(wal_path)
        self._wal_writer.open()

    def _replay_wal(self) -> None:
        """Replay WAL entries from disk into the memtable for crash recovery.

        If a checkpoint exists, its memtable state is restored first and only
        WAL entries written *after* the checkpoint are replayed.
        """
        ckpt = self._checkpoint_mgr.load()
        if ckpt is not None:
            logger.info(
                "Restoring checkpoint: %d docs, wal_offset=%d",
                len(ckpt["docs"]), ckpt["wal_offset"],
            )
            for doc_id, vecs in ckpt["docs"].items():
                self._memtable.insert(doc_id, vecs, ckpt["payloads"].get(doc_id))
            for doc_id in ckpt["tombstones"]:
                self._memtable.delete(doc_id)
            self._next_doc_id = max(self._next_doc_id, ckpt["next_doc_id"])
            sealed_ckpt = ckpt.get("sealed_payloads")
            if sealed_ckpt and hasattr(self, "_sealed_payloads"):
                self._sealed_payloads.update(sealed_ckpt)

        wal_path = self._path / "wal.bin"
        if not wal_path.exists():
            return
        entries = WalReader(wal_path).replay()
        if not entries:
            return
        wal_skip = ckpt["wal_offset"] if ckpt else 0
        entries = entries[wal_skip:]
        if not entries:
            return
        logger.info("Replaying %d WAL entries for crash recovery", len(entries))
        for entry in entries:
            if entry.op == WalOp.INSERT:
                if entry.vectors is not None:
                    self._memtable.insert(entry.doc_id, entry.vectors, entry.payload)
                    self._next_doc_id = max(self._next_doc_id, entry.doc_id + 1)
            elif entry.op == WalOp.DELETE:
                self._memtable.delete(entry.doc_id)
            elif entry.op == WalOp.UPSERT:
                if entry.vectors is not None:
                    self._memtable.upsert(entry.doc_id, entry.vectors, entry.payload)
                    self._next_doc_id = max(self._next_doc_id, entry.doc_id + 1)
            elif entry.op == WalOp.UPDATE_PAYLOAD:
                if entry.payload is not None:
                    self._memtable.upsert_payload(entry.doc_id, entry.payload)
                    if hasattr(self, '_sealed_payloads') and self._sealed_payloads and entry.doc_id in self._sealed_payloads:
                        self._sealed_payloads[entry.doc_id].update(entry.payload)
        logger.info("WAL replay done: memtable has %d docs, %d tombstones",
                    self._memtable.size, self._memtable.tombstone_count)

    def _build_and_save_doc_means(
        self, all_vecs: np.ndarray, doc_offsets: List[Tuple[int, int]], ids: List[int],
    ) -> None:
        """Compute mean-pooled doc embeddings and persist for proxy scoring."""
        dim = all_vecs.shape[1]
        means = np.zeros((len(ids), dim), dtype=np.float16)
        for i, (s, e) in enumerate(doc_offsets):
            if e > s:
                means[i] = all_vecs[s:e].astype(np.float32).mean(axis=0).astype(np.float16)
        np.savez_compressed(
            self._path / "doc_means.npz",
            means=means,
            ids=np.array(ids, dtype=np.int64),
        )
        dev = torch.device(self._device if torch.cuda.is_available() else "cpu")
        self._doc_means = torch.from_numpy(means).to(dev)
        self._doc_mean_id_to_idx = {int(did): i for i, did in enumerate(ids)}
        logger.info("Doc-mean proxy embeddings saved: %d docs, %.1f MB",
                     len(ids), means.nbytes / 1e6)

    def _load_doc_means(self) -> None:
        """Load mean-pooled doc embeddings for proxy scoring (if available)."""
        path = self._path / "doc_means.npz"
        if not path.exists():
            return
        try:
            with np.load(path) as data:
                means = data["means"]
                ids = data["ids"]
            dev = torch.device(self._device if torch.cuda.is_available() else "cpu")
            self._doc_means = torch.from_numpy(means.copy()).to(dev)
            self._doc_mean_id_to_idx = {int(did): i for i, did in enumerate(ids)}
            logger.info("Doc-mean proxy embeddings loaded: %d docs on %s", len(ids), dev)
        except Exception as exc:
            logger.warning("Could not load doc means for proxy scoring: %s", exc)

    def _init_rust_index(self) -> None:
        """Initialize the Rust ShardIndex for centroid-code approximate scoring."""
        centroids_path = self._path / "centroids.npy"
        doc_index_path = self._path / "doc_index.npz"
        if not centroids_path.exists() or not doc_index_path.exists():
            logger.info("Centroid codes not available — skipping Rust approx scoring")
            return
        try:
            import latence_shard_engine
        except ImportError:
            logger.warning("latence_shard_engine not installed — skipping Rust approx scoring")
            return
        try:
            centroids = np.load(str(centroids_path)).astype(np.float32)
            doc_index = np.load(str(doc_index_path), allow_pickle=True)
            doc_ids = doc_index["doc_ids"]
            shard_ids = doc_index["shard_ids"]
            local_starts = doc_index["local_starts"]
            local_ends = doc_index["local_ends"]

            shard_dir = self._path / "shards"
            shard_files = sorted(shard_dir.glob("shard_*.safetensors"))
            if not shard_files:
                logger.warning("No shard files found — skipping Rust approx scoring")
                return

            import tempfile, os, shutil
            if self._rust_tmpdir and os.path.isdir(self._rust_tmpdir):
                shutil.rmtree(self._rust_tmpdir, ignore_errors=True)
            tmpdir = tempfile.mkdtemp(prefix="shard_idx_")
            self._rust_tmpdir = tmpdir
            for i, sf in enumerate(shard_files):
                os.symlink(str(sf), os.path.join(tmpdir, f"shard_{i}.safetensors"))
            open(os.path.join(tmpdir, "shard.wal"), "wb").close()

            rust_idx = latence_shard_engine.ShardIndex(tmpdir, self._dim)
            rust_idx.register_shard_docs(
                doc_ids.tolist(),
                shard_ids.astype(np.uint32).tolist(),
                local_starts.tolist(),
                local_ends.tolist(),
            )
            rust_idx.set_centroids(centroids)
            self._rust_index = rust_idx
            logger.info(
                "Rust ShardIndex ready: %d docs, %d centroids for approx scoring",
                rust_idx.doc_count, len(centroids),
            )
        except Exception as exc:
            logger.warning("Failed to initialize Rust ShardIndex: %s", exc)
            self._rust_index = None

    def _load_sealed_payloads(self) -> Dict[int, dict]:
        """Load sealed payloads from payloads.json."""
        payload_path = self._path / "payloads.json"
        if not payload_path.exists():
            return {}
        try:
            with open(payload_path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Failed to load sealed payloads: %s", exc)
            return {}

    def _load_sealed_vectors(self) -> Optional[List[np.ndarray]]:
        """Load sealed doc vectors from the shard store for GPU preload (single batch)."""
        if not self._store or not self._doc_ids:
            return None
        try:
            fetched = self._store.fetch_docs(self._doc_ids)
            all_vecs = []
            missing = 0
            for did in self._doc_ids:
                if did in fetched:
                    all_vecs.append(fetched[did])
                else:
                    missing += 1
                    all_vecs.append(np.zeros((1, self._dim), dtype=np.float16))
            if missing > 0:
                logger.warning("%d docs missing from shard store during GPU preload", missing)
            return all_vecs
        except Exception as exc:
            logger.warning("Could not load sealed vectors for GPU preload: %s", exc)
            return None
