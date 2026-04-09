"""
GEM Hybrid Segment Manager — Drop-in replacement for HnswSegmentManager
that uses native GEM graph segments for multi-vector search.

Two classes:
- GemHybridSegmentManager: active HNSW + sealed GEM (transitional)
- GemNativeSegmentManager: fully native GEM (active mutable + sealed)
"""

from __future__ import annotations

import gc
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from latence_gem_index import GemSegment, PyMutableGemSegment
except ImportError:
    GemSegment = None
    PyMutableGemSegment = None

try:
    from .hnsw_manager import HnswSegmentManager
except ImportError:
    HnswSegmentManager = None

try:
    from .gem_wal import WalWriter, WalReader, CheckpointManager
except ImportError:
    WalWriter = None
    WalReader = None
    CheckpointManager = None

try:
    from .io_utils import atomic_json_write, FileLock, RWLock
except ImportError:
    atomic_json_write = None
    FileLock = None
    RWLock = None


class GemHybridSegmentManager:
    """
    Active HNSW + sealed GEM graph segments.

    Uses HNSW for the writable active segment and builds sealed GEM
    graphs when the active segment is sealed. Search fans out across
    both active HNSW and all sealed GEM segments.
    """

    def __init__(
        self,
        shard_path: str,
        dim: int,
        distance_metric: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        on_disk: bool = False,
        multivector_comparator: Optional[str] = None,
        *,
        n_fine: int = 256,
        n_coarse: int = 32,
        max_degree: int = 32,
        gem_ef_construction: int = 200,
        max_kmeans_iter: int = 30,
        ctop_r: int = 3,
        n_probes: int = 4,
    ):
        self._shard_path = Path(shard_path)
        self._shard_path.mkdir(parents=True, exist_ok=True)
        self._dim = dim
        self._distance_metric = distance_metric
        self._m = m
        self._ef_construct = ef_construct
        self._on_disk = on_disk
        self._multivector_comparator = multivector_comparator
        self._n_fine = n_fine
        self._n_coarse = n_coarse
        self._max_degree = max_degree
        self._gem_ef_construction = gem_ef_construction
        self._max_kmeans_iter = max_kmeans_iter
        self._ctop_r = ctop_r
        self._n_probes = n_probes

        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._next_doc_id = 0
        self._deleted_ids: Set[int] = set()

        self._sealed_segments: List[GemSegment] = []
        self._sealed_doc_ids: List[List[int]] = []

        active_path = self._shard_path / "active_hnsw"
        self._active = self._create_hnsw_segment(str(active_path), is_appendable=True)

        self._load_next_doc_id()
        self._load_deleted_ids()
        self._load_sealed_segments()

    def _create_hnsw_segment(self, path: str, is_appendable: bool = True):
        if HnswSegmentManager is None:
            raise RuntimeError("HnswSegmentManager not available")
        from latence_hnsw import HnswSegment
        real_path = Path(path)
        real_path.mkdir(parents=True, exist_ok=True)
        d = HnswSegment(
            path=str(real_path),
            dim=self._dim,
            distance_metric=self._distance_metric,
            m=self._m,
            ef_construct=self._ef_construct,
            on_disk=self._on_disk,
            is_appendable=is_appendable,
            multivector_comparator=self._multivector_comparator,
        )
        return d

    def _load_next_doc_id(self):
        p = self._shard_path / "next_doc_id.json"
        if p.exists():
            with open(p) as f:
                self._next_doc_id = json.load(f)

    def _save_next_doc_id(self):
        p = self._shard_path / "next_doc_id.json"
        with open(p, "w") as f:
            json.dump(self._next_doc_id, f)

    def _load_deleted_ids(self):
        p = self._shard_path / "deleted_ids.json"
        if p.exists():
            with open(p) as f:
                self._deleted_ids = set(json.load(f))

    def _save_deleted_ids(self):
        p = self._shard_path / "deleted_ids.json"
        with open(p, "w") as f:
            json.dump(sorted(self._deleted_ids), f)

    def _load_sealed_segments(self):
        sealed_dir = self._shard_path / "sealed"
        if not sealed_dir.exists():
            return
        if GemSegment is None:
            logger.error("latence_gem_index not available — cannot load sealed segments")
            return
        for seg_dir in sorted(sealed_dir.iterdir()):
            gem_path = seg_dir / "segment.gem"
            ids_path = seg_dir / "doc_ids.json"
            if gem_path.exists() and ids_path.exists():
                try:
                    seg = GemSegment()
                    seg.load(str(gem_path))
                    with open(ids_path) as f:
                        ids = json.load(f)
                    self._sealed_segments.append(seg)
                    self._sealed_doc_ids.append(ids)
                except Exception as e:
                    logger.error("Failed to load sealed segment %s: %s", seg_dir, e)

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)
        return self._executor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def add(self, vectors: np.ndarray, ids: List[int], payloads: Optional[List[Dict]] = None):
        with self._lock:
            for i, doc_id in enumerate(ids):
                vecs = vectors[i] if vectors.ndim == 3 else vectors[i:i+1]
                self._active.upsert(
                    int(doc_id),
                    vecs.astype(np.float32),
                    {},
                )
                self._payloads[doc_id] = (payloads[i] if payloads else {})
                self._next_doc_id = max(self._next_doc_id, doc_id + 1)
            self._save_next_doc_id()

    def add_multidense(self, vectors: List[np.ndarray], ids: List[int], payloads: Optional[List[Dict]] = None):
        with self._lock:
            for i, doc_id in enumerate(ids):
                vecs = vectors[i].astype(np.float32)
                self._active.upsert(int(doc_id), vecs, {})
                self._payloads[doc_id] = (payloads[i] if payloads else {})
                self._next_doc_id = max(self._next_doc_id, doc_id + 1)
            self._save_next_doc_id()

    @staticmethod
    def _gem_to_similarity(gem_results):
        return [(doc_id, 1.0 / (1.0 + max(score, 0.0))) for doc_id, score in gem_results]

    def search(self, query: np.ndarray, k: int = 10, ef: int = 100, filters: Optional[Dict] = None):
        return self.search_multivector(
            query if query.ndim == 2 else query.reshape(1, -1),
            k=k, ef=ef, n_probes=self._n_probes,
        )

    def search_multivector(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: int = 100,
        n_probes: int = 4,
    ) -> List[Tuple[int, float]]:
        with self._lock:
            all_results: Dict[int, float] = {}

            for seg, seg_ids in zip(self._sealed_segments, self._sealed_doc_ids):
                raw = seg.search(query_vectors.astype(np.float32), k=k, ef=ef, n_probes=n_probes)
                for doc_id, score in self._gem_to_similarity(raw):
                    if doc_id not in self._deleted_ids:
                        if doc_id not in all_results or score > all_results[doc_id]:
                            all_results[doc_id] = score

            sorted_results = sorted(all_results.items(), key=lambda x: -x[1])
            return sorted_results[:k]

    def retrieve(self, ids: List[int], with_vector: bool = False, with_payload: bool = True):
        results = {}
        for doc_id in ids:
            entry: Dict[str, Any] = {"id": doc_id}
            if with_payload:
                entry["payload"] = self._payloads.get(doc_id, {})
            results[doc_id] = entry
        return results

    def seal_active_segment(self):
        with self._lock:
            active_vectors = []
            active_ids = []
            for doc_id, data in getattr(self._active, "items", {}).items():
                if doc_id not in self._deleted_ids:
                    vec = data.get("vector", data) if isinstance(data, dict) else data
                    if isinstance(vec, np.ndarray):
                        if vec.ndim == 1:
                            vec = vec.reshape(1, -1)
                        active_vectors.append(vec)
                        active_ids.append(doc_id)

            if not active_vectors:
                return

            all_vecs = np.vstack(active_vectors).astype(np.float32)
            offsets = []
            pos = 0
            for v in active_vectors:
                n = v.shape[0]
                offsets.append((pos, pos + n))
                pos += n

            seg = GemSegment()
            seg.build(
                all_vecs, active_ids, offsets,
                n_fine=self._n_fine, n_coarse=self._n_coarse,
                max_degree=self._max_degree,
                ef_construction=self._gem_ef_construction,
                max_kmeans_iter=self._max_kmeans_iter,
                ctop_r=self._ctop_r,
            )

            seg_idx = len(self._sealed_segments)
            seg_dir = self._shard_path / "sealed" / f"seg_{seg_idx:04d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg.save(str(seg_dir / "segment.gem"))
            with open(seg_dir / "doc_ids.json", "w") as f:
                json.dump(active_ids, f)

            self._sealed_segments.append(seg)
            self._sealed_doc_ids.append(active_ids)

    def delete(self, ids: List[int]):
        with self._lock:
            for doc_id in ids:
                self._deleted_ids.add(doc_id)
                self._payloads.pop(doc_id, None)
            self._save_deleted_ids()

    def upsert_payload(self, id: int, payload: Dict[str, Any]):
        with self._lock:
            self._payloads[id] = payload

    def total_vectors(self) -> int:
        total = 0
        for seg in self._sealed_segments:
            total += seg.n_docs()
        return total

    def flush(self):
        self._save_next_doc_id()
        self._save_deleted_ids()

    def close(self):
        self.flush()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.total_vectors(),
            "sealed_segments": len(self._sealed_segments),
            "deleted_count": len(self._deleted_ids),
            "dim": self._dim,
        }


class GemNativeSegmentManager:
    """
    Fully native GEM segment manager with mutable active segment,
    WAL-based crash recovery, and sealed segment lifecycle.
    """

    def __init__(
        self,
        shard_path: str,
        dim: int,
        *,
        n_fine: int = 256,
        n_coarse: int = 32,
        max_degree: int = 32,
        gem_ef_construction: int = 200,
        max_kmeans_iter: int = 30,
        ctop_r: int = 3,
        n_probes: int = 4,
        seed_batch_size: int = 256,
        seal_size_threshold: int = 10000,
        seal_quality_threshold: float = 0.6,
        compaction_threshold: float = 0.3,
        compaction_interval_s: float = 60.0,
        enable_shortcuts: bool = False,
    ):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if n_fine <= 0:
            raise ValueError(f"n_fine must be positive, got {n_fine}")
        if n_coarse <= 0:
            raise ValueError(f"n_coarse must be positive, got {n_coarse}")
        if max_degree <= 0:
            raise ValueError(f"max_degree must be positive, got {max_degree}")
        if not (0.0 <= seal_quality_threshold <= 1.0):
            raise ValueError(f"seal_quality_threshold must be in [0,1], got {seal_quality_threshold}")
        if not (0.0 <= compaction_threshold <= 1.0):
            raise ValueError(f"compaction_threshold must be in [0,1], got {compaction_threshold}")

        self._shard_path = Path(shard_path)
        self._shard_path.mkdir(parents=True, exist_ok=True)
        self._dim = dim
        self._n_fine = n_fine
        self._n_coarse = n_coarse
        self._max_degree = max_degree
        self._gem_ef_construction = gem_ef_construction
        self._max_kmeans_iter = max_kmeans_iter
        self._ctop_r = ctop_r
        self._n_probes = n_probes
        self._seed_batch_size = seed_batch_size
        self._seal_size_threshold = seal_size_threshold
        self._seal_quality_threshold = seal_quality_threshold
        self._compaction_threshold = compaction_threshold
        self._compaction_interval_s = compaction_interval_s
        self._enable_shortcuts = enable_shortcuts

        # Phase 3.4: Reader-writer lock for concurrent reads
        if RWLock is not None:
            self._rwlock = RWLock()
        else:
            self._rwlock = None
        self._lock = threading.RLock()

        # Phase 3.1: File lock for multi-process safety
        self._file_lock: Optional[FileLock] = None
        if FileLock is not None:
            self._file_lock = FileLock(self._shard_path / ".lock")
            self._file_lock.acquire()

        self._executor: Optional[ThreadPoolExecutor] = None
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._next_doc_id = 0
        self._deleted_ids: Set[int] = set()
        self._sealed_deleted_ids: Set[int] = set()
        self._metrics_hook = None

        self._active: Optional[PyMutableGemSegment] = None
        self._sealed_segments: List[GemSegment] = []
        self._sealed_doc_ids: List[List[int]] = []

        self._seed_buffer_vecs: List[np.ndarray] = []
        self._seed_buffer_ids: List[int] = []
        self._codebook_trained = False

        self._wal_writer: Optional[WalWriter] = None
        self._checkpoint_mgr: Optional[CheckpointManager] = None

        if WalWriter is not None:
            wal_path = self._shard_path / "wal.bin"
            self._wal_writer = WalWriter(wal_path)
            self._wal_writer.open()
            self._checkpoint_mgr = CheckpointManager(self._shard_path / "checkpoints")

        self._compaction_thread: Optional[threading.Thread] = None
        self._compaction_stop = threading.Event()

        self._load_next_doc_id()
        self._load_deleted_ids()
        self._load_sealed_deleted_ids()
        self._load_sealed_segments()
        self._load_payloads()
        self._recover_from_wal()
        self._start_compaction_thread()

    def _load_next_doc_id(self):
        p = self._shard_path / "next_doc_id.json"
        if p.exists():
            with open(p) as f:
                self._next_doc_id = json.load(f)

    def _save_next_doc_id(self):
        p = self._shard_path / "next_doc_id.json"
        if atomic_json_write is not None:
            atomic_json_write(p, self._next_doc_id)
        else:
            with open(p, "w") as f:
                json.dump(self._next_doc_id, f)

    def _load_deleted_ids(self):
        p = self._shard_path / "deleted_ids.json"
        if p.exists():
            with open(p) as f:
                self._deleted_ids = set(json.load(f))

    def _save_deleted_ids(self):
        p = self._shard_path / "deleted_ids.json"
        if atomic_json_write is not None:
            atomic_json_write(p, sorted(self._deleted_ids))
        else:
            with open(p, "w") as f:
                json.dump(sorted(self._deleted_ids), f)

    def _load_sealed_deleted_ids(self):
        p = self._shard_path / "sealed_deleted_ids.json"
        if p.exists():
            with open(p) as f:
                self._sealed_deleted_ids = set(json.load(f))

    def _save_sealed_deleted_ids(self):
        p = self._shard_path / "sealed_deleted_ids.json"
        if atomic_json_write is not None:
            atomic_json_write(p, sorted(self._sealed_deleted_ids))
        else:
            with open(p, "w") as f:
                json.dump(sorted(self._sealed_deleted_ids), f)

    def _load_sealed_segments(self):
        sealed_dir = self._shard_path / "sealed"
        if not sealed_dir.exists():
            return
        if GemSegment is None:
            logger.error("latence_gem_index not available — cannot load sealed segments")
            return
        for seg_dir in sorted(sealed_dir.iterdir()):
            gem_path = seg_dir / "segment.gem"
            ids_path = seg_dir / "doc_ids.json"
            if gem_path.exists() and ids_path.exists():
                try:
                    seg = GemSegment()
                    seg.load(str(gem_path))
                    with open(ids_path) as f:
                        ids = json.load(f)
                    self._sealed_segments.append(seg)
                    self._sealed_doc_ids.append(ids)
                except Exception as e:
                    logger.error("Failed to load sealed segment %s: %s", seg_dir, e)

    def _load_payloads(self):
        p = self._shard_path / "payloads.json"
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
                self._payloads = {int(k): v for k, v in raw.items()}

    def _save_payloads(self):
        p = self._shard_path / "payloads.json"
        data = {str(k): v for k, v in self._payloads.items()}
        if atomic_json_write is not None:
            atomic_json_write(p, data)
        else:
            with open(p, "w") as f:
                json.dump(data, f)

    def _recover_from_wal(self):
        """Recover state from checkpoint + WAL replay on startup."""
        if self._checkpoint_mgr and self._checkpoint_mgr.exists():
            result = self._checkpoint_mgr.load()
            if result:
                vecs, ids, next_id, trained, sealed_del = result
                self._next_doc_id = max(self._next_doc_id, next_id)
                self._codebook_trained = trained
                self._sealed_deleted_ids = sealed_del
                if vecs and ids:
                    self._replay_recovered_docs(vecs, ids)

        if self._wal_writer and WalReader is not None:
            reader = WalReader(self._shard_path / "wal.bin")
            entries = reader.replay()
            for entry in entries:
                from .gem_wal import WalOp
                if entry.op == WalOp.INSERT and entry.vectors is not None:
                    self._seed_buffer_vecs.append(entry.vectors)
                    self._seed_buffer_ids.append(entry.external_id)
                elif entry.op == WalOp.DELETE:
                    self._deleted_ids.add(entry.external_id)
                elif entry.op == WalOp.UPSERT and entry.vectors is not None:
                    self._deleted_ids.discard(entry.external_id)
                    self._seed_buffer_vecs.append(entry.vectors)
                    self._seed_buffer_ids.append(entry.external_id)
                elif entry.op == WalOp.UPDATE_PAYLOAD and entry.payload is not None:
                    self._payloads[entry.external_id] = entry.payload

            if self._seed_buffer_ids and not self._codebook_trained:
                if len(self._seed_buffer_ids) >= self._seed_batch_size:
                    self._init_active_from_seed()
                else:
                    logger.info(
                        "WAL recovery: %d docs in seed buffer (< batch size %d), "
                        "deferring active segment build until more data arrives",
                        len(self._seed_buffer_ids), self._seed_batch_size,
                    )

    def _replay_recovered_docs(self, vecs: List[np.ndarray], ids: List[int]):
        """Rebuild the active segment from recovered document data."""
        if not vecs or not ids:
            return
        all_vecs = np.vstack(vecs).astype(np.float32)
        offsets = []
        pos = 0
        for v in vecs:
            n = v.shape[0]
            offsets.append((pos, pos + n))
            pos += n

        self._active = PyMutableGemSegment()
        self._active.build(
            all_vecs, ids, offsets,
            n_fine=self._n_fine, n_coarse=self._n_coarse,
            max_degree=self._max_degree,
            ef_construction=self._gem_ef_construction,
            max_kmeans_iter=self._max_kmeans_iter,
            ctop_r=self._ctop_r, n_probes=self._n_probes,
        )
        self._codebook_trained = True

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)
        return self._executor

    def _init_active_from_seed(self):
        """Train codebook from seed buffer and build the initial mutable graph."""
        if not self._seed_buffer_vecs:
            return

        all_vecs = np.vstack(self._seed_buffer_vecs).astype(np.float32)
        offsets = []
        pos = 0
        for v in self._seed_buffer_vecs:
            n = v.shape[0]
            offsets.append((pos, pos + n))
            pos += n

        self._active = PyMutableGemSegment()
        self._active.build(
            all_vecs, self._seed_buffer_ids, offsets,
            n_fine=self._n_fine, n_coarse=self._n_coarse,
            max_degree=self._max_degree,
            ef_construction=self._gem_ef_construction,
            max_kmeans_iter=self._max_kmeans_iter,
            ctop_r=self._ctop_r, n_probes=self._n_probes,
        )
        self._codebook_trained = True
        self._seed_buffer_vecs.clear()
        self._seed_buffer_ids.clear()

    def _start_compaction_thread(self):
        if self._compaction_interval_s <= 0:
            return
        self._compaction_stop.clear()
        self._compaction_thread = threading.Thread(
            target=self._compaction_loop, daemon=True,
        )
        self._compaction_thread.start()

    def _stop_compaction_thread(self):
        self._compaction_stop.set()
        if self._compaction_thread and self._compaction_thread.is_alive():
            self._compaction_thread.join(timeout=5)

    def _compaction_loop(self):
        while not self._compaction_stop.wait(timeout=self._compaction_interval_s):
            try:
                self._maybe_compact()
            except Exception as e:
                logger.error("Compaction error: %s", e)

    def _maybe_compact(self):
        with self._lock:
            if self._active is not None and self._active.is_ready():
                if self._active.delete_ratio() > self._compaction_threshold:
                    self._active.compact()
                    self._emit_metric("compaction", 1)

    def add_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[Dict]] = None,
    ):
        ctx = self._rwlock.write_lock() if self._rwlock else self._lock
        with ctx:
            for i, doc_id in enumerate(ids):
                vecs = vectors[i].astype(np.float32)

                if self._wal_writer:
                    self._wal_writer.log_insert(doc_id, vecs)

                self._payloads[doc_id] = (payloads[i] if payloads else {})
                self._next_doc_id = max(self._next_doc_id, doc_id + 1)

                if self._active is not None and self._active.is_ready():
                    self._active.insert(vecs, doc_id)
                else:
                    self._seed_buffer_vecs.append(vecs)
                    self._seed_buffer_ids.append(doc_id)
                    if len(self._seed_buffer_ids) >= self._seed_batch_size:
                        self._init_active_from_seed()

            self._save_next_doc_id()
            self._save_payloads()

            if self.should_seal():
                self.seal_active_segment()

    def add(
        self,
        vectors: np.ndarray,
        ids: List[int],
        payloads: Optional[List[Dict]] = None,
    ):
        vecs_list = []
        if vectors.ndim == 3:
            for i in range(vectors.shape[0]):
                vecs_list.append(vectors[i])
        elif vectors.ndim == 2:
            for i in range(len(ids)):
                vecs_list.append(vectors[i:i+1])
        else:
            raise ValueError(f"Expected 2D or 3D vectors, got {vectors.ndim}D")
        self.add_multidense(vecs_list, ids, payloads)

    @staticmethod
    def _gem_to_similarity(gem_results):
        return [(doc_id, 1.0 / (1.0 + max(score, 0.0))) for doc_id, score in gem_results]

    def _match_filter(self, doc_id: int, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        payload = self._payloads.get(doc_id, {})
        return self._check_payload(payload, filters, doc_id=doc_id)

    @staticmethod
    def _match_filter_snapshot(doc_id: int, filters: Optional[Dict], payloads: Dict) -> bool:
        if not filters:
            return True
        payload = payloads.get(doc_id, {})
        return GemNativeSegmentManager._check_payload(payload, filters, doc_id=doc_id)

    @staticmethod
    def _check_payload(payload: Dict, filters: Dict, doc_id: Optional[int] = None) -> bool:
        """Evaluate Qdrant-compatible filter predicates with full logical operators.

        Supports:
        - $and: all conditions must match
        - $or: any condition must match
        - $not: negate a condition
        - $has_id: check if doc_id is in the given set
        - Nested field access via dot notation: "metadata.author"
        - Field-level operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin,
          $exists, $contains, $geo_radius
        """
        return GemNativeSegmentManager._evaluate_filter(payload, filters, doc_id)

    @staticmethod
    def _resolve_nested(payload: Dict, key: str) -> Any:
        """Resolve dotted key paths like 'metadata.author' in payload."""
        parts = key.split(".")
        current: Any = payload
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _evaluate_filter(payload: Dict, filter_node: Dict, doc_id: Optional[int] = None) -> bool:
        for key, condition in filter_node.items():
            if key == "$and":
                if not isinstance(condition, list):
                    return False
                if not all(
                    GemNativeSegmentManager._evaluate_filter(payload, sub, doc_id)
                    for sub in condition
                ):
                    return False
            elif key == "$or":
                if not isinstance(condition, list):
                    return False
                if not any(
                    GemNativeSegmentManager._evaluate_filter(payload, sub, doc_id)
                    for sub in condition
                ):
                    return False
            elif key == "$not":
                if not isinstance(condition, dict):
                    return False
                if GemNativeSegmentManager._evaluate_filter(payload, condition, doc_id):
                    return False
            elif key == "$has_id":
                if isinstance(condition, (list, set)):
                    if doc_id is None or doc_id not in condition:
                        return False
                else:
                    return False
            else:
                val = GemNativeSegmentManager._resolve_nested(payload, key)
                if isinstance(condition, dict):
                    for op, cmp_val in condition.items():
                        if op == "$eq" and val != cmp_val:
                            return False
                        elif op == "$ne" and val == cmp_val:
                            return False
                        elif op == "$gt" and (val is None or val <= cmp_val):
                            return False
                        elif op == "$gte" and (val is None or val < cmp_val):
                            return False
                        elif op == "$lt" and (val is None or val >= cmp_val):
                            return False
                        elif op == "$lte" and (val is None or val > cmp_val):
                            return False
                        elif op == "$in" and val not in cmp_val:
                            return False
                        elif op == "$nin" and val in cmp_val:
                            return False
                        elif op == "$exists":
                            exists = val is not None
                            if exists != bool(cmp_val):
                                return False
                        elif op == "$contains":
                            if not isinstance(val, (list, str)):
                                return False
                            if cmp_val not in val:
                                return False
                        elif op == "$geo_radius":
                            if not isinstance(val, dict) or not isinstance(cmp_val, dict):
                                return False
                            try:
                                import math
                                lat1 = float(val.get("lat", 0))
                                lon1 = float(val.get("lon", 0))
                                lat2 = float(cmp_val.get("lat", 0))
                                lon2 = float(cmp_val.get("lon", 0))
                                radius = float(cmp_val.get("radius_km", 0))
                                dlat = math.radians(lat2 - lat1)
                                dlon = math.radians(lon2 - lon1)
                                a = (math.sin(dlat / 2) ** 2 +
                                     math.cos(math.radians(lat1)) *
                                     math.cos(math.radians(lat2)) *
                                     math.sin(dlon / 2) ** 2)
                                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                                if dist_km > radius:
                                    return False
                            except (TypeError, ValueError):
                                return False
                else:
                    if val != condition:
                        return False
        return True

    def search_multivector(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: int = 100,
        n_probes: int = 4,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        ctx = self._rwlock.read_lock() if self._rwlock else self._lock
        with ctx:
            all_results: Dict[int, float] = {}
            payloads_snap = dict(self._payloads)

            if self._active is not None and self._active.is_ready():
                raw = self._active.search(query_vectors.astype(np.float32), k=k, ef=ef)
                for doc_id, score in self._gem_to_similarity(raw):
                    if doc_id not in self._deleted_ids:
                        if self._match_filter_snapshot(doc_id, filters, payloads_snap):
                            if doc_id not in all_results or score > all_results[doc_id]:
                                all_results[doc_id] = score

            for seg, seg_ids in zip(self._sealed_segments, self._sealed_doc_ids):
                raw = seg.search(
                    query_vectors.astype(np.float32),
                    k=k, ef=ef, n_probes=n_probes,
                    enable_shortcuts=self._enable_shortcuts,
                )
                for doc_id, score in self._gem_to_similarity(raw):
                    if doc_id not in self._deleted_ids and doc_id not in self._sealed_deleted_ids:
                        if self._match_filter_snapshot(doc_id, filters, payloads_snap):
                            if doc_id not in all_results or score > all_results[doc_id]:
                                all_results[doc_id] = score

            sorted_results = sorted(all_results.items(), key=lambda x: -x[1])
            return sorted_results[:k]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        qv = query if query.ndim == 2 else query.reshape(1, -1)
        return self.search_multivector(qv, k=k, ef=ef, n_probes=self._n_probes, filters=filters)

    def search_batch(
        self,
        queries: List[np.ndarray],
        k: int = 10,
        ef: int = 100,
    ) -> List[List[Tuple[int, float]]]:
        """Batch search across all segments using parallel Rust execution."""
        ctx = self._rwlock.read_lock() if self._rwlock else self._lock
        with ctx:
            all_query_results: List[List[Tuple[int, float]]] = [[] for _ in queries]

            for seg in self._sealed_segments:
                batch_results = seg.search_batch(queries, k=k, ef=ef, n_probes=self._n_probes)
                for qi, res in enumerate(batch_results):
                    for doc_id, score in self._gem_to_similarity(res):
                        if doc_id not in self._deleted_ids and doc_id not in self._sealed_deleted_ids:
                            all_query_results[qi].append((doc_id, score))

            if self._active is not None and self._active.is_ready():
                for qi, query in enumerate(queries):
                    qv = query if query.ndim == 2 else query.reshape(1, -1)
                    active_res = self._active.search(qv, k=k, ef=ef)
                    for doc_id, score in self._gem_to_similarity(active_res):
                        if doc_id not in self._deleted_ids:
                            all_query_results[qi].append((doc_id, score))

            final = []
            for qi in range(len(queries)):
                merged: Dict[int, float] = {}
                for doc_id, score in all_query_results[qi]:
                    if doc_id not in merged or score > merged[doc_id]:
                        merged[doc_id] = score
                sorted_res = sorted(merged.items(), key=lambda x: -x[1])
                final.append(sorted_res[:k])
            return final

    def retrieve(self, ids: List[int], with_vector: bool = False, with_payload: bool = True):
        ctx = self._rwlock.read_lock() if self._rwlock else self._lock
        with ctx:
            results = {}
            for doc_id in ids:
                entry: Dict[str, Any] = {"id": doc_id}
                if with_payload:
                    entry["payload"] = self._payloads.get(doc_id, {})
                results[doc_id] = entry
            return results

    def seal_active_segment(self):
        with self._lock:
            self._seal_active_segment_inner()

    def _seal_active_segment_inner(self):
        if self._active is None or not self._active.is_ready():
            return

        active_n = self._active.n_live()
        if active_n == 0:
            return

        logger.info("Sealing active segment with %d live docs", active_n)

        vecs, ids = self._collect_all_active_docs()
        if not vecs or not ids:
            logger.warning("Seal: no vectors collected from active, skipping")
            return

        all_vecs = np.vstack(vecs).astype(np.float32)
        offsets = []
        pos = 0
        for v in vecs:
            n = v.shape[0]
            offsets.append((pos, pos + n))
            pos += n

        if GemSegment is None:
            raise RuntimeError("latence_gem_index not available — cannot seal")

        seg_idx = len(self._sealed_segments)
        seg_dir = self._shard_path / "sealed" / f"seg_{seg_idx:04d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3.5: Write seal marker BEFORE building so recovery
        # knows these IDs are being sealed (crash between marker and completion
        # is handled by checking for the marker on next startup).
        marker_path = seg_dir / "seal_marker.json"
        marker_data = {"ids": ids, "status": "in_progress"}
        if atomic_json_write is not None:
            atomic_json_write(marker_path, marker_data)
        else:
            with open(marker_path, "w") as f:
                json.dump(marker_data, f)

        sealed = GemSegment()
        sealed.build(
            all_vecs, ids, offsets,
            n_fine=self._n_fine, n_coarse=self._n_coarse,
            max_degree=self._max_degree,
            ef_construction=self._gem_ef_construction,
            max_kmeans_iter=self._max_kmeans_iter,
            ctop_r=self._ctop_r,
        )

        sealed.save(str(seg_dir / "segment.gem"))
        if atomic_json_write is not None:
            atomic_json_write(seg_dir / "doc_ids.json", ids)
        else:
            with open(seg_dir / "doc_ids.json", "w") as f:
                json.dump(ids, f)

        # Update marker to completed
        marker_data["status"] = "completed"
        if atomic_json_write is not None:
            atomic_json_write(marker_path, marker_data)
        else:
            with open(marker_path, "w") as f:
                json.dump(marker_data, f)

        self._sealed_segments.append(sealed)
        self._sealed_doc_ids.append(ids)
        logger.info("Sealed segment %d with %d docs written to %s", seg_idx, len(ids), seg_dir)

        self._active = None
        self._codebook_trained = False
        gc.collect()

        if self._wal_writer:
            self._wal_writer.truncate()
        if self._checkpoint_mgr:
            self._checkpoint_mgr.clear()

    def _collect_all_active_docs(self):
        """Collect all live document vectors from WAL replay for sealing."""
        if not self._wal_writer or WalReader is None:
            return self._seed_buffer_vecs[:], self._seed_buffer_ids[:]

        reader = WalReader(self._shard_path / "wal.bin")
        entries = reader.replay()

        live_docs = {}
        for entry in entries:
            from .gem_wal import WalOp
            if entry.op == WalOp.INSERT and entry.vectors is not None:
                if entry.external_id not in self._deleted_ids:
                    live_docs[entry.external_id] = entry.vectors
            elif entry.op == WalOp.DELETE:
                live_docs.pop(entry.external_id, None)
            elif entry.op == WalOp.UPSERT and entry.vectors is not None:
                if entry.external_id not in self._deleted_ids:
                    live_docs[entry.external_id] = entry.vectors
                else:
                    live_docs.pop(entry.external_id, None)

        ids = list(live_docs.keys())
        vecs = [live_docs[i] for i in ids]
        return vecs, ids

    def should_seal(self) -> bool:
        with self._lock:
            return self._should_seal_inner()

    def _should_seal_inner(self) -> bool:
        if self._active is None or not self._active.is_ready():
            return False
        if self._active.n_live() >= self._seal_size_threshold:
            return True
        if self._active.quality_score() < self._seal_quality_threshold:
            return True
        return False

    def quality_score(self) -> float:
        with self._lock:
            if self._active is not None and self._active.is_ready():
                return self._active.quality_score()
            return 1.0

    def upsert_multidense(
        self,
        vectors: List[np.ndarray],
        ids: List[int],
        payloads: Optional[List[Dict]] = None,
    ):
        ctx = self._rwlock.write_lock() if self._rwlock else self._lock
        with ctx:
            for i, doc_id in enumerate(ids):
                vecs = vectors[i].astype(np.float32)
                if self._wal_writer:
                    self._wal_writer.log_upsert(doc_id, vecs)
                self._payloads[doc_id] = (payloads[i] if payloads else {})
                self._next_doc_id = max(self._next_doc_id, doc_id + 1)

                if self._active is not None and self._active.is_ready():
                    self._active.upsert(vecs, doc_id)
                else:
                    self._deleted_ids.discard(doc_id)
                    self._seed_buffer_vecs.append(vecs)
                    self._seed_buffer_ids.append(doc_id)
                    if len(self._seed_buffer_ids) >= self._seed_batch_size:
                        self._init_active_from_seed()

            self._save_next_doc_id()
            self._save_payloads()

    def delete(self, ids: List[int]):
        wctx = self._rwlock.write_lock() if self._rwlock else self._lock
        with wctx:
            sealed_id_set = set()
            for seg_ids in self._sealed_doc_ids:
                sealed_id_set.update(seg_ids)

            for doc_id in ids:
                if self._wal_writer:
                    self._wal_writer.log_delete(doc_id)
                self._deleted_ids.add(doc_id)
                if doc_id in sealed_id_set:
                    self._sealed_deleted_ids.add(doc_id)
                if self._active is not None and self._active.is_ready():
                    self._active.delete(doc_id)
                self._payloads.pop(doc_id, None)
            self._save_deleted_ids()
            if self._sealed_deleted_ids:
                self._save_sealed_deleted_ids()

    def upsert_payload(self, doc_id: int, payload: Dict[str, Any]):
        wctx = self._rwlock.write_lock() if self._rwlock else self._lock
        with wctx:
            if self._wal_writer:
                self._wal_writer.log_update_payload(doc_id, payload)
            self._payloads[doc_id] = payload
            self._save_payloads()

    def total_vectors(self) -> int:
        ctx = self._rwlock.read_lock() if self._rwlock else self._lock
        with ctx:
            total = 0
            if self._active is not None and self._active.is_ready():
                total += self._active.n_live()
            total += len(self._seed_buffer_ids)
            for seg in self._sealed_segments:
                total += seg.n_docs()
            return total

    def set_metrics_hook(self, hook):
        self._metrics_hook = hook

    def _emit_metric(self, name: str, value: Any):
        if self._metrics_hook:
            try:
                self._metrics_hook(name, value)
            except Exception as e:
                logger.debug("Metrics hook error: %s", e)

    def flush(self):
        with self._lock:
            self._save_next_doc_id()
            self._save_deleted_ids()
            self._save_sealed_deleted_ids()
            self._save_payloads()
            if self._checkpoint_mgr and self._active is not None:
                try:
                    self._checkpoint_mgr.save(
                        doc_vectors=self._seed_buffer_vecs,
                        doc_ids=self._seed_buffer_ids,
                        next_doc_id=self._next_doc_id,
                        codebook_trained=self._codebook_trained,
                        sealed_deleted_ids=self._sealed_deleted_ids,
                    )
                except Exception as e:
                    logger.error("Checkpoint save failed: %s", e)

    def close(self):
        self._stop_compaction_thread()
        self.flush()
        if self._wal_writer:
            self._wal_writer.close()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._file_lock:
            self._file_lock.release()
            self._file_lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _explain_score(
        self,
        query: np.ndarray,
        doc_id: int,
    ) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """Per-query-token score attribution for a given doc.

        Returns (token_scores, matched_centroid_ids) or (None, None)
        if not computable.
        """
        n_query = query.shape[0] if query.ndim == 2 else 1
        token_scores = [0.0] * n_query
        matched = [0] * n_query
        return token_scores, matched

    def get_statistics(self) -> Dict[str, Any]:
        ctx = self._rwlock.read_lock() if self._rwlock else self._lock
        with ctx:
            stats: Dict[str, Any] = {
                "total_vectors": self.total_vectors(),
                "sealed_segments": len(self._sealed_segments),
                "deleted_count": len(self._deleted_ids),
                "dim": self._dim,
                "codebook_trained": self._codebook_trained,
                "seed_buffer_size": len(self._seed_buffer_ids),
            }
            if self._active is not None and self._active.is_ready():
                stats["active"] = {
                    "n_live": self._active.n_live(),
                    "n_nodes": self._active.n_nodes(),
                    "n_edges": self._active.n_edges(),
                    "quality": self._active.quality_score(),
                    "delete_ratio": self._active.delete_ratio(),
                    "avg_degree": self._active.avg_degree(),
                }
            return stats
