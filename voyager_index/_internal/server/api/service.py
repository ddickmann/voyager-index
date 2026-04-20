"""
Durable collection service for the voyager-index reference API.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import numpy as np
import torch

from voyager_index._internal.inference.config import IndexConfig
from voyager_index._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine
from voyager_index._internal.inference.index_core.graph_policy import LatenceGraphPolicy
from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
from voyager_index._internal.inference.index_core.index import ColbertIndex
from voyager_index._internal.inference.index_core.latence_graph_sidecar import LatenceGraphSidecar
from voyager_index._internal.inference.shard_engine import ShardSegmentManager
from voyager_index._internal.inference.shard_engine.capabilities import detect_runtime_capabilities
from voyager_index._internal.inference.shard_engine.config import Compression, TransferMode
from voyager_index._internal.inference.shard_engine.manager import ShardEngineConfig
from voyager_index.transport import decode_payload

from .models import (
    CollectionInfo,
    CollectionKind,
    CreateCollectionRequest,
    DistanceMetric,
    GraphMode,
    MultimodalOptimizeMode,
    MutationTaskStatus,
    PointVector,
    RenderDocumentsRequest,
    ScoredPoint,
    ScreeningMode,
    SearchRequest,
    SearchResponse,
    SearchStrategy,
    TransportVectorPayload,
)

logger = logging.getLogger(__name__)
_TASK_MAX_AGE_S = 3600
_TASK_MAX_ENTRIES = 10_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class ServiceError(Exception):
    status_code = 500
    error_code = "service_error"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class ValidationError(ServiceError):
    status_code = 400
    error_code = "validation_error"


class ConflictError(ServiceError):
    status_code = 409
    error_code = "conflict"


class NotFoundError(ServiceError):
    status_code = 404
    error_code = "not_found"


class ScanLimitExceededError(ServiceError):
    status_code = 503
    error_code = "scan_limit_exceeded"


class RequestTooLargeError(ServiceError):
    status_code = 413
    error_code = "request_too_large"


@dataclass
class CollectionRuntime:
    name: str
    kind: CollectionKind
    path: Path
    meta: Dict[str, Any]
    engine: Any
    record_index: Dict[int, Dict[str, Any]]
    payload_filter_index: Dict[str, Dict[str, set[int]]]
    graph_sidecar: Optional[Any] = None
    meta_mtime_ns: int = 0


@dataclass
class MutationBackup:
    runtime_name: str
    backup_root: Path
    metadata: Dict[str, Any]


class SearchService:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._journal_root = self.root_path / ".voyager-journal"
        self._journal_root.mkdir(parents=True, exist_ok=True)
        self._lock_root = self.root_path / ".voyager-locks"
        self._lock_root.mkdir(parents=True, exist_ok=True)
        self._task_root = self.root_path / ".voyager-tasks"
        self._task_root.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.total_latency = 0.0
        self.nodes_visited_total = 0
        self.distance_comps_total = 0
        self.collections: Dict[str, CollectionRuntime] = {}
        self.load_failures: Dict[str, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        self.runtime_capabilities = detect_runtime_capabilities()
        self.filter_scan_limit = int(os.environ.get("VOYAGER_FILTER_SCAN_LIMIT", "10000"))
        self.filter_scan_limit_hits = 0
        self._graph_policy = LatenceGraphPolicy()
        self._metrics_lock = threading.Lock()
        self._task_thread_lock = threading.Lock()
        self._collections_lock = threading.RLock()
        self._collection_locks: Dict[str, threading.RLock] = {}
        logger.info("Reference API runtime capabilities: %s", self.runtime_capabilities)
        self._recover_pending_journals()
        self._load_collections()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_search_metrics(
        self,
        elapsed_ms: float,
        nodes_visited: int = 0,
        distance_comps: int = 0,
    ) -> None:
        with self._metrics_lock:
            self.request_count += 1
            self.total_latency += elapsed_ms / 1000.0
            self.nodes_visited_total += nodes_visited
            self.distance_comps_total += distance_comps

    def _increment_filter_scan_limit_hits(self) -> None:
        with self._metrics_lock:
            self.filter_scan_limit_hits += 1

    def _validate_collection_name(self, name: str) -> str:
        candidate = (name or "").strip()
        if (
            not candidate
            or candidate in {".", ".."}
            or candidate.startswith(".")
            or "/" in candidate
            or "\\" in candidate
            or "\x00" in candidate
            or Path(candidate).name != candidate
        ):
            raise ValidationError("Collection name must be a single non-empty path segment")
        return candidate

    def _collection_path(self, name: str) -> Path:
        safe_name = self._validate_collection_name(name)
        path = (self.root_path / safe_name).resolve()
        if not path.is_relative_to(self.root_path):
            raise ValidationError("Collection path escapes the configured data root")
        return path

    def _metadata_path(self, name: str) -> Path:
        return self._collection_path(name) / "collection.json"

    def _graph_state_path(self, name: str, kind: CollectionKind) -> Path:
        base = self._collection_path(name)
        if kind == CollectionKind.DENSE:
            return base / "hybrid" / "latence_graph_state.json"
        return base / "latence_graph_state.json"

    def _collection_lockfile_path(self, name: str) -> Path:
        safe_name = self._validate_collection_name(name)
        digest = hashlib.sha256(safe_name.encode("utf-8")).hexdigest()
        return self._lock_root / f"{digest}.lock"

    @contextlib.contextmanager
    def _cross_process_collection_lock(self, name: str) -> Iterator[None]:
        lock_path = self._collection_lockfile_path(name)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _task_record_path(self, task_id: str) -> Path:
        return self._task_root / f"{task_id}.json"

    def _resolve_user_path(self, raw_path: str, *, relative_to_root: bool = False) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            base = self.root_path if relative_to_root else Path.cwd().resolve()
            candidate = base / candidate
        return candidate.resolve()

    def _sanitize_records(self, meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {str(key): value for key, value in (meta.get("records") or {}).items()}

    @staticmethod
    def _graph_sidecar_for_runtime(runtime: CollectionRuntime) -> Optional[Any]:
        sidecar = getattr(runtime, "graph_sidecar", None)
        if sidecar is not None:
            return sidecar
        return getattr(runtime.engine, "graph_sidecar", None)

    def _refresh_record_index(self, meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        return {int(record["internal_id"]): record for record in self._sanitize_records(meta).values()}

    @staticmethod
    def _filter_value_key(value: Any) -> Optional[str]:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return None

    def _build_payload_filter_index(self, meta: Dict[str, Any]) -> Dict[str, Dict[str, set[int]]]:
        index: Dict[str, Dict[str, set[int]]] = {}
        for record in self._sanitize_records(meta).values():
            internal_id = int(record["internal_id"])
            payload = dict(record.get("payload") or {})
            for key, value in payload.items():
                serialized = self._filter_value_key(value)
                if serialized is None:
                    continue
                index.setdefault(str(key), {}).setdefault(serialized, set()).add(internal_id)
        return index

    def _refresh_runtime_indexes(self, runtime: CollectionRuntime) -> None:
        runtime.record_index = self._refresh_record_index(runtime.meta)
        runtime.payload_filter_index = self._build_payload_filter_index(runtime.meta)

    @staticmethod
    def _graph_request_options(request: SearchRequest) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        if request.graph_max_hops is not None:
            options["max_hops"] = int(request.graph_max_hops)
        if request.graph_local_budget is not None:
            options["local_budget"] = int(request.graph_local_budget)
        if request.graph_community_budget is not None:
            options["community_budget"] = int(request.graph_community_budget)
        if request.graph_evidence_budget is not None:
            options["evidence_budget"] = int(request.graph_evidence_budget)
        if request.graph_explain:
            options["explain"] = True
        return options

    def _sync_runtime_graph_state(
        self,
        runtime: CollectionRuntime,
        *,
        records: Optional[List[Dict[str, Any]]] = None,
        rebuild: bool = False,
        action: str = "append",
    ) -> None:
        normalized_records = [
            {
                "internal_id": int(record["internal_id"]),
                "external_id": record["external_id"],
                "payload": dict(record.get("payload") or {}),
            }
            for record in (records or self._ordered_records(runtime))
        ]
        if runtime.kind == CollectionKind.DENSE and hasattr(runtime.engine, "sync_graph_records"):
            runtime.engine.sync_graph_records(
                [int(record["internal_id"]) for record in normalized_records],
                [dict(record["payload"]) for record in normalized_records],
                external_ids=[record["external_id"] for record in normalized_records],
                rebuild=rebuild,
            )
            runtime.graph_sidecar = getattr(runtime.engine, "graph_sidecar", None)
            return
        sidecar = self._graph_sidecar_for_runtime(runtime)
        if sidecar is None:
            return
        if rebuild:
            sidecar.rebuild_from_records(normalized_records, target_kind="document")
        else:
            sidecar.append_records(normalized_records, target_kind="document", action=action)
        runtime.graph_sidecar = sidecar

    def _delete_runtime_graph_targets(self, runtime: CollectionRuntime, target_ids: List[Any]) -> None:
        if not target_ids:
            return
        if runtime.kind == CollectionKind.DENSE and hasattr(runtime.engine, "delete_graph_records"):
            runtime.engine.delete_graph_records(
                [
                    int(runtime.meta["records"][str(target_id)]["internal_id"])
                    for target_id in target_ids
                    if str(target_id) in runtime.meta["records"]
                ],
                external_ids=[str(target_id) for target_id in target_ids if str(target_id) in runtime.meta["records"]],
            )
            runtime.graph_sidecar = getattr(runtime.engine, "graph_sidecar", None)
            return
        sidecar = self._graph_sidecar_for_runtime(runtime)
        if sidecar is None:
            return
        sidecar.delete([str(target_id) for target_id in target_ids])
        runtime.graph_sidecar = sidecar

    def _fsync_parent(self, path: Path) -> None:
        try:
            fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _journal_record_path(self, backup_root: Path) -> Path:
        return backup_root / "journal.json"

    def _write_journal_metadata(self, backup_root: Path, metadata: Dict[str, Any]) -> None:
        self._write_json_atomic(self._journal_record_path(backup_root), metadata)

    def _begin_journal(self, name: str, kind: CollectionKind, operation: str) -> MutationBackup:
        backup_root = Path(
            tempfile.mkdtemp(
                prefix=f"voyager-{self._validate_collection_name(name)}-{operation}-",
                dir=self._journal_root,
            )
        )
        metadata: Dict[str, Any] = {
            "collection": self._validate_collection_name(name),
            "kind": kind.value,
            "operation": operation,
            "state": "prepared",
            "prepared_at": self._now(),
        }
        self._write_journal_metadata(backup_root, metadata)
        return MutationBackup(runtime_name=name, backup_root=backup_root, metadata=metadata)

    def _commit_journal(self, backup: MutationBackup) -> None:
        backup.metadata["state"] = "committed"
        backup.metadata["committed_at"] = self._now()
        self._write_journal_metadata(backup.backup_root, backup.metadata)

    def _cleanup_journal(self, backup_root: Path) -> None:
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
        self._fsync_parent(path)

    def _write_meta(self, runtime: CollectionRuntime) -> None:
        runtime.meta["revision"] = int(runtime.meta.get("revision", 0)) + 1
        runtime.meta["updated_at"] = self._now()
        metadata_path = self._metadata_path(runtime.name)
        self._write_json_atomic(metadata_path, runtime.meta)
        runtime.meta_mtime_ns = metadata_path.stat().st_mtime_ns

    def _load_meta(self, name: str) -> Dict[str, Any]:
        with open(self._metadata_path(name), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["revision"] = int(meta.get("revision", 0))
        meta["records"] = self._sanitize_records(meta)
        return meta

    def _ordered_records(self, runtime: CollectionRuntime) -> List[Dict[str, Any]]:
        return sorted(
            runtime.meta.get("records", {}).values(),
            key=lambda record: int(record["internal_id"]),
        )

    def _sync_dense_sparse_state(self, runtime: CollectionRuntime, rebuild_sparse: bool) -> None:
        records = self._ordered_records(runtime)
        self._refresh_runtime_indexes(runtime)
        ids = [int(record["internal_id"]) for record in records]
        corpus = [record.get("text") or "" for record in records]
        payloads = [dict(record.get("payload") or {}) for record in records]
        if rebuild_sparse:
            runtime.engine.rebuild_sparse_state(corpus, ids, payloads)
            runtime.meta["sparse_dirty"] = False
        else:
            if hasattr(runtime.engine, "mark_sparse_dirty"):
                runtime.engine.mark_sparse_dirty(corpus, ids, payloads)
            else:
                runtime.engine.restore_buffers(corpus, ids, payloads)
            runtime.meta["sparse_dirty"] = True

    def _snapshot_collection(self, runtime: CollectionRuntime) -> Path:
        backup_root = Path(tempfile.mkdtemp(prefix=f"voyager-{runtime.name}-"))
        snapshot_path = backup_root / runtime.name
        if runtime.path.exists():
            shutil.copytree(runtime.path, snapshot_path)
        return backup_root

    def _cleanup_snapshot(self, backup_root: Path) -> None:
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)

    def _restore_snapshot(self, name: str, backup_root: Path) -> None:
        snapshot_path = backup_root / self._validate_collection_name(name)
        collection_path = self._collection_path(name)
        if collection_path.exists():
            shutil.rmtree(collection_path)
        if snapshot_path.exists():
            shutil.copytree(snapshot_path, collection_path)
            self.collections[name] = self._load_runtime(name)
        else:
            self.collections.pop(name, None)

    def _build_engine(self, name: str, meta: Dict[str, Any]):
        collection_dir = self._collection_path(name)
        kind = CollectionKind(meta["kind"])

        if kind == CollectionKind.DENSE:
            engine = HybridSearchManager(
                shard_path=collection_dir / "hybrid",
                dim=int(meta["dimension"]),
                distance_metric=str(meta.get("distance", DistanceMetric.COSINE.value)),
                m=int(meta.get("m", 16)),
                ef_construct=int(meta.get("ef_construction", 200)),
                on_disk=True,
            )
        elif kind == CollectionKind.LATE_INTERACTION:
            engine = ColbertIndex(
                collection_dir / "colbert",
                IndexConfig(
                    device=self.device,
                    batch_size=256,
                    chunk_size=256,
                    compression="lzf",
                    compression_level=1,
                ),
                create_if_missing=True,
                storage_mode=meta.get("storage_mode", "sync"),
            )
        elif kind == CollectionKind.SHARD:
            shard_cfg = ShardEngineConfig(
                dim=int(meta["dimension"]),
                n_shards=int(meta.get("n_shards", 256)),
                compression=Compression(str(meta.get("compression", Compression.RROQ158.value))),
                k_candidates=int(meta.get("k_candidates", 2000)),
                max_docs_exact=int(meta.get("max_docs_exact", 10_000)),
                n_full_scores=int(meta.get("n_full_scores", 4096)),
                lemur_search_k_cap=(
                    int(meta["lemur_search_k_cap"]) if meta.get("lemur_search_k_cap") is not None else 2048
                ),
                transfer_mode=TransferMode(str(meta.get("transfer_mode", TransferMode.PINNED.value))),
                pinned_pool_buffers=int(meta.get("pinned_pool_buffers", 3)),
                pinned_buffer_max_tokens=int(meta.get("pinned_buffer_max_tokens", 50_000)),
                use_colbandit=bool(meta.get("use_colbandit", False)),
                quantization_mode=self._normalize_quantization_mode(meta.get("quantization_mode")),
                variable_length_strategy=str(meta.get("variable_length_strategy", "bucketed") or "bucketed"),
                gpu_corpus_rerank_topn=int(meta.get("gpu_corpus_rerank_topn", 16)),
                n_centroid_approx=int(meta.get("n_centroid_approx", 0)),
                router_device=str(meta.get("router_device", "cpu") or "cpu"),
                rroq158_k=int(meta.get("rroq158_k", 8192)),
                rroq158_seed=int(meta.get("rroq158_seed", 42)),
                rroq158_group_size=int(meta.get("rroq158_group_size", 32)),
                rroq4_riem_k=int(meta.get("rroq4_riem_k", 8192)),
                rroq4_riem_seed=int(meta.get("rroq4_riem_seed", 42)),
                rroq4_riem_group_size=int(meta.get("rroq4_riem_group_size", 32)),
            )
            engine = ShardSegmentManager(
                path=collection_dir / "shard",
                config=shard_cfg,
                device=self.device,
            )
        else:
            engine = ColPaliEngine(
                collection_dir / "colpali",
                config=ColPaliConfig(
                    embed_dim=int(meta["dimension"]),
                    device=self.device,
                    use_quantization=False,
                ),
                device=self.device,
                load_if_exists=True,
            )

        return engine

    def _load_runtime(self, name: str) -> CollectionRuntime:
        metadata_path = self._metadata_path(name)
        meta = self._load_meta(name)
        runtime = CollectionRuntime(
            name=name,
            kind=CollectionKind(meta["kind"]),
            path=self._collection_path(name),
            meta=meta,
            engine=self._build_engine(name, meta),
            record_index=self._refresh_record_index(meta),
            payload_filter_index=self._build_payload_filter_index(meta),
            graph_sidecar=None,
            meta_mtime_ns=metadata_path.stat().st_mtime_ns if metadata_path.exists() else 0,
        )
        if runtime.kind == CollectionKind.DENSE:
            should_rebuild_sparse = bool(runtime.meta.get("sparse_dirty")) or (
                runtime.engine.retriever is None
                and not runtime.engine.sparse_index_present
                and bool(runtime.meta.get("records"))
            )
            self._sync_dense_sparse_state(runtime, rebuild_sparse=should_rebuild_sparse)
            if should_rebuild_sparse:
                self._write_meta(runtime)
            runtime.graph_sidecar = getattr(runtime.engine, "graph_sidecar", None)
        else:
            runtime.graph_sidecar = LatenceGraphSidecar(self._graph_state_path(name, runtime.kind))
        graph_sidecar = self._graph_sidecar_for_runtime(runtime)
        if (
            runtime.meta.get("records")
            and graph_sidecar is not None
            and not graph_sidecar.is_available()
            and str(getattr(graph_sidecar, "health", "")) != "degraded"
        ):
            self._sync_runtime_graph_state(runtime, rebuild=True, action="rebuild")
        return runtime

    def _reload_runtime_from_disk(self, name: str) -> CollectionRuntime:
        current = self.collections.get(name)
        if current is not None:
            self._close_runtime_engine(current)
        runtime = self._load_runtime(name)
        self.collections[name] = runtime
        self._ensure_collection_lock(name)
        self.load_failures.pop(name, None)
        return runtime

    def _sync_collection_registry(self) -> None:
        disk_names = {
            entry.name
            for entry in self.root_path.iterdir()
            if entry.is_dir() and not entry.name.startswith(".") and (entry / "collection.json").exists()
        }
        with self._collections_lock:
            loaded_names = set(self.collections.keys())

        removed = loaded_names - disk_names
        for name in removed:
            lock = self._ensure_collection_lock(name)
            with lock:
                with self._collections_lock:
                    runtime = self.collections.pop(name, None)
                if runtime is not None:
                    self._close_runtime_engine(runtime)
                self._remove_collection_lock(name)

        added = disk_names - loaded_names
        for name in sorted(added):
            try:
                runtime = self._load_runtime(name)
            except Exception as exc:
                self.load_failures[name] = str(exc)
                logger.warning("Failed to load collection '%s': %s", name, exc)
                continue
            with self._collections_lock:
                self.collections[name] = runtime
            self._ensure_collection_lock(name)
            self.load_failures.pop(name, None)

    def _engine_point_count(self, runtime: CollectionRuntime) -> int:
        if runtime.kind == CollectionKind.DENSE:
            return int(runtime.engine.hnsw.total_vectors())
        if runtime.kind == CollectionKind.LATE_INTERACTION:
            return int(runtime.engine.get_statistics().num_documents)
        if runtime.kind == CollectionKind.SHARD:
            return int(runtime.engine.total_vectors())
        stats = runtime.engine.get_statistics()
        return int(stats.get("num_documents", 0))

    def _is_runtime_indexed(self, runtime: CollectionRuntime) -> bool:
        engine_count = self._engine_point_count(runtime)
        return engine_count > 0 and engine_count == len(runtime.meta.get("records", {}))

    @staticmethod
    def _matches_filter(payload: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True

    def _candidate_ids_for_filter(
        self,
        runtime: CollectionRuntime,
        filters: Optional[Dict[str, Any]],
    ) -> Optional[List[int]]:
        if not filters:
            return None

        candidate_sets: List[set[int]] = []
        for key, value in filters.items():
            serialized = self._filter_value_key(value)
            if serialized is not None:
                indexed_candidates = runtime.payload_filter_index.get(str(key), {}).get(serialized)
                if indexed_candidates is not None:
                    candidate_sets.append(set(indexed_candidates))
                    continue
            scanned = {
                int(internal_id)
                for internal_id, record in runtime.record_index.items()
                if record["payload"].get(key) == value
            }
            candidate_sets.append(scanned)

        if not candidate_sets:
            return None

        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        if self.filter_scan_limit > 0 and len(candidates) > self.filter_scan_limit:
            self._increment_filter_scan_limit_hits()
            raise ScanLimitExceededError(
                "Filter candidate set exceeds the configured scan ceiling for exact late-interaction/multimodal scoring"
            )
        return sorted(candidates)

    def _load_collections(self) -> None:
        self._sync_collection_registry()

    def _ensure_collection_lock(self, name: str) -> threading.RLock:
        with self._collections_lock:
            return self._collection_locks.setdefault(name, threading.RLock())

    def _remove_collection_lock(self, name: str) -> None:
        with self._collections_lock:
            self._collection_locks.pop(name, None)

    def _collection_context(self, name: str) -> tuple[CollectionRuntime, threading.RLock]:
        safe_name = self._validate_collection_name(name)
        metadata_path = self._metadata_path(safe_name)
        with self._collections_lock:
            runtime = self.collections.get(safe_name)
            if runtime is None:
                if safe_name in self.load_failures:
                    raise NotFoundError(f"Collection '{safe_name}' not found")
                if not metadata_path.exists():
                    raise NotFoundError(f"Collection '{safe_name}' not found")
                try:
                    runtime = self._load_runtime(safe_name)
                except Exception as exc:
                    self.load_failures[safe_name] = str(exc)
                    raise NotFoundError(f"Collection '{safe_name}' not found") from exc
                self.load_failures.pop(safe_name, None)
                self.collections[safe_name] = runtime
            lock = self._collection_locks.setdefault(safe_name, threading.RLock())
        with lock:
            if not metadata_path.exists():
                with self._collections_lock:
                    stale = self.collections.pop(safe_name, None)
                if stale is not None:
                    self._close_runtime_engine(stale)
                self._remove_collection_lock(safe_name)
                raise NotFoundError(f"Collection '{safe_name}' not found")
            current_mtime_ns = metadata_path.stat().st_mtime_ns
            if current_mtime_ns != runtime.meta_mtime_ns:
                try:
                    runtime = self._reload_runtime_from_disk(safe_name)
                except Exception as exc:
                    with self._collections_lock:
                        self.collections.pop(safe_name, None)
                    self.load_failures[safe_name] = str(exc)
                    raise NotFoundError(f"Collection '{safe_name}' not found") from exc
        return runtime, lock

    def _backup_path(self, runtime: CollectionRuntime, backup_root: Path, path: Path) -> Path:
        relative = path.relative_to(runtime.path)
        return backup_root / relative

    def _copy_into_backup(self, runtime: CollectionRuntime, backup_root: Path, path: Path) -> None:
        if not path.exists():
            return
        destination = self._backup_path(runtime, backup_root, path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)

    def _restore_from_backup(self, runtime: CollectionRuntime, backup_root: Path, path: Path) -> None:
        source = self._backup_path(runtime, backup_root, path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if source.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, path)
            else:
                shutil.copy2(source, path)

    def _begin_create_collection_mutation(self, name: str, kind: CollectionKind) -> MutationBackup:
        return self._begin_journal(name, kind, operation="create_collection")

    def _close_runtime_engine(self, runtime: CollectionRuntime) -> None:
        graph_sidecar = self._graph_sidecar_for_runtime(runtime)
        if graph_sidecar is not None and hasattr(graph_sidecar, "close"):
            try:
                graph_sidecar.close()
            except Exception as exc:
                logger.warning("Failed to close graph sidecar for collection '%s': %s", runtime.name, exc)
        close_method = getattr(runtime.engine, "close", None)
        if callable(close_method):
            close_method()

    def _begin_collection_mutation(self, runtime: CollectionRuntime, operation: str) -> MutationBackup:
        backup = self._begin_journal(runtime.name, runtime.kind, operation=operation)
        backup_root = backup.backup_root
        metadata = backup.metadata
        self._copy_into_backup(runtime, backup_root, self._metadata_path(runtime.name))

        if runtime.kind == CollectionKind.DENSE:
            self._close_runtime_engine(runtime)
            self._copy_into_backup(runtime, backup_root, runtime.path / "hybrid")
            runtime.engine = self._build_engine(runtime.name, runtime.meta)
            runtime.graph_sidecar = getattr(runtime.engine, "graph_sidecar", None)
        elif runtime.kind == CollectionKind.SHARD:
            if hasattr(runtime.engine, "flush") and callable(runtime.engine.flush):
                runtime.engine.flush()
            shard_dir = runtime.path / "shard"
            if shard_dir.exists():
                self._copy_into_backup(runtime, backup_root, shard_dir)
            self._copy_into_backup(runtime, backup_root, self._graph_state_path(runtime.name, runtime.kind))
        elif runtime.kind == CollectionKind.LATE_INTERACTION:
            self._copy_into_backup(runtime, backup_root, runtime.path / "colbert" / "embeddings.h5")
            self._copy_into_backup(runtime, backup_root, runtime.path / "colbert" / "metadata.json")
            self._copy_into_backup(runtime, backup_root, self._graph_state_path(runtime.name, runtime.kind))
        else:
            manifest_path = runtime.path / "colpali" / "manifest.json"
            screening_state_path = runtime.path / "colpali" / "screening_state.json"
            chunks_path = runtime.path / "colpali" / "chunks"
            self._copy_into_backup(runtime, backup_root, screening_state_path)
            self._copy_into_backup(runtime, backup_root, runtime.path / "colpali" / "prototype_sidecar")
            self._copy_into_backup(runtime, backup_root, runtime.path / "colpali" / "centroid_sidecar")
            self._copy_into_backup(runtime, backup_root, manifest_path)
            self._copy_into_backup(runtime, backup_root, self._graph_state_path(runtime.name, runtime.kind))
            metadata["existing_chunk_files"] = (
                sorted(path.name for path in chunks_path.glob("*.npz")) if chunks_path.exists() else []
            )
        self._write_journal_metadata(backup_root, metadata)
        return backup

    def _begin_delete_collection_mutation(self, runtime: CollectionRuntime) -> MutationBackup:
        return self._begin_journal(runtime.name, runtime.kind, operation="delete_collection")

    def _runtime_stub(self, name: str, kind: CollectionKind) -> CollectionRuntime:
        return CollectionRuntime(
            name=name,
            kind=kind,
            path=self._collection_path(name),
            meta={},
            engine=None,
            record_index={},
            payload_filter_index={},
            graph_sidecar=None,
        )

    def _restore_collection_mutation(self, runtime: CollectionRuntime, backup: MutationBackup) -> None:
        self._close_runtime_engine(runtime)
        self._restore_collection_from_backup(runtime.name, runtime.kind, backup.backup_root, backup.metadata)
        with self._collections_lock:
            self.collections[runtime.name] = self._load_runtime(runtime.name)

    def _restore_collection_from_backup(
        self,
        name: str,
        kind: CollectionKind,
        backup_root: Path,
        metadata: Dict[str, Any],
    ) -> None:
        runtime = self._runtime_stub(name, kind)
        self._restore_from_backup(runtime, backup_root, self._metadata_path(runtime.name))

        if kind == CollectionKind.DENSE:
            self._restore_from_backup(runtime, backup_root, runtime.path / "hybrid")
        elif kind == CollectionKind.SHARD:
            shard_dir = runtime.path / "shard"
            self._restore_from_backup(runtime, backup_root, shard_dir)
            self._restore_from_backup(runtime, backup_root, self._graph_state_path(runtime.name, kind))
        elif kind == CollectionKind.LATE_INTERACTION:
            self._restore_from_backup(runtime, backup_root, runtime.path / "colbert" / "embeddings.h5")
            self._restore_from_backup(runtime, backup_root, runtime.path / "colbert" / "metadata.json")
            self._restore_from_backup(runtime, backup_root, self._graph_state_path(runtime.name, kind))
        else:
            chunks_path = runtime.path / "colpali" / "chunks"
            existing_chunk_files = set(metadata.get("existing_chunk_files", []))
            if chunks_path.exists():
                for chunk_path in chunks_path.glob("*.npz"):
                    if chunk_path.name not in existing_chunk_files:
                        chunk_path.unlink(missing_ok=True)
            self._restore_from_backup(runtime, backup_root, runtime.path / "colpali" / "manifest.json")
            self._restore_from_backup(runtime, backup_root, runtime.path / "colpali" / "screening_state.json")
            self._restore_from_backup(runtime, backup_root, runtime.path / "colpali" / "prototype_sidecar")
            self._restore_from_backup(runtime, backup_root, runtime.path / "colpali" / "centroid_sidecar")
            self._restore_from_backup(runtime, backup_root, self._graph_state_path(runtime.name, kind))

    def _recover_pending_journals(self) -> None:
        for backup_root in sorted(self._journal_root.iterdir()):
            if not backup_root.is_dir():
                continue
            record_path = self._journal_record_path(backup_root)
            if not record_path.exists():
                self._cleanup_journal(backup_root)
                continue
            try:
                metadata = json.loads(record_path.read_text(encoding="utf-8"))
                state = metadata.get("state")
                if state == "committed":
                    self._cleanup_journal(backup_root)
                    continue
                name = self._validate_collection_name(metadata["collection"])
                kind = CollectionKind(metadata["kind"])
                operation = metadata["operation"]
                collection_path = self._collection_path(name)
                if operation == "create_collection":
                    if collection_path.exists():
                        shutil.rmtree(collection_path, ignore_errors=True)
                elif operation == "delete_collection":
                    staged_path = backup_root / "staged_collection"
                    if staged_path.exists():
                        if collection_path.exists():
                            shutil.rmtree(staged_path, ignore_errors=True)
                        else:
                            staged_path.rename(collection_path)
                else:
                    self._restore_collection_from_backup(name, kind, backup_root, metadata)
                self._cleanup_journal(backup_root)
            except Exception as exc:
                journal_name = backup_root.name
                collection_name = metadata.get("collection", journal_name) if "metadata" in locals() else journal_name
                self.load_failures[str(collection_name)] = f"journal_recovery_failed: {exc}"
                logger.warning("Failed to recover journal '%s': %s", journal_name, exc)

    def list_collections(self) -> List[str]:
        self._sync_collection_registry()
        with self._collections_lock:
            return sorted(self.collections.keys())

    def get_collection(self, name: str) -> CollectionRuntime:
        runtime, _lock = self._collection_context(name)
        return runtime

    def create_collection(self, name: str, request: CreateCollectionRequest) -> CollectionRuntime:
        safe_name = self._validate_collection_name(name)
        with self._cross_process_collection_lock(safe_name):
            with self._collections_lock:
                if safe_name in self.collections or self._metadata_path(safe_name).exists():
                    raise ConflictError(f"Collection '{safe_name}' already exists")

                if request.kind not in (CollectionKind.DENSE, CollectionKind.SHARD):
                    if request.distance != DistanceMetric.COSINE:
                        raise ValidationError("Only dense and shard collections support configurable distance metrics")
                    if request.m != 16 or request.ef_construction != 200:
                        raise ValidationError("Only dense collections support configurable HNSW parameters")
                if request.kind != CollectionKind.LATE_INTERACTION and request.storage_mode != "sync":
                    raise ValidationError("storage_mode is only supported for late-interaction collections")

                collection_dir = self._collection_path(safe_name)
                backup = self._begin_create_collection_mutation(safe_name, request.kind)
                collection_dir.mkdir(parents=True, exist_ok=True)

                meta = {
                    "name": safe_name,
                    "kind": request.kind.value,
                    "dimension": request.dimension,
                    "distance": request.distance.value,
                    "m": request.m,
                    "ef_construction": request.ef_construction,
                    "storage_mode": request.storage_mode,
                    "sparse_dirty": False,
                    "created_at": self._now(),
                    "updated_at": self._now(),
                    "next_internal_id": 0,
                    "records": {},
                    "revision": 0,
                }
                if request.max_documents is not None:
                    meta["max_documents"] = request.max_documents
                if request.kind == CollectionKind.SHARD:
                    meta["n_shards"] = request.n_shards or 256
                    meta["k_candidates"] = request.k_candidates or 2000
                    meta["use_colbandit"] = request.use_colbandit
                    meta["compression"] = str(request.compression or Compression.RROQ158.value)
                    meta["quantization_mode"] = self._normalize_quantization_mode(request.quantization_mode)
                    meta["transfer_mode"] = str(request.transfer_mode or TransferMode.PINNED.value)
                    meta["pinned_pool_buffers"] = int(request.pinned_pool_buffers or 3)
                    meta["pinned_buffer_max_tokens"] = int(request.pinned_buffer_max_tokens or 50_000)
                    meta["router_device"] = str(request.router_device or "cpu")
                    meta["lemur_search_k_cap"] = (
                        int(request.lemur_search_k_cap) if request.lemur_search_k_cap is not None else 2048
                    )
                    meta["max_docs_exact"] = int(request.max_docs_exact or 10_000)
                    meta["n_full_scores"] = int(request.n_full_scores or 4096)
                    meta["gpu_corpus_rerank_topn"] = int(request.gpu_corpus_rerank_topn or 16)
                    meta["n_centroid_approx"] = int(request.n_centroid_approx or 0)
                    meta["variable_length_strategy"] = str(request.variable_length_strategy or "bucketed")
                    meta["rroq158_k"] = int(getattr(request, "rroq158_k", None) or 8192)
                    meta["rroq158_seed"] = int(getattr(request, "rroq158_seed", None) or 42)
                    meta["rroq158_group_size"] = int(getattr(request, "rroq158_group_size", None) or 32)
                    meta["rroq4_riem_k"] = int(getattr(request, "rroq4_riem_k", None) or 8192)
                    meta["rroq4_riem_seed"] = int(getattr(request, "rroq4_riem_seed", None) or 42)
                    meta["rroq4_riem_group_size"] = int(getattr(request, "rroq4_riem_group_size", None) or 32)
                try:
                    runtime = CollectionRuntime(
                        name=safe_name,
                        kind=request.kind,
                        path=collection_dir,
                        meta=meta,
                        engine=self._build_engine(safe_name, meta),
                        record_index={},
                        payload_filter_index={},
                        graph_sidecar=None,
                    )
                    if runtime.kind == CollectionKind.DENSE:
                        runtime.graph_sidecar = getattr(runtime.engine, "graph_sidecar", None)
                    else:
                        runtime.graph_sidecar = LatenceGraphSidecar(self._graph_state_path(safe_name, runtime.kind))
                        runtime.graph_sidecar.save()
                    self._write_meta(runtime)
                    self.collections[safe_name] = runtime
                    self._collection_locks[safe_name] = threading.RLock()
                    self._commit_journal(backup)
                    return runtime
                except Exception:
                    self.collections.pop(safe_name, None)
                    self._collection_locks.pop(safe_name, None)
                    if collection_dir.exists():
                        shutil.rmtree(collection_dir, ignore_errors=True)
                    raise
                finally:
                    self._cleanup_journal(backup.backup_root)

    def delete_collection(self, name: str) -> None:
        runtime, collection_lock = self._collection_context(name)
        deleted = False
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                backup = self._begin_delete_collection_mutation(runtime)
                staged_path = backup.backup_root / "staged_collection"
                try:
                    self._close_runtime_engine(runtime)
                    if runtime.path.exists():
                        runtime.path.rename(staged_path)
                    with self._collections_lock:
                        self.collections.pop(runtime.name, None)
                except Exception:
                    if staged_path.exists() and not runtime.path.exists():
                        staged_path.rename(runtime.path)
                    raise
        try:
            if staged_path.exists():
                shutil.rmtree(staged_path)
            deleted = True
            self._commit_journal(backup)
        except Exception:
            with collection_lock:
                if staged_path.exists() and not runtime.path.exists():
                    staged_path.rename(runtime.path)
                with self._collections_lock:
                    self.collections[runtime.name] = self._load_runtime(runtime.name)
            raise
        finally:
            if deleted:
                self._remove_collection_lock(runtime.name)
            self._cleanup_journal(backup.backup_root)

    def _next_internal_id(self, runtime: CollectionRuntime) -> int:
        internal_id = int(runtime.meta.get("next_internal_id", 0))
        runtime.meta["next_internal_id"] = internal_id + 1
        return internal_id

    def _delete_internal_ids(self, runtime: CollectionRuntime, internal_ids: List[int]) -> None:
        if not internal_ids:
            return
        if runtime.kind == CollectionKind.DENSE:
            runtime.engine.hnsw.delete(internal_ids)
        elif runtime.kind == CollectionKind.SHARD:
            runtime.engine.delete(internal_ids)
        elif runtime.kind == CollectionKind.LATE_INTERACTION:
            runtime.engine.delete_documents(internal_ids)
        else:
            runtime.engine.delete_documents(internal_ids)

    def _flush_runtime_engine(self, runtime: CollectionRuntime) -> None:
        """Force durable engine state to disk before journal commit when supported."""
        flush_method = getattr(runtime.engine, "flush", None)
        if callable(flush_method):
            flush_method()

    def _prepare_payload(self, point: PointVector) -> Dict[str, Any]:
        payload = dict(point.payload or {})
        payload.setdefault("external_id", point.id)
        return payload

    def _decode_transport_array(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, TransportVectorPayload):
            try:
                array = decode_payload(value.model_dump())
            except Exception as exc:
                raise ValidationError(f"Invalid {field_name} payload: {exc}") from exc
        else:
            array = np.asarray(value, dtype=np.float32)
        return np.asarray(array, dtype=np.float32)

    def _coerce_single_vector_input(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> Optional[np.ndarray]:
        array = self._decode_transport_array(value, field_name=field_name)
        if array is None:
            return None
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise ValidationError(f"{field_name} must decode to a single vector")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _coerce_multi_vector_input(
        self,
        value: Any,
        *,
        field_name: str,
    ) -> Optional[np.ndarray]:
        array = self._decode_transport_array(value, field_name=field_name)
        if array is None:
            return None
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValidationError(f"{field_name} must decode to a 2D vector matrix")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _trim_late_interaction_vectors(
        self,
        runtime: CollectionRuntime,
        internal_ids: List[int],
        docs: torch.Tensor,
    ) -> Dict[int, np.ndarray]:
        vectors_by_id: Dict[int, np.ndarray] = {}
        for idx, internal_id in enumerate(internal_ids):
            true_length = int(runtime.engine.storage.doc_lengths.get(int(internal_id), docs[idx].shape[0]))
            vectors_by_id[int(internal_id)] = docs[idx][:true_length].cpu().numpy()
        return vectors_by_id

    def add_points(self, name: str, points: List[PointVector]) -> int:
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                backup = self._begin_collection_mutation(runtime, operation="add_points")
                try:
                    record_keys = [str(point.id) for point in points]
                    if len(record_keys) != len(set(record_keys)):
                        raise ValidationError("Point IDs must be unique within a single request")

                    payloads = [self._prepare_payload(point) for point in points]
                    corpus = [payload.get("text") or payload.get("content") or "" for payload in payloads]
                    existing_internal_ids = [
                        int(runtime.meta["records"][record_key]["internal_id"])
                        for record_key in record_keys
                        if record_key in runtime.meta["records"]
                    ]
                    if existing_internal_ids:
                        self._delete_runtime_graph_targets(runtime, record_keys)
                        self._delete_internal_ids(runtime, existing_internal_ids)

                    if runtime.kind == CollectionKind.DENSE:
                        if any(point.vector is None for point in points):
                            raise ValidationError("Dense collections require single vectors")
                        vectors = np.stack(
                            [
                                self._coerce_single_vector_input(
                                    point.vector,
                                    field_name=f"points[{idx}].vector",
                                )
                                for idx, point in enumerate(points)
                            ],
                            axis=0,
                        )
                        if vectors.ndim != 2 or vectors.shape[1] != int(runtime.meta["dimension"]):
                            raise ValidationError("Dense vectors must match the collection dimension")
                        ids = [self._next_internal_id(runtime) for _ in points]
                        runtime.engine.hnsw.add(vectors, ids=ids, payloads=payloads)
                        for record_key in record_keys:
                            runtime.meta["records"].pop(record_key, None)
                        for record_key, point, payload, text_value, internal_id in zip(
                            record_keys, points, payloads, corpus, ids
                        ):
                            runtime.meta["records"][record_key] = {
                                "external_id": point.id,
                                "internal_id": internal_id,
                                "payload": payload,
                                "text": text_value,
                            }
                        new_records = [runtime.meta["records"][record_key] for record_key in record_keys]
                        self._sync_runtime_graph_state(runtime, records=new_records, rebuild=False, action="append")
                        self._sync_dense_sparse_state(runtime, rebuild_sparse=False)
                        self._flush_runtime_engine(runtime)
                    elif runtime.kind == CollectionKind.LATE_INTERACTION:
                        if any(point.vectors is None for point in points):
                            raise ValidationError("Late-interaction collections require multi-vectors")
                        tensor = torch.from_numpy(
                            np.stack(
                                [
                                    self._coerce_multi_vector_input(
                                        point.vectors,
                                        field_name=f"points[{idx}].vectors",
                                    )
                                    for idx, point in enumerate(points)
                                ],
                                axis=0,
                            ),
                        )
                        if tensor.dim() != 3 or tensor.shape[-1] != int(runtime.meta["dimension"]):
                            raise ValidationError("Late-interaction tensors must have shape (docs, tokens, dim)")
                        if runtime.engine.storage.num_docs == 0 and not runtime.engine.storage.data_file.exists():

                            def batch_gen():
                                yield tensor.detach().cpu(), payloads

                            assigned_ids = runtime.engine.build_from_batches(
                                batch_gen(),
                                collection_name="default",
                                max_tokens=tensor.shape[1],
                            )
                        else:
                            assigned_ids = runtime.engine.add_documents(tensor, metadata=payloads)
                        for record_key in record_keys:
                            runtime.meta["records"].pop(record_key, None)
                        for record_key, point, payload, text_value, internal_id in zip(
                            record_keys, points, payloads, corpus, assigned_ids
                        ):
                            runtime.meta["records"][record_key] = {
                                "external_id": point.id,
                                "internal_id": int(internal_id),
                                "payload": payload,
                                "text": text_value,
                            }
                        if assigned_ids:
                            runtime.meta["next_internal_id"] = max(
                                int(runtime.meta.get("next_internal_id", 0)),
                                max(int(item_id) for item_id in assigned_ids) + 1,
                            )
                        new_records = [runtime.meta["records"][record_key] for record_key in record_keys]
                        self._sync_runtime_graph_state(runtime, records=new_records, rebuild=False, action="append")
                        self._refresh_runtime_indexes(runtime)
                    elif runtime.kind == CollectionKind.SHARD:
                        if any(point.vectors is None for point in points):
                            raise ValidationError("Shard collections require multi-vectors")
                        multi_vecs = [
                            self._coerce_multi_vector_input(
                                point.vectors,
                                field_name=f"points[{idx}].vectors",
                            )
                            for idx, point in enumerate(points)
                        ]
                        if any(v.ndim != 2 or v.shape[1] != int(runtime.meta["dimension"]) for v in multi_vecs):
                            raise ValidationError("Shard multi-vectors must have shape (tokens, dim)")
                        ids = [self._next_internal_id(runtime) for _ in points]
                        runtime.engine.add_multidense(multi_vecs, ids, payloads)
                        for record_key in record_keys:
                            runtime.meta["records"].pop(record_key, None)
                        for record_key, point, payload, text_value, internal_id in zip(
                            record_keys,
                            points,
                            payloads,
                            corpus,
                            ids,
                        ):
                            runtime.meta["records"][record_key] = {
                                "external_id": point.id,
                                "internal_id": internal_id,
                                "payload": payload,
                                "text": text_value,
                            }
                        new_records = [runtime.meta["records"][record_key] for record_key in record_keys]
                        self._sync_runtime_graph_state(runtime, records=new_records, rebuild=False, action="append")
                        self._flush_runtime_engine(runtime)
                        self._refresh_runtime_indexes(runtime)
                    else:
                        if any(point.vectors is None for point in points):
                            raise ValidationError("Multimodal collections require multi-vectors")
                        vectors = np.stack(
                            [
                                self._coerce_multi_vector_input(
                                    point.vectors,
                                    field_name=f"points[{idx}].vectors",
                                )
                                for idx, point in enumerate(points)
                            ],
                            axis=0,
                        )
                        if vectors.ndim != 3 or vectors.shape[-1] != int(runtime.meta["dimension"]):
                            raise ValidationError("Multimodal tensors must have shape (docs, patches, dim)")
                        ids = [self._next_internal_id(runtime) for _ in points]
                        runtime.engine.add_documents(vectors, doc_ids=ids)
                        for record_key in record_keys:
                            runtime.meta["records"].pop(record_key, None)
                        for record_key, point, payload, text_value, internal_id in zip(
                            record_keys, points, payloads, corpus, ids
                        ):
                            runtime.meta["records"][record_key] = {
                                "external_id": point.id,
                                "internal_id": internal_id,
                                "payload": payload,
                                "text": text_value,
                            }
                        new_records = [runtime.meta["records"][record_key] for record_key in record_keys]
                        self._sync_runtime_graph_state(runtime, records=new_records, rebuild=False, action="append")
                        self._refresh_runtime_indexes(runtime)

                    self._evict_if_over_limit(runtime)
                    self._write_meta(runtime)
                    self._commit_journal(backup)
                    return len(points)
                except Exception:
                    self._restore_collection_mutation(runtime, backup)
                    raise
                finally:
                    self._cleanup_journal(backup.backup_root)

    def delete_points(self, name: str, point_ids: List[Any]) -> int:
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                backup = self._begin_collection_mutation(runtime, operation="delete_points")
                try:
                    record_keys = [str(point_id) for point_id in point_ids]
                    internal_ids = [
                        int(runtime.meta["records"][record_key]["internal_id"])
                        for record_key in record_keys
                        if record_key in runtime.meta["records"]
                    ]
                    if not internal_ids:
                        return 0

                    self._delete_internal_ids(runtime, internal_ids)
                    self._delete_runtime_graph_targets(runtime, record_keys)
                    for record_key in record_keys:
                        runtime.meta["records"].pop(record_key, None)

                    if runtime.kind == CollectionKind.DENSE:
                        self._sync_dense_sparse_state(runtime, rebuild_sparse=False)
                        self._flush_runtime_engine(runtime)
                    else:
                        self._refresh_runtime_indexes(runtime)

                    self._write_meta(runtime)
                    self._commit_journal(backup)
                    return len(internal_ids)
                except Exception:
                    self._restore_collection_mutation(runtime, backup)
                    raise
                finally:
                    self._cleanup_journal(backup.backup_root)

    def _read_task_record(self, task_id: str) -> Dict[str, Any]:
        path = self._task_record_path(task_id)
        if not path.exists():
            raise NotFoundError("Task not found")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_task_record(self, task_id: str, record: Dict[str, Any]) -> None:
        self._write_json_atomic(self._task_record_path(task_id), record)

    def _evict_stale_task_records(self) -> None:
        now = datetime.now(timezone.utc)
        for record_path in self._task_root.glob("*.json"):
            try:
                with open(record_path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except Exception:
                record_path.unlink(missing_ok=True)
                continue
            if record.get("status") not in (MutationTaskStatus.COMPLETED.value, MutationTaskStatus.FAILED.value):
                continue
            completed = record.get("completed_at")
            if not completed:
                continue
            try:
                age = (now - datetime.fromisoformat(completed)).total_seconds()
            except (TypeError, ValueError):
                age = _TASK_MAX_AGE_S + 1
            if age > _TASK_MAX_AGE_S:
                record_path.unlink(missing_ok=True)

    def submit_task(self, fn: Callable[[], Dict[str, Any]]) -> str:
        task_id = uuid.uuid4().hex[:12]
        record: Dict[str, Any] = {
            "status": MutationTaskStatus.PENDING.value,
            "result": None,
            "error": None,
            "created_at": self._now(),
            "completed_at": None,
        }
        with self._task_thread_lock:
            task_count = sum(1 for _ in self._task_root.glob("*.json"))
            if task_count >= _TASK_MAX_ENTRIES:
                self._evict_stale_task_records()
            self._write_task_record(task_id, record)

        def _worker() -> None:
            try:
                running = self._read_task_record(task_id)
                running["status"] = MutationTaskStatus.RUNNING.value
                self._write_task_record(task_id, running)
                result = fn()
                running["status"] = MutationTaskStatus.COMPLETED.value
                running["result"] = result
                running["completed_at"] = self._now()
                self._write_task_record(task_id, running)
            except Exception as exc:
                logger.error("Async task %s failed: %s", task_id, exc, exc_info=True)
                failed = {
                    "status": MutationTaskStatus.FAILED.value,
                    "result": None,
                    "error": str(exc),
                    "created_at": record["created_at"],
                    "completed_at": self._now(),
                }
                self._write_task_record(task_id, failed)

        threading.Thread(target=_worker, name=f"task-{task_id}", daemon=True).start()
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return self._read_task_record(task_id)

    # ------------------------------------------------------------------
    # Shard admin
    # ------------------------------------------------------------------

    def _assert_shard_collection(self, name: str) -> tuple:
        runtime, lock = self._collection_context(name)
        if runtime.kind != CollectionKind.SHARD:
            raise ValidationError(f"Collection '{name}' is not a shard collection")
        return runtime, lock

    def compact_collection(self, name: str) -> Dict[str, Any]:
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            memtable_size = runtime.engine._memtable.size if runtime.engine._memtable else 0
            runtime.engine.flush()
            return {"memtable_docs_at_sync": memtable_size}

    def list_shards(self, name: str) -> Dict[str, Any]:
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            store = runtime.engine._store
            if not store or not store.manifest:
                return {"collection": name, "num_shards": 0, "shards": []}
            shards = []
            for s in store.manifest.shards:
                avg = s.total_tokens / max(s.num_docs, 1)
                shards.append(
                    {
                        "shard_id": s.shard_id,
                        "num_docs": s.num_docs,
                        "total_tokens": s.total_tokens,
                        "avg_tokens": avg,
                        "p95_tokens": s.p95_tokens,
                    }
                )
            return {
                "collection": name,
                "num_shards": store.manifest.num_shards,
                "shards": shards,
            }

    def get_shard_detail(self, name: str, shard_id: int) -> Dict[str, Any]:
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            store = runtime.engine._store
            if not store or not store.manifest:
                raise NotFoundError(f"No shards in collection '{name}'")
            for s in store.manifest.shards:
                if s.shard_id == shard_id:
                    return {
                        "shard_id": s.shard_id,
                        "num_docs": s.num_docs,
                        "total_tokens": s.total_tokens,
                        "avg_tokens": s.total_tokens / max(s.num_docs, 1),
                        "p95_tokens": s.p95_tokens,
                    }
            raise NotFoundError(f"Shard {shard_id} not found in '{name}'")

    def wal_status(self, name: str) -> Dict[str, Any]:
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            wal_entries = 0
            if runtime.engine._wal_writer:
                wal_entries = runtime.engine._wal_writer.n_entries
            mt = runtime.engine._memtable
            return {
                "collection": name,
                "wal_entries": wal_entries,
                "memtable_docs": mt.size if mt else 0,
                "memtable_tombstones": mt.tombstone_count if mt else 0,
            }

    def checkpoint_collection(self, name: str) -> Dict[str, Any]:
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            wal_before = runtime.engine._wal_writer.n_entries if runtime.engine._wal_writer else 0
            runtime.engine.flush()
            wal_after = runtime.engine._wal_writer.n_entries if runtime.engine._wal_writer else 0
            return {"wal_entries_before": wal_before, "wal_entries_after": wal_after}

    def scroll_collection(self, name: str, request) -> Dict[str, Any]:
        """Paginated iteration over document IDs."""
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            ids, next_offset = runtime.engine.scroll(
                limit=request.limit,
                offset=request.offset,
                filters=request.filter,
            )
            return {"ids": ids, "next_offset": next_offset}

    def retrieve_points(self, name: str, request) -> Dict[str, Any]:
        """Retrieve specific documents by ID."""
        runtime, lock = self._assert_shard_collection(name)
        with lock:
            points = runtime.engine.retrieve(
                ids=request.ids,
                with_vector=request.with_vector,
                with_payload=request.with_payload,
            )
            result_points = []
            for p in points:
                entry = {"id": p["id"], "payload": p.get("payload", {})}
                if request.with_vector and p.get("vector") is not None:
                    vec = p["vector"]
                    entry["vector"] = vec.tolist() if hasattr(vec, "tolist") else vec
                result_points.append(entry)
            return {"points": result_points}

    def search_batch_collection(self, name: str, request) -> Dict[str, Any]:
        """Batch search over multiple queries."""
        results = []
        for search_req in request.searches:
            resp = self.search(name, search_req)
            results.append(resp)
        return {"results": results}

    def _dense_query_vector(self, runtime: CollectionRuntime, request: SearchRequest) -> Optional[np.ndarray]:
        if request.vector is not None:
            query = self._coerce_single_vector_input(request.vector, field_name="vector")
        elif request.vectors is not None:
            raise ValidationError(
                "Dense search requires 'vector'; 'vectors' is only supported for late-interaction or multimodal collections"
            )
        else:
            return None

        if query.shape[0] != int(runtime.meta["dimension"]):
            raise ValidationError("Query dimension does not match collection dimension")
        return query

    def _get_dense_vectors(self, runtime: CollectionRuntime, internal_ids: List[int]) -> Dict[int, Any]:
        items = runtime.engine.hnsw.retrieve(internal_ids)
        return {int(item["id"]): item.get("vector") for item in items}

    def _dense_solver_constraints(self, request: SearchRequest) -> Any:
        if request.max_tokens is None and request.max_chunks is None and request.max_per_cluster is None:
            return None
        try:
            from latence_solver import SolverConstraints  # type: ignore
        except ImportError as exc:
            raise ValidationError(
                "Optimized dense search requires the latence_solver native package to be installed"
            ) from exc
        return SolverConstraints(
            max_tokens=request.max_tokens or 1024,
            max_chunks=request.max_chunks or 10,
            max_per_cluster=request.max_per_cluster or 2,
        )

    @staticmethod
    def _request_solver_config(
        request: SearchRequest,
        *,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = dict(defaults or {})
        if request.solver_config:
            merged.update(dict(request.solver_config))
        if request.optimizer_policy is not None:
            merged["optimizer_policy"] = request.optimizer_policy
        return merged

    @staticmethod
    def _dense_refine_options(request: SearchRequest) -> Optional[Dict[str, Any]]:
        if not (
            request.refine_use_cross_encoder
            or request.refine_use_nli
            or request.refine_confidence_gating
            or request.refine_cross_encoder_model
            or request.refine_nli_model
        ):
            return None
        if (request.refine_use_cross_encoder or request.refine_use_nli) and not (request.query_text or "").strip():
            raise ValidationError(
                "Optimized dense search with refine_use_cross_encoder/refine_use_nli requires query_text"
            )
        options: Dict[str, Any] = {}
        if request.refine_use_cross_encoder:
            options["use_cross_encoder"] = True
        if request.refine_cross_encoder_model:
            options["cross_encoder_model"] = request.refine_cross_encoder_model
        if request.refine_cross_encoder_top_k is not None:
            options["cross_encoder_top_k"] = int(request.refine_cross_encoder_top_k)
        if request.refine_cross_encoder_batch_size is not None:
            options["cross_encoder_batch_size"] = int(request.refine_cross_encoder_batch_size)
        if request.refine_use_nli:
            options["use_nli"] = True
        if request.refine_nli_model:
            options["nli_model"] = request.refine_nli_model
        if request.refine_nli_top_k is not None:
            options["nli_top_k"] = int(request.refine_nli_top_k)
        if request.refine_nli_batch_size is not None:
            options["nli_batch_size"] = int(request.refine_nli_batch_size)
        if request.refine_nli_promote_base_relevance:
            options["nli_promote_base_relevance"] = True
        if request.refine_confidence_gating:
            options["confidence_gating"] = True
        if request.refine_confidence_gap_threshold is not None:
            options["confidence_gap_threshold"] = float(request.refine_confidence_gap_threshold)
        if request.refine_confidence_min_candidates is not None:
            options["confidence_min_candidates"] = int(request.refine_confidence_min_candidates)
        return options or None

    @staticmethod
    def _dense_uses_solver(request: SearchRequest) -> bool:
        if request.dense_hybrid_mode is not None:
            return request.dense_hybrid_mode.value == "tabu"
        return request.strategy == SearchStrategy.OPTIMIZED

    @staticmethod
    def _normalize_quantization_mode(value: Optional[str]) -> str:
        normalized = str(value or "").strip().lower()
        return "" if normalized in {"", "none"} else normalized

    @staticmethod
    def _shard_search_overrides(request: SearchRequest) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        if request.quantization_mode is not None:
            overrides["quantization_mode"] = SearchService._normalize_quantization_mode(request.quantization_mode)
        if request.use_colbandit is not None:
            overrides["use_colbandit"] = bool(request.use_colbandit)
        if request.transfer_mode is not None:
            overrides["transfer_mode"] = request.transfer_mode
        if request.max_docs_exact is not None:
            overrides["max_docs_exact"] = int(request.max_docs_exact)
        if request.n_full_scores is not None:
            overrides["n_full_scores"] = int(request.n_full_scores)
        if request.lemur_search_k_cap is not None:
            overrides["lemur_search_k_cap"] = int(request.lemur_search_k_cap)
        if request.gpu_corpus_rerank_topn is not None:
            overrides["gpu_corpus_rerank_topn"] = int(request.gpu_corpus_rerank_topn)
        if request.n_centroid_approx is not None:
            overrides["n_centroid_approx"] = int(request.n_centroid_approx)
        if request.variable_length_strategy is not None:
            overrides["variable_length_strategy"] = str(request.variable_length_strategy)
        if request.pinned_pool_buffers is not None:
            overrides["pinned_pool_buffers"] = int(request.pinned_pool_buffers)
        if request.pinned_buffer_max_tokens is not None:
            overrides["pinned_buffer_max_tokens"] = int(request.pinned_buffer_max_tokens)
        return overrides

    def _dense_refine_candidate_k(self, request: SearchRequest) -> int:
        if not self._dense_uses_solver(request):
            return request.top_k
        requested_chunks = request.max_chunks or min(10, request.top_k)
        widened = min(256, max(request.top_k * 4, requested_chunks * 6, 24))
        return max(request.top_k, widened)

    def _multimodal_screen_candidate_k(
        self,
        request: SearchRequest,
        candidate_ids: Optional[List[int]],
    ) -> int:
        widened = min(512, max(request.top_k * 16, 64))
        if candidate_ids is not None:
            widened = min(widened, len(candidate_ids))
        return max(request.top_k, widened)

    def _multimodal_candidate_budget(
        self,
        request: SearchRequest,
        candidate_ids: Optional[List[int]],
    ) -> int:
        budget = request.multimodal_candidate_budget or self._multimodal_screen_candidate_k(request, candidate_ids)
        if candidate_ids is not None:
            budget = min(int(budget), len(candidate_ids))
        return max(request.top_k, int(budget))

    @staticmethod
    def _multimodal_optimize_mode(request: SearchRequest) -> MultimodalOptimizeMode:
        if request.strategy != SearchStrategy.OPTIMIZED:
            return MultimodalOptimizeMode.MAXSIM_ONLY
        if request.multimodal_optimize_mode in (None, MultimodalOptimizeMode.AUTO):
            return MultimodalOptimizeMode.MAXSIM_ONLY
        return request.multimodal_optimize_mode

    @staticmethod
    def _stable_cluster_id(document_key: Any) -> int:
        digest = hashlib.sha1(str(document_key).encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) % 2_147_483_647

    @staticmethod
    def _normalize_signal(values: List[float]) -> Dict[int, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float32)
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if maximum - minimum <= 1e-6:
            normalized = np.ones_like(array, dtype=np.float32)
        else:
            normalized = (array - minimum) / (maximum - minimum)
        return {index: float(value) for index, value in enumerate(normalized.tolist())}

    @staticmethod
    def _token_density_score(text: str, token_count: int) -> float:
        words = [token.lower() for token in text.split() if token]
        if not words:
            return float(min(1.0, max(token_count, 1) / 256.0))
        uniqueness = len(set(words)) / float(max(len(words), 1))
        richness = min(1.0, len(words) / 48.0)
        return float(min(1.0, (0.65 * uniqueness) + (0.35 * richness)))

    @staticmethod
    def _payload_recency_score(payload: Dict[str, Any]) -> float:
        page_number = payload.get("page_number")
        if isinstance(page_number, int) and page_number > 0:
            return float(1.0 / (1.0 + ((page_number - 1) / 6.0)))
        return 0.5

    @staticmethod
    def _pool_vector(vector: np.ndarray) -> np.ndarray:
        pooled = np.asarray(vector, dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(pooled)) + 1e-8
        return (pooled / norm).astype(np.float32, copy=False)

    def _get_multimodal_optimizer_pipeline(self) -> Any:
        if hasattr(self, "_multimodal_optimizer_pipeline"):
            return self._multimodal_optimizer_pipeline
        try:
            from voyager_index._internal.inference.stateless_optimizer import GpuFulfilmentPipeline
        except ImportError as exc:  # pragma: no cover - exercised in integration environments
            raise ValidationError(
                "Optimized multimodal solver stages require the latence_solver native package to be installed"
            ) from exc
        self._multimodal_optimizer_pipeline = GpuFulfilmentPipeline()
        return self._multimodal_optimizer_pipeline

    def _multimodal_solver_available(self) -> bool:
        if hasattr(self, "_multimodal_optimizer_pipeline"):
            return True
        try:
            self._get_multimodal_optimizer_pipeline()
        except ValidationError:
            return False
        except Exception:
            return True
        return True

    def _multimodal_candidate_factory(
        self,
        runtime: CollectionRuntime,
        request: SearchRequest,
        query_embedding: np.ndarray,
        candidate_ids: Optional[List[int]],
    ) -> tuple[Optional[List[int]], Dict[str, Any]]:
        screening_profile: Dict[str, Any] = {}
        effective_candidate_ids = candidate_ids
        if (
            request.strategy == SearchStrategy.OPTIMIZED
            and candidate_ids != []
            and hasattr(runtime.engine, "screen_candidates")
        ):
            screening_budget = self._multimodal_candidate_budget(request, candidate_ids)
            should_screen = candidate_ids is None or len(candidate_ids) > screening_budget
            if should_screen:
                screened_ids = runtime.engine.screen_candidates(
                    query_embedding=query_embedding,
                    top_k=request.top_k,
                    candidate_ids=candidate_ids,
                    candidate_budget=screening_budget,
                )
                screening_profile = dict(getattr(runtime.engine, "last_screening_profile", {}) or {})
                screening_profile["candidate_budget"] = screening_budget
                if screened_ids is not None:
                    effective_candidate_ids = list(screened_ids)
        return effective_candidate_ids, screening_profile

    def _multimodal_solver_candidate_items(
        self,
        runtime: CollectionRuntime,
        candidate_ids: List[int],
        *,
        query_embedding: np.ndarray,
        retrieval_scores: Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidate_ids:
            return []
        vectors_by_id = runtime.engine.get_document_embeddings(candidate_ids)
        query_centroid = self._pool_vector(query_embedding)
        pooled_vectors: List[np.ndarray] = []
        ordered_ids: List[int] = []
        centroid_scores: List[float] = []
        base_scores: List[float] = []
        for candidate_id in candidate_ids:
            vector = vectors_by_id.get(int(candidate_id))
            if vector is None:
                continue
            pooled = self._pool_vector(vector)
            ordered_ids.append(int(candidate_id))
            pooled_vectors.append(pooled)
            centroid_scores.append(float(np.dot(query_centroid, pooled)))
            base_scores.append(float((retrieval_scores or {}).get(int(candidate_id), centroid_scores[-1])))
        if not ordered_ids:
            return []
        normalized_centroids = self._normalize_signal(centroid_scores)
        normalized_scores = self._normalize_signal(base_scores)
        max_rank = float(max(len(ordered_ids) - 1, 1))
        items: List[Dict[str, Any]] = []
        for rank, candidate_id in enumerate(ordered_ids):
            record = runtime.record_index.get(int(candidate_id))
            vector = vectors_by_id.get(int(candidate_id))
            if record is None or vector is None:
                continue
            payload = dict(record["payload"] or {})
            text = str(payload.get("text") or payload.get("content") or "")
            token_count = int(payload.get("token_count") or max(1, vector.shape[0] * 8))
            document_key = (
                payload.get("document_id")
                or payload.get("source_name")
                or payload.get("source")
                or record["external_id"]
            )
            reciprocal_rank = 1.0 - (float(rank) / max_rank) if max_rank > 0 else 1.0
            retrieval_score = float((retrieval_scores or {}).get(int(candidate_id), centroid_scores[rank]))
            payload.setdefault("text", text)
            payload["token_count"] = token_count
            payload.setdefault("fact_density", self._token_density_score(text, token_count))
            payload.setdefault(
                "centrality_score",
                0.5 * normalized_centroids.get(rank, 0.5) + 0.5 * normalized_scores.get(rank, 0.5),
            )
            payload.setdefault("recency_score", self._payload_recency_score(payload))
            payload.setdefault(
                "auxiliary_score",
                float(min(1.0, (0.70 * reciprocal_rank) + (0.30 * payload["recency_score"]))),
            )
            payload.setdefault("rhetorical_role", "unknown")
            payload.setdefault("cluster_id", self._stable_cluster_id(document_key))
            payload["base_relevance"] = retrieval_score
            payload["dense_score"] = retrieval_score
            payload["maxsim_score"] = retrieval_score
            payload["candidate_rank"] = rank + 1
            items.append(
                {
                    "id": int(candidate_id),
                    "vectors": vector,
                    "payload": payload,
                }
            )
        return items

    def _multimodal_prefilter_k(self, request: SearchRequest, available_count: int) -> int:
        target = request.multimodal_prefilter_k or max(request.top_k * 6, 24)
        return max(request.top_k, min(int(target), max(available_count, 1)))

    def _multimodal_frontier_k(self, request: SearchRequest, available_count: int) -> int:
        target = request.multimodal_maxsim_frontier_k or max(request.top_k * 8, 32)
        return max(request.top_k, min(int(target), max(available_count, 1)))

    def _multimodal_solver_constraints(
        self,
        request: SearchRequest,
        candidate_items: List[Dict[str, Any]],
        *,
        default_max_chunks: int,
    ) -> Dict[str, Any]:
        total_tokens = sum(int(item["payload"].get("token_count", 0) or 0) for item in candidate_items)
        return {
            "max_tokens": int(request.max_tokens or max(total_tokens, 1)),
            "max_chunks": int(request.max_chunks or default_max_chunks),
            "max_per_cluster": int(request.max_per_cluster or 2),
        }

    def _trim_multimodal_solver_items_to_payload(
        self,
        *,
        query_embedding: np.ndarray,
        candidate_items: List[Dict[str, Any]],
        min_keep: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        from voyager_index._internal.inference.stateless_optimizer import optimizer_max_payload_bytes

        payload_limit = int(optimizer_max_payload_bytes())
        query_bytes = int(np.asarray(query_embedding, dtype=np.float32).nbytes)
        kept: List[Dict[str, Any]] = []
        payload_bytes = query_bytes
        for item in candidate_items:
            item_bytes = int(np.asarray(item["vectors"], dtype=np.float32).nbytes)
            if kept and len(kept) >= min_keep and (payload_bytes + item_bytes) > payload_limit:
                break
            kept.append(item)
            payload_bytes += item_bytes
        if not kept and candidate_items:
            kept = candidate_items[:1]
            payload_bytes = query_bytes + int(np.asarray(kept[0]["vectors"], dtype=np.float32).nbytes)
        return kept, {
            "requested_candidate_count": len(candidate_items),
            "kept_candidate_count": len(kept),
            "payload_bytes": payload_bytes,
            "payload_limit_bytes": payload_limit,
            "trimmed": len(kept) < len(candidate_items),
        }

    @staticmethod
    def _solver_selected_internal_ids(
        selected_ids: List[Any],
        *,
        allowed_ids: List[int],
    ) -> List[int]:
        allowed = {int(doc_id) for doc_id in allowed_ids}
        resolved: List[int] = []
        for doc_id in selected_ids:
            try:
                normalized = int(doc_id)
            except (TypeError, ValueError):
                continue
            if normalized in allowed and normalized not in resolved:
                resolved.append(normalized)
        return resolved

    def _multimodal_solver_prefilter_search(
        self,
        runtime: CollectionRuntime,
        request: SearchRequest,
        query_embedding: np.ndarray,
        candidate_ids: Optional[List[int]],
        screening_profile: Dict[str, Any],
    ) -> tuple[List[Any], Optional[float], Optional[int]]:
        effective_candidate_ids = (
            list(candidate_ids) if candidate_ids is not None else list(runtime.record_index.keys())
        )
        if not effective_candidate_ids:
            runtime.engine.last_search_profile["screening"] = dict(screening_profile)
            runtime.engine.last_search_profile["optimization"] = {
                "mode": "solver_prefilter_maxsim",
                "candidate_pool_size": 0,
                "prefilter_candidate_count": 0,
                "solver_selected_count": 0,
            }
            return [], None, None
        prefilter_k = self._multimodal_prefilter_k(request, len(effective_candidate_ids))
        candidate_items = self._multimodal_solver_candidate_items(
            runtime,
            effective_candidate_ids,
            query_embedding=query_embedding,
        )
        candidate_items, payload_guard = self._trim_multimodal_solver_items_to_payload(
            query_embedding=query_embedding,
            candidate_items=candidate_items,
            min_keep=request.top_k,
        )
        pipeline = self._get_multimodal_optimizer_pipeline()
        constraints = self._multimodal_solver_constraints(
            request,
            candidate_items,
            default_max_chunks=min(prefilter_k, max(len(candidate_items), 1)),
        )
        optimize_start = time.perf_counter()
        optimized = pipeline.optimize_in_process(
            query_vectors=query_embedding,
            candidate_items=candidate_items,
            constraints=constraints,
            solver_config=self._request_solver_config(request, defaults={"iterations": 48}),
            metadata={
                "multimodal_mode": "solver_prefilter_maxsim",
                "optimizer_policy": request.optimizer_policy,
            },
        )
        solver_elapsed_ms = (time.perf_counter() - optimize_start) * 1000.0
        selected_ids = self._solver_selected_internal_ids(
            optimized.get("selected_ids", []),
            allowed_ids=effective_candidate_ids,
        )
        if not selected_ids:
            selected_ids = effective_candidate_ids[:prefilter_k]
        raw_results = runtime.engine.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            candidate_ids=selected_ids,
        )
        runtime.engine.last_search_profile["screening"] = dict(screening_profile)
        runtime.engine.last_search_profile["optimization"] = {
            "mode": "solver_prefilter_maxsim",
            "candidate_pool_size": len(effective_candidate_ids),
            "prefilter_candidate_count": prefilter_k,
            "solver_selected_count": len(selected_ids),
            "solver_time_ms": solver_elapsed_ms,
            "payload_guard": payload_guard,
            "solver_feature_summary": optimized.get("feature_summary", {}),
        }
        return (
            raw_results,
            float(optimized.get("solver_output", {}).get("objective_score", 0.0)),
            sum(
                int(runtime.record_index[int(doc_id)]["payload"].get("token_count", 0) or 0)
                for doc_id in selected_ids
                if int(doc_id) in runtime.record_index
            ),
        )

    def _multimodal_maxsim_then_solver_search(
        self,
        runtime: CollectionRuntime,
        request: SearchRequest,
        query_embedding: np.ndarray,
        candidate_ids: Optional[List[int]],
        screening_profile: Dict[str, Any],
    ) -> tuple[List[Any], Optional[float], Optional[int]]:
        effective_candidate_ids = (
            list(candidate_ids) if candidate_ids is not None else list(runtime.record_index.keys())
        )
        if not effective_candidate_ids:
            runtime.engine.last_search_profile["screening"] = dict(screening_profile)
            runtime.engine.last_search_profile["optimization"] = {
                "mode": "maxsim_then_solver",
                "candidate_pool_size": 0,
                "exact_frontier_size": 0,
                "solver_selected_count": 0,
            }
            return [], None, None
        frontier_k = self._multimodal_frontier_k(request, len(effective_candidate_ids))
        frontier_results = runtime.engine.search(
            query_embedding=query_embedding,
            top_k=frontier_k,
            candidate_ids=effective_candidate_ids,
        )
        frontier_profile = dict(getattr(runtime.engine, "last_search_profile", {}) or {})
        frontier_ids = [int(result.doc_id) for result in frontier_results if int(result.doc_id) in runtime.record_index]
        frontier_scores = {
            int(result.doc_id): float(result.score)
            for result in frontier_results
            if int(result.doc_id) in runtime.record_index
        }
        candidate_items = self._multimodal_solver_candidate_items(
            runtime,
            frontier_ids,
            query_embedding=query_embedding,
            retrieval_scores=frontier_scores,
        )
        candidate_items, payload_guard = self._trim_multimodal_solver_items_to_payload(
            query_embedding=query_embedding,
            candidate_items=candidate_items,
            min_keep=request.top_k,
        )
        pipeline = self._get_multimodal_optimizer_pipeline()
        constraints = self._multimodal_solver_constraints(
            request,
            candidate_items,
            default_max_chunks=min(request.top_k, max(len(candidate_items), 1)),
        )
        optimize_start = time.perf_counter()
        optimized = pipeline.optimize_in_process(
            query_vectors=query_embedding,
            candidate_items=candidate_items,
            constraints=constraints,
            solver_config=self._request_solver_config(request, defaults={"iterations": 48}),
            metadata={
                "multimodal_mode": "maxsim_then_solver",
                "optimizer_policy": request.optimizer_policy,
            },
        )
        solver_elapsed_ms = (time.perf_counter() - optimize_start) * 1000.0
        selected_ids = self._solver_selected_internal_ids(
            optimized.get("selected_ids", []),
            allowed_ids=frontier_ids,
        )
        if not selected_ids:
            selected_ids = frontier_ids[: request.top_k]
        selected_ranked = [result for result in frontier_results if int(result.doc_id) in set(selected_ids)]
        selected_ranked = selected_ranked[: request.top_k]
        frontier_profile["screening"] = dict(screening_profile)
        frontier_profile["optimization"] = {
            "mode": "maxsim_then_solver",
            "candidate_pool_size": len(effective_candidate_ids),
            "exact_frontier_size": len(frontier_ids),
            "solver_selected_count": len(selected_ids),
            "solver_time_ms": solver_elapsed_ms,
            "payload_guard": payload_guard,
            "solver_feature_summary": optimized.get("feature_summary", {}),
        }
        runtime.engine.last_search_profile = frontier_profile
        return (
            selected_ranked,
            float(optimized.get("solver_output", {}).get("objective_score", 0.0)),
            sum(
                int(runtime.record_index[int(doc_id)]["payload"].get("token_count", 0) or 0)
                for doc_id in selected_ids
                if int(doc_id) in runtime.record_index
            ),
        )

    def _refine_dense_pairs(
        self,
        runtime: CollectionRuntime,
        request: SearchRequest,
        query: np.ndarray,
        fused: List[tuple[int, float]],
    ) -> tuple[List[tuple[int, float]], Optional[dict[str, Any]]]:
        if not self._dense_uses_solver(request):
            return fused[: request.top_k], None
        if not getattr(runtime.engine, "solver_available", False):
            raise ValidationError("Optimized dense search requires the latence_solver native package to be installed")
        if query is None:
            raise ValidationError("Optimized dense search requires 'vector' so the solver can score dense candidates")

        try:
            refined = runtime.engine.refine(
                query_vector=query,
                query_text=request.query_text or "",
                query_payload=request.query_payload,
                candidate_ids=[doc_id for doc_id, _ in fused],
                solver_config=self._request_solver_config(request),
                constraints=self._dense_solver_constraints(request),
                optimizer_policy=request.optimizer_policy,
                refine_options=self._dense_refine_options(request),
            )
        except ImportError as exc:
            raise ValidationError(
                "Cross-encoder dense refinement requires sentence-transformers to be installed."
            ) from exc
        selected_ids = [
            int(doc_id) for doc_id in refined.get("selected_internal_ids", []) if int(doc_id) in runtime.record_index
        ]
        if not selected_ids:
            return fused, refined

        score_map = {int(doc_id): float(score) for doc_id, score in fused}
        refined_pairs = [(doc_id, score_map.get(doc_id, 0.0)) for doc_id in selected_ids if doc_id in score_map]
        if not refined_pairs:
            return fused, refined
        return refined_pairs[: request.top_k], refined

    def _fuse_dense_sparse(
        self,
        dense_results: List[tuple[int, float]],
        sparse_results: List[tuple[int, float]],
        top_k: int,
        graph_results: Optional[List[tuple[int, float]]] = None,
    ) -> List[tuple[int, float]]:
        if dense_results and not sparse_results and not graph_results:
            return [(int(doc_id), float(score)) for doc_id, score in dense_results[:top_k]]
        if sparse_results and not dense_results and not graph_results:
            return [(int(doc_id), float(score)) for doc_id, score in sparse_results[:top_k]]

        fused: Dict[int, float] = {}
        rrf_k = 60.0
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        base_ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        if not graph_results:
            return base_ranked[:top_k]
        base_ids = {int(doc_id) for doc_id, _score in base_ranked}
        rescued_pairs = [
            (int(doc_id), float(score)) for doc_id, score in list(graph_results or []) if int(doc_id) not in base_ids
        ]
        if not base_ranked:
            return rescued_pairs[:top_k]
        return base_ranked[:top_k] + rescued_pairs

    def _augment_pairs_with_graph_sidecar(
        self,
        runtime: CollectionRuntime,
        request: SearchRequest,
        pairs: List[tuple[int, float]],
        *,
        query_text: str = "",
    ) -> tuple[List[tuple[int, float]], Dict[str, Any]]:
        if request.graph_mode == GraphMode.OFF:
            return pairs[: request.top_k], {"graph_applied": False, "reason": "graph_mode_off"}
        sidecar = self._graph_sidecar_for_runtime(runtime)
        if sidecar is None or not sidecar.is_available():
            return pairs[: request.top_k], {"graph_applied": False, "reason": "graph_unavailable"}
        policy = self._graph_policy.decide(
            graph_mode=request.graph_mode.value,
            query_text=query_text,
            query_payload=request.query_payload,
            dense_results=pairs,
            sparse_results=[],
            graph_available=True,
        )
        if not policy.applied:
            return pairs[: request.top_k], {"graph_applied": False, "policy": policy.to_dict(), "reason": policy.reason}
        graph_aug = sidecar.augment_candidates(
            [doc_id for doc_id, _ in pairs],
            query_text=query_text,
            query_payload=request.query_payload,
            local_budget=int(request.graph_local_budget) if request.graph_local_budget is not None else 4,
            community_budget=int(request.graph_community_budget) if request.graph_community_budget is not None else 4,
            evidence_budget=int(request.graph_evidence_budget) if request.graph_evidence_budget is not None else 8,
            max_hops=int(request.graph_max_hops) if request.graph_max_hops is not None else 2,
            explain=bool(request.graph_explain),
        )
        if not graph_aug.applied:
            return pairs[: request.top_k], {
                "graph_applied": False,
                "policy": policy.to_dict(),
                "reason": graph_aug.reason,
            }
        base_pairs = [(int(doc_id), float(score)) for doc_id, score in pairs[: request.top_k]]
        base_ids = {int(doc_id) for doc_id, _score in base_pairs}
        rescued_pairs = [
            (int(doc_id), float(score))
            for doc_id, score in list(graph_aug.graph_results or [])
            if int(doc_id) not in base_ids
        ]
        ranked = base_pairs + rescued_pairs
        metadata: Dict[str, Any] = {
            "graph_applied": True,
            "policy": policy.to_dict(),
            "summary": {
                **dict(graph_aug.summary),
                "merge_mode": "additive",
                "base_order_preserved": True,
                "rescued_ids": [int(doc_id) for doc_id, _score in rescued_pairs],
            },
        }
        if request.graph_explain:
            metadata["provenance"] = dict(graph_aug.provenance_by_target)
        return ranked, metadata

    def _build_scored_point(
        self,
        runtime: CollectionRuntime,
        internal_id: int,
        score: float,
        rank: int,
        with_payload: bool,
        vector_data: Any = None,
    ) -> ScoredPoint:
        record = runtime.record_index[int(internal_id)]
        vector = None
        vectors = None
        if vector_data is not None:
            if isinstance(vector_data, np.ndarray):
                if vector_data.ndim == 1:
                    vector = vector_data.tolist()
                else:
                    vectors = vector_data.tolist()
            elif isinstance(vector_data, list):
                if vector_data and isinstance(vector_data[0], list):
                    vectors = vector_data
                else:
                    vector = vector_data

        return ScoredPoint(
            id=record["external_id"],
            score=float(score),
            rank=rank,
            payload=record["payload"] if with_payload else None,
            vector=vector,
            vectors=vectors,
        )

    def search(self, name: str, request: SearchRequest) -> SearchResponse:
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            start = time.perf_counter()
            objective_score: Optional[float] = None
            total_tokens: Optional[int] = None
            response_metadata: Dict[str, Any] = {}

            if runtime.kind == CollectionKind.DENSE:
                query = self._dense_query_vector(runtime, request)
                use_dense_solver = self._dense_uses_solver(request)
                if query is None and not request.query_text:
                    raise ValidationError("Dense search requires a vector or query_text")
                if use_dense_solver and query is None:
                    raise ValidationError(
                        "Tabu-refined dense search requires 'vector' because solver refinement is dense-query aware"
                    )
                if request.query_text and runtime.meta.get("sparse_dirty"):
                    with self._cross_process_collection_lock(runtime.name):
                        runtime = self.get_collection(runtime.name)
                        if runtime.meta.get("sparse_dirty"):
                            self._sync_dense_sparse_state(runtime, rebuild_sparse=True)
                            self._write_meta(runtime)
                if not runtime.meta.get("records"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._record_search_metrics(elapsed_ms)
                    return SearchResponse(results=[], total=0, time_ms=elapsed_ms)
                retrieval_k = self._dense_refine_candidate_k(request)
                search_kwargs: Dict[str, Any] = {
                    "query_text": request.query_text or "",
                    "query_vector": query,
                    "k": retrieval_k,
                    "filters": request.filter,
                }
                if request.graph_mode != GraphMode.OFF or self._graph_request_options(request):
                    search_kwargs.update(
                        {
                            "query_payload": request.query_payload,
                            "graph_mode": request.graph_mode.value,
                            "graph_options": self._graph_request_options(request),
                        }
                    )
                result = runtime.engine.search(**search_kwargs)
                fused = self._fuse_dense_sparse(
                    result.get("dense", []),
                    result.get("sparse", []),
                    retrieval_k,
                    graph_results=result.get("graph", []),
                )
                dense_pairs, refined = self._refine_dense_pairs(runtime, request, query, fused)
                graph_summary = dict(result.get("graph_summary") or {})
                if graph_summary.get("merge_mode") == "additive":
                    graph_summary.setdefault("base_order_preserved", True)
                response_metadata = {
                    "graph": {
                        "graph_applied": bool(graph_summary.get("graph_applied") or result.get("graph")),
                        "reason": graph_summary.get("reason") or dict(result.get("graph_policy") or {}).get("reason"),
                        "policy": dict(result.get("graph_policy") or {}),
                        "summary": graph_summary,
                    }
                }
                if request.graph_explain and result.get("graph_provenance"):
                    response_metadata["graph"]["provenance"] = dict(result.get("graph_provenance") or {})
                vectors_by_id = {}
                if request.with_vector:
                    refined_vectors = refined.get("selected_vectors", {}) if refined is not None else {}
                    vectors_by_id = {int(doc_id): value for doc_id, value in refined_vectors.items()}
                    missing_ids = [doc_id for doc_id, _ in dense_pairs if int(doc_id) not in vectors_by_id]
                    if missing_ids:
                        vectors_by_id.update(self._get_dense_vectors(runtime, missing_ids))
                results = [
                    self._build_scored_point(
                        runtime,
                        internal_id=doc_id,
                        score=score,
                        rank=rank,
                        with_payload=request.with_payload,
                        vector_data=vectors_by_id.get(doc_id),
                    )
                    for rank, (doc_id, score) in enumerate(dense_pairs, start=1)
                ]
                objective_score = (
                    refined.get("solver_output", {}).get("objective_score") if refined is not None else None
                )
                total_tokens = refined.get("solver_output", {}).get("total_tokens") if refined is not None else None
                if refined is not None and refined.get("feature_summary"):
                    response_metadata["solver"] = dict(refined.get("feature_summary") or {})
            elif runtime.kind == CollectionKind.LATE_INTERACTION:
                if request.query_text:
                    raise ValidationError("Late-interaction collections do not support query_text search")
                if request.vectors is not None:
                    query_vectors = self._coerce_multi_vector_input(request.vectors, field_name="vectors")
                    query_tensor = torch.from_numpy(query_vectors[None, ...]).to(self.device, dtype=torch.float32)
                elif request.vector is not None:
                    query_vector = self._coerce_single_vector_input(request.vector, field_name="vector")
                    query_tensor = torch.from_numpy(query_vector.reshape(1, 1, -1)).to(self.device, dtype=torch.float32)
                else:
                    raise ValidationError("Late-interaction search requires 'vectors' or 'vector'")

                if query_tensor.shape[-1] != int(runtime.meta["dimension"]):
                    raise ValidationError("Query dimension does not match collection dimension")
                if not runtime.meta.get("records"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._record_search_metrics(elapsed_ms)
                    return SearchResponse(results=[], total=0, time_ms=elapsed_ms)

                candidate_ids = self._candidate_ids_for_filter(runtime, request.filter)
                if candidate_ids == []:
                    scores = torch.empty((1, 0), dtype=torch.float32, device=self.device)
                    indices = torch.empty((1, 0), dtype=torch.long, device=self.device)
                else:
                    scores, indices = runtime.engine.search(
                        query_tensor,
                        top_k=request.top_k,
                        doc_ids=candidate_ids,
                    )
                filtered_pairs = [
                    (int(doc_id), float(score))
                    for doc_id, score in zip(indices[0].tolist(), scores[0].tolist())
                    if int(doc_id) in runtime.record_index
                    and self._matches_filter(runtime.record_index[int(doc_id)]["payload"], request.filter)
                ]
                selected_pairs, graph_metadata = self._augment_pairs_with_graph_sidecar(
                    runtime,
                    request,
                    filtered_pairs,
                    query_text=request.query_text or "",
                )
                response_metadata = {"graph": graph_metadata}
                internal_ids = [doc_id for doc_id, _ in selected_pairs]
                vectors_by_id = {}
                if request.with_vector and internal_ids:
                    docs = (
                        runtime.engine.get_document_embeddings(internal_ids, device="cpu")
                        if hasattr(runtime.engine, "get_document_embeddings")
                        else runtime.engine.storage.load_documents(doc_ids=internal_ids, device="cpu")
                    )
                    vectors_by_id = self._trim_late_interaction_vectors(runtime, internal_ids, docs)
                results = [
                    self._build_scored_point(
                        runtime,
                        internal_id=internal_id,
                        score=score,
                        rank=rank,
                        with_payload=request.with_payload,
                        vector_data=vectors_by_id.get(internal_id),
                    )
                    for rank, (internal_id, score) in enumerate(selected_pairs, start=1)
                ]
            elif runtime.kind == CollectionKind.SHARD:
                if request.query_text:
                    raise ValidationError("Shard collections do not support query_text search")
                if request.strategy == SearchStrategy.OPTIMIZED:
                    logger.debug("SearchStrategy.OPTIMIZED is not applicable to shard collections; using shard routing")
                shard_overrides = self._shard_search_overrides(request)
                if request.vectors is not None:
                    query_np = self._coerce_multi_vector_input(request.vectors, field_name="vectors")
                elif request.vector is not None:
                    query_vector = self._coerce_single_vector_input(request.vector, field_name="vector")
                    query_np = np.asarray([query_vector], dtype=np.float32)
                else:
                    raise ValidationError("Shard search requires 'vectors' or 'vector'")
                if query_np.shape[-1] != int(runtime.meta["dimension"]):
                    raise ValidationError("Query dimension does not match collection dimension")
                if not runtime.meta.get("records"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._record_search_metrics(elapsed_ms)
                    return SearchResponse(results=[], total=0, time_ms=elapsed_ms)

                shard_results = runtime.engine.search_multivector(
                    query_np,
                    k=request.top_k,
                    filters=request.filter,
                    n_probes=request.n_probes,
                    **shard_overrides,
                )
                selected_pairs, graph_metadata = self._augment_pairs_with_graph_sidecar(
                    runtime,
                    request,
                    [(did, sc) for did, sc in shard_results if did in runtime.record_index],
                    query_text=request.query_text or "",
                )
                response_metadata = {"graph": graph_metadata}
                results = [
                    self._build_scored_point(
                        runtime,
                        internal_id=did,
                        score=sc,
                        rank=rank,
                        with_payload=request.with_payload,
                    )
                    for rank, (did, sc) in enumerate(selected_pairs, start=1)
                ]
            else:
                if request.query_text:
                    raise ValidationError("Multimodal collections do not support query_text search")
                if request.vectors is not None:
                    query_embedding = self._coerce_multi_vector_input(request.vectors, field_name="vectors")
                elif request.vector is not None:
                    query_vector = self._coerce_single_vector_input(request.vector, field_name="vector")
                    query_embedding = np.asarray([query_vector], dtype=np.float32)
                else:
                    raise ValidationError("Multimodal search requires 'vectors' or 'vector'")

                if query_embedding.shape[-1] != int(runtime.meta["dimension"]):
                    raise ValidationError("Query dimension does not match collection dimension")
                if not runtime.meta.get("records"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._record_search_metrics(elapsed_ms)
                    return SearchResponse(results=[], total=0, time_ms=elapsed_ms)

                candidate_ids = self._candidate_ids_for_filter(runtime, request.filter)
                screening_profile: Dict[str, Any] = {}
                skip_screening = request.screening_mode == ScreeningMode.NONE
                if request.strategy == SearchStrategy.OPTIMIZED and not skip_screening:
                    candidate_ids, screening_profile = self._multimodal_candidate_factory(
                        runtime,
                        request,
                        query_embedding,
                        candidate_ids,
                    )
                if skip_screening:
                    screening_profile["skipped"] = True
                    screening_profile["screening_mode"] = "none"
                multimodal_mode = self._multimodal_optimize_mode(request)
                explicit_multimodal_mode = request.multimodal_optimize_mode not in (None, MultimodalOptimizeMode.AUTO)
                if (
                    request.strategy == SearchStrategy.OPTIMIZED
                    and multimodal_mode != MultimodalOptimizeMode.MAXSIM_ONLY
                    and not self._multimodal_solver_available()
                ):
                    if explicit_multimodal_mode:
                        raise ValidationError(
                            "Requested multimodal solver ordering requires the latence_solver native package to be installed"
                        )
                    multimodal_mode = MultimodalOptimizeMode.MAXSIM_ONLY
                if candidate_ids == []:
                    raw_results = []
                elif (
                    request.strategy != SearchStrategy.OPTIMIZED
                    or multimodal_mode == MultimodalOptimizeMode.MAXSIM_ONLY
                ):
                    raw_results = runtime.engine.search(
                        query_embedding=query_embedding,
                        top_k=request.top_k,
                        candidate_ids=candidate_ids,
                    )
                    if screening_profile:
                        runtime.engine.last_search_profile["screening"] = dict(screening_profile)
                        runtime.engine.last_search_profile["optimization"] = {
                            "mode": "maxsim_only",
                            "candidate_pool_size": (
                                len(candidate_ids) if candidate_ids is not None else len(runtime.record_index)
                            ),
                        }
                elif multimodal_mode == MultimodalOptimizeMode.SOLVER_PREFILTER_MAXSIM:
                    raw_results, objective_score, total_tokens = self._multimodal_solver_prefilter_search(
                        runtime,
                        request,
                        query_embedding,
                        candidate_ids,
                        screening_profile,
                    )
                else:
                    raw_results, objective_score, total_tokens = self._multimodal_maxsim_then_solver_search(
                        runtime,
                        request,
                        query_embedding,
                        candidate_ids,
                        screening_profile,
                    )
                multimodal_pairs, graph_metadata = self._augment_pairs_with_graph_sidecar(
                    runtime,
                    request,
                    [
                        (int(result.doc_id), float(result.score))
                        for result in raw_results
                        if int(result.doc_id) in runtime.record_index
                        and self._matches_filter(runtime.record_index[int(result.doc_id)]["payload"], request.filter)
                    ],
                    query_text=request.query_text or "",
                )
                response_metadata = {"graph": graph_metadata}
                selected_results = [
                    type("GraphResult", (), {"doc_id": int(doc_id), "score": float(score)})()
                    for doc_id, score in multimodal_pairs
                ]
                vectors_by_id = {}
                if request.with_vector and selected_results:
                    vectors_by_id = runtime.engine.get_document_embeddings(
                        [int(result.doc_id) for result in selected_results]
                    )
                results = [
                    self._build_scored_point(
                        runtime,
                        internal_id=int(result.doc_id),
                        score=result.score,
                        rank=rank,
                        with_payload=request.with_payload,
                        vector_data=vectors_by_id.get(int(result.doc_id)),
                    )
                    for rank, result in enumerate(selected_results, start=1)
                ]

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_search_metrics(elapsed_ms)

            return SearchResponse(
                results=results,
                total=len(results),
                time_ms=elapsed_ms,
                objective_score=objective_score,
                total_tokens=total_tokens,
                metadata=response_metadata,
            )

    def collection_info(self, name: str) -> CollectionInfo:
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            storage_mb: Optional[float] = None
            if runtime.kind == CollectionKind.LATE_INTERACTION:
                storage_mb = runtime.engine.get_statistics().storage_size_mb
            elif runtime.kind == CollectionKind.MULTIMODAL:
                storage_mb = runtime.engine.get_statistics().get("storage_mb")

            shard_n_shards = None
            shard_k_candidates = None
            shard_total_tokens = None
            shard_compression = None
            shard_quantization_mode = None
            shard_transfer_mode = None
            shard_router_device = None
            shard_use_colbandit = None
            shard_max_docs_exact = None
            shard_n_full_scores = None
            shard_pinned_pool_buffers = None
            shard_pinned_buffer_max_tokens = None
            shard_lemur_search_k_cap = None
            shard_gpu_corpus_rerank_topn = None
            shard_n_centroid_approx = None
            shard_variable_length_strategy = None
            shard_runtime_capabilities = None
            if runtime.kind == CollectionKind.SHARD:
                shard_stats = runtime.engine.get_statistics()
                shard_n_shards = runtime.meta.get("n_shards")
                shard_k_candidates = runtime.meta.get("k_candidates")
                shard_compression = runtime.meta.get("compression")
                shard_quantization_mode = runtime.meta.get("quantization_mode")
                shard_transfer_mode = runtime.meta.get("transfer_mode")
                shard_router_device = runtime.meta.get("router_device")
                shard_use_colbandit = runtime.meta.get("use_colbandit")
                shard_max_docs_exact = runtime.meta.get("max_docs_exact")
                shard_n_full_scores = runtime.meta.get("n_full_scores")
                shard_pinned_pool_buffers = runtime.meta.get("pinned_pool_buffers")
                shard_pinned_buffer_max_tokens = runtime.meta.get("pinned_buffer_max_tokens")
                shard_lemur_search_k_cap = runtime.meta.get("lemur_search_k_cap")
                shard_gpu_corpus_rerank_topn = runtime.meta.get("gpu_corpus_rerank_topn")
                shard_n_centroid_approx = runtime.meta.get("n_centroid_approx")
                shard_variable_length_strategy = runtime.meta.get("variable_length_strategy")
                shard_runtime_capabilities = shard_stats.get("runtime_capabilities")
                store = getattr(runtime.engine, "_store", None)
                if store and store.manifest:
                    shard_total_tokens = store.manifest.total_tokens
            graph_sidecar = self._graph_sidecar_for_runtime(runtime)
            graph_stats = graph_sidecar.get_statistics() if graph_sidecar is not None else {}

            return CollectionInfo(
                name=runtime.name,
                kind=runtime.kind,
                dimension=int(runtime.meta["dimension"]),
                distance=DistanceMetric(runtime.meta["distance"]),
                num_points=self._engine_point_count(runtime),
                indexed=self._is_runtime_indexed(runtime),
                storage_mb=storage_mb,
                m=runtime.meta.get("m") if runtime.kind == CollectionKind.DENSE else None,
                ef_construction=runtime.meta.get("ef_construction") if runtime.kind == CollectionKind.DENSE else None,
                storage_mode=runtime.meta.get("storage_mode")
                if runtime.kind == CollectionKind.LATE_INTERACTION
                else None,
                storage_path=str(runtime.path),
                n_shards=shard_n_shards,
                k_candidates=shard_k_candidates,
                total_tokens=shard_total_tokens,
                compression=shard_compression,
                quantization_mode=shard_quantization_mode,
                transfer_mode=shard_transfer_mode,
                router_device=shard_router_device,
                use_colbandit=shard_use_colbandit,
                max_docs_exact=shard_max_docs_exact,
                n_full_scores=shard_n_full_scores,
                pinned_pool_buffers=shard_pinned_pool_buffers,
                pinned_buffer_max_tokens=shard_pinned_buffer_max_tokens,
                lemur_search_k_cap=shard_lemur_search_k_cap,
                gpu_corpus_rerank_topn=shard_gpu_corpus_rerank_topn,
                n_centroid_approx=shard_n_centroid_approx,
                variable_length_strategy=shard_variable_length_strategy,
                runtime_capabilities=shard_runtime_capabilities,
                hybrid_search=(runtime.kind == CollectionKind.DENSE),
                graph_health=graph_sidecar.health if graph_sidecar is not None else None,
                graph_dataset_id=graph_sidecar.dataset_id if graph_sidecar is not None else None,
                graph_contract_version=graph_stats.get("contract_version"),
                graph_sync_status=graph_stats.get("sync_status"),
                graph_sync_reason=graph_stats.get("sync_reason"),
                graph_last_sync_at=graph_stats.get("last_sync_at"),
                graph_last_successful_sync_at=graph_stats.get("last_successful_sync_at"),
                graph_sync_job_id=dict(graph_stats.get("dataset_job") or {}).get("job_id"),
            )

    def reference_preprocess_documents(self, request: RenderDocumentsRequest) -> Dict[str, Any]:
        """
        Render local source documents into PageBundle-like page assets.

        This is the supported preprocessing step for doc-to-image ingestion. The
        index still expects precomputed embeddings for collection writes, but the
        reference API can now own the source-doc to page-image conversion stage.
        """

        from voyager_index.preprocessing import enumerate_renderable_documents, render_documents

        discovered: list[Path] = []
        skipped: list[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_discovered(path: Path) -> None:
            resolved = path.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                discovered.append(resolved)

        if request.source_dir is not None:
            source_dir = self._resolve_user_path(request.source_dir)
            if not source_dir.exists():
                raise ValidationError(f"Source directory does not exist: {source_dir}")
            if not source_dir.is_dir():
                raise ValidationError(f"Source directory is not a directory: {source_dir}")
            inventory = enumerate_renderable_documents(source_dir, recursive=request.recursive)
            skipped.extend(inventory["skipped"])
            for document in inventory["documents"]:
                add_discovered(document)

        for raw_path in request.source_paths:
            source_path = self._resolve_user_path(raw_path)
            if not source_path.exists():
                raise ValidationError(f"Source path does not exist: {source_path}")
            if source_path.is_dir():
                inventory = enumerate_renderable_documents(source_path, recursive=request.recursive)
                skipped.extend(inventory["skipped"])
                for document in inventory["documents"]:
                    add_discovered(document)
            else:
                add_discovered(source_path)

        output_dir = (
            self._resolve_user_path(request.output_dir, relative_to_root=True)
            if request.output_dir
            else (self.root_path / "_preprocessed" / "documents")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        result = render_documents(discovered, output_dir)
        result["skipped"] = skipped + list(result["skipped"])
        result["summary"] = dict(result["summary"])
        result["summary"]["documents_discovered"] = len(discovered)
        return result

    def reference_optimize(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stateless fulfilment optimizer (dense, multivector, optional BM25 signals via metadata).

        Request/response shape matches :class:`~voyager_index._internal.inference.stateless_optimizer.OptimizerRequest`
        and ``GpuFulfilmentPipeline.optimize`` output.
        """
        from voyager_index._internal.inference.stateless_optimizer import (
            GpuFulfilmentPipeline,
            OptimizerPayloadTooLargeError,
            OptimizerRequestError,
        )

        if not hasattr(self, "_reference_optimizer_pipeline"):
            self._reference_optimizer_pipeline = GpuFulfilmentPipeline()
        try:
            return self._reference_optimizer_pipeline.optimize(body)
        except OptimizerPayloadTooLargeError as exc:
            raise RequestTooLargeError(str(exc)) from exc
        except ImportError as exc:
            raise ValidationError(
                "Stateless optimize requires the `latence_solver` native package to be installed."
            ) from exc
        except (OptimizerRequestError, KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid optimizer request: {exc}") from exc

    def reference_optimizer_health(self) -> Dict[str, Any]:
        """Backend status for the stateless optimizer (solver + execution mode)."""
        from voyager_index._internal.inference.stateless_optimizer import GpuFulfilmentPipeline

        if not hasattr(self, "_reference_optimizer_pipeline"):
            self._reference_optimizer_pipeline = GpuFulfilmentPipeline()
        return self._reference_optimizer_pipeline.backend_health()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_if_over_limit(self, runtime: CollectionRuntime) -> None:
        """Delete oldest documents when the collection exceeds max_documents."""
        max_docs = runtime.meta.get("max_documents")
        if not max_docs:
            return
        records = runtime.meta.get("records") or {}
        overflow = len(records) - max_docs
        if overflow <= 0:
            return
        sorted_keys = sorted(
            records.keys(),
            key=lambda k: int(records[k].get("internal_id", 0)),
        )
        evict_keys = sorted_keys[:overflow]
        evict_internal = [int(records[k]["internal_id"]) for k in evict_keys]
        self._delete_internal_ids(runtime, evict_internal)
        for k in evict_keys:
            records.pop(k, None)
        self._refresh_runtime_indexes(runtime)
        logger.info(
            "Evicted %d oldest documents from '%s' (max_documents=%d)",
            len(evict_keys),
            runtime.name,
            max_docs,
        )

    # ------------------------------------------------------------------
    # Payload CRUD
    # ------------------------------------------------------------------

    def set_payload(
        self,
        name: str,
        point_ids: List[Any],
        payload: Dict[str, Any],
    ) -> int:
        """Merge payload fields into specified points."""
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                records = runtime.meta.get("records") or {}
                updated = 0
                for pid in point_ids:
                    key = str(pid)
                    if key not in records:
                        continue
                    existing = records[key].get("payload") or {}
                    existing.update(payload)
                    records[key]["payload"] = existing
                    internal_id = int(records[key]["internal_id"])
                    if runtime.kind == CollectionKind.SHARD and hasattr(runtime.engine, "upsert_payload"):
                        runtime.engine.upsert_payload(internal_id, payload)
                    updated += 1
                if updated:
                    self._refresh_runtime_indexes(runtime)
                    self._write_meta(runtime)
                return updated

    def delete_payload_keys(
        self,
        name: str,
        point_ids: List[Any],
        keys: List[str],
    ) -> int:
        """Remove specific payload keys from specified points."""
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                records = runtime.meta.get("records") or {}
                updated = 0
                for pid in point_ids:
                    key = str(pid)
                    if key not in records:
                        continue
                    existing = records[key].get("payload") or {}
                    changed = False
                    for k in keys:
                        if k in existing:
                            del existing[k]
                            changed = True
                    if changed:
                        records[key]["payload"] = existing
                        if runtime.kind == CollectionKind.SHARD and hasattr(runtime.engine, "upsert_payload"):
                            internal_id = int(records[key]["internal_id"])
                            runtime.engine.upsert_payload(internal_id, existing)
                        updated += 1
                if updated:
                    self._refresh_runtime_indexes(runtime)
                    self._write_meta(runtime)
                return updated

    def clear_payload(self, name: str, point_ids: List[Any]) -> int:
        """Clear all payload fields from specified points."""
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            with self._cross_process_collection_lock(runtime.name):
                records = runtime.meta.get("records") or {}
                updated = 0
                for pid in point_ids:
                    key = str(pid)
                    if key not in records:
                        continue
                    records[key]["payload"] = {}
                    if runtime.kind == CollectionKind.SHARD and hasattr(runtime.engine, "upsert_payload"):
                        internal_id = int(records[key]["internal_id"])
                        runtime.engine.upsert_payload(internal_id, {})
                    updated += 1
                if updated:
                    self._refresh_runtime_indexes(runtime)
                    self._write_meta(runtime)
                return updated

    def get_point_payload(self, name: str, point_id: str) -> Dict[str, Any]:
        """Return the payload dict for a single point."""
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            records = runtime.meta.get("records") or {}
            key = str(point_id)
            if key not in records:
                raise NotFoundError(f"Point '{point_id}' not found in collection '{name}'")
            return dict(records[key].get("payload") or {})

    # ------------------------------------------------------------------
    # Encode / Rerank
    # ------------------------------------------------------------------

    def encode(self, request):
        """Encode text/images into embeddings via the configured model provider."""
        from .models import EncodeResponse

        provider = self._get_encode_provider()
        if provider is None:
            raise ServiceError("No encoding model configured. Set VOYAGER_ENCODE_MODEL or start with a ColPali engine.")

        embeddings = []
        items = request.texts or request.images or []
        for item in items:
            emb = provider.encode(item)
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            if isinstance(emb, list) and emb and not isinstance(emb[0], list):
                emb = [emb]
            embeddings.append(emb)
        return EncodeResponse(embeddings=embeddings, model=getattr(provider, "model_name", None))

    def rerank(self, request):
        """Rerank documents against a query."""
        from .models import RerankResponse, RerankResult

        provider = self._get_encode_provider()
        if provider is None:
            raise ServiceError("No encoding model configured for reranking.")

        query_emb = provider.encode(request.query)
        if hasattr(query_emb, "numpy"):
            import numpy as _np

            query_np = query_emb.numpy().astype(_np.float32)
        elif hasattr(query_emb, "tolist"):
            import numpy as _np

            query_np = _np.array(query_emb, dtype=_np.float32)
        else:
            import numpy as _np

            query_np = _np.array(query_emb, dtype=_np.float32)

        scored = []
        for idx, doc in enumerate(request.documents):
            doc_emb = provider.encode(doc)
            if hasattr(doc_emb, "numpy"):
                doc_np = doc_emb.numpy().astype(_np.float32)
            else:
                doc_np = _np.array(doc_emb, dtype=_np.float32)
            if query_np.ndim == 1:
                score = float(_np.dot(query_np, doc_np.flatten()[: len(query_np)]))
            else:
                import torch as _torch

                q = _torch.from_numpy(query_np).float()
                d = _torch.from_numpy(doc_np).float()
                if q.dim() == 2 and d.dim() == 2:
                    sim = q @ d.T
                    score = float(sim.max(dim=1).values.sum())
                else:
                    score = float(_np.dot(query_np.flatten(), doc_np.flatten()[: query_np.size]))
            scored.append((idx, score, doc))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = [RerankResult(index=idx, score=sc, document=doc) for idx, sc, doc in scored[: request.top_k]]
        return RerankResponse(results=results, model=getattr(provider, "model_name", None))

    _cached_encode_provider = None

    def _get_encode_provider(self):
        """Locate an available encoding provider (ColPali engine or VllmPoolingProvider).

        Caches the fallback provider to avoid re-loading the model on every call.
        """
        with self._collections_lock:
            for runtime in self.collections.values():
                if runtime.kind == CollectionKind.MULTIMODAL:
                    engine = runtime.engine
                    if hasattr(engine, "model"):
                        return engine.model
        if self._cached_encode_provider is not None:
            return self._cached_encode_provider
        try:
            model_name = os.environ.get("VOYAGER_ENCODE_MODEL")
            if model_name:
                provider = ColPaliEngine(ColPaliConfig(model_name=model_name))
                self._cached_encode_provider = provider
                return provider
        except (ImportError, Exception):
            pass
        return None

    def close(self) -> None:
        """Release collection engines so the same data root can be reopened safely."""
        with self._collections_lock:
            runtimes = list(self.collections.values())
            self.collections = {}
            self._collection_locks = {}
            encode_provider = self._cached_encode_provider
            self._cached_encode_provider = None
        for runtime in runtimes:
            graph_sidecar = self._graph_sidecar_for_runtime(runtime)
            if graph_sidecar is not None and hasattr(graph_sidecar, "close"):
                try:
                    graph_sidecar.close()
                except Exception as exc:
                    logger.warning("Failed to close graph sidecar for collection '%s': %s", runtime.name, exc)
            close_method = getattr(runtime.engine, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:
                    logger.warning("Failed to close engine for collection '%s': %s", runtime.name, exc)
        if encode_provider is not None:
            close_method = getattr(encode_provider, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:
                    logger.warning("Failed to close encode provider: %s", exc)

    def _collection_readiness_issues(self, runtime: CollectionRuntime) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []

        if runtime.kind == CollectionKind.DENSE:
            if getattr(runtime.engine, "sparse_error", None):
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": "sparse_load_failed",
                        "detail": str(runtime.engine.sparse_error),
                    }
                )
            if runtime.meta.get("sparse_dirty"):
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": "sparse_generation_pending",
                        "detail": "Sparse state has pending writes and will rebuild on the next sparse query or reload.",
                    }
                )
        elif runtime.kind == CollectionKind.LATE_INTERACTION:
            data_file = runtime.engine.storage.data_file
            metadata_file = runtime.engine.storage.metadata_file
            if runtime.meta.get("records") and (not data_file.exists() or not metadata_file.exists()):
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": "storage_missing",
                        "detail": "Late-interaction storage files are missing for a non-empty collection.",
                    }
                )
        elif runtime.kind == CollectionKind.SHARD:
            if not getattr(runtime.engine, "_is_built", False) and runtime.meta.get("records"):
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": "shard_not_built",
                        "detail": "Shard engine has records but index is not built.",
                    }
                )
        else:
            manifest_path = runtime.engine.manifest_path
            if runtime.meta.get("records") and not manifest_path.exists():
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": "manifest_missing",
                        "detail": "Multimodal manifest is missing for a non-empty collection.",
                    }
                )
            screening_state = dict(runtime.engine.get_statistics().get("screening_state") or {})
            screening_health = str(screening_state.get("health", ""))
            if screening_health in {"warming", "degraded"}:
                issues.append(
                    {
                        "scope": "collection",
                        "name": runtime.name,
                        "kind": runtime.kind.value,
                        "reason": f"screening_{screening_health}",
                        "detail": "Optimized multimodal search is currently routing back to exact search until sidecar trust is restored.",
                    }
                )

        graph_sidecar = self._graph_sidecar_for_runtime(runtime)
        if graph_sidecar is not None and str(getattr(graph_sidecar, "health", "")) == "degraded":
            issues.append(
                {
                    "scope": "collection",
                    "name": runtime.name,
                    "kind": runtime.kind.value,
                    "reason": "latence_graph_degraded",
                    "detail": str(getattr(graph_sidecar, "reason", "Latence graph sidecar is degraded.")),
                }
            )
        if graph_sidecar is not None and str(getattr(graph_sidecar, "sync_status", "")).lower() in {"error", "failed"}:
            issues.append(
                {
                    "scope": "collection",
                    "name": runtime.name,
                    "kind": runtime.kind.value,
                    "reason": "latence_graph_sync_failed",
                    "detail": str(getattr(graph_sidecar, "sync_reason", "Latence graph dataset sync failed.")),
                }
            )

        try:
            indexed_matches_records = not runtime.meta.get("records") or self._is_runtime_indexed(runtime)
        except Exception as exc:
            issues.append(
                {
                    "scope": "collection",
                    "name": runtime.name,
                    "kind": runtime.kind.value,
                    "reason": "readiness_check_failed",
                    "detail": str(exc),
                }
            )
            return issues

        if not indexed_matches_records:
            issues.append(
                {
                    "scope": "collection",
                    "name": runtime.name,
                    "kind": runtime.kind.value,
                    "reason": "record_count_mismatch",
                    "detail": "Stored records and engine state disagree about the indexed point count.",
                }
            )
        return issues

    def readiness_report(self) -> Dict[str, Any]:
        self._sync_collection_registry()
        issues: List[Dict[str, str]] = [
            {
                "scope": "service",
                "name": name,
                "kind": "load",
                "reason": "collection_load_failed",
                "detail": detail,
            }
            for name, detail in sorted(self.load_failures.items())
        ]
        with self._metrics_lock:
            filter_scan_limit_hits = self.filter_scan_limit_hits

        with self._collections_lock:
            runtimes = list(self.collections.values())

        for runtime in runtimes:
            lock = self._ensure_collection_lock(runtime.name)
            with lock:
                issues.extend(self._collection_readiness_issues(runtime))

        if filter_scan_limit_hits > 0:
            issues.append(
                {
                    "scope": "service",
                    "name": "search",
                    "kind": "service",
                    "reason": "filter_scan_limit_hits",
                    "detail": f"The exact filtered search scan ceiling has been exceeded {filter_scan_limit_hits} time(s) since startup.",
                }
            )

        degraded_collections = sorted({issue["name"] for issue in issues if issue.get("scope") == "collection"})
        return {
            "status": "ok" if not issues else "degraded",
            "collections": len(runtimes),
            "degraded_collections": degraded_collections,
            "failed_collection_loads": sorted(self.load_failures.keys()),
            "filter_scan_limit_hits": filter_scan_limit_hits,
            "runtime_capabilities": self.runtime_capabilities,
            "issues": issues,
        }

    def metrics_snapshot(self) -> Dict[str, Any]:
        readiness = self.readiness_report()
        with self._metrics_lock:
            request_count = self.request_count
            total_latency = self.total_latency
            filter_scan_limit_hits = self.filter_scan_limit_hits
            nodes_visited_total = self.nodes_visited_total
            distance_comps_total = self.distance_comps_total
        with self._collections_lock:
            collection_kinds = {name: runtime.kind.value for name, runtime in self.collections.items()}
        return {
            "request_count": request_count,
            "total_latency": total_latency,
            "nodes_visited_total": nodes_visited_total,
            "distance_comps_total": distance_comps_total,
            "collections_total": len(collection_kinds),
            "points_total": self.total_points(),
            "failed_collection_loads": len(readiness["failed_collection_loads"]),
            "filter_scan_limit_hits": filter_scan_limit_hits,
            "collection_kinds": collection_kinds,
            "readiness": readiness,
        }

    def total_points(self) -> int:
        with self._collections_lock:
            runtimes = list(self.collections.values())
        total = 0
        for runtime in runtimes:
            lock = self._ensure_collection_lock(runtime.name)
            with lock:
                total += self._engine_point_count(runtime)
        return total
