"""
Durable collection service for the voyager-index reference API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from voyager_index._internal.inference.config import IndexConfig
from voyager_index._internal.inference.engines.colpali import ColPaliConfig, ColPaliEngine
from voyager_index._internal.inference.index_core.hybrid_manager import HybridSearchManager
from voyager_index._internal.inference.index_core.index import ColbertIndex

from .models import (
    CollectionInfo,
    CollectionKind,
    CreateCollectionRequest,
    DistanceMetric,
    MultimodalOptimizeMode,
    PointVector,
    RenderDocumentsRequest,
    ScoredPoint,
    ScreeningMode,
    SearchRequest,
    SearchResponse,
    SearchStrategy,
)

logger = logging.getLogger(__name__)


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
        self.request_count = 0
        self.total_latency = 0.0
        self.collections: Dict[str, CollectionRuntime] = {}
        self.load_failures: Dict[str, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        self.filter_scan_limit = int(os.environ.get("VOYAGER_FILTER_SCAN_LIMIT", "10000"))
        self.filter_scan_limit_hits = 0
        self._metrics_lock = threading.Lock()
        self._collections_lock = threading.RLock()
        self._collection_locks: Dict[str, threading.RLock] = {}
        self._recover_pending_journals()
        self._load_collections()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_search_metrics(self, elapsed_ms: float) -> None:
        with self._metrics_lock:
            self.request_count += 1
            self.total_latency += elapsed_ms / 1000.0

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

    def _resolve_user_path(self, raw_path: str, *, relative_to_root: bool = False) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            base = self.root_path if relative_to_root else Path.cwd().resolve()
            candidate = base / candidate
        return candidate.resolve()

    def _sanitize_records(self, meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            str(key): value
            for key, value in (meta.get("records") or {}).items()
        }

    def _refresh_record_index(self, meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        return {
            int(record["internal_id"]): record
            for record in self._sanitize_records(meta).values()
        }

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
        runtime.meta["updated_at"] = self._now()
        metadata_path = self._metadata_path(runtime.name)
        self._write_json_atomic(metadata_path, runtime.meta)

    def _load_meta(self, name: str) -> Dict[str, Any]:
        with open(self._metadata_path(name), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
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
        corpus = [
            record.get("text") or ""
            for record in records
        ]
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
        meta = self._load_meta(name)
        runtime = CollectionRuntime(
            name=name,
            kind=CollectionKind(meta["kind"]),
            path=self._collection_path(name),
            meta=meta,
            engine=self._build_engine(name, meta),
            record_index=self._refresh_record_index(meta),
            payload_filter_index=self._build_payload_filter_index(meta),
        )
        if runtime.kind == CollectionKind.DENSE:
            should_rebuild_sparse = (
                bool(runtime.meta.get("sparse_dirty"))
                or (
                    runtime.engine.retriever is None
                    and not runtime.engine.sparse_index_present
                    and bool(runtime.meta.get("records"))
                )
            )
            self._sync_dense_sparse_state(runtime, rebuild_sparse=should_rebuild_sparse)
            if should_rebuild_sparse:
                self._write_meta(runtime)
        return runtime

    def _engine_point_count(self, runtime: CollectionRuntime) -> int:
        if runtime.kind == CollectionKind.DENSE:
            return int(runtime.engine.hnsw.total_vectors())
        if runtime.kind == CollectionKind.LATE_INTERACTION:
            return int(runtime.engine.get_statistics().num_documents)
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
        for collection_dir in self.root_path.iterdir():
            if not collection_dir.is_dir() or collection_dir.name.startswith("."):
                continue
            metadata_path = collection_dir / "collection.json"
            if not metadata_path.exists():
                continue

            name = collection_dir.name
            try:
                runtime = self._load_runtime(name)
            except Exception as exc:
                self.load_failures[name] = str(exc)
                logger.warning("Failed to load collection '%s': %s", name, exc)
                continue
            self.load_failures.pop(name, None)
            self.collections[name] = runtime
            self._ensure_collection_lock(name)
            logger.info("Loaded collection '%s' (%s)", name, runtime.kind.value)

    def _ensure_collection_lock(self, name: str) -> threading.RLock:
        with self._collections_lock:
            return self._collection_locks.setdefault(name, threading.RLock())

    def _remove_collection_lock(self, name: str) -> None:
        with self._collections_lock:
            self._collection_locks.pop(name, None)

    def _collection_context(self, name: str) -> tuple[CollectionRuntime, threading.RLock]:
        safe_name = self._validate_collection_name(name)
        with self._collections_lock:
            runtime = self.collections.get(safe_name)
            if runtime is None:
                raise NotFoundError(f"Collection '{safe_name}' not found")
            lock = self._collection_locks.setdefault(safe_name, threading.RLock())
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
        elif runtime.kind == CollectionKind.LATE_INTERACTION:
            self._copy_into_backup(runtime, backup_root, runtime.path / "colbert" / "embeddings.h5")
            self._copy_into_backup(runtime, backup_root, runtime.path / "colbert" / "metadata.json")
        else:
            manifest_path = runtime.path / "colpali" / "manifest.json"
            screening_state_path = runtime.path / "colpali" / "screening_state.json"
            chunks_path = runtime.path / "colpali" / "chunks"
            self._copy_into_backup(runtime, backup_root, screening_state_path)
            self._copy_into_backup(runtime, backup_root, runtime.path / "colpali" / "prototype_sidecar")
            self._copy_into_backup(runtime, backup_root, runtime.path / "colpali" / "centroid_sidecar")
            self._copy_into_backup(runtime, backup_root, manifest_path)
            metadata["existing_chunk_files"] = (
                sorted(path.name for path in chunks_path.glob("*.npz"))
                if chunks_path.exists()
                else []
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
        elif kind == CollectionKind.LATE_INTERACTION:
            self._restore_from_backup(runtime, backup_root, runtime.path / "colbert" / "embeddings.h5")
            self._restore_from_backup(runtime, backup_root, runtime.path / "colbert" / "metadata.json")
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
        with self._collections_lock:
            return sorted(self.collections.keys())

    def get_collection(self, name: str) -> CollectionRuntime:
        with self._collections_lock:
            safe_name = self._validate_collection_name(name)
            if safe_name not in self.collections:
                raise NotFoundError(f"Collection '{safe_name}' not found")
            return self.collections[safe_name]

    def create_collection(self, name: str, request: CreateCollectionRequest) -> CollectionRuntime:
        with self._collections_lock:
            safe_name = self._validate_collection_name(name)
            if safe_name in self.collections:
                raise ConflictError(f"Collection '{safe_name}' already exists")

            if request.kind != CollectionKind.DENSE:
                if request.distance != DistanceMetric.COSINE:
                    raise ValidationError("Only dense collections support configurable distance metrics")
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
            }
            try:
                runtime = CollectionRuntime(
                    name=safe_name,
                    kind=request.kind,
                    path=collection_dir,
                    meta=meta,
                    engine=self._build_engine(safe_name, meta),
                    record_index={},
                    payload_filter_index={},
                )
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
            backup = self._begin_collection_mutation(runtime, operation="add_points")
            try:
                record_keys = [str(point.id) for point in points]
                if len(record_keys) != len(set(record_keys)):
                    raise ValidationError("Point IDs must be unique within a single request")

                payloads = [self._prepare_payload(point) for point in points]
                corpus = [
                    payload.get("text") or payload.get("content") or ""
                    for payload in payloads
                ]
                existing_internal_ids = [
                    int(runtime.meta["records"][record_key]["internal_id"])
                    for record_key in record_keys
                    if record_key in runtime.meta["records"]
                ]
                if existing_internal_ids:
                    self._delete_internal_ids(runtime, existing_internal_ids)

                if runtime.kind == CollectionKind.DENSE:
                    if any(point.vector is None for point in points):
                        raise ValidationError("Dense collections require single vectors")
                    vectors = np.asarray([point.vector for point in points], dtype=np.float32)
                    if vectors.ndim != 2 or vectors.shape[1] != int(runtime.meta["dimension"]):
                        raise ValidationError("Dense vectors must match the collection dimension")
                    ids = [self._next_internal_id(runtime) for _ in points]
                    runtime.engine.hnsw.add(vectors, ids=ids, payloads=payloads)
                    for record_key in record_keys:
                        runtime.meta["records"].pop(record_key, None)
                    for record_key, point, payload, text_value, internal_id in zip(record_keys, points, payloads, corpus, ids):
                        runtime.meta["records"][record_key] = {
                            "external_id": point.id,
                            "internal_id": internal_id,
                            "payload": payload,
                            "text": text_value,
                        }
                    self._sync_dense_sparse_state(runtime, rebuild_sparse=False)
                    self._flush_runtime_engine(runtime)
                elif runtime.kind == CollectionKind.LATE_INTERACTION:
                    if any(point.vectors is None for point in points):
                        raise ValidationError("Late-interaction collections require multi-vectors")
                    tensor = torch.from_numpy(np.asarray([point.vectors for point in points], dtype=np.float32))
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
                    for record_key, point, payload, text_value, internal_id in zip(record_keys, points, payloads, corpus, assigned_ids):
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
                    self._refresh_runtime_indexes(runtime)
                else:
                    if any(point.vectors is None for point in points):
                        raise ValidationError("Multimodal collections require multi-vectors")
                    vectors = np.asarray([point.vectors for point in points], dtype=np.float32)
                    if vectors.ndim != 3 or vectors.shape[-1] != int(runtime.meta["dimension"]):
                        raise ValidationError("Multimodal tensors must have shape (docs, patches, dim)")
                    ids = [self._next_internal_id(runtime) for _ in points]
                    runtime.engine.add_documents(vectors, doc_ids=ids)
                    for record_key in record_keys:
                        runtime.meta["records"].pop(record_key, None)
                    for record_key, point, payload, text_value, internal_id in zip(record_keys, points, payloads, corpus, ids):
                        runtime.meta["records"][record_key] = {
                            "external_id": point.id,
                            "internal_id": internal_id,
                            "payload": payload,
                            "text": text_value,
                        }
                    self._refresh_runtime_indexes(runtime)

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

    def _dense_query_vector(self, runtime: CollectionRuntime, request: SearchRequest) -> Optional[np.ndarray]:
        if request.vector is not None:
            query = np.asarray(request.vector, dtype=np.float32)
        elif request.vectors is not None:
            raise ValidationError("Dense search requires 'vector'; 'vectors' is only supported for late-interaction or multimodal collections")
        else:
            return None

        if query.shape[0] != int(runtime.meta["dimension"]):
            raise ValidationError("Query dimension does not match collection dimension")
        return query

    def _get_dense_vectors(self, runtime: CollectionRuntime, internal_ids: List[int]) -> Dict[int, Any]:
        items = runtime.engine.hnsw.retrieve(internal_ids)
        return {
            int(item["id"]): item.get("vector")
            for item in items
        }

    def _dense_solver_constraints(self, request: SearchRequest) -> Any:
        if request.max_tokens is None and request.max_chunks is None:
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
            max_per_cluster=2,
        )

    def _dense_refine_candidate_k(self, request: SearchRequest) -> int:
        if request.strategy != SearchStrategy.OPTIMIZED:
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
            import latence_solver  # type: ignore  # noqa: F401
        except ImportError:
            return False
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
            "max_per_cluster": 2,
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
            list(candidate_ids)
            if candidate_ids is not None
            else list(runtime.record_index.keys())
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
            solver_config={"iterations": 48},
            metadata={"multimodal_mode": "solver_prefilter_maxsim"},
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
            list(candidate_ids)
            if candidate_ids is not None
            else list(runtime.record_index.keys())
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
        frontier_ids = [
            int(result.doc_id)
            for result in frontier_results
            if int(result.doc_id) in runtime.record_index
        ]
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
            solver_config={"iterations": 48},
            metadata={"multimodal_mode": "maxsim_then_solver"},
        )
        solver_elapsed_ms = (time.perf_counter() - optimize_start) * 1000.0
        selected_ids = self._solver_selected_internal_ids(
            optimized.get("selected_ids", []),
            allowed_ids=frontier_ids,
        )
        if not selected_ids:
            selected_ids = frontier_ids[: request.top_k]
        selected_ranked = [
            result
            for result in frontier_results
            if int(result.doc_id) in set(selected_ids)
        ]
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
        if request.strategy != SearchStrategy.OPTIMIZED:
            return fused, None
        if not getattr(runtime.engine, "solver_available", False):
            raise ValidationError(
                "Optimized dense search requires the latence_solver native package to be installed"
            )
        if query is None:
            raise ValidationError("Optimized dense search requires 'vector' so the solver can score dense candidates")

        refined = runtime.engine.refine(
            query_vector=query,
            query_text=request.query_text or "",
            candidate_ids=[doc_id for doc_id, _ in fused],
            constraints=self._dense_solver_constraints(request),
        )
        selected_ids = [
            int(doc_id)
            for doc_id in refined.get("selected_internal_ids", [])
            if int(doc_id) in runtime.record_index
        ]
        if not selected_ids:
            return fused, refined

        score_map = {int(doc_id): float(score) for doc_id, score in fused}
        refined_pairs = [
            (doc_id, score_map.get(doc_id, 0.0))
            for doc_id in selected_ids
            if doc_id in score_map
        ]
        if not refined_pairs:
            return fused, refined
        return refined_pairs[: request.top_k], refined

    def _fuse_dense_sparse(
        self,
        dense_results: List[tuple[int, float]],
        sparse_results: List[tuple[int, float]],
        top_k: int,
    ) -> List[tuple[int, float]]:
        if dense_results and not sparse_results:
            return [(int(doc_id), float(score)) for doc_id, score in dense_results[:top_k]]
        if sparse_results and not dense_results:
            return [(int(doc_id), float(score)) for doc_id, score in sparse_results[:top_k]]

        fused: Dict[int, float] = {}
        rrf_k = 60.0
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            fused[int(doc_id)] = fused.get(int(doc_id), 0.0) + 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]

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

            if runtime.kind == CollectionKind.DENSE:
                query = self._dense_query_vector(runtime, request)
                if query is None and not request.query_text:
                    raise ValidationError("Dense search requires a vector or query_text")
                if request.strategy == SearchStrategy.OPTIMIZED and query is None:
                    raise ValidationError("Optimized dense search requires 'vector' because solver refinement is dense-query aware")
                if request.query_text and runtime.meta.get("sparse_dirty"):
                    self._sync_dense_sparse_state(runtime, rebuild_sparse=True)
                    self._write_meta(runtime)
                if not runtime.meta.get("records"):
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._record_search_metrics(elapsed_ms)
                    return SearchResponse(results=[], total=0, time_ms=elapsed_ms)
                retrieval_k = self._dense_refine_candidate_k(request)
                result = runtime.engine.search(
                    query_text=request.query_text or "",
                    query_vector=query,
                    k=retrieval_k,
                    filters=request.filter,
                )
                fused = self._fuse_dense_sparse(result.get("dense", []), result.get("sparse", []), retrieval_k)
                dense_pairs, refined = self._refine_dense_pairs(runtime, request, query, fused)
                vectors_by_id = {}
                if request.with_vector:
                    refined_vectors = refined.get("selected_vectors", {}) if refined is not None else {}
                    vectors_by_id = {int(doc_id): value for doc_id, value in refined_vectors.items()}
                    missing_ids = [
                        doc_id for doc_id, _ in dense_pairs
                        if int(doc_id) not in vectors_by_id
                    ]
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
                    refined.get("solver_output", {}).get("objective_score")
                    if refined is not None
                    else None
                )
                total_tokens = (
                    refined.get("solver_output", {}).get("total_tokens")
                    if refined is not None
                    else None
                )
            elif runtime.kind == CollectionKind.LATE_INTERACTION:
                if request.query_text:
                    raise ValidationError("Late-interaction collections do not support query_text search")
                if request.vectors is not None:
                    query_tensor = torch.tensor([request.vectors], dtype=torch.float32, device=self.device)
                elif request.vector is not None:
                    query_tensor = torch.tensor([[request.vector]], dtype=torch.float32, device=self.device)
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
                selected_pairs = filtered_pairs[:request.top_k]
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
            else:
                if request.query_text:
                    raise ValidationError("Multimodal collections do not support query_text search")
                if request.vectors is not None:
                    query_embedding = np.asarray(request.vectors, dtype=np.float32)
                elif request.vector is not None:
                    query_embedding = np.asarray([request.vector], dtype=np.float32)
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
                elif request.strategy != SearchStrategy.OPTIMIZED or multimodal_mode == MultimodalOptimizeMode.MAXSIM_ONLY:
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
                                len(candidate_ids)
                                if candidate_ids is not None
                                else len(runtime.record_index)
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
                selected_results = [
                    result
                    for result in raw_results
                    if int(result.doc_id) in runtime.record_index
                    and self._matches_filter(runtime.record_index[int(result.doc_id)]["payload"], request.filter)
                ][:request.top_k]
                vectors_by_id = {}
                if request.with_vector and selected_results:
                    vectors_by_id = runtime.engine.get_document_embeddings([int(result.doc_id) for result in selected_results])
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
            )

    def collection_info(self, name: str) -> CollectionInfo:
        runtime, collection_lock = self._collection_context(name)
        with collection_lock:
            storage_mb: Optional[float] = None
            if runtime.kind == CollectionKind.LATE_INTERACTION:
                storage_mb = runtime.engine.get_statistics().storage_size_mb
            elif runtime.kind == CollectionKind.MULTIMODAL:
                storage_mb = runtime.engine.get_statistics().get("storage_mb")

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
                storage_mode=runtime.meta.get("storage_mode") if runtime.kind == CollectionKind.LATE_INTERACTION else None,
                storage_path=str(runtime.path),
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

    def close(self) -> None:
        """Release collection engines so the same data root can be reopened safely."""
        with self._collections_lock:
            runtimes = list(self.collections.values())
            self.collections = {}
            self._collection_locks = {}
        for runtime in runtimes:
            close_method = getattr(runtime.engine, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:
                    logger.warning("Failed to close engine for collection '%s': %s", runtime.name, exc)

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

        try:
            indexed_matches_records = (
                not runtime.meta.get("records") or self._is_runtime_indexed(runtime)
            )
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

        degraded_collections = sorted(
            {
                issue["name"]
                for issue in issues
                if issue.get("scope") == "collection"
            }
        )
        return {
            "status": "ok" if not issues else "degraded",
            "collections": len(runtimes),
            "degraded_collections": degraded_collections,
            "failed_collection_loads": sorted(self.load_failures.keys()),
            "filter_scan_limit_hits": filter_scan_limit_hits,
            "issues": issues,
        }

    def metrics_snapshot(self) -> Dict[str, Any]:
        readiness = self.readiness_report()
        with self._metrics_lock:
            request_count = self.request_count
            total_latency = self.total_latency
            filter_scan_limit_hits = self.filter_scan_limit_hits
        with self._collections_lock:
            collection_kinds = {
                name: runtime.kind.value
                for name, runtime in self.collections.items()
            }
        return {
            "request_count": request_count,
            "total_latency": total_latency,
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
