"""
API routes for the voyager-index reference server.
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .models import (
    AddPointsRequest,
    ClearPayloadRequest,
    CollectionInfo,
    CollectionListResponse,
    CreateCollectionRequest,
    DeletePayloadKeysRequest,
    DeletePointsRequest,
    EncodeRequest,
    EncodeResponse,
    ErrorResponse,
    HealthResponse,
    MutationTaskResponse,
    MutationTaskStatus,
    OptimizeRequest,
    RenderDocumentsRequest,
    RenderDocumentsResponse,
    RerankRequest,
    RerankResponse,
    RetrieveRequest,
    RetrieveResponse,
    ScrollRequest,
    ScrollResponse,
    SearchBatchRequest,
    SearchBatchResponse,
    SearchRequest,
    SearchResponse,
    SetPayloadRequest,
    ShardListResponse,
    TaskStatusResponse,
    WalStatusResponse,
)
from .service import SearchService, ServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(request: Request) -> SearchService:
    return request.app.state.search_service


def _raise_service_error(exc: ServiceError, request: Request | None = None) -> None:
    rid = ""
    if request and hasattr(request, "state"):
        rid = getattr(request.state, "request_id", "")
    raise HTTPException(
        status_code=exc.status_code,
        detail=ErrorResponse(
            code=getattr(exc, "error_code", "service_error"),
            message=exc.detail,
            request_id=rid,
            error=getattr(exc, "error_code", "service_error"),
            detail=exc.detail,
            status_code=exc.status_code,
        ).model_dump(),
    ) from exc


def _metric_labels(**labels: str) -> str:
    escaped = []
    for key, value in labels.items():
        sanitized = str(value).replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'{key}="{sanitized}"')
    return "{" + ",".join(escaped) + "}"


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request, service: SearchService = Depends(get_service)):
    return HealthResponse(
        status="ok",
        version=request.app.version,
        collections=len(service.collections),
        gpu_available=service.gpu_available,
    )


@router.get("/ready", tags=["System"])
async def readiness_check(service: SearchService = Depends(get_service)):
    report = service.readiness_report()
    status_code = 200 if report["status"] == "ok" else 503
    return JSONResponse(content=report, status_code=status_code)


@router.get("/reference/optimize/health", tags=["Optimizer"])
async def reference_optimizer_health(service: SearchService = Depends(get_service)):
    """Health and backend status for the stateless fulfilment optimizer."""
    return service.reference_optimizer_health()


@router.post("/reference/optimize", tags=["Optimizer"])
async def reference_optimize(
    body: Dict[str, Any],
    service: SearchService = Depends(get_service),
):
    """
    Canonical stateless optimizer API (dense, multivector / MaxSim precompute, optional BM25 via metadata).

    Body matches the canonical OSS ``OptimizerRequest`` JSON shape: ``query_text``, ``query_vectors``,
    ``candidates``, ``constraints``, ``solver_config``, ``metadata``.
    """
    try:
        return service.reference_optimize(body)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post(
    "/reference/preprocess/documents",
    response_model=RenderDocumentsResponse,
    tags=["Preprocessing"],
)
async def reference_preprocess_documents(
    request: RenderDocumentsRequest,
    service: SearchService = Depends(get_service),
):
    """Render local source documents into page images before embedding/ingestion."""
    try:
        return service.reference_preprocess_documents(request)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def prometheus_metrics(service: SearchService = Depends(get_service)):
    snapshot = service.metrics_snapshot()
    readiness = snapshot["readiness"]
    collection_kinds = snapshot["collection_kinds"]
    lines = [
        "# HELP voyager_search_requests_total Total search requests",
        "# TYPE voyager_search_requests_total counter",
        f"voyager_search_requests_total {snapshot['request_count']}",
        "",
        "# HELP voyager_search_latency_seconds Search latency in seconds",
        "# TYPE voyager_search_latency_seconds summary",
        f"voyager_search_latency_seconds_sum {snapshot['total_latency']:.6f}",
        f"voyager_search_latency_seconds_count {snapshot['request_count']}",
        "",
        "# HELP voyager_collections_total Number of collections",
        "# TYPE voyager_collections_total gauge",
        f"voyager_collections_total {snapshot['collections_total']}",
        "",
        "# HELP voyager_points_total Total indexed points",
        "# TYPE voyager_points_total gauge",
        f"voyager_points_total {snapshot['points_total']}",
        "",
        "# HELP voyager_collection_load_failures Number of collections that failed to load",
        "# TYPE voyager_collection_load_failures gauge",
        f"voyager_collection_load_failures {snapshot['failed_collection_loads']}",
        "",
        "# HELP voyager_filter_scan_limit_hits_total Number of filtered searches rejected by the exact scan ceiling",
        "# TYPE voyager_filter_scan_limit_hits_total counter",
        f"voyager_filter_scan_limit_hits_total {snapshot['filter_scan_limit_hits']}",
        "",
        "# HELP voyager_search_nodes_visited_total Cumulative graph nodes visited across all searches",
        "# TYPE voyager_search_nodes_visited_total counter",
        f"voyager_search_nodes_visited_total {snapshot['nodes_visited_total']}",
        "",
        "# HELP voyager_search_distance_comps_total Cumulative distance computations across all searches",
        "# TYPE voyager_search_distance_comps_total counter",
        f"voyager_search_distance_comps_total {snapshot['distance_comps_total']}",
        "",
        "# HELP voyager_collection_status Current collection readiness status (1 for the reported state)",
        "# TYPE voyager_collection_status gauge",
    ]

    issues_by_name = {name: [] for name in collection_kinds.keys()}
    kinds_by_name = dict(collection_kinds)
    for issue in readiness["issues"]:
        if issue.get("scope") == "collection":
            issues_by_name.setdefault(issue["name"], []).append(issue)
            kinds_by_name.setdefault(issue["name"], issue.get("kind", "unknown"))
    for name in sorted(collection_kinds.keys()):
        status = "degraded" if issues_by_name.get(name) else "ok"
        lines.append(
            f"voyager_collection_status{_metric_labels(collection=name, kind=kinds_by_name[name], status=status)} 1"
        )
    return "\n".join(lines) + "\n"


@router.get("/collections", response_model=CollectionListResponse, tags=["Collections"])
async def list_collections(service: SearchService = Depends(get_service)):
    return CollectionListResponse(collections=service.list_collections())


@router.post("/collections/{name}", tags=["Collections"])
async def create_collection(
    name: str,
    request: CreateCollectionRequest,
    service: SearchService = Depends(get_service),
):
    if request.name is not None and request.name != name:
        raise HTTPException(status_code=400, detail="Path name and request.name must match")
    try:
        service.create_collection(name, request)
    except ServiceError as exc:
        _raise_service_error(exc)

    logger.info("Created collection '%s' (%s)", name, request.kind.value)
    return {"status": "ok", "name": name}


@router.get("/collections/{name}/info", response_model=CollectionInfo, tags=["Collections"])
async def get_collection_info(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        return service.collection_info(name)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.delete("/collections/{name}", tags=["Collections"])
async def delete_collection(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        service.delete_collection(name)
    except ServiceError as exc:
        _raise_service_error(exc)

    logger.info("Deleted collection '%s'", name)
    return {"status": "ok", "name": name}


@router.post("/collections/{name}/points", tags=["Points"])
async def add_points(
    name: str,
    request: AddPointsRequest,
    service: SearchService = Depends(get_service),
):
    try:
        added = service.add_points(name, request.points)
    except ServiceError as exc:
        _raise_service_error(exc)

    logger.info("Added %s points to '%s'", added, name)
    return {"status": "ok", "added": added}


@router.delete("/collections/{name}/points", tags=["Points"])
async def delete_points(
    name: str,
    request: DeletePointsRequest,
    service: SearchService = Depends(get_service),
):
    try:
        deleted = service.delete_points(name, request.ids)
    except ServiceError as exc:
        _raise_service_error(exc)

    logger.info("Deleted %s points from '%s'", deleted, name)
    return {"status": "ok", "deleted": deleted}


@router.post("/collections/{name}/search", response_model=SearchResponse, tags=["Search"])
async def search(
    name: str,
    request: SearchRequest,
    service: SearchService = Depends(get_service),
):
    try:
        return service.search(name, request)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/collections/{name}/scroll", response_model=ScrollResponse, tags=["Points"])
async def scroll_collection(
    name: str,
    request: ScrollRequest,
    service: SearchService = Depends(get_service),
):
    try:
        return service.scroll_collection(name, request)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/collections/{name}/retrieve", response_model=RetrieveResponse, tags=["Points"])
async def retrieve_points(
    name: str,
    request: RetrieveRequest,
    service: SearchService = Depends(get_service),
):
    try:
        return service.retrieve_points(name, request)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/collections/{name}/search/batch", response_model=SearchBatchResponse, tags=["Search"])
async def search_batch(
    name: str,
    request: SearchBatchRequest,
    service: SearchService = Depends(get_service),
):
    try:
        return service.search_batch_collection(name, request)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/collections/{name}/optimize", response_model=SearchResponse, tags=["Search"])
async def optimize_search(
    name: str,
    request: OptimizeRequest,
    service: SearchService = Depends(get_service),
):
    raise HTTPException(
        status_code=501,
        detail="Use /reference/optimize for the canonical OSS solver API.",
    )


# ------------------------------------------------------------------
# Shard admin endpoints
# ------------------------------------------------------------------


@router.post("/collections/{name}/compact", tags=["Shard Admin"])
async def compact_collection(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        result = service.compact_collection(name)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"status": "ok", **result}


@router.get("/collections/{name}/shards", response_model=ShardListResponse, tags=["Shard Admin"])
async def list_shards(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        return service.list_shards(name)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.get("/collections/{name}/shards/{shard_id}", tags=["Shard Admin"])
async def get_shard_detail(
    name: str,
    shard_id: int,
    service: SearchService = Depends(get_service),
):
    try:
        return service.get_shard_detail(name, shard_id)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.get("/collections/{name}/wal/status", response_model=WalStatusResponse, tags=["Shard Admin"])
async def wal_status(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        return service.wal_status(name)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/collections/{name}/checkpoint", tags=["Shard Admin"])
async def checkpoint_collection(
    name: str,
    service: SearchService = Depends(get_service),
):
    try:
        result = service.checkpoint_collection(name)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"status": "ok", **result}


# ------------------------------------------------------------------
# Async mutation task queue
# ------------------------------------------------------------------

_task_store: Dict[str, Dict[str, Any]] = {}
_task_lock = threading.Lock()
_TASK_MAX_AGE_S = 3600  # evict completed tasks older than 1 hour
_TASK_MAX_ENTRIES = 10_000


def _evict_stale_tasks() -> None:
    """Remove completed/failed tasks older than _TASK_MAX_AGE_S (called under lock)."""
    now = datetime.now(timezone.utc)
    stale = []
    for tid, rec in _task_store.items():
        if rec["status"] in (MutationTaskStatus.COMPLETED, MutationTaskStatus.FAILED):
            completed = rec.get("completed_at")
            if completed:
                try:
                    age = (now - datetime.fromisoformat(completed)).total_seconds()
                    if age > _TASK_MAX_AGE_S:
                        stale.append(tid)
                except (ValueError, TypeError):
                    stale.append(tid)
    for tid in stale:
        _task_store.pop(tid, None)


def _submit_task(fn: Callable) -> str:
    """Run *fn* in a background thread, return a task_id immediately."""
    task_id = uuid.uuid4().hex[:12]
    record: Dict[str, Any] = {
        "status": MutationTaskStatus.PENDING,
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    with _task_lock:
        if len(_task_store) >= _TASK_MAX_ENTRIES:
            _evict_stale_tasks()
        _task_store[task_id] = record

    def _worker():
        with _task_lock:
            _task_store[task_id]["status"] = MutationTaskStatus.RUNNING
        try:
            res = fn()
            with _task_lock:
                _task_store[task_id]["status"] = MutationTaskStatus.COMPLETED
                _task_store[task_id]["result"] = res
                _task_store[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.error("Async task %s failed: %s", task_id, exc, exc_info=True)
            with _task_lock:
                _task_store[task_id]["status"] = MutationTaskStatus.FAILED
                _task_store[task_id]["error"] = str(exc)
                _task_store[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=_worker, name=f"task-{task_id}", daemon=True).start()
    return task_id


@router.post(
    "/collections/{name}/points/async",
    response_model=MutationTaskResponse,
    status_code=202,
    tags=["Points"],
)
async def add_points_async(
    name: str,
    request_body: AddPointsRequest,
    service: SearchService = Depends(get_service),
):
    """Accept points for asynchronous ingestion; returns 202 with a task ID."""
    task_id = _submit_task(
        lambda: {"added": service.add_points(name, request_body.points)},
    )
    return MutationTaskResponse(task_id=task_id, status=MutationTaskStatus.PENDING)


@router.delete(
    "/collections/{name}/points/async",
    response_model=MutationTaskResponse,
    status_code=202,
    tags=["Points"],
)
async def delete_points_async(
    name: str,
    request_body: DeletePointsRequest,
    service: SearchService = Depends(get_service),
):
    """Accept point deletions for asynchronous processing."""
    task_id = _submit_task(
        lambda: {"deleted": service.delete_points(name, request_body.ids)},
    )
    return MutationTaskResponse(task_id=task_id, status=MutationTaskStatus.PENDING)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Poll the status of an async mutation task."""
    with _task_lock:
        record = _task_store.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(task_id=task_id, **record)


# ------------------------------------------------------------------
# Metadata / Payload CRUD
# ------------------------------------------------------------------

@router.post("/collections/{name}/points/payload", tags=["Payload"])
async def set_payload(
    name: str,
    request_body: SetPayloadRequest,
    service: SearchService = Depends(get_service),
):
    """Set or merge payload fields on specific points."""
    try:
        updated = service.set_payload(name, request_body.points, request_body.payload)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"status": "ok", "updated": updated}


@router.post("/collections/{name}/points/payload/delete", tags=["Payload"])
async def delete_payload_keys(
    name: str,
    request_body: DeletePayloadKeysRequest,
    service: SearchService = Depends(get_service),
):
    """Delete specific payload keys from points."""
    try:
        updated = service.delete_payload_keys(name, request_body.points, request_body.keys)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"status": "ok", "updated": updated}


@router.post("/collections/{name}/points/payload/clear", tags=["Payload"])
async def clear_payload(
    name: str,
    request_body: ClearPayloadRequest,
    service: SearchService = Depends(get_service),
):
    """Clear all payload fields from points."""
    try:
        updated = service.clear_payload(name, request_body.points)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"status": "ok", "updated": updated}


@router.get("/collections/{name}/points/{point_id}/payload", tags=["Payload"])
async def get_point_payload(
    name: str,
    point_id: str,
    service: SearchService = Depends(get_service),
):
    """Get the payload for a single point by external ID."""
    try:
        payload = service.get_point_payload(name, point_id)
    except ServiceError as exc:
        _raise_service_error(exc)
    return {"id": point_id, "payload": payload}


# ------------------------------------------------------------------
# Encode / Rerank
# ------------------------------------------------------------------

@router.post("/encode", response_model=EncodeResponse, tags=["Inference"])
async def encode(
    request_body: EncodeRequest,
    service: SearchService = Depends(get_service),
):
    """Encode text or images into multi-vector embeddings."""
    try:
        return service.encode(request_body)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/rerank", response_model=RerankResponse, tags=["Inference"])
async def rerank(
    request_body: RerankRequest,
    service: SearchService = Depends(get_service),
):
    """Rerank documents against a query using late-interaction scoring."""
    try:
        return service.rerank(request_body)
    except ServiceError as exc:
        _raise_service_error(exc)

