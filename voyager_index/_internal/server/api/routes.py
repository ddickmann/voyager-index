"""
API routes for the voyager-index reference server.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .models import (
    AddPointsRequest,
    CollectionInfo,
    CollectionListResponse,
    CreateCollectionRequest,
    DeletePointsRequest,
    HealthResponse,
    OptimizeRequest,
    RenderDocumentsRequest,
    RenderDocumentsResponse,
    SearchRequest,
    SearchResponse,
    ShardListResponse,
    WalStatusResponse,
)
from .service import SearchService, ServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(request: Request) -> SearchService:
    return request.app.state.search_service


def _raise_service_error(exc: ServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


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

