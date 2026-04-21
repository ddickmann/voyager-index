"""
ColSearch Reference API

FastAPI server for deploying the local colsearch reference service.

Usage:
    # Development
    uvicorn colsearch.server:app --reload --port 8080

    # Local service
    uvicorn colsearch.server:app --host 127.0.0.1 --port 8080

Author: Latence Team
License: CC-BY-NC-4.0
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from colsearch import __version__ as PACKAGE_VERSION

from .api.routes import router
from .api.service import SearchService
from .middleware import (
    ConcurrencyLimitMiddleware,
    RateLimitMiddleware,
    RequestIdMiddleware,
    structured_error_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Factory
# ============================================================================


def create_app(
    title: str = "ColSearch Reference API",
    version: str = PACKAGE_VERSION,
    enable_cors: bool = False,
    index_path: Optional[str] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        version: API version
        enable_cors: Enable CORS middleware
        index_path: Path for index storage

    Returns:
        Configured FastAPI application
    """

    if "VOYAGER_INDEX_PATH" in os.environ and "COLSEARCH_INDEX_PATH" not in os.environ:
        logger.warning(
            "VOYAGER_INDEX_PATH is deprecated; use COLSEARCH_INDEX_PATH. "
            "The legacy name will be removed in 0.3.0."
        )
    index_dir = (
        index_path
        or os.environ.get("COLSEARCH_INDEX_PATH")
        or os.environ.get("VOYAGER_INDEX_PATH")
        or os.environ.get("LATENCE_INDEX_PATH")
        or "/data/colsearch"
    )
    os.makedirs(index_dir, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events."""
        logger.info(f"Starting {title} v{version}")
        logger.info(f"Index directory: {index_dir}")
        app.state.search_service = SearchService(index_dir)

        # Check GPU availability
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Running on CPU")
        except ImportError:
            logger.info("PyTorch not available, running on CPU")

        yield

        # Shutdown
        close_service = getattr(app.state.search_service, "close", None)
        if callable(close_service):
            close_service()
        logger.info("Shutting down server")

    app = FastAPI(
        title=title,
        description="""
## ColSearch Reference API

Single-host production reference service for the open-source `colsearch` runtime.

### Features

- Durable collection metadata on disk
- Single-host multi-worker runtime reloads for collection mutations and async task polling
- Dense, late-interaction, and multimodal collection types
- Hybrid dense+sparse retrieval for dense collections with explicit `dense_hybrid_mode: "rrf"` or `"tabu"`
- Base64 or JSON float-array vector transport on collection ingest/search
- Shard-engine CPU/GPU routing with ColBANDIT and INT8/FP8/ROQ4 selection
- MaxSim-backed late-interaction retrieval
- Stateless fulfilment optimizer at `/reference/optimize` (canonical contract; needs `latence_solver`)
- Source-doc preprocessing at `/reference/preprocess/documents` for PDF, DOCX, XLSX, and image inputs
- Local multimodal retrieval over precomputed embeddings

### Quick Start

1. Render source docs into page images:
```bash
curl -X POST http://localhost:8080/reference/preprocess/documents \\
  -H "Content-Type: application/json" \\
  -d '{"source_paths": ["/data/source/invoice.pdf"]}'
```

2. Create a collection:
```bash
curl -X POST http://localhost:8080/collections/docs \\
  -H "Content-Type: application/json" \\
  -d '{"dimension": 128, "kind": "dense"}'
```

3. Add points (base64 is the preferred transport for larger vectors):
```bash
curl -X POST http://localhost:8080/collections/docs/points \\
  -H "Content-Type: application/json" \\
  -d '{"points": [{"id": "1", "vector": {"encoding":"float32","shape":[1,128],"dtype":"float32","data_b64":"..."}}]}'
```

4. Search:
```bash
curl -X POST http://localhost:8080/collections/docs/search \\
  -H "Content-Type: application/json" \\
  -d '{"vector": {"encoding":"float32","shape":[1,128],"dtype":"float32","data_b64":"..."}, "top_k": 10}'
```
        """,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.index_dir = index_dir

    # --- Middleware stack (outermost first) ---
    # Request-ID must be outermost so all downstream middleware can read it.
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ConcurrencyLimitMiddleware)

    if enable_cors:
        allowed_origins = os.environ.get(
            "VOYAGER_CORS_ORIGINS",
            "http://127.0.0.1,http://localhost",
        ).split(",")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        )

    # --- Structured error handlers ---
    @app.exception_handler(HTTPException)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        rid = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
        payload = exc.detail if isinstance(exc.detail, dict) else {}
        raw_details = payload.get("details")
        details = raw_details if isinstance(raw_details, dict) else None
        message = str(payload.get("message") or payload.get("detail") or exc.detail)
        response = structured_error_response(
            exc.status_code,
            code=str(payload.get("code") or f"http_{exc.status_code}"),
            message=message,
            details=details,
            request_id=rid,
            detail=str(payload.get("detail") or message),
            include_status_code=True,
        )
        if exc.headers:
            response.headers.update({key: value for key, value in exc.headers.items() if value is not None})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
        return structured_error_response(
            422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()},
            request_id=rid,
            detail="Request validation failed",
            include_status_code=True,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=True)
        rid = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
        return structured_error_response(
            500,
            code="internal_error",
            message="Internal server error",
            request_id=rid,
            detail="Internal server error",
            include_status_code=True,
        )

    app.include_router(router, prefix="")

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8080))
    default_workers = min(max(os.cpu_count() or 1, 2), 4)
    workers = min(int(os.environ.get("WORKERS", default_workers)), 8)
    if workers < 1:
        workers = 1

    uvicorn.run(
        "colsearch.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


# Default application instance
app = create_app()


if __name__ == "__main__":
    main()
