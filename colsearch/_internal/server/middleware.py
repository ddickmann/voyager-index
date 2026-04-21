"""Production middleware for the colsearch reference API.

Provides:
- X-Request-ID propagation (generate if missing)
- Token-bucket rate limiting per client IP
- Concurrency limiter (cap in-flight requests)
- Structured JSON error wrapper
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


# ------------------------------------------------------------------
# X-Request-ID
# ------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID into every request/response.

    If the client sends one, it is preserved; otherwise a UUID4 is generated.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


# ------------------------------------------------------------------
# Rate limiter (token-bucket per client IP)
# ------------------------------------------------------------------


class _TokenBucket:
    __slots__ = ("capacity", "rate", "tokens", "last_refill", "_lock")

    def __init__(self, capacity: float, rate: float):
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


_BUCKET_TTL = 300.0  # evict idle IPs after 5 minutes
_MAX_BUCKETS = 10_000


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP token-bucket rate limiter.

    Env vars:
        VOYAGER_RATE_LIMIT       – max requests per second per IP (0=disabled)
        VOYAGER_RATE_LIMIT_BURST – burst capacity (default = 2× rate)
    """

    def __init__(self, app, rate: float = 0, burst: float = 0):
        super().__init__(app)
        self._rate = rate or float(os.environ.get("VOYAGER_RATE_LIMIT", "0"))
        self._burst = burst or float(os.environ.get("VOYAGER_RATE_LIMIT_BURST", "0")) or self._rate * 2
        self._buckets: dict[str, _TokenBucket] = {}
        self._buckets_lock = threading.Lock()
        self._enabled = self._rate > 0

    def _get_bucket(self, client: str) -> _TokenBucket:
        with self._buckets_lock:
            bucket = self._buckets.get(client)
            if bucket is None:
                if len(self._buckets) >= _MAX_BUCKETS:
                    self._evict_stale()
                bucket = _TokenBucket(self._burst, self._rate)
                self._buckets[client] = bucket
            return bucket

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [k for k, v in self._buckets.items() if now - v.last_refill > _BUCKET_TTL]
        for k in stale:
            del self._buckets[k]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._enabled:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        bucket = self._get_bucket(client)
        if not bucket.try_consume():
            request_id = getattr(request.state, "request_id", "")
            return JSONResponse(
                status_code=429,
                content={
                    "code": "rate_limited",
                    "message": "Too many requests",
                    "details": {"retry_after_seconds": round(1.0 / self._rate, 2)},
                    "request_id": request_id,
                },
                headers={
                    "Retry-After": str(int(1.0 / self._rate) + 1),
                    _REQUEST_ID_HEADER: request_id,
                },
            )
        return await call_next(request)


# ------------------------------------------------------------------
# Concurrency limiter
# ------------------------------------------------------------------


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Cap the number of in-flight requests.

    Env var:
        VOYAGER_MAX_CONCURRENT – max concurrent requests (0=disabled)
    """

    def __init__(self, app, max_concurrent: int = 0):
        super().__init__(app)
        limit = max_concurrent or int(os.environ.get("VOYAGER_MAX_CONCURRENT", "0"))
        self._semaphore = asyncio.Semaphore(limit) if limit > 0 else None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._semaphore is None:
            return await call_next(request)
        if self._semaphore.locked():
            request_id = getattr(request.state, "request_id", "")
            return JSONResponse(
                status_code=503,
                content={
                    "code": "overloaded",
                    "message": "Server is at capacity, try again shortly",
                    "request_id": request_id,
                },
                headers={_REQUEST_ID_HEADER: request_id},
            )
        async with self._semaphore:
            return await call_next(request)


# ------------------------------------------------------------------
# Structured error formatter
# ------------------------------------------------------------------


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def structured_error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    details: dict | None = None,
    request_id: str = "",
    detail: str | None = None,
    include_status_code: bool = False,
) -> JSONResponse:
    """Build a consistent JSON error envelope."""
    body: dict = {"code": code, "message": message}
    if details:
        body["details"] = _json_safe(details)
    if request_id:
        body["request_id"] = request_id
    if detail is not None:
        body["detail"] = detail
    if include_status_code:
        body["status_code"] = status_code
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={_REQUEST_ID_HEADER: request_id} if request_id else {},
    )
