import os
import time
from collections import defaultdict
from uuid import uuid4
from fastapi import Request
from fastapi.responses import JSONResponse

from autopilot.observability import get_tracer


async def otel_tracing_middleware(request: Request, call_next):
    """
    OpenTelemetry Tracing Middleware for FastAPI.
    Wraps HTTP requests in spans and measures latency.
    """
    tracer = get_tracer()
    request_id = request.headers.get("X-Request-ID", uuid4().hex[:16])
    path = request.url.path
    method = request.method

    if path in ("/health", "/metrics"):
        return await call_next(request)

    with tracer.start_as_current_span(
        f"http.{method.lower()}.{path}",
        attributes={
            "http.method": method,
            "http.path": path,
            "http.query": str(request.url.query),
            "http.request_id": request_id,
        },
    ) as span:
        start = time.monotonic()
        try:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.set_attribute(
                "http.latency_ms", round((time.monotonic() - start) * 1000)
            )


# ── Rate Limiting Middleware ─────────────────────────────────────────


_RATE_LIMIT_RPM = int(os.getenv("API_RATE_LIMIT_RPM", "60"))
_RATE_WINDOW = 60.0  # seconds
_rate_store: dict[str, list[float]] = defaultdict(list)


async def rate_limit_middleware(request: Request, call_next):
    """Sliding-window rate limiter per client IP.

    Reads API_RATE_LIMIT_RPM env var (default: 60 requests/min).
    Set to 0 to disable rate limiting entirely.
    Skips /health, /metrics, and webhook paths.
    """
    if _RATE_LIMIT_RPM <= 0:
        return await call_next(request)

    path = request.url.path
    if (
        path in ("/health", "/metrics")
        or path.startswith("/gmail/")
        or path.startswith("/telegram/")
    ):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()

    # Prune expired entries
    _rate_store[client_ip] = [
        t for t in _rate_store[client_ip] if now - t < _RATE_WINDOW
    ]

    if len(_rate_store[client_ip]) >= _RATE_LIMIT_RPM:
        retry_after = int(_RATE_WINDOW - (now - _rate_store[client_ip][0])) + 1
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded ({_RATE_LIMIT_RPM} requests/min)",
                    "detail": f"Retry after {retry_after} seconds",
                    "retryable": True,
                    "http_status": 429,
                }
            },
            headers={"Retry-After": str(retry_after)},
        )

    _rate_store[client_ip].append(now)
    return await call_next(request)


# ── API Versioning Middleware ────────────────────────────────────────


async def api_versioning_middleware(request: Request, call_next):
    """Inject API version and docs headers on /api/v1/* responses.

    Headers added:
      - API-Version: v1
      - X-API-Docs: /docs
    """
    response = await call_next(request)

    if request.url.path.startswith("/api/v1"):
        response.headers["API-Version"] = "v1"
        response.headers["X-API-Docs"] = "/docs"

    return response
