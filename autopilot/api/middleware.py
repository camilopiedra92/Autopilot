import time
from uuid import uuid4
from fastapi import Request

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
