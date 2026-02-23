"""
Platform-level observability — OpenTelemetry tracing + generic Prometheus metrics.

Provides:
- Distributed tracing with spans per agent stage
- Generic agent-level counters and histograms
- FastAPI middleware for request tracing
- Prometheus scraping utilities

Workflow-specific metrics (pipeline, YNAB, cache) live in each workflow's
own metrics module.
"""

import os
import time
import structlog
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = structlog.get_logger(__name__)

# ── OpenTelemetry Setup ──────────────────────────────────────────────

_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.

    Exporter hierarchy (3-tier):
      1. OTLP endpoint provided → OTLPSpanExporter
      2. Cloud Run (K_SERVICE set) → CloudTraceSpanExporter
      3. Default → ConsoleSpanExporter (local dev)

    Args:
        service_name: Name of the service (appears in traces).
                      Defaults to K_SERVICE env var (Cloud Run) or APP_NAME.
        otlp_endpoint: OTLP collector endpoint.
    """
    global _tracer

    from autopilot.version import APP_NAME, VERSION

    if service_name is None:
        service_name = os.getenv("K_SERVICE", APP_NAME.lower())

    resource = Resource.create(
        {"service.name": service_name, "service.version": VERSION}
    )
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("otel_otlp_configured", endpoint=otlp_endpoint)
        except ImportError:
            logger.warning("otel_otlp_unavailable_fallback_console")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif os.getenv("K_SERVICE"):
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            exporter = CloudTraceSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("otel_cloud_trace_configured", service=service_name)
        except ImportError:
            logger.warning("otel_cloud_trace_unavailable_fallback_console")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Console exporter for local development
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    logger.info("otel_tracing_initialized", service=service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get or initialize the tracer."""
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


@contextmanager
def trace_agent_stage(
    stage_name: str, pipeline_name: str = "autopilot", **attributes
) -> Generator:
    """
    Context manager to trace an agent pipeline stage.

    Usage:
        with trace_agent_stage("email_parser", pipeline_name="bank_to_ynab", email_length=len(body)):
            result = await email_parser.run(...)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"pipeline.{stage_name}",
        attributes={
            "agent.stage": stage_name,
            "agent.pipeline": pipeline_name,
            **{k: str(v) for k, v in attributes.items()},
        },
    ) as span:
        start = time.monotonic()
        try:
            yield span
            span.set_attribute("agent.status", "success")
        except Exception as e:
            span.set_attribute("agent.status", "error")
            span.set_attribute("agent.error", str(e))
            span.record_exception(e)
            raise
        finally:
            latency = (time.monotonic() - start) * 1000
            span.set_attribute("agent.latency_ms", round(latency))


# ── Generic Agent Metrics ────────────────────────────────────────────

AGENT_CALLS = Counter(
    "agent_calls_total",
    "Total calls per agent stage",
    ["agent_name", "status"],
    namespace="autopilot",
)

AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Per-agent latency",
    ["agent_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
    namespace="autopilot",
)


# ── Metric Factories ─────────────────────────────────────────────────


def create_pipeline_metrics(namespace: str) -> tuple[Counter, Histogram]:
    """
    Create standard pipeline execution metrics.

    Args:
        namespace: Prometheus namespace (e.g., "bank_to_ynab").

    Returns:
        (pipeline_requests, pipeline_latency)
    """
    requests = Counter(
        "pipeline_requests_total",
        "Total pipeline invocations",
        ["status"],
        namespace=namespace,
    )
    latency = Histogram(
        "pipeline_latency_seconds",
        "Pipeline end-to-end latency",
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
        namespace=namespace,
    )
    return requests, latency


def create_connector_metrics(
    namespace: str, connector_name: str
) -> tuple[Counter, Histogram]:
    """
    Create standard connector/API metrics.

    Args:
        namespace: Prometheus namespace (e.g., "bank_to_ynab").
        connector_name: Name of the connector (e.g., "ynab_api").

    Returns:
        (connector_requests, connector_latency)
    """
    requests = Counter(
        f"{connector_name}_requests_total",
        f"Total {connector_name} requests",
        ["method", "path", "status"],
        namespace=namespace,
    )
    latency = Histogram(
        f"{connector_name}_latency_seconds",
        f"{connector_name} request latency",
        ["method"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
        namespace=namespace,
    )
    return requests, latency


# ── Prometheus Scraping ──────────────────────────────────────────────


def get_metrics() -> bytes:
    """Generate Prometheus metrics for scraping."""
    return generate_latest()


def get_metrics_content_type() -> str:
    """Content type for Prometheus metrics response."""
    return CONTENT_TYPE_LATEST
