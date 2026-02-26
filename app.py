"""
AutoPilot v5.0.0 — Multi-Workflow AI Automation Platform

Single entry point for the entire platform. This file is GENERIC —
all workflow-specific logic lives inside each workflow package.

The platform:
  - Auto-discovers workflows from the workflows/ directory
  - Calls register_routes() so each workflow mounts its own endpoints
  - Provides the management dashboard at /dashboard
  - Exposes a unified API at /api/workflows

Run with:
  uvicorn app:app --reload --port 8080
"""

import os

# ── gRPC configuration ───────────────────────────────────────────────
# Disable false-positive fork warnings (async threads, not actual forks)
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import structlog
import uvicorn

from autopilot.version import VERSION, APP_NAME
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

from autopilot.config import get_platform_settings as get_settings

# Platform imports
from autopilot.registry import get_registry
from autopilot.connectors import get_connector_registry
from autopilot.errors import AutoPilotError
from autopilot.api.errors import autopilot_error_handler
from autopilot.api.v1.routes import router as v1_router
# ...

logger = structlog.get_logger(__name__)

settings = get_settings()


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Platform lifespan — discover workflows, setup, register routes, shutdown."""

    # 1. Initialize connectors (shared integration blocks)
    connector_registry = get_connector_registry()
    logger.info(
        "connectors_initialized",
        count=len(connector_registry),
        names=connector_registry.names,
    )

    # 1.5 Auto-expose connector methods as platform tools
    from autopilot.core.tools import register_all_connector_tools

    try:
        register_all_connector_tools(connector_registry)
    except Exception as e:
        logger.warning("tool_registration_failed_silently_continuing", error=str(e))

    # 2. Discover workflows
    registry = get_registry()
    discovered = registry.discover()

    logger.info(
        "platform_starting",
        version=VERSION,
        platform=APP_NAME,
        workflows_discovered=discovered,
        connectors=connector_registry.names,
    )

    # 3. Setup all connectors
    await connector_registry.setup_all()

    # 4. Setup all workflows (each handles its own services)
    await registry.setup_all()

    # 4. Each workflow registers its own routes
    for name in discovered:
        try:
            wf = registry.get(name)
            if wf:
                wf.register_routes(app)
                logger.info("workflow_routes_registered", workflow=name)
        except Exception as e:
            logger.error(
                "workflow_routes_registration_failed",
                workflow=name,
                error=str(e),
                exc_info=True,
            )

    logger.info(
        "platform_ready",
        workflows=registry.list_names(),
        workflow_count=registry.count,
    )

    # 5. Mount A2A Protocol Server (agent discovery + JSON-RPC)
    from autopilot.api.a2a import mount_a2a_routes

    mount_a2a_routes(app, registry)

    yield

    # Shutdown — teardown connectors and workflows
    logger.info("platform_shutting_down")
    await registry.teardown_all()
    await connector_registry.teardown_all()
    logger.info("platform_shutdown_complete")


# ── OpenAPI Tag Metadata ─────────────────────────────────────────────
# Groups endpoints in /docs and /redoc for developer navigation.
OPENAPI_TAGS = [
    {
        "name": "Workflows",
        "description": "Discover, inspect, and manage registered workflows. "
        "Each workflow is a self-contained automation pipeline with its own "
        "agents, triggers, and execution history.",
    },
    {
        "name": "Runs",
        "description": "Durable run history powered by `RunLogService`. "
        "Supports cursor-based pagination, per-step artifact traces, and "
        "aggregate stats (total, success rate).",
    },
    {
        "name": "Events",
        "description": "EventBus history and real-time SSE stream. "
        "Subscribe to live platform events via Server-Sent Events with "
        "automatic reconnection and `Last-Event-ID` replay.",
    },
    {
        "name": "HITL",
        "description": "Human-In-The-Loop controls. List paused runs awaiting "
        "human action and resume them with custom payloads.",
    },
    {
        "name": "Copilot",
        "description": "Platform observability meta-agent. Ask natural language "
        "questions about workflow failures, run history, and event timelines. "
        "Powered by a ReAct reasoning loop with read-only tools.",
    },
    {
        "name": "System",
        "description": "Health checks, Prometheus metrics, and platform info.",
    },
    {
        "name": "Webhooks",
        "description": "Inbound event adapters for external triggers. "
        "Thin adapters that publish typed events to the EventBus — "
        "workflows subscribe and react independently.",
    },
    {
        "name": "Gmail Watch",
        "description": "Gmail push notification lifecycle management. "
        "Renew, inspect, and stop Gmail API watches for event-driven workflows.",
    },
]

API_DESCRIPTION = """\
## Autopilot Headless API

A **pure backend API** (JSON + SSE) for orchestrating multi-agent AI workflows.
There is no internal frontend — the API is the product.

### Authentication

All `/api/v1/*` endpoints require the `X-API-Key` header:

```
X-API-Key: <your-api-key>
```

The key is validated against the `API_KEY_SECRET` environment variable using
timing-safe comparison ([HMAC](https://docs.python.org/3/library/hmac.html)).

### Base URL

| Environment | Base URL |
|---|---|
| Local | `http://localhost:8080` |
| Cloud Run | `https://<service>-<hash>.run.app` |

### Error Format

All errors return a structured JSON envelope:

```json
{
  "error": {
    "error_code": "WORKFLOW_NOT_FOUND",
    "message": "Workflow 'xyz' not found",
    "detail": "Available: ['bank_to_ynab', 'conversational_assistant']",
    "retryable": false,
    "http_status": 404
  }
}
```

Every error carries `retryable` (boolean) and `error_code` (machine-readable)
for automated retry decisions.

### Real-Time Events

Connect to `/api/v1/events/stream` via SSE for live platform events.
The stream supports `Last-Event-ID` replay and sends keepalive pings every 30s.
Connections are intentionally closed after 5 minutes (Edge LB safety) —
clients should auto-reconnect.
"""

# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="AutoPilot — AI Workflow Platform",
    description=API_DESCRIPTION,
    version=VERSION,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    license_info={"name": "Private", "identifier": "LicenseRef-Private"},
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────
# Headless API: CORS is disabled by default (empty = no browser
# origins allowed).  Opt-in via API_CORS_ORIGINS="https://admin.example.com".
cors_origins_str = os.getenv("API_CORS_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in cors_origins_str.split(",") if origin.strip()
]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register global exception handler
app.add_exception_handler(AutoPilotError, autopilot_error_handler)

# Mount webhooks
from autopilot.api.webhooks import router as webhooks_router

app.include_router(webhooks_router)

# Mount generic platform routes (health, metrics, root)
from autopilot.api.system import router as system_router

app.include_router(system_router)

# Mount versioned API routes
app.include_router(v1_router)


# ── Middlewares ───────────────────────────────────────────
from autopilot.api.middleware import (
    otel_tracing_middleware,
    rate_limit_middleware,
    api_versioning_middleware,
)

app.add_middleware(BaseHTTPMiddleware, dispatch=otel_tracing_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=api_versioning_middleware)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
