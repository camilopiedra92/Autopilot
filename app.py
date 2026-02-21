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

import structlog
import uvicorn

from autopilot.version import VERSION, APP_NAME
import os
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

    yield

    # Shutdown — teardown connectors and workflows
    logger.info("platform_shutting_down")
    await registry.teardown_all()
    await connector_registry.teardown_all()
    logger.info("platform_shutdown_complete")


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(
    title="AutoPilot — AI Workflow Platform",
    description="Multi-workflow AI automation platform powered by Google ADK",
    version=VERSION,
    lifespan=lifespan,
)

cors_origins_str = os.getenv("API_CORS_ORIGINS", "*")
allowed_origins = [
    origin.strip() for origin in cors_origins_str.split(",") if origin.strip()
]

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
from autopilot.api.middleware import otel_tracing_middleware

app.add_middleware(BaseHTTPMiddleware, dispatch=otel_tracing_middleware)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
