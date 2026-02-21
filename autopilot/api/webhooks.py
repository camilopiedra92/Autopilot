"""
Platform Routes — Unified HTTP endpoints for the Autopilot platform.

Handles:
  - Generic Webhooks: POST /api/webhook/{path}
  - Gmail Push: POST /gmail/webhook (Pub/Sub)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, Body, Depends
from typing import Any

from autopilot.router import get_router
from autopilot.connectors import get_connector_registry

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── Generic Webhook ──────────────────────────────────────────────────

@router.post("/api/webhook/{webhook_path:path}")
async def generic_webhook(webhook_path: str, request: Request):
    """
    Route a webhook request to the workflow that handles the given path.
    ID is resolved via TriggerConfig.path matches.
    """
    # Parse body (JSON or Form)
    try:
        data = await request.json()
    except Exception:
        data = {}

    # Add headers/query for completeness if needed
    data.update({
        "_headers": dict(request.headers),
        "_query": dict(request.query_params),
    })

    router_svc = get_router()
    try:
        # Prepend slash if missing, or handle strict matching
        path_key = f"/{webhook_path}" if not webhook_path.startswith("/") else webhook_path
        
        run = await router_svc.route_webhook(path_key, data)
        return {
            "status": "success",
            "run_id": run.id,
            "workflow_id": run.workflow_id,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No workflow found for webhook path: {path_key}")
    except Exception as e:
        logger.error("webhook_failed", path=webhook_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Gmail Pub/Sub Webhook ────────────────────────────────────────────

@router.post("/gmail/webhook")
async def gmail_push_webhook(request: Request):
    """
    Handle Google Cloud Pub/Sub push notifications for Gmail.
    
    Flow:
      1. Receive Pub/Sub message
      2. PubSubConnector decodes & fetches new emails
      3. For each email, WorkflowRouter routes to matching workflows
    """
    try:
        pubsub_message = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. Decode & Fetch
    registry = get_connector_registry()
    try:
        pubsub = registry.get("pubsub")
    except KeyError:
        raise HTTPException(status_code=503, detail="PubSub connector not initialized")

    try:
        # Returns list of dicts: {'id', 'snippet', 'body', 'from', 'subject', ...}
        emails = await pubsub.handle_notification(pubsub_message)
    except Exception as e:
        logger.error("pubsub_handler_failed", error=str(e))
        # Return 200 to acknowledge Pub/Sub (otherwise it retries endlessly)
        return {"status": "error", "message": "Handler failed, acking to stop retry"}

    if not emails:
        return {"status": "ignored", "reason": "no_new_emails"}

    # 2. Route
    router_svc = get_router()
    summary = {"processed": 0, "triggered_runs": 0}
    
    for email in emails:
        # route_gmail_push now returns list[WorkflowRun]
        runs = await router_svc.route_gmail_push(email)
        summary["processed"] += 1
        summary["triggered_runs"] += len(runs)

    return {"status": "ok", "summary": summary}



