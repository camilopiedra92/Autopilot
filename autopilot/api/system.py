from fastapi import APIRouter, Response
from pydantic import BaseModel

from autopilot.version import VERSION, APP_NAME
from autopilot.config import get_platform_settings as get_settings
from autopilot.registry import get_registry
from autopilot.observability import get_metrics, get_metrics_content_type

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    version: str
    platform: str
    model: str
    workflows_registered: int
    workflows_enabled: int
    workflow_details: dict


@router.get("/health", response_model=HealthResponse)
async def health():
    registry = get_registry()
    workflows = registry.list_all()

    # Collect health info from each workflow
    details = {}
    for info in workflows:
        wf = registry.get(info.name)
        if wf and hasattr(wf, "get_health_info"):
            details[info.name] = wf.get_health_info()
        else:
            details[info.name] = {"status": "registered"}

    return HealthResponse(
        status="healthy",
        version=VERSION,
        platform=APP_NAME,
        model=settings.model_name,
        workflows_registered=registry.count,
        workflows_enabled=sum(1 for w in workflows if w.enabled),
        workflow_details=details,
    )


@router.get("/")
async def root():
    return {
        "api": "AutoPilot Headless API",
        "version": VERSION,
        "status": "online"
    }


@router.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type=get_metrics_content_type())


# ── Gmail Watch Management ───────────────────────────────────────────


def _get_pubsub_connector():
    """Get the PubSub connector from the registry, or None."""
    try:
        from autopilot.connectors import get_connector_registry
        return get_connector_registry().get("pubsub")
    except Exception:
        return None


@router.get("/gmail/watch/status")
async def gmail_watch_status():
    """Diagnostic endpoint: returns the current Gmail watch state."""
    pubsub = _get_pubsub_connector()
    if not pubsub:
        return {"error": "PubSub connector not available", "active": False}
    return pubsub.watch_status


@router.post("/gmail/watch/renew")
async def gmail_watch_renew():
    """Force re-register the Gmail watch.

    Designed to be called by Cloud Scheduler (daily cron) to ensure
    the watch stays alive even when Cloud Run scales to zero and
    the in-process renewal loop dies.
    """
    pubsub = _get_pubsub_connector()
    if not pubsub:
        return {"error": "PubSub connector not available", "renewed": False}

    try:
        status = await pubsub.force_rewatch()
        return {"renewed": True, **status}
    except Exception as e:
        return {"renewed": False, "error": str(e)}


@router.post("/gmail/watch/stop")
async def gmail_watch_stop():
    """Force stop the Gmail watch.
    
    Useful when migrating environments or fixing the 'Only one user push notification 
    client allowed per developer' error.
    """
    pubsub = _get_pubsub_connector()
    if not pubsub:
        return {"error": "PubSub connector not available", "stopped": False}

    try:
        await pubsub.stop_watching()
        return {"stopped": True}
    except Exception as e:
        return {"stopped": False, "error": str(e)}

