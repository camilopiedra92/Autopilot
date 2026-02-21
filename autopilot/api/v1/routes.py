"""
V1 API Routes — Public-facing endpoints for the AutoPilot platform.

Provides programmatic access to execute workflows, query state, and manage runs.
All routes are protected by the X-API-Key header.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Body
import structlog

from autopilot.auth.api_security import get_api_key
from autopilot.registry import get_registry
from autopilot.router import get_router

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_api_key)])


@router.get("/workflows")
async def list_workflows():
    """List all registered workflows along with their status and trigger info."""
    registry = get_registry()
    workflows = registry.list_all()
    
    return {
        "workflows": [
            {
                "id": info.name,
                "name": info.display_name,
                "version": info.version,
                "enabled": info.enabled,
                "description": info.description,
                "triggers": [t.model_dump() for t in info.triggers],
                "tags": info.tags
            }
            for info in workflows
        ],
        "total": registry.count
    }


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get the full manifest details of a specific workflow."""
    registry = get_registry()
    wf = registry.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    return wf.manifest.model_dump()


@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    input_data: Dict[str, Any] = Body(default_factory=dict)
):
    """
    Programmatically trigger a workflow execution.
    This simulates a manual invocation with the provided input payload.
    """
    router_svc = get_router()
    try:
        run = await router_svc.route_manual(workflow_id, input_data)
        return {
            "status": "success",
            "run_id": run.id,
            "workflow_id": run.workflow_id,
            "run_status": run.status.value,
            "result": run.result,
            "error": run.error
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("api_execution_failed", workflow=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")


@router.get("/workflows/{workflow_id}/runs")
async def list_workflow_runs(workflow_id: str):
    """Get the recent runs for a specific workflow."""
    registry = get_registry()
    wf = registry.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    return {
        "workflow_id": workflow_id,
        "runs": [
            {
                "id": run.id,
                "status": run.status.value,
                "duration_ms": run.duration_ms,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "error": run.error,
                "trigger_type": run.trigger_type.value
            }
            for run in wf.recent_runs
        ]
    }


@router.get("/workflows/{workflow_id}/runs/{run_id}")
async def get_workflow_run(workflow_id: str, run_id: str):
    """Get details of a specific execution run."""
    registry = get_registry()
    wf = registry.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    for run in wf.recent_runs:
        if run.id == run_id:
            return run.model_dump()
            
    raise HTTPException(status_code=404, detail=f"Run {run_id} not found for workflow {workflow_id}")
