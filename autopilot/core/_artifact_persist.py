"""
Shared artifact persistence helper for Pipeline and DAG engines.

Provides a single ``persist_node_artifact`` coroutine that both the
sequential Pipeline and DAG runners call after each step/node completes.

Design decisions:
  - Fire-and-forget: failures are logged but never block execution.
  - JSON-safe serialization handles Pydantic models, datetimes, etc.
  - Each artifact includes metadata (node name, engine, duration_ms).
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from google.genai import types

from autopilot.core.context import AgentContext

logger = structlog.get_logger(__name__)


async def persist_node_artifact(
    ctx: AgentContext,
    *,
    engine_name: str,
    node_name: str,
    output: dict[str, Any],
    duration_ms: float,
) -> None:
    """Persist a step/node's output as a versioned JSON artifact.

    Saves ``{node_name}.json`` scoped to the current execution_id.
    Never blocks the pipeline — failures are logged and swallowed.

    Args:
        ctx: The current execution context (provides artifact_service).
        engine_name: "dag" or "pipeline" — included in the artifact envelope.
        node_name: Name of the step/node whose output is being saved.
        output: The dict returned by the step/node.
        duration_ms: How long the step/node took to execute.
    """
    try:
        payload = {
            "node": node_name,
            "engine": engine_name,
            "execution_id": ctx.execution_id,
            "pipeline": ctx.pipeline_name,
            "duration_ms": duration_ms,
            "output": _json_safe(output),
        }
        artifact = types.Part(text=json.dumps(payload, ensure_ascii=False))
        version = await ctx.save_artifact(f"{node_name}.json", artifact)
        ctx.logger.debug(
            "artifact_persisted",
            node=node_name,
            filename=f"{node_name}.json",
            version=version,
        )
    except Exception as exc:
        # Never block the pipeline for artifact failures
        ctx.logger.warning(
            "artifact_persist_failed",
            node=node_name,
            error=str(exc),
        )


def _json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
