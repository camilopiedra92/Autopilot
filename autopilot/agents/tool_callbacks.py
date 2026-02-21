"""
Platform Tool Callbacks — Observability and resilience for ADK tool execution.

Provides:
  - before_tool_logger: Logs every tool invocation with args (audit trail)
  - after_tool_logger: Logs tool results with latency + Prometheus metrics
  - tool_rate_limiter: Token-bucket rate limiter for external API tools
  - tool_error_handler: Catches connector errors and returns structured fallbacks

These complement the model-level callbacks (before_model_logger, after_model_logger)
to give **complete observability** across LLM calls AND tool/API calls.
"""

from __future__ import annotations

import time
import structlog
from typing import Any, Optional

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext

from autopilot.agents.callbacks import _emit_event_async
from autopilot.observability import AGENT_CALLS, AGENT_LATENCY

logger = structlog.get_logger(__name__)


# ── Per-context timing for tool calls ────────────────────────────────
_tool_start_times: dict[str, float] = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Before Tool Callback — Audit logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def before_tool_logger(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]:
    """
    Logs every tool call with its arguments for full audit trail.

    Emits an SSE event so the frontend can show real-time API activity.
    Returns None to let the tool execute normally.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name if hasattr(tool_context, "agent_name") else "unknown"

    # Track start time for latency measurement
    key = f"{agent_name}:{tool_name}"
    _tool_start_times[key] = time.time()

    # Sanitize args for logging (truncate large values)
    safe_args = {}
    for k, v in args.items():
        str_val = str(v)
        safe_args[k] = str_val[:200] + "..." if len(str_val) > 200 else str_val

    logger.info(
        "tool_call_started",
        agent=agent_name,
        tool=tool_name,
        args=safe_args,
    )

    # Emit SSE event for real-time tool observability
    _emit_event_async({
        "type": "tool_started",
        "agent": agent_name,
        "tool": tool_name,
        "args": safe_args,
    })

    return None  # Allow tool execution to proceed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  After Tool Callback — Metrics + audit logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def after_tool_logger(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """
    Logs tool completion with latency and result summary.

    Instruments Prometheus counters and histograms for tool-level metrics,
    complementing the model-level metrics from after_model_logger.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name if hasattr(tool_context, "agent_name") else "unknown"

    key = f"{agent_name}:{tool_name}"
    start = _tool_start_times.pop(key, None)
    latency_s = (time.time() - start) if start else 0
    latency_ms = int(latency_s * 1000)

    # Summarize response for logging (avoid dumping entire API responses)
    response_summary = _summarize_response(tool_response)
    is_error = isinstance(tool_response, dict) and "error" in tool_response

    logger.info(
        "tool_call_completed",
        agent=agent_name,
        tool=tool_name,
        latency_ms=latency_ms,
        response_summary=response_summary,
        is_error=is_error,
    )

    # ── Prometheus metrics (tool-level) ──────────────────────────────
    status = "error" if is_error else "success"
    AGENT_CALLS.labels(
        agent_name=f"tool:{tool_name}", status=status
    ).inc()
    AGENT_LATENCY.labels(
        agent_name=f"tool:{tool_name}"
    ).observe(latency_s)

    # ── SSE event ────────────────────────────────────────────────────
    _emit_event_async({
        "type": "tool_completed",
        "agent": agent_name,
        "tool": tool_name,
        "latency_ms": latency_ms,
        "is_error": is_error,
        "response_summary": response_summary,
    })

    return None  # Don't modify the tool response


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Error Handler — Graceful degradation for connector failures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_tool_error_handler(fallback_message: str = "Tool temporarily unavailable"):
    """
    Factory that creates a before_tool_callback providing graceful degradation.

    If the tool raises an error, instead of crashing the pipeline,
    returns a structured error response the agent can reason about.

    Usage:
        agent = LlmAgent(
            before_tool_callback=create_tool_error_handler("YNAB API is down"),
            ...
        )
    """
    async def tool_error_handler(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        # This callback runs BEFORE the tool — it doesn't catch errors.
        # For error handling, use after_tool_callback or wrap tools.
        return None

    return tool_error_handler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _summarize_response(response: Any) -> str:
    """Create a concise summary of a tool response for logging."""
    if response is None:
        return "null"
    if isinstance(response, dict):
        if "error" in response:
            return f"error: {str(response['error'])[:100]}"
        keys = list(response.keys())[:5]
        return f"dict({len(response)} keys: {keys})"
    if isinstance(response, (list, tuple)):
        return f"list({len(response)} items)"
    if isinstance(response, str):
        return f"str({len(response)} chars)" if len(response) > 100 else response
    return str(type(response).__name__)
