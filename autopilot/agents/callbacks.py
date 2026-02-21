"""
Platform Callbacks — Observability and composition for ADK agent callbacks.

Provides:
  - before_model_logger / after_model_logger: structured logging + Prometheus + SSE
  - pipeline_session_id ContextVar: async-safe session tracking
  - create_chained_before_callback / create_chained_after_callback: callback composition
"""

from __future__ import annotations

import time
import asyncio
import contextvars
import structlog
from typing import Optional, Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

from autopilot.observability import AGENT_CALLS, AGENT_LATENCY

logger = structlog.get_logger(__name__)

# ── Async-safe call timing using ContextVar ──────────────────────────
# Stores a dict[agent_name, start_time] per async context so concurrent
# requests don't clobber each other's timing data.
_call_start_times: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar(
    "_call_start_times", default=None
)

# ── Event bus session ID for the current pipeline context ────────────
# Set by PipelineRunner before invoking the pipeline so callbacks
# can emit events to the correct SSE stream.
pipeline_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pipeline_session_id", default=None
)


def _get_times() -> dict[str, float]:
    """Get or initialize the per-context timing dict."""
    times = _call_start_times.get(None)
    if times is None:
        times = {}
        _call_start_times.set(times)
    return times


def _emit_event_async(event: dict) -> None:
    """
    Fire-and-forget emit an event to the pipeline's event bus.

    Non-blocking: creates a task if a running loop exists,
    otherwise silently skips (e.g. in tests or sync contexts).
    """
    session_id = pipeline_session_id.get(None)
    if not session_id:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # No loop — skip (sync context / tests)

    async def _emit():
        try:
            from autopilot.services.event_bus import get_event_bus

            bus = get_event_bus()
            await bus.emit(session_id, event)
        except Exception:
            pass  # Never let event bus errors break the pipeline

    loop.create_task(_emit())


def before_model_logger(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Logs every LLM call with request details for observability."""
    agent_name = callback_context.agent_name
    times = _get_times()
    times[agent_name] = time.time()

    # Count tools available
    tools = getattr(llm_request, "tools", None)
    tool_count = len(tools) if tools else 0

    # Count messages in context
    contents = getattr(llm_request, "contents", None)
    message_count = len(contents) if contents else 0

    logger.info(
        "llm_call_started",
        agent=agent_name,
        tools_available=tool_count,
        context_messages=message_count,
    )

    # Emit "started" event to SSE stream
    _emit_event_async(
        {
            "type": "stage_started",
            "stage": agent_name,
            "status": "running",
            "tools_available": tool_count,
            "context_messages": message_count,
        }
    )

    return None  # Always proceed


def after_model_logger(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Logs LLM response with latency metrics and instruments Prometheus."""
    agent_name = callback_context.agent_name
    times = _get_times()
    start = times.pop(agent_name, None)
    latency_s = (time.time() - start) if start else 0
    latency_ms = int(latency_s * 1000)

    # Check if response includes tool calls
    has_tool_calls = False
    response_length = 0
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.function_call:
                has_tool_calls = True
            if part.text:
                response_length += len(part.text)

    logger.info(
        "llm_call_completed",
        agent=agent_name,
        latency_ms=latency_ms,
        response_length=response_length,
        has_tool_calls=has_tool_calls,
    )

    # ── Prometheus metrics ────────────────────────────────────────────
    AGENT_CALLS.labels(agent_name=agent_name, status="success").inc()
    AGENT_LATENCY.labels(agent_name=agent_name).observe(latency_s)

    # ── Emit "completed" event to SSE stream ──────────────────────────
    _emit_event_async(
        {
            "type": "stage_completed",
            "stage": agent_name,
            "status": "completed",
            "latency_ms": latency_ms,
            "response_length": response_length,
            "has_tool_calls": has_tool_calls,
        }
    )

    return None  # Never modify the response


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Callback Composition Factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BeforeCallback = Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]
AfterCallback = Callable[[CallbackContext, LlmResponse], Optional[LlmResponse]]


def create_chained_before_callback(
    *callbacks: BeforeCallback,
) -> BeforeCallback:
    """
    Creates a composite before_model_callback that chains multiple callbacks.

    Each callback is called in order. If any callback returns a non-None
    LlmResponse (i.e., it blocks the request), that response is returned
    immediately and subsequent callbacks are skipped.
    """

    def chained_before_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        for cb in callbacks:
            result = cb(callback_context, llm_request)
            if result is not None:
                return result  # Blocked — short-circuit
        return None  # All passed

    return chained_before_callback


def create_chained_after_callback(
    *callbacks: AfterCallback,
) -> AfterCallback:
    """
    Creates a composite after_model_callback that chains multiple callbacks.

    Each callback is called in order. If any callback returns a non-None
    LlmResponse (i.e., it modifies/blocks the response), that modified
    response is returned immediately.
    """

    def chained_after_callback(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        for cb in callbacks:
            result = cb(callback_context, llm_response)
            if result is not None:
                return result  # Modified/blocked
        return None  # All passed, no modifications

    return chained_after_callback
