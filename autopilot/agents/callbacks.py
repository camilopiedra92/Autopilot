"""Platform Callbacks — Observability and composition for ADK agent callbacks.

Provides:
  - before_model_logger / after_model_logger: structured logging + Prometheus + SSE + cost tracking
  - pipeline_session_id ContextVar: async-safe session tracking
  - create_chained_before_callback / create_chained_after_callback: callback composition
  - create_budget_guardrail: budget-based LLM call gating
"""

from __future__ import annotations

import time
import asyncio
import contextvars
import structlog
from typing import Optional, Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest

from autopilot.observability import (
    AGENT_CALLS,
    AGENT_LATENCY,
    TOKEN_USAGE,
    ESTIMATED_COST_USD,
)
from autopilot.core.cost import get_cost_tracker

logger = structlog.get_logger(__name__)

# ── Async-safe call timing using ContextVar ──────────────────────────
# Stores a dict[agent_name, start_time] per async context so concurrent
# requests don't clobber each other's timing data.
_call_start_times: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar(
    "_call_start_times", default=None
)

# ── Event bus session ID for the current pipeline context ────────────
# Set by ADKRunner before invoking the pipeline so callbacks
# can publish events with the correct correlation ID.
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


def _publish_event_async(topic: str, payload: dict) -> None:
    """
    Fire-and-forget publish an event to the unified EventBus.

    Non-blocking: creates a task if a running loop exists,
    otherwise silently skips (e.g. in tests or sync contexts).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # No loop — skip (sync context / tests)

    async def _do_publish():
        try:
            from autopilot.core.bus import get_event_bus

            bus = get_event_bus()
            session_id = pipeline_session_id.get(None)
            await bus.publish(
                topic,
                {**payload, "session_id": session_id or ""},
                sender="adk_callback",
            )
        except Exception:
            pass  # Never let event bus errors break the pipeline

    loop.create_task(_do_publish())


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

    # Publish "started" event to unified bus
    _publish_event_async(
        "model.started",
        {
            "agent": agent_name,
            "tools_available": tool_count,
            "context_messages": message_count,
        },
    )

    return None  # Always proceed


def after_model_logger(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Logs LLM response with latency metrics, cost tracking, and Prometheus."""
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

    # ── Cost tracking (usage_metadata extraction) ─────────────────────
    usage = llm_response.usage_metadata
    prompt_tokens = 0
    candidates_tokens = 0
    cached_tokens = 0

    if usage is not None:
        prompt_tokens = usage.prompt_token_count or 0
        candidates_tokens = usage.candidates_token_count or 0
        cached_tokens = usage.cached_content_token_count or 0

        # Record in the per-execution CostTracker
        model_name = getattr(llm_response, "model_version", "") or ""
        tracker = get_cost_tracker()
        cost_before = tracker.estimated_cost_usd
        tracker.record(agent_name, usage, model_name)
        call_cost = tracker.estimated_cost_usd - cost_before

        # ── Prometheus token metrics ──────────────────────────────────
        TOKEN_USAGE.labels(agent_name=agent_name, token_type="prompt").inc(
            prompt_tokens
        )
        TOKEN_USAGE.labels(agent_name=agent_name, token_type="candidates").inc(
            candidates_tokens
        )
        if cached_tokens:
            TOKEN_USAGE.labels(agent_name=agent_name, token_type="cached").inc(
                cached_tokens
            )
        if call_cost > 0:
            ESTIMATED_COST_USD.labels(agent_name=agent_name, model=model_name).inc(
                call_cost
            )

    logger.info(
        "llm_call_completed",
        agent=agent_name,
        latency_ms=latency_ms,
        response_length=response_length,
        has_tool_calls=has_tool_calls,
        prompt_tokens=prompt_tokens,
        candidates_tokens=candidates_tokens,
        cached_tokens=cached_tokens,
    )

    # ── Prometheus latency metrics ────────────────────────────────────
    AGENT_CALLS.labels(agent_name=agent_name, status="success").inc()
    AGENT_LATENCY.labels(agent_name=agent_name).observe(latency_s)

    # ── Publish "completed" event to unified bus ──────────────────────
    _publish_event_async(
        "model.completed",
        {
            "agent": agent_name,
            "latency_ms": latency_ms,
            "response_length": response_length,
            "has_tool_calls": has_tool_calls,
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "cached_tokens": cached_tokens,
        },
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Budget Guardrail Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_budget_guardrail(max_cost_usd: float) -> BeforeCallback:
    """Create a before_model_callback that blocks LLM calls when budget exceeded.

    When the accumulated cost of the current execution exceeds
    ``max_cost_usd``, the callback returns a blocked ``LlmResponse``
    with an error message instead of allowing the LLM call to proceed.

    Args:
        max_cost_usd: Maximum allowed cost in USD for this execution.

    Returns:
        A ``BeforeCallback`` suitable for ``create_chained_before_callback``.
    """
    from google.genai import types as genai_types

    def budget_guardrail(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        tracker = get_cost_tracker()
        current_cost = tracker.estimated_cost_usd

        if current_cost >= max_cost_usd:
            agent_name = callback_context.agent_name
            logger.warning(
                "budget_exceeded",
                agent=agent_name,
                current_cost_usd=round(current_cost, 6),
                max_cost_usd=max_cost_usd,
                llm_calls=tracker.llm_calls,
            )
            _publish_event_async(
                "budget.exceeded",
                {
                    "agent": agent_name,
                    "current_cost_usd": round(current_cost, 6),
                    "max_cost_usd": max_cost_usd,
                },
            )
            return LlmResponse(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            text=f"Budget exceeded: ${current_cost:.4f} >= ${max_cost_usd:.4f}. "
                            f"LLM call blocked for agent '{agent_name}'."
                        )
                    ],
                ),
                error_code="BUDGET_EXCEEDED",
                error_message=f"Execution budget of ${max_cost_usd} exceeded.",
            )

        return None  # Under budget — proceed

    return budget_guardrail
