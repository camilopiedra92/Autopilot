"""
ADKRunner — Platform-level ADK Runtime Bridge.

Wraps Google ADK's `Runner` + `SessionService` into a single entry point
for executing any native ADK agent (LlmAgent, SequentialAgent, ParallelAgent,
LoopAgent, Custom agents).

This is the ONLY way ADK agents are executed within the platform. It adds:
  - Retry with exponential backoff (503, 429, timeout resilience)
  - EventBus integration (pipeline.adk_started, adk_completed, error)
  - OpenTelemetry tracing spans
  - Structured result extraction (PipelineResult)
  - Function call response handling (Native Output Schema)

Usage:
    from autopilot.core.adk_runner import get_adk_runner

    runner = get_adk_runner()
    result = await runner.run(
        agent=my_adk_agent,
        message="Process this input...",
        initial_state={"key": "value"},
    )
    print(result.parsed_json)
"""

from __future__ import annotations

import time
import structlog
from typing import Any
from uuid import uuid4

from opentelemetry import trace

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from autopilot.agents.callbacks import pipeline_session_id
from autopilot.agents.json_utils import extract_json
from autopilot.errors import PipelineEmptyResponseError
from autopilot.models import PipelineResult
from autopilot.utils.resilience import retry_with_backoff
from google.api_core.exceptions import (
    ServiceUnavailable,
    ResourceExhausted,
    DeadlineExceeded,
)
from autopilot.errors import LLMRateLimitError, ConnectorError, ToolExecutionError
from autopilot.core.bus import get_event_bus

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ADKRunner:
    def __init__(self, app_name: str, user_id: str):
        self._app_name = app_name
        self._user_id = user_id
        self._session_service = InMemorySessionService()

    @retry_with_backoff(
        retries=3,
        initial_delay=2.0,
        max_delay=15.0,
        backoff_factor=2.0,
        retryable_exceptions=(
            ServiceUnavailable,  # 503 from Google AI
            ResourceExhausted,  # 429 Quota Exceeded / Rate Limit
            DeadlineExceeded,  # 504 Gateway Timeout
            LLMRateLimitError,  # Custom wrapper for rate limits
            ConnectorError,  # Transient connector errors (5xx, 429)
            ToolExecutionError,  # Tool failures may be transient
            ConnectionError,  # Network blips
            TimeoutError,  # Asyncio timeouts
        ),
    )
    async def run(
        self,
        pipeline,
        message: str,
        *,
        initial_state: dict[str, Any] | None = None,
        stream_session_id: str | None = None,
    ) -> PipelineResult:
        """
        Run an ADK agent and return a structured result.

        Args:
            pipeline: An ADK agent (LlmAgent, SequentialAgent, etc.).
            message: The user message to send to the agent.
            initial_state: Optional initial session state.
            stream_session_id: If provided, used as the SSE event bus session ID.

        Returns:
            PipelineResult with final text, parsed JSON, session state, and timing.
        """
        session_id = f"pipeline_{uuid4().hex[:12]}"
        effective_stream_id = stream_session_id or session_id

        return await self._run_adk_agent(
            pipeline, message, initial_state, effective_stream_id
        )

    async def _run_adk_agent(
        self,
        pipeline,
        message: str,
        initial_state: dict[str, Any] | None,
        effective_stream_id: str,
    ) -> PipelineResult:
        """Execute the ADK agent through Runner + SessionService."""
        with tracer.start_as_current_span(
            "adk_runner.run",
            attributes={
                "app_name": self._app_name,
                "agent_name": getattr(pipeline, "name", "unknown"),
            },
        ) as span:
            session_id = f"pipeline_{uuid4().hex[:12]}"
            span.set_attribute("session_id", session_id)
            bus = get_event_bus()

        # Set the event bus session context so callbacks can stream events
        token = pipeline_session_id.set(effective_stream_id)

        start = time.monotonic()

        try:
            runner = Runner(
                app_name=self._app_name,
                agent=pipeline,
                session_service=self._session_service,
            )

            # Create a fresh session with optional initial state
            session = await self._session_service.create_session(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session_id,
                state=initial_state or {},
            )

            # Build user message
            user_message = types.Content(
                role="user",
                parts=[types.Part(text=message)],
            )

            logger.info(
                "adk_runner_started",
                app_name=self._app_name,
                session_id=session_id,
                stream_session_id=effective_stream_id,
                message_length=len(message),
                agent_name=getattr(pipeline, "name", "unknown"),
            )

            # Publish pipeline_started event to unified bus
            await bus.publish(
                "pipeline.adk_started",
                {
                    "session_id": session_id,
                    "agent_name": getattr(pipeline, "name", "unknown"),
                },
                sender="adk_runner",
            )

            span.add_event("adk_runner_started")

            # Run the agent
            final_text = ""
            async for event in runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        # Extract text parts
                        text_parts = [p.text for p in event.content.parts if p.text]
                        if text_parts:
                            final_text = "\n".join(text_parts)

                        # Extract function call parts (Native Output Schema)
                        if not final_text:
                            for p in event.content.parts:
                                if p.function_call and p.function_call.args:
                                    try:
                                        import json

                                        fc_data = dict(p.function_call.args)
                                        final_text = json.dumps(fc_data)
                                        break
                                    except Exception as e:
                                        logger.warning(
                                            "failed_to_serialize_func_call",
                                            error=str(e),
                                        )

            elapsed_ms = round((time.monotonic() - start) * 1000, 2)

            if not final_text:
                await bus.publish(
                    "pipeline.error",
                    {
                        "error": "Pipeline returned no final response.",
                        "session_id": session_id,
                    },
                    sender="adk_runner",
                )
                raise PipelineEmptyResponseError(
                    "Pipeline returned no final response.",
                    detail=f"agent={getattr(pipeline, 'name', 'unknown')}, session={session_id}",
                )

            # Extract JSON with robust fallback
            try:
                parsed_json = extract_json(final_text)
            except ValueError:
                parsed_json = {}

            # Retrieve final session state
            session = await self._session_service.get_session(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            final_state = (
                dict(session.state) if session and hasattr(session, "state") else {}
            )

            logger.info(
                "adk_runner_completed",
                app_name=self._app_name,
                session_id=session_id,
                agent_name=getattr(pipeline, "name", "unknown"),
                duration_ms=elapsed_ms,
                final_text_length=len(final_text),
                has_parsed_json=bool(parsed_json),
            )

            # Publish pipeline_completed event to unified bus
            await bus.publish(
                "pipeline.adk_completed",
                {
                    "session_id": session_id,
                    "duration_ms": elapsed_ms,
                },
                sender="adk_runner",
            )

            return PipelineResult(
                session_id=session_id,
                final_text=final_text,
                parsed_json=parsed_json,
                state=final_state,
                duration_ms=elapsed_ms,
            )

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            await bus.publish(
                "pipeline.error",
                {
                    "error": str(e),
                    "session_id": effective_stream_id,
                },
                sender="adk_runner",
            )
            raise

        finally:
            # Reset pipeline context
            pipeline_session_id.reset(token)


# ── Singleton ─────────────────────────────────────────────────────────

_runners: dict[str, ADKRunner] = {}


def get_adk_runner(
    app_name: str = "autopilot",
    user_id: str = "default_user",
) -> ADKRunner:
    """
    Get or create an ADKRunner singleton for the given app_name.

    Each unique app_name gets its own Runner with its own session service.
    """
    key = f"{app_name}:{user_id}"
    if key not in _runners:
        _runners[key] = ADKRunner(app_name=app_name, user_id=user_id)
    return _runners[key]
