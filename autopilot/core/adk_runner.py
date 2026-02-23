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
  - Per-agent context caching via App wrapper (when opted in)
  - Context window compression (ADK-native SlidingWindow)
  - Cross-session memory transfer (add_session_to_memory)

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

import json
import os
import time
import structlog
from typing import Any
from uuid import uuid4

from opentelemetry import trace

from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.runners import Runner
from autopilot.agents.context_cache import (
    has_cache_context,
    create_context_cache_config,
)
from autopilot.core.session import create_session_service
from autopilot.core.memory import create_memory_service
from autopilot.core.artifact import create_artifact_service
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


# ── Context Window Compression Factory ────────────────────────────────
# ADK-native SlidingWindow compression.  When the token count of session
# events exceeds ``trigger_tokens``, older history is compressed down to
# ``target_tokens``.  Configured via 12-Factor env vars.  Set both to 0
# to disable entirely (no RunConfig overhead).

_DEFAULT_TRIGGER_TOKENS = 100_000  # ~78 % of Gemini 128k window
_DEFAULT_TARGET_TOKENS = 80_000  # ~62 % of Gemini 128k window


def _create_run_config() -> RunConfig | None:
    """Build a RunConfig with SlidingWindow compression (if enabled).

    Returns ``None`` when compression is disabled (both thresholds == 0),
    which tells the Runner to use its defaults with zero overhead.
    """
    trigger = int(
        os.environ.get("CONTEXT_COMPRESSION_TRIGGER_TOKENS", _DEFAULT_TRIGGER_TOKENS)
    )
    target = int(
        os.environ.get("CONTEXT_COMPRESSION_TARGET_TOKENS", _DEFAULT_TARGET_TOKENS)
    )

    if trigger == 0 and target == 0:
        return None

    return RunConfig(
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=trigger,
            sliding_window=types.SlidingWindow(
                target_tokens=target,
            ),
        ),
    )


class ADKRunner:
    def __init__(self, app_name: str, user_id: str):
        self._app_name = app_name
        self._user_id = user_id
        self._session_service = create_session_service()
        self._memory_service = create_memory_service()
        self._artifact_service = create_artifact_service()

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
        session_id: str | None = None,
        persist_memory: bool = False,
    ) -> PipelineResult:
        """
        Run an ADK agent and return a structured result.

        Args:
            pipeline: An ADK agent (LlmAgent, SequentialAgent, etc.).
            message: The user message to send to the agent.
            initial_state: Optional initial session state.
            stream_session_id: If provided, used as the SSE event bus session ID.
            session_id: If provided, reuses this session for conversation continuity.
                        If the session exists, its history is preserved (multi-turn).
                        If not, a new session is created with this ID.
            persist_memory: If True, transfer session events to long-term memory
                           after execution. Controlled by manifest ``memory`` flag.

        Returns:
            PipelineResult with final text, parsed JSON, session state, and timing.
        """
        effective_session_id = session_id or f"pipeline_{uuid4().hex[:12]}"
        effective_stream_id = stream_session_id or effective_session_id

        return await self._run_adk_agent(
            pipeline,
            message,
            initial_state,
            effective_stream_id,
            session_id=effective_session_id,
            persist_memory=persist_memory,
        )

    async def _run_adk_agent(
        self,
        pipeline,
        message: str,
        initial_state: dict[str, Any] | None,
        effective_stream_id: str,
        *,
        session_id: str | None = None,
        persist_memory: bool = False,
    ) -> PipelineResult:
        """Execute the ADK agent through Runner + SessionService."""
        with tracer.start_as_current_span(
            "adk_runner.run",
            attributes={
                "app_name": self._app_name,
                "agent_name": getattr(pipeline, "name", "unknown"),
            },
        ) as span:
            session_id = session_id or f"pipeline_{uuid4().hex[:12]}"
            span.set_attribute("session_id", session_id)
            bus = get_event_bus()

        # Set the event bus session context so callbacks can stream events
        token = pipeline_session_id.set(effective_stream_id)

        start = time.monotonic()

        try:
            # Conditionally wrap agent in App for context caching
            if has_cache_context(pipeline):
                cache_config = create_context_cache_config()
                app = App(
                    name=self._app_name,
                    root_agent=pipeline,
                    context_cache_config=cache_config,
                )
                runner = Runner(
                    app=app,
                    session_service=self._session_service,
                    memory_service=self._memory_service,
                    artifact_service=self._artifact_service,
                )
            else:
                runner = Runner(
                    app_name=self._app_name,
                    agent=pipeline,
                    session_service=self._session_service,
                    memory_service=self._memory_service,
                    artifact_service=self._artifact_service,
                )

            # Try to resume an existing session (multi-turn), otherwise create new
            session = await self._session_service.get_session(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session_id,
            )
            if session is None:
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

            # Run the agent and extract final text response.
            # When output_schema is set, ADK activates Gemini native JSON mode
            # (response_schema + response_mime_type=application/json), which
            # guarantees structured JSON text output. ADK also disables tools
            # and function calls in this mode, so only text parts are emitted.
            final_text = ""
            run_config = _create_run_config()
            async for event in runner.run_async(
                user_id=self._user_id,
                session_id=session_id,
                new_message=user_message,
                run_config=run_config,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        text_parts = [p.text for p in event.content.parts if p.text]
                        if text_parts:
                            final_text = "\n".join(text_parts)

            elapsed_ms = round((time.monotonic() - start) * 1000, 2)

            if not final_text:
                # ReAct agents (like conversational_assistant) may complete
                # their work entirely via tool calls (e.g. sending a Telegram
                # message, creating a Todoist task) without emitting a final
                # text response.  Only raise for agents with output_schema,
                # which MUST produce structured JSON text.
                has_output_schema = getattr(pipeline, "output_schema", None) is not None
                if has_output_schema:
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
                else:
                    final_text = "[Agent completed via tool calls]"
                    logger.info(
                        "adk_runner_no_final_text",
                        agent_name=getattr(pipeline, "name", "unknown"),
                        session_id=session_id,
                        reason="ReAct agent completed via tool calls without text response",
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

            pipeline_result = PipelineResult(
                session_id=session_id,
                final_text=final_text,
                parsed_json=parsed_json,
                state=final_state,
                duration_ms=elapsed_ms,
            )

            # ── Persist LLM result as a versioned artifact ────────────
            await self._persist_llm_artifact(
                agent_name=getattr(pipeline, "name", "unknown"),
                session_id=session_id,
                stream_session_id=effective_stream_id,
                result=pipeline_result,
            )

            # ── Transfer session events to memory (workflow opt-in) ──
            if persist_memory:
                await self._transfer_session_to_memory(session)

            return pipeline_result

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

    async def _persist_llm_artifact(
        self,
        agent_name: str,
        session_id: str,
        stream_session_id: str,
        result: PipelineResult,
    ) -> None:
        """Persist an LLM agent's result as a versioned artifact.

        Saves ``{agent_name}.llm.json`` scoped to the execution via
        stream_session_id. Never blocks — failures are logged and swallowed.
        """
        try:
            payload = {
                "agent": agent_name,
                "app_name": self._app_name,
                "session_id": session_id,
                "duration_ms": result.duration_ms,
                "final_text": result.final_text,
                "parsed_json": result.parsed_json,
            }
            artifact = types.Part(
                text=json.dumps(payload, ensure_ascii=False, default=str)
            )
            version = await self._artifact_service.save_artifact(
                app_name=self._app_name,
                user_id="default",
                session_id=stream_session_id,
                filename=f"{agent_name}.llm.json",
                artifact=artifact,
            )
            logger.debug(
                "llm_artifact_persisted",
                agent=agent_name,
                filename=f"{agent_name}.llm.json",
                version=version,
                session_id=stream_session_id,
            )
        except Exception as exc:
            logger.warning(
                "llm_artifact_persist_failed",
                agent=agent_name,
                error=str(exc),
            )

    async def _transfer_session_to_memory(self, session) -> None:
        """Transfer session events to memory for cross-session recall.

        Fire-and-forget: failures are logged and swallowed (same pattern as
        ``_persist_llm_artifact``).  When ``MEMORY_BACKEND=memory`` this
        stores events in-process for keyword-based recall across sessions.
        When ``MEMORY_BACKEND=firestore`` it embeds events via Gemini and
        persists to Firestore for durable semantic vector search.
        When ``MEMORY_BACKEND=vertexai`` it persists to Vertex AI Memory Bank.
        """
        if not self._memory_service:
            return
        try:
            await self._memory_service.add_session_to_memory(session)
            logger.debug(
                "session_memory_transferred",
                session_id=session.id,
                event_count=len(session.events) if session.events else 0,
            )
        except Exception as exc:
            logger.warning(
                "session_memory_transfer_failed",
                session_id=getattr(session, "id", "unknown"),
                error=str(exc),
            )


# ── Singleton ─────────────────────────────────────────────────────────

_runners: dict[str, ADKRunner] = {}


def get_adk_runner(
    app_name: str = "autopilot",
    user_id: str = "default",
) -> ADKRunner:
    """
    Get or create an ADKRunner singleton for the given app_name.

    Each unique app_name gets its own Runner with its own session service.
    """
    key = f"{app_name}:{user_id}"
    if key not in _runners:
        _runners[key] = ADKRunner(app_name=app_name, user_id=user_id)
    return _runners[key]
