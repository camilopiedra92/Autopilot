"""
AgentContext — Rich execution context for every agent invocation.

Every agent in the platform receives an `AgentContext` that provides:
  - execution_id: Unique ID for this pipeline run (correlation)
  - logger: Structured logger pre-bound with execution metadata
  - emit(): Publish events to the EventBus for real-time streaming
  - state: Accumulated pipeline state (typed dict)
  - bus: Typed pub/sub Agent Bus for inter-agent communication (A2A)
  - metadata: Immutable run-level metadata (trigger, workflow, etc.)
  - session: Short-term key-value state scoped to this execution
  - memory: Long-term semantic memory shared across executions

Design:
  - Session and Memory are always present (auto-provisioned if not injected).
  - Cheap to create; one per pipeline run.
  - Compatible with asyncio, zero thread-local hacks.
"""

from __future__ import annotations

import time
import structlog
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from autopilot.services.event_bus import get_event_bus
from autopilot.core.session import BaseSessionService, InMemorySessionService
from autopilot.core.memory import BaseMemoryService, InMemoryMemoryService, Observation


@dataclass(frozen=False)
class AgentContext:
    """
    Execution context passed to every agent in a pipeline.

    Created once per pipeline run by the PipelineRunner and threaded
    through each agent invocation.  Agents can read state, emit events,
    use session for short-term KV storage, and use memory for long-term
    semantic recall — all without importing anything.

    Attributes:
        execution_id: UUID for this specific pipeline execution.
        pipeline_name: Human-readable name of the pipeline being run.
        logger: structlog logger pre-bound with execution metadata.
        state: Mutable dict that accumulates agent outputs across steps.
        metadata: Immutable, user-provided run metadata (trigger info, etc.).
        session: Short-term session service (always present).
        memory: Long-term memory service (always present).
        _stream_id: Internal stream ID for the EventBus.
        _started_at: Monotonic timestamp for duration tracking.
    """

    execution_id: str = field(default_factory=lambda: uuid4().hex[:16])
    pipeline_name: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # V3 — Session & Memory (auto-provisioned if not injected)
    session: BaseSessionService = field(default=None)
    memory: BaseMemoryService = field(default=None)

    # Internal — not part of the public API but not hidden either.
    _stream_id: str = ""
    _started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # Auto-provision session and memory if not injected
        if self.session is None:
            self.session = InMemorySessionService()
        if self.memory is None:
            self.memory = InMemoryMemoryService()

        self.logger = structlog.get_logger("autopilot.core").bind(
            execution_id=self.execution_id,
            pipeline=self.pipeline_name,
        )
        if not self._stream_id:
            object.__setattr__(self, "_stream_id", self.execution_id)

    # ── Event Emission ───────────────────────────────────────────────

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """
        Publish an event to the platform EventBus.

        Events are delivered in real-time to SSE subscribers (dashboard, CLI)
        and can be used for monitoring, alerting, and debugging.

        Args:
            event_type: Short slug like "step_started", "step_completed".
            data: Optional payload merged into the event envelope.
        """
        event_bus = get_event_bus()
        payload = {
            "type": event_type,
            "execution_id": self.execution_id,
            "pipeline": self.pipeline_name,
            **(data or {}),
        }
        await event_bus.emit(self._stream_id, payload)

    # ── State Management ─────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from the accumulated pipeline state."""
        return self.state.get(key, default)

    def update_state(self, updates: dict[str, Any]) -> None:
        """Merge updates into the pipeline state (shallow merge)."""
        self.state.update(updates)

    # ── Memory Convenience Methods ───────────────────────────────────

    async def remember(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """
        Record an observation in long-term memory.

        Args:
            text: Natural language content to remember.
            metadata: Optional context (agent name, tags, etc.).

        Returns:
            The created Observation.
        """
        return await self.memory.add_observation(text, metadata)

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Observation]:
        """
        Retrieve relevant observations from long-term memory.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.

        Returns:
            List of Observations sorted by descending relevance.
        """
        return await self.memory.search_relevant(query, top_k=top_k)

    # ── Timing ───────────────────────────────────────────────────────

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since context creation."""
        return round((time.monotonic() - self._started_at) * 1000, 2)

    # ── Tool Registry Access ─────────────────────────────────────────

    @property
    def tools(self):
        """
        Access the global ToolRegistry from any agent.

        Returns:
            The platform-wide ``ToolRegistry`` singleton.

        Usage::

            # Inside an agent's run() method:
            ynab_tool = ctx.tools.get("ynab.create_transaction")
            all_tools = ctx.tools.list_all()
        """
        from autopilot.core.tools.registry import get_tool_registry

        return get_tool_registry()

    # ── Agent Bus (A2A) ──────────────────────────────────────────────

    @property
    def bus(self):
        """
        Access the global Agent Bus for inter-agent messaging.

        Returns:
            The platform-wide ``AgentBus`` singleton.

        Usage::

            # Subscribe to events:
            ctx.bus.subscribe("agent.error", my_handler)

            # Publish events:
            await ctx.bus.publish("agent.completed", {"result": ...})
        """
        from autopilot.core.bus import get_agent_bus

        return get_agent_bus()

    async def publish(self, topic: str, payload: dict | None = None) -> None:
        """
        Convenience: publish a message to the Agent Bus.

        Automatically sets ``sender`` to the current pipeline name.
        """
        await self.bus.publish(topic, payload or {}, sender=self.pipeline_name)

    def subscribe(self, topic: str, handler) -> Any:
        """
        Convenience: subscribe to Agent Bus messages.

        Returns a ``Subscription`` handle for unsubscribing.
        """
        return self.bus.subscribe(topic, handler)

    # ── Child Context ────────────────────────────────────────────────

    def for_step(self, step_name: str) -> AgentContext:
        """
        Create a child context for a specific pipeline step.

        Shares the same execution_id, state, session, and memory
        but binds step-level metadata to the logger for richer traces.
        """
        child = AgentContext(
            execution_id=self.execution_id,
            pipeline_name=self.pipeline_name,
            state=self.state,  # Shared reference — intentional
            metadata=self.metadata,
            session=self.session,  # Shared — same session across steps
            memory=self.memory,  # Shared — same memory across steps
            _stream_id=self._stream_id,
            _started_at=self._started_at,
        )
        child.logger = self.logger.bind(step=step_name)
        return child
