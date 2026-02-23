"""
AgentContext — Rich execution context for every agent invocation.

Every agent in the platform receives an ``AgentContext`` that provides:
  - execution_id: Unique ID for this pipeline run (correlation)
  - logger: Structured logger pre-bound with execution metadata
  - publish(): Publish events to the unified EventBus
  - state: Accumulated pipeline state (typed dict)
  - bus: The unified EventBus for all event communication
  - metadata: Immutable run-level metadata (trigger, workflow, etc.)
  - session_service: ADK-native SessionService for full lifecycle
  - session: ADK Session object (id, app_name, user_id, state, events)
  - memory: Long-term semantic memory shared across executions

Design:
  - Session and Memory are always present (auto-provisioned if not injected).
  - ADK session is created lazily via ensure_session() — called automatically
    by Pipeline/DAG executors before running agents.
  - Agents interact with session.state directly (dict), exactly as ADK intended.
  - Cheap to create; one per pipeline run.
  - Compatible with asyncio, zero thread-local hacks.
"""

from __future__ import annotations

import time
import structlog
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from autopilot.core.session import (
    BaseSessionService,
    InMemorySessionService,
    Session,
)
from autopilot.core.memory import BaseMemoryService, InMemoryMemoryService, Observation


@dataclass(frozen=False)
class AgentContext:
    """
    Execution context passed to every agent in a pipeline.

    Created once per pipeline run and threaded through each agent
    invocation.  Agents read/write session state via ``ctx.session.state``
    (a plain dict), exactly as Google ADK intended.

    Attributes:
        execution_id: UUID for this specific pipeline execution.
        pipeline_name: Human-readable name of the pipeline being run.
        logger: structlog logger pre-bound with execution metadata.
        state: Mutable dict that accumulates agent outputs across steps.
        metadata: Immutable, user-provided run metadata (trigger info, etc.).
        session_service: ADK-native SessionService (always present).
        session: ADK Session object — ``session.state`` is the KV store.
        memory: Long-term memory service (always present).
    """

    execution_id: str = field(default_factory=lambda: uuid4().hex[:16])
    pipeline_name: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ADK-native session — no wrappers
    session_service: BaseSessionService = field(default=None)
    session: Session = field(default=None)

    # Long-term memory
    memory: BaseMemoryService = field(default=None)

    # Internal
    _started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        # Auto-provision session service if not injected
        if self.session_service is None:
            self.session_service = InMemorySessionService()

        # Auto-provision memory if not injected
        if self.memory is None:
            self.memory = InMemoryMemoryService()

        self.logger = structlog.get_logger("autopilot.core").bind(
            execution_id=self.execution_id,
            pipeline=self.pipeline_name,
        )

    # ── ADK Session Lifecycle ────────────────────────────────────────

    async def ensure_session(self) -> None:
        """Initialize ADK session lazily.

        Called automatically by Pipeline/DAG executors before running
        agents.  Safe to call multiple times — subsequent calls are no-ops.

        Creates an ADK ``Session`` via ``session_service.create_session()``.
        After this, ``ctx.session.state`` is the live KV store.
        """
        if self.session is not None:
            return

        self.session = await self.session_service.create_session(
            app_name="autopilot",
            user_id="default",
            session_id=self.execution_id,
            state=dict(self.state) if self.state else {},
        )

    # ── Event Bus (Unified) ──────────────────────────────────────────

    @property
    def bus(self):
        """Access the global EventBus for all event communication."""
        from autopilot.core.bus import get_event_bus

        return get_event_bus()

    async def publish(self, topic: str, payload: dict | None = None) -> None:
        """
        Convenience: publish an event to the unified EventBus.

        Automatically sets ``sender`` to the current pipeline name
        and ``correlation_id`` to the execution_id.
        """
        await self.bus.publish(
            topic,
            {
                "execution_id": self.execution_id,
                "pipeline": self.pipeline_name,
                **(payload or {}),
            },
            sender=self.pipeline_name,
            correlation_id=self.execution_id,
        )

    def subscribe(self, topic: str, handler) -> Any:
        """Convenience: subscribe to EventBus messages."""
        return self.bus.subscribe(topic, handler)

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
        """Record an observation in long-term memory."""
        return await self.memory.add_observation(text, metadata)

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Observation]:
        """Retrieve relevant observations from long-term memory."""
        return await self.memory.search_relevant(query, top_k=top_k)

    # ── Timing ───────────────────────────────────────────────────────

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since context creation."""
        return round((time.monotonic() - self._started_at) * 1000, 2)

    # ── Tool Registry Access ─────────────────────────────────────────

    @property
    def tools(self):
        """Access the global ToolRegistry from any agent."""
        from autopilot.core.tools.registry import get_tool_registry

        return get_tool_registry()

    # ── Child Context ────────────────────────────────────────────────

    def for_step(self, step_name: str) -> AgentContext:
        """
        Create a child context for a specific pipeline step.

        Shares the same execution_id, state, session_service, session,
        and memory but binds step-level metadata to the logger.
        """
        child = AgentContext(
            execution_id=self.execution_id,
            pipeline_name=self.pipeline_name,
            state=self.state,  # Shared reference — intentional
            metadata=self.metadata,
            session_service=self.session_service,
            session=self.session,  # Same ADK Session
            memory=self.memory,
            _started_at=self._started_at,
        )
        child.logger = self.logger.bind(step=step_name)
        return child
