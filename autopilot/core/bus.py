"""
Agent Bus — Typed pub/sub messaging for inter-agent communication (A2A).

Provides a lightweight, async message bus that enables decoupled,
topic-based communication between agents.  Any agent (or platform
component) can publish typed messages to a topic, and any number
of subscribers receive them concurrently.

Key features:
  - Typed messages via ``AgentMessage`` (Pydantic model)
  - Wildcard topic matching (``"agent.*"`` matches ``"agent.error"``)
  - Dead-letter logging: handler errors are logged, never block others
  - Per-topic ring-buffer history for debugging / replay
  - Singleton accessor: ``get_agent_bus()``

Usage::

    bus = get_agent_bus()

    # Subscribe
    async def on_error(msg: AgentMessage) -> None:
        print(f"Error from {msg.sender}: {msg.payload}")

    sub = bus.subscribe("agent.error", on_error)

    # Publish
    await bus.publish("agent.error", {"detail": "timeout"}, sender="parser")

    # Unsubscribe
    bus.unsubscribe(sub)

Design:
  - Pure asyncio — no external broker required.
  - Handlers execute concurrently via ``asyncio.gather``.
  - Topic wildcards use ``fnmatch`` (stdlib, zero dependencies).
  - Thread-safe via asyncio Lock (no thread-local hacks).
"""

from __future__ import annotations

import asyncio
import fnmatch
import structlog
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from opentelemetry import trace

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentMessage — Typed envelope for bus messages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AgentMessage(BaseModel):
    """
    Typed message envelope for the Agent Bus.

    Every message published through the bus is wrapped in this model,
    giving consumers a consistent, introspectable structure.

    Attributes:
        topic: The topic this message was published to.
        sender: Identifier of the publishing agent/component.
        payload: Arbitrary data payload.
        timestamp: UTC ISO-8601 timestamp of publication.
        correlation_id: UUID for tracing related messages across the bus.
    """

    topic: str
    sender: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = Field(default_factory=lambda: uuid4().hex[:16])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Subscription — Handle for unsubscribing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Type alias for handler functions
MessageHandler = Callable[[AgentMessage], Awaitable[None]]


@dataclass(frozen=True)
class Subscription:
    """
    Opaque handle returned by ``AgentBus.subscribe()``.

    Pass this to ``AgentBus.unsubscribe()`` to remove the subscription.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    topic_pattern: str = ""
    handler: MessageHandler | None = field(default=None, repr=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentBus — The pub/sub bus
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default max messages per topic in history ring buffer
_DEFAULT_HISTORY_LIMIT = 100


class AgentBus:
    """
    Async pub/sub message bus for inter-agent communication.

    Agents publish typed ``AgentMessage`` objects to topics.
    Subscribers register handlers that are invoked concurrently
    for each matching message.

    Features:
      - **Topic wildcards**: ``"agent.*"`` matches ``"agent.error"``
      - **Dead-letter logging**: handler exceptions are logged, not propagated
      - **Per-topic history**: ring buffer of recent messages for replay/debug
      - **Concurrent dispatch**: handlers run via ``asyncio.gather``
    """

    def __init__(self, *, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        self._history: dict[str, deque[AgentMessage]] = {}
        self._history_limit = history_limit
        self._lock = asyncio.Lock()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}

    # ── Subscribe ────────────────────────────────────────────────────

    def subscribe(
        self,
        topic_pattern: str,
        handler: MessageHandler,
    ) -> Subscription:
        """
        Register an async handler for messages matching a topic pattern.

        Args:
            topic_pattern: Topic string, may include wildcards
                           (``*`` matches anything, ``?`` matches one char).
                           Examples: ``"agent.error"``, ``"agent.*"``, ``"*"``.
            handler: Async callable ``(AgentMessage) -> None``.

        Returns:
            A ``Subscription`` handle for later unsubscribe.
        """
        sub = Subscription(
            topic_pattern=topic_pattern,
            handler=handler,
        )
        self._subscriptions[sub.id] = sub
        logger.debug(
            "bus_subscribed",
            sub_id=sub.id,
            topic_pattern=topic_pattern,
        )
        return sub

    # ── Unsubscribe ──────────────────────────────────────────────────

    def unsubscribe(self, subscription: Subscription) -> bool:
        """
        Remove a subscription.

        Args:
            subscription: The ``Subscription`` handle from ``subscribe()``.

        Returns:
            ``True`` if the subscription was found and removed, ``False`` otherwise.
        """
        removed = self._subscriptions.pop(subscription.id, None)
        if removed:
            logger.debug("bus_unsubscribed", sub_id=subscription.id)
        return removed is not None

    # ── Publish ──────────────────────────────────────────────────────

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        sender: str = "",
        correlation_id: str | None = None,
    ) -> AgentMessage:
        """
        Publish a typed message to a topic.

        Builds an ``AgentMessage`` and dispatches it to all handlers
        whose ``topic_pattern`` matches the given ``topic``.

        Args:
            topic: The topic to publish to (e.g. ``"agent.completed"``).
            payload: Arbitrary data dictionary.
            sender: Identifier of the publishing agent.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            The published ``AgentMessage``.
        """
        msg = AgentMessage(
            topic=topic,
            sender=sender,
            payload=payload or {},
            **({"correlation_id": correlation_id} if correlation_id else {}),
        )

        # Record in history
        async with self._lock:
            if topic not in self._history:
                self._history[topic] = deque(maxlen=self._history_limit)
            self._history[topic].append(msg)

        self._stats["published"] += 1

        # Find matching handlers
        matching: list[MessageHandler] = []
        for sub in self._subscriptions.values():
            if fnmatch.fnmatch(topic, sub.topic_pattern):
                if sub.handler is not None:
                    matching.append(sub.handler)

        with tracer.start_as_current_span(
            "bus.publish",
            attributes={
                "topic": topic,
                "sender": sender,
                "subscribers": len(matching),
                "correlation_id": msg.correlation_id,
            }
        ) as span:
            if not matching:
                logger.debug("bus_no_subscribers", topic=topic, sender=sender)
                span.set_attribute("delivered", 0)
                span.set_attribute("errors", 0)
                return msg

            # Dispatch concurrently with dead-letter isolation
            results = await asyncio.gather(
                *(self._safe_invoke(handler, msg) for handler in matching),
                return_exceptions=True,
            )

            delivered = sum(1 for r in results if r is None)
            errors = sum(1 for r in results if r is not None)
            self._stats["delivered"] += delivered
            self._stats["errors"] += errors

            logger.debug(
                "bus_published",
                topic=topic,
                sender=sender,
                handlers=len(matching),
                delivered=delivered,
                errors=errors,
            )
            
            span.set_attribute("delivered", delivered)
            span.set_attribute("errors", errors)

        return msg

    # ── History ──────────────────────────────────────────────────────

    def history(
        self,
        topic: str,
        *,
        limit: int = 50,
    ) -> list[AgentMessage]:
        """
        Return recent messages published to a topic.

        Args:
            topic: The exact topic to query (no wildcards).
            limit: Max number of messages to return (most recent first).

        Returns:
            List of ``AgentMessage`` objects, newest first.
        """
        buf = self._history.get(topic)
        if buf is None:
            return []
        return list(buf)[-limit:][::-1]

    # ── Introspection ────────────────────────────────────────────────

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)

    @property
    def stats(self) -> dict[str, int]:
        """Bus-level statistics (published, delivered, errors)."""
        return dict(self._stats)

    # ── Reset ────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all subscriptions and history. Useful for tests."""
        self._subscriptions.clear()
        self._history.clear()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}
        logger.debug("bus_cleared")

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    async def _safe_invoke(
        handler: MessageHandler,
        msg: AgentMessage,
    ) -> None:
        """
        Invoke a handler with dead-letter isolation.

        If the handler raises, the exception is logged but not propagated.
        This ensures one failing subscriber never blocks others.
        """
        with tracer.start_as_current_span(
            f"bus.handler",
            attributes={
                "topic": msg.topic,
                "handler": handler.__name__ if hasattr(handler, "__name__") else str(handler),
            }
        ) as span:
            try:
                await handler(msg)
            except Exception as exc:
                logger.error(
                    "bus_handler_error",
                    topic=msg.topic,
                    sender=msg.sender,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise  # Re-raise so asyncio.gather can collect it

    def __repr__(self) -> str:
        return (
            f"AgentBus(subscriptions={self.subscription_count}, "
            f"stats={self._stats})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton — Module-level accessor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_agent_bus: AgentBus | None = None


def get_agent_bus() -> AgentBus:
    """Get or create the global AgentBus singleton."""
    global _agent_bus
    if _agent_bus is None:
        _agent_bus = AgentBus()
    return _agent_bus


def reset_agent_bus() -> None:
    """
    Reset the global AgentBus singleton.

    Useful in tests to ensure a clean bus between test cases.
    """
    global _agent_bus
    if _agent_bus is not None:
        _agent_bus.clear()
    _agent_bus = None
