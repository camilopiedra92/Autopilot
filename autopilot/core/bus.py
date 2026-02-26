"""
EventBus — Unified typed pub/sub messaging for the Autopilot platform.

The single event backbone that handles ALL event communication:
  - Inter-agent messaging (A2A)
  - Pipeline/stage/tool observability events
  - External trigger routing (email.received, webhook.*)
  - Reactive subscribers (transaction.created → Telegram)

Key features:
  - ``EventBusProtocol`` ABC for swappable backends
  - ``EventBus`` — in-memory implementation (dev/test/single-instance prod)
  - Typed messages via ``AgentMessage`` (Pydantic)
  - Wildcard topic matching (``"agent.*"`` → ``"agent.error"``)
  - Dead-letter isolation: handler errors never block others
  - Per-topic ring-buffer history for replay/debug
  - Middleware chain for event persistence, filtering, metrics
  - Singleton accessor: ``get_event_bus()``

Topic naming convention: ``domain.verb``
  - ``pipeline.started``, ``pipeline.completed``, ``pipeline.error``
  - ``stage.started``, ``stage.completed``
  - ``tool.started``, ``tool.completed``
  - ``email.received``, ``transaction.created``
  - ``agent.error``, ``agent.completed``

Usage::

    bus = get_event_bus()

    # Subscribe
    async def on_error(msg: AgentMessage) -> None:
        print(f"Error from {msg.sender}: {msg.payload}")

    sub = bus.subscribe("agent.error", on_error)

    # Publish
    await bus.publish("agent.error", {"detail": "timeout"}, sender="parser")

    # Middleware (persistence, metrics, filtering)
    async def log_all(msg: AgentMessage) -> AgentMessage | None:
        logger.info("event", topic=msg.topic)
        return msg  # Pass through (return None to filter)

    bus.use(log_all)

    # Unsubscribe
    bus.unsubscribe(sub)
"""

import abc
import asyncio
import fnmatch
import os
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
    Typed message envelope for the Event Bus.

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

# Type aliases
MessageHandler = Callable[[AgentMessage], Awaitable[None]]
EventMiddleware = Callable[[AgentMessage], Awaitable[AgentMessage | None]]


@dataclass(frozen=True)
class Subscription:
    """
    Opaque handle returned by ``EventBus.subscribe()``.

    Pass this to ``EventBus.unsubscribe()`` to remove the subscription.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    topic_pattern: str = ""
    handler: MessageHandler | None = field(default=None, repr=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EventBusProtocol — Abstract contract for swappable backends
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EventBusProtocol(abc.ABC):
    """
    Abstract Event Bus protocol — swappable backends.

    All event bus implementations MUST implement this interface.
    Backend selection is config-driven via ``EVENTBUS_BACKEND`` env var.

    Implementations:
      - EventBus (InMemory): dev/test — asyncio, zero deps, deterministic
      - CloudPubSubEventBus: production — GCP Pub/Sub, fully managed,
        cross-instance fanout, 7-day retention, scale-to-zero native
    """

    @abc.abstractmethod
    def subscribe(
        self, topic_pattern: str, handler: MessageHandler
    ) -> Subscription: ...

    @abc.abstractmethod
    def unsubscribe(self, subscription: Subscription) -> bool: ...

    @abc.abstractmethod
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        sender: str = "",
        correlation_id: str | None = None,
    ) -> AgentMessage: ...

    @abc.abstractmethod
    def history(self, topic: str, *, limit: int = 50) -> list[AgentMessage]: ...

    @abc.abstractmethod
    async def replay(
        self,
        topic: str,
        *,
        since: str | None = None,
        handler: MessageHandler | None = None,
    ) -> list[AgentMessage]:
        """Replay events from history/store for a topic.

        In-memory: replays from ring buffer history.
        Redis/Pub/Sub: replays from persistent stream with optional cursor.

        Args:
            topic: Topic pattern to replay.
            since: Optional ISO-8601 timestamp to replay from.
            handler: Optional handler to invoke for each replayed event.

        Returns:
            List of replayed AgentMessage objects.
        """
        ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @property
    @abc.abstractmethod
    def subscription_count(self) -> int: ...

    @property
    @abc.abstractmethod
    def stats(self) -> dict[str, int]: ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EventBus — In-memory implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DEFAULT_HISTORY_LIMIT = 100


class EventBus(EventBusProtocol):
    """
    In-memory event bus for the Autopilot platform.

    Correct for single-instance Cloud Run (scale-to-zero) because:
      - Subscriptions are re-registered on every cold start via setup()
      - Each HTTP request is self-contained within a single instance
      - History is only needed for debugging within a single lifecycle

    Features:
      - **Topic wildcards**: ``"agent.*"`` matches ``"agent.error"``
      - **Dead-letter logging**: handler exceptions are logged, not propagated
      - **Per-topic history**: ring buffer for replay/debug
      - **Concurrent dispatch**: handlers run via ``asyncio.gather``
      - **Middleware chain**: intercept, filter, persist events
    """

    def __init__(self, *, history_limit: int = _DEFAULT_HISTORY_LIMIT) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        self._history: dict[str, deque[AgentMessage]] = {}
        self._history_limit = history_limit
        self._lock = asyncio.Lock()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}
        self._middleware: list[EventMiddleware] = []

    # ── Middleware ────────────────────────────────────────────────────

    def use(self, middleware: EventMiddleware) -> None:
        """Register middleware that intercepts every published event.

        Middleware runs in order before dispatch. If middleware returns None,
        the event is suppressed (filtered). Otherwise the returned event
        (possibly modified) is dispatched.

        Use cases: event logging, metrics, persistence, filtering.
        """
        self._middleware.append(middleware)

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
            ``True`` if found and removed, ``False`` otherwise.
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

        Builds an ``AgentMessage``, runs middleware chain, then dispatches
        to all handlers whose ``topic_pattern`` matches.

        Args:
            topic: The topic to publish to (e.g. ``"pipeline.started"``).
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

        # Run middleware chain
        for mw in self._middleware:
            result = await mw(msg)
            if result is None:
                return msg  # Filtered by middleware
            msg = result

        # Record in history
        async with self._lock:
            if topic not in self._history:
                self._history[topic] = deque(maxlen=self._history_limit)
            self._history[topic].append(msg)

        self._stats["published"] += 1

        # Find matching handlers
        matching: list[MessageHandler] = [
            sub.handler
            for sub in self._subscriptions.values()
            if sub.handler is not None and fnmatch.fnmatch(topic, sub.topic_pattern)
        ]

        with tracer.start_as_current_span(
            "bus.publish",
            attributes={
                "topic": topic,
                "sender": sender,
                "subscribers": len(matching),
                "correlation_id": msg.correlation_id,
            },
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

    # ── History & Replay ─────────────────────────────────────────────

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

    async def replay(
        self,
        topic: str,
        *,
        since: str | None = None,
        handler: MessageHandler | None = None,
    ) -> list[AgentMessage]:
        """Replay events from in-memory history buffer.

        Args:
            topic: Exact topic to replay from.
            since: Optional ISO-8601 timestamp — only replay events after this.
            handler: Optional async handler to invoke for each replayed event.

        Returns:
            List of replayed AgentMessage objects (oldest first).
        """
        buf = self._history.get(topic)
        if buf is None:
            return []

        messages = list(buf)
        if since:
            messages = [m for m in messages if m.timestamp > since]

        if handler:
            for msg in messages:
                await self._safe_invoke(handler, msg)

        return messages

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
        """Remove all subscriptions, history, and middleware. Useful for tests."""
        self._subscriptions.clear()
        self._history.clear()
        self._middleware.clear()
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

        If the handler raises, the exception is logged but not propagated
        to other subscribers. This ensures one failure never blocks others.
        """
        with tracer.start_as_current_span(
            "bus.handler",
            attributes={
                "topic": msg.topic,
                "handler": handler.__name__
                if hasattr(handler, "__name__")
                else str(handler),
            },
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
            f"EventBus(subscriptions={self.subscription_count}, "
            f"middleware={len(self._middleware)}, stats={self._stats})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Factory + Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_event_bus(backend: str | None = None) -> EventBusProtocol:
    """Factory for creating the appropriate bus backend.

    Backend selection follows 12-Factor App (Factor III: Config):
      - ``"memory"`` (default): In-memory bus for dev/test
      - ``"pubsub"``: Cloud Pub/Sub for production (cross-instance, durable)

    Args:
        backend: Override backend choice. Defaults to ``EVENTBUS_BACKEND``
                 env var, falling back to ``"memory"``.

    Returns:
        An ``EventBusProtocol`` implementation.
    """
    backend = backend or os.getenv("EVENTBUS_BACKEND", "memory")
    logger.info("event_bus_backend_selected", backend=backend)

    if backend == "pubsub":
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        return CloudPubSubEventBus.from_env()

    return EventBus()


_event_bus: EventBusProtocol | None = None


def get_event_bus() -> EventBusProtocol:
    """Get or create the global EventBus singleton.

    The backend is selected by ``create_event_bus()`` on first call.
    Subsequent calls return the same instance.
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = create_event_bus()
    return _event_bus


def reset_event_bus() -> None:
    """
    Reset the global EventBus singleton.

    Useful in tests to ensure a clean bus between test cases.
    """
    global _event_bus
    if _event_bus is not None:
        _event_bus.clear()
    _event_bus = None
