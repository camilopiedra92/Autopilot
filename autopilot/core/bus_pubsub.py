"""
CloudPubSubEventBus — Durable event bus backed by Google Cloud Pub/Sub.

Production-grade EventBusProtocol implementation for Cloud Run deployments.
Uses Cloud Pub/Sub for persistent, cross-instance event delivery with
native scale-to-zero support.

Architecture (Hybrid Dispatch):
  - publish() writes to Cloud Pub/Sub AND dispatches to local subscribers
  - Local dispatch = zero-latency, same as InMemoryEventBus
  - Pub/Sub layer = persistence + cross-instance fanout (async, fire-and-forget)

  This dual path ensures no latency regression for same-instance workflows
  while adding cross-instance visibility and 7-day message retention.

Environment Variables:
  - GOOGLE_CLOUD_PROJECT: GCP project ID (required for Pub/Sub)
  - EVENTBUS_PUBSUB_TOPIC: Pub/Sub topic name (default: "autopilot-events")

Usage::

    # Production (Cloud Run) — via factory
    os.environ["EVENTBUS_BACKEND"] = "pubsub"
    bus = get_event_bus()  # → CloudPubSubEventBus

    # Or direct instantiation
    bus = CloudPubSubEventBus.from_env()
    await bus.publish("transaction.created", {"amount": 50000}, sender="pipeline")
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import structlog
from collections import deque
from typing import Any

from google.cloud import pubsub_v1
from opentelemetry import trace

from autopilot.core.bus import (
    AgentMessage,
    EventBusProtocol,
    MessageHandler,
    Subscription,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_DEFAULT_TOPIC = "autopilot-events"
_DEFAULT_HISTORY_LIMIT = 100


class CloudPubSubEventBus(EventBusProtocol):
    """
    Durable event bus backed by Google Cloud Pub/Sub.

    Hybrid dispatch model:
      - **Local**: In-process subscribers fire immediately via asyncio.gather
        (identical to EventBus — zero latency for same-request workflows)
      - **Remote**: Messages published to Cloud Pub/Sub for cross-instance
        fanout and 7-day persistence

    Features:
      - Topic wildcards (fnmatch) for local subscribers
      - Dead-letter isolation (handler errors never block others)
      - In-memory ring buffer history (same-lifecycle debugging)
      - Concurrent local dispatch via asyncio.gather
      - Idempotent topic creation on startup (safe for cold starts)
      - OTel tracing on publish
    """

    def __init__(
        self,
        *,
        project_id: str,
        topic_name: str = _DEFAULT_TOPIC,
        history_limit: int = _DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self._project_id = project_id
        self._topic_name = topic_name
        self._topic_path = f"projects/{project_id}/topics/{topic_name}"

        # Cloud Pub/Sub publisher (batched, thread-safe)
        self._publisher = pubsub_v1.PublisherClient()

        # Local in-process state (identical to EventBus)
        self._subscriptions: dict[str, Subscription] = {}
        self._history: dict[str, deque[AgentMessage]] = {}
        self._history_limit = history_limit
        self._lock = asyncio.Lock()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}
        self._middleware = []

        # Ensure topic exists (idempotent)
        self._ensure_topic()

    @classmethod
    def from_env(cls) -> CloudPubSubEventBus:
        """Create from environment variables.

        Reads:
          - GOOGLE_CLOUD_PROJECT: GCP project ID (required)
          - EVENTBUS_PUBSUB_TOPIC: topic name (default: "autopilot-events")
        """
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            # Cloud Run sets this automatically
            project_id = os.environ.get("GCLOUD_PROJECT", "")
        if not project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT env var required for CloudPubSubEventBus"
            )

        return cls(
            project_id=project_id,
            topic_name=os.getenv("EVENTBUS_PUBSUB_TOPIC", _DEFAULT_TOPIC),
        )

    # ── Topic Management ─────────────────────────────────────────────

    def _ensure_topic(self) -> None:
        """Idempotent topic creation — safe for cold starts."""
        try:
            self._publisher.create_topic(request={"name": self._topic_path})
            logger.info("pubsub_topic_created", topic=self._topic_path)
        except Exception as exc:
            # 409 ALREADY_EXISTS is expected and fine
            if "ALREADY_EXISTS" in str(exc) or "409" in str(exc):
                logger.debug("pubsub_topic_exists", topic=self._topic_path)
            else:
                logger.warning(
                    "pubsub_topic_create_error",
                    topic=self._topic_path,
                    error=str(exc),
                )

    # ── Middleware ────────────────────────────────────────────────────

    def use(self, middleware) -> None:
        """Register middleware (same API as EventBus)."""
        self._middleware.append(middleware)

    # ── Subscribe ────────────────────────────────────────────────────

    def subscribe(
        self,
        topic_pattern: str,
        handler: MessageHandler,
    ) -> Subscription:
        """Register a local in-process handler for messages matching a topic.

        This is the same local subscription model as EventBus.
        Cross-instance delivery happens via Cloud Pub/Sub pull subscribers
        (managed separately at the infrastructure level).
        """
        sub = Subscription(topic_pattern=topic_pattern, handler=handler)
        self._subscriptions[sub.id] = sub
        logger.debug(
            "pubsub_bus_subscribed",
            sub_id=sub.id,
            topic_pattern=topic_pattern,
        )
        return sub

    # ── Unsubscribe ──────────────────────────────────────────────────

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a local subscription."""
        removed = self._subscriptions.pop(subscription.id, None)
        if removed:
            logger.debug("pubsub_bus_unsubscribed", sub_id=subscription.id)
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
        """Publish a message — hybrid local + Cloud Pub/Sub dispatch.

        1. Builds AgentMessage, runs middleware chain
        2. Dispatches to local in-process subscribers (zero latency)
        3. Publishes to Cloud Pub/Sub (persistence + cross-instance)

        Cloud Pub/Sub publish is fire-and-forget — local delivery is
        never blocked by Pub/Sub latency.
        """
        msg = AgentMessage(
            topic=topic,
            sender=sender,
            payload=payload or {},
            **({} if correlation_id is None else {"correlation_id": correlation_id}),
        )

        # Run middleware chain
        for mw in self._middleware:
            result = await mw(msg)
            if result is None:
                return msg  # Filtered
            msg = result

        # Record in local history
        async with self._lock:
            if topic not in self._history:
                self._history[topic] = deque(maxlen=self._history_limit)
            self._history[topic].append(msg)

        self._stats["published"] += 1

        with tracer.start_as_current_span(
            "pubsub_bus.publish",
            attributes={
                "topic": topic,
                "sender": sender,
                "correlation_id": msg.correlation_id,
            },
        ) as span:
            # 1. Local dispatch (zero latency, same as EventBus)
            await self._dispatch_local(msg)

            # 2. Cloud Pub/Sub (fire-and-forget for persistence/fanout)
            self._publish_to_pubsub(msg)

            span.set_attribute("delivered_local", self._stats["delivered"])

        return msg

    def _publish_to_pubsub(self, msg: AgentMessage) -> None:
        """Fire-and-forget publish to Cloud Pub/Sub.

        Uses the PublisherClient's built-in batching — messages are
        accumulated and sent in batches for efficiency. The future
        result is not awaited to avoid blocking local dispatch.
        """
        try:
            data = msg.model_dump_json().encode("utf-8")
            future = self._publisher.publish(
                self._topic_path,
                data,
                # Message attributes for server-side filtering
                event_topic=msg.topic,
                sender=msg.sender,
                correlation_id=msg.correlation_id,
            )
            # Add error callback (non-blocking)
            future.add_done_callback(self._on_pubsub_publish_done)
        except Exception as exc:
            # Never let Pub/Sub errors break the pipeline
            logger.warning(
                "pubsub_publish_error",
                topic=msg.topic,
                error=str(exc),
            )

    @staticmethod
    def _on_pubsub_publish_done(future) -> None:
        """Callback for Pub/Sub publish completion."""
        try:
            future.result()  # Raises if publish failed
        except Exception as exc:
            logger.warning("pubsub_publish_failed", error=str(exc))

    # ── Local Dispatch ───────────────────────────────────────────────

    async def _dispatch_local(self, msg: AgentMessage) -> None:
        """Dispatch to in-process subscribers (identical to EventBus)."""
        matching: list[MessageHandler] = [
            sub.handler
            for sub in self._subscriptions.values()
            if sub.handler is not None and fnmatch.fnmatch(msg.topic, sub.topic_pattern)
        ]

        if not matching:
            logger.debug(
                "pubsub_bus_no_local_subscribers",
                topic=msg.topic,
                sender=msg.sender,
            )
            return

        results = await asyncio.gather(
            *(self._safe_invoke(handler, msg) for handler in matching),
            return_exceptions=True,
        )

        delivered = sum(1 for r in results if r is None)
        errors = sum(1 for r in results if r is not None)
        self._stats["delivered"] += delivered
        self._stats["errors"] += errors

        logger.debug(
            "pubsub_bus_dispatched_local",
            topic=msg.topic,
            handlers=len(matching),
            delivered=delivered,
            errors=errors,
        )

    @staticmethod
    async def _safe_invoke(handler: MessageHandler, msg: AgentMessage) -> None:
        """Invoke handler with dead-letter isolation (same as EventBus)."""
        with tracer.start_as_current_span(
            "pubsub_bus.handler",
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
                    "pubsub_bus_handler_error",
                    topic=msg.topic,
                    sender=msg.sender,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise

    # ── History & Replay ─────────────────────────────────────────────

    def history(
        self,
        topic: str,
        *,
        limit: int = 50,
    ) -> list[AgentMessage]:
        """Return recent messages from in-memory ring buffer.

        Same-lifecycle debugging only. For persistent history,
        use Cloud Pub/Sub's message retention + replay().
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

        For production replay from Cloud Pub/Sub retained messages,
        use the Pub/Sub subscription seek API at the infrastructure level.
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
        """Number of active local subscriptions."""
        return len(self._subscriptions)

    @property
    def stats(self) -> dict[str, int]:
        """Bus-level statistics."""
        return dict(self._stats)

    # ── Reset ────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all local subscriptions and history."""
        self._subscriptions.clear()
        self._history.clear()
        self._middleware.clear()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}
        logger.debug("pubsub_bus_cleared")

    def __repr__(self) -> str:
        return (
            f"CloudPubSubEventBus("
            f"project={self._project_id!r}, "
            f"topic={self._topic_name!r}, "
            f"subscriptions={self.subscription_count})"
        )
