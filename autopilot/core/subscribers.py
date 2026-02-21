"""
SubscriberRegistry — Platform-level reactive event subscriber management.

Provides a centralized registry for managing event subscribers on the
``AgentBus``.  Workflows register handlers during their ``setup()``
lifecycle hook, and the registry wires them to the global bus.

Key features:
  - Singleton pattern via ``get_subscriber_registry()``
  - Automatic wiring to the ``AgentBus`` singleton
  - Named subscriptions for introspection and debugging
  - Bulk unregister for clean teardown (tests, shutdown)

Usage::

    from autopilot.core.subscribers import get_subscriber_registry

    registry = get_subscriber_registry()

    # Register a reactive handler
    registry.register(
        "transaction.created",
        on_transaction_created,
        name="telegram_notifier",
    )

    # Introspect active subscriptions
    print(registry.registered)

    # Teardown
    registry.unregister_all()

Design:
  - Handlers receive ``AgentMessage`` (from ``core.bus``).
  - Dead-letter isolation is handled by the bus itself.
  - OTel tracing propagates through the bus's span hierarchy.
"""

from __future__ import annotations

import structlog
from typing import Any

from autopilot.core.bus import (
    MessageHandler,
    Subscription,
    get_agent_bus,
)

logger = structlog.get_logger(__name__)


class SubscriberRegistry:
    """
    Manages lifecycle of event subscribers on the AgentBus.

    Subscribers are async handlers that react to bus events.
    The registry tracks them by name for introspection, debugging,
    and bulk teardown.
    """

    def __init__(self) -> None:
        self._entries: list[_RegistryEntry] = []

    def register(
        self,
        topic: str,
        handler: MessageHandler,
        *,
        name: str,
    ) -> Subscription:
        """
        Register a reactive event handler on the global AgentBus.

        Args:
            topic: Topic pattern to subscribe to (supports wildcards).
            handler: Async callable ``(AgentMessage) -> None``.
            name: Human-readable name for logging and introspection.

        Returns:
            A ``Subscription`` handle for manual unsubscribe if needed.
        """
        bus = get_agent_bus()
        sub = bus.subscribe(topic, handler)
        self._entries.append(
            _RegistryEntry(name=name, topic=topic, subscription=sub)
        )
        logger.info(
            "subscriber_registered",
            name=name,
            topic=topic,
            sub_id=sub.id,
        )
        return sub

    def unregister_all(self) -> None:
        """Remove all registered subscribers from the bus."""
        bus = get_agent_bus()
        for entry in self._entries:
            bus.unsubscribe(entry.subscription)
            logger.debug(
                "subscriber_unregistered",
                name=entry.name,
                topic=entry.topic,
            )
        self._entries.clear()
        logger.info("subscriber_registry_cleared")

    @property
    def registered(self) -> list[dict[str, Any]]:
        """Introspection: list all active subscriber registrations."""
        return [
            {
                "name": e.name,
                "topic": e.topic,
                "sub_id": e.subscription.id,
            }
            for e in self._entries
        ]

    @property
    def count(self) -> int:
        """Number of active subscriber registrations."""
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SubscriberRegistry(count={self.count})"


class _RegistryEntry:
    """Internal: tracks a single subscriber registration."""

    __slots__ = ("name", "topic", "subscription")

    def __init__(self, name: str, topic: str, subscription: Subscription):
        self.name = name
        self.topic = topic
        self.subscription = subscription


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_subscriber_registry: SubscriberRegistry | None = None


def get_subscriber_registry() -> SubscriberRegistry:
    """Get or create the global SubscriberRegistry singleton."""
    global _subscriber_registry
    if _subscriber_registry is None:
        _subscriber_registry = SubscriberRegistry()
    return _subscriber_registry


def reset_subscriber_registry() -> None:
    """
    Reset the global SubscriberRegistry singleton.

    Unregisters all subscribers and creates a fresh registry.
    Useful in tests to ensure clean state.
    """
    global _subscriber_registry
    if _subscriber_registry is not None:
        _subscriber_registry.unregister_all()
    _subscriber_registry = None
