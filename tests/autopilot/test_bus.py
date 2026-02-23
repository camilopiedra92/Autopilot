"""
Tests for autopilot.core.bus — Unified Event Bus typed pub/sub messaging.

Covers:
  - AgentMessage: creation, defaults, serialization
  - EventBus: subscribe, publish, unsubscribe, history, clear, middleware, replay
  - Wildcard topic matching
  - Dead-letter isolation (handler errors don't block others)
  - Singleton: get_event_bus / reset_event_bus
  - AgentContext integration: ctx.bus, ctx.publish(), ctx.subscribe()
"""

import pytest
from unittest.mock import AsyncMock

from autopilot.core.bus import (
    EventBus,
    AgentMessage,
    Subscription,
    get_event_bus,
    reset_event_bus,
)
from autopilot.core.context import AgentContext
from autopilot.errors import BusError, BusTimeoutError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def bus():
    """Fresh EventBus for each test."""
    return EventBus()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is reset between tests."""
    reset_event_bus()
    yield
    reset_event_bus()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentMessage Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentMessage:
    def test_creation_with_defaults(self):
        msg = AgentMessage(topic="test.event")
        assert msg.topic == "test.event"
        assert msg.sender == ""
        assert msg.payload == {}
        assert msg.timestamp  # Non-empty ISO string
        assert msg.correlation_id  # Non-empty UUID

    def test_creation_with_all_fields(self):
        msg = AgentMessage(
            topic="agent.error",
            sender="parser",
            payload={"detail": "timeout"},
            correlation_id="abc123",
        )
        assert msg.topic == "agent.error"
        assert msg.sender == "parser"
        assert msg.payload == {"detail": "timeout"}
        assert msg.correlation_id == "abc123"

    def test_serialization(self):
        msg = AgentMessage(topic="test", sender="a")
        d = msg.model_dump()
        assert d["topic"] == "test"
        assert d["sender"] == "a"
        assert "timestamp" in d
        assert "correlation_id" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Subscription Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSubscription:
    def test_subscription_created(self, bus):
        handler = AsyncMock()
        sub = bus.subscribe("test.topic", handler)
        assert isinstance(sub, Subscription)
        assert sub.topic_pattern == "test.topic"
        assert sub.handler is handler
        assert bus.subscription_count == 1

    def test_subscription_id_unique(self, bus):
        h1 = AsyncMock()
        h2 = AsyncMock()
        sub1 = bus.subscribe("a", h1)
        sub2 = bus.subscribe("b", h2)
        assert sub1.id != sub2.id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Publish / Subscribe Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPubSub:
    @pytest.mark.asyncio
    async def test_publish_subscribe_basic(self, bus):
        """A handler receives a published message."""
        received = []

        async def handler(msg: AgentMessage):
            received.append(msg)

        bus.subscribe("agent.completed", handler)
        msg = await bus.publish("agent.completed", {"result": 42}, sender="test")

        assert len(received) == 1
        assert received[0].topic == "agent.completed"
        assert received[0].payload == {"result": 42}
        assert received[0].sender == "test"
        assert isinstance(msg, AgentMessage)

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        """Multiple handlers on the same topic all fire."""
        count = {"a": 0, "b": 0}

        async def handler_a(msg):
            count["a"] += 1

        async def handler_b(msg):
            count["b"] += 1

        bus.subscribe("event", handler_a)
        bus.subscribe("event", handler_b)
        await bus.publish("event")

        assert count["a"] == 1
        assert count["b"] == 1

    @pytest.mark.asyncio
    async def test_no_subscribers_noop(self, bus):
        """Publishing to a topic with no subscribers succeeds silently."""
        msg = await bus.publish("nobody.listening", {"x": 1})
        assert msg.topic == "nobody.listening"
        assert bus.stats["published"] == 1
        assert bus.stats["delivered"] == 0

    @pytest.mark.asyncio
    async def test_publish_returns_message(self, bus):
        """publish() returns the AgentMessage that was created."""
        msg = await bus.publish(
            "test", {"key": "val"}, sender="s", correlation_id="cid"
        )
        assert msg.topic == "test"
        assert msg.payload == {"key": "val"}
        assert msg.sender == "s"
        assert msg.correlation_id == "cid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Wildcard Topic Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWildcardTopics:
    @pytest.mark.asyncio
    async def test_star_wildcard(self, bus):
        """'agent.*' matches 'agent.error' and 'agent.started'."""
        received = []

        async def handler(msg):
            received.append(msg.topic)

        bus.subscribe("agent.*", handler)

        await bus.publish("agent.error")
        await bus.publish("agent.started")
        await bus.publish("workflow.error")  # Should NOT match

        assert "agent.error" in received
        assert "agent.started" in received
        assert "workflow.error" not in received
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_catch_all_wildcard(self, bus):
        """'*' matches everything."""
        received = []

        async def handler(msg):
            received.append(msg.topic)

        bus.subscribe("*", handler)

        await bus.publish("a")
        await bus.publish("b")

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_question_mark_wildcard(self, bus):
        """'agent.?' matches 'agent.a' but not 'agent.ab'."""
        received = []

        async def handler(msg):
            received.append(msg.topic)

        bus.subscribe("agent.?", handler)

        await bus.publish("agent.a")
        await bus.publish("agent.ab")

        assert "agent.a" in received
        assert "agent.ab" not in received


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unsubscribe Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self, bus):
        """After unsubscribe, handler stops receiving messages."""
        received = []

        async def handler(msg):
            received.append(msg)

        sub = bus.subscribe("event", handler)

        await bus.publish("event")
        assert len(received) == 1

        result = bus.unsubscribe(sub)
        assert result is True
        assert bus.subscription_count == 0

        await bus.publish("event")
        assert len(received) == 1  # No new message

    def test_unsubscribe_unknown_returns_false(self, bus):
        """Unsubscribing an unknown subscription returns False."""
        fake_sub = Subscription(id="nonexistent", topic_pattern="x")
        assert bus.unsubscribe(fake_sub) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dead-Letter Isolation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeadLetterIsolation:
    @pytest.mark.asyncio
    async def test_handler_error_does_not_block_others(self, bus):
        """A failing handler doesn't block other handlers."""
        success_received = []

        async def bad_handler(msg):
            raise ValueError("I broke!")

        async def good_handler(msg):
            success_received.append(msg)

        bus.subscribe("event", bad_handler)
        bus.subscribe("event", good_handler)

        await bus.publish("event", {"test": True})

        # Good handler still received the message
        assert len(success_received) == 1
        assert bus.stats["errors"] == 1
        assert bus.stats["delivered"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  History Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHistory:
    @pytest.mark.asyncio
    async def test_message_history(self, bus):
        """history() returns recent messages in reverse chronological order."""
        await bus.publish("events", {"n": 1})
        await bus.publish("events", {"n": 2})
        await bus.publish("events", {"n": 3})

        history = bus.history("events")
        assert len(history) == 3
        # Newest first
        assert history[0].payload["n"] == 3
        assert history[1].payload["n"] == 2
        assert history[2].payload["n"] == 1

    @pytest.mark.asyncio
    async def test_history_limit(self):
        """Ring buffer respects max capacity."""
        bus = EventBus(history_limit=3)

        for i in range(10):
            await bus.publish("topic", {"n": i})

        history = bus.history("topic")
        assert len(history) == 3
        # Only the last 3
        assert [m.payload["n"] for m in history] == [9, 8, 7]

    @pytest.mark.asyncio
    async def test_history_empty_topic(self, bus):
        """history() for unrecognized topic returns empty list."""
        assert bus.history("nonexistent") == []

    @pytest.mark.asyncio
    async def test_history_limit_param(self, bus):
        """history(limit=N) restricts results."""
        for i in range(5):
            await bus.publish("t", {"i": i})

        history = bus.history("t", limit=2)
        assert len(history) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Clear / Reset Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClearReset:
    @pytest.mark.asyncio
    async def test_clear(self, bus):
        """clear() removes all subscriptions and history."""
        bus.subscribe("a", AsyncMock())
        await bus.publish("a")

        bus.clear()

        assert bus.subscription_count == 0
        assert bus.history("a") == []
        assert bus.stats == {"published": 0, "delivered": 0, "errors": 0}

    def test_repr(self, bus):
        r = repr(bus)
        assert "EventBus" in r
        assert "subscriptions=0" in r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSingleton:
    def test_get_event_bus_returns_same_instance(self):
        """get_event_bus() returns the same instance each time."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus_provides_fresh_instance(self):
        """reset_event_bus() provides a fresh instance."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentContext Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestContextIntegration:
    def test_context_bus_property(self):
        """ctx.bus returns the global EventBus."""
        ctx = AgentContext(pipeline_name="test")
        bus = ctx.bus
        assert isinstance(bus, EventBus)
        assert bus is get_event_bus()

    @pytest.mark.asyncio
    async def test_context_publish_convenience(self):
        """ctx.publish() delegates to bus.publish with correct sender."""
        received = []

        async def handler(msg):
            received.append(msg)

        ctx = AgentContext(pipeline_name="my_pipeline")
        ctx.bus.subscribe("agent.done", handler)
        await ctx.publish("agent.done", {"status": "ok"})

        assert len(received) == 1
        assert received[0].sender == "my_pipeline"
        assert "status" in received[0].payload
        assert received[0].payload["status"] == "ok"

    def test_context_subscribe_convenience(self):
        """ctx.subscribe() returns a Subscription handle."""
        ctx = AgentContext(pipeline_name="test")
        sub = ctx.subscribe("topic", AsyncMock())
        assert isinstance(sub, Subscription)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Error Taxonomy Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBusErrors:
    def test_bus_error_attributes(self):
        err = BusError("bus failed")
        assert err.error_code == "BUS_ERROR"
        assert err.http_status == 500
        assert err.retryable is False

    def test_bus_timeout_error_attributes(self):
        err = BusTimeoutError("timed out")
        assert err.error_code == "BUS_TIMEOUT"
        assert err.http_status == 504
        assert err.retryable is True
        assert isinstance(err, BusError)

    def test_bus_error_serialization(self):
        err = BusError("something broke")
        d = err.to_dict()
        assert d["error_code"] == "BUS_ERROR"
        assert d["message"] == "something broke"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration: End-to-End Agent Communication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestE2EAgentCommunication:
    """Simulates real cross-agent communication via the bus."""

    @pytest.mark.asyncio
    async def test_monitor_agent_receives_error_events(self, bus):
        """
        Scenario: A monitoring agent subscribes to 'agent.*' errors.
        When the parser agent publishes 'agent.error', the monitor
        captures it and logs the alert.
        """
        alerts = []

        async def monitor_handler(msg: AgentMessage):
            alerts.append(
                {
                    "from": msg.sender,
                    "error": msg.payload.get("error"),
                    "topic": msg.topic,
                }
            )

        bus.subscribe("agent.*", monitor_handler)

        # Parser agent reports an error
        await bus.publish(
            "agent.error",
            {"error": "Failed to parse email", "email_id": "msg-123"},
            sender="email_parser",
        )

        # Categorizer agent reports completion
        await bus.publish(
            "agent.completed",
            {"category": "Shopping"},
            sender="categorizer",
        )

        assert len(alerts) == 2
        assert alerts[0]["from"] == "email_parser"
        assert alerts[0]["error"] == "Failed to parse email"
        assert alerts[1]["from"] == "categorizer"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, bus):
        """Stats accurately track published, delivered, and errors."""

        async def good(msg):
            pass

        async def bad(msg):
            raise RuntimeError("boom")

        bus.subscribe("t", good)
        bus.subscribe("t", bad)

        await bus.publish("t")
        await bus.publish("t")

        assert bus.stats["published"] == 2
        assert bus.stats["delivered"] == 2  # 1 good per publish
        assert bus.stats["errors"] == 2  # 1 bad per publish
