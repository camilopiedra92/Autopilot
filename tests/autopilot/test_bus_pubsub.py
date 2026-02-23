"""
Tests for CloudPubSubEventBus and create_event_bus factory.

Covers:
  - CloudPubSubEventBus: protocol compliance, hybrid dispatch, subscribe/unsub,
    wildcards, dead-letter isolation, history, stats, from_env()
  - create_event_bus(): factory backend selection via arg and env var
  - get_event_bus(): singleton returns EventBusProtocol
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from autopilot.core.bus import (
    AgentMessage,
    EventBus,
    EventBusProtocol,
    Subscription,
    create_event_bus,
    get_event_bus,
    reset_event_bus,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def mock_publisher():
    """Mock the Pub/Sub PublisherClient."""
    with patch("autopilot.core.bus_pubsub.pubsub_v1.PublisherClient") as mock_cls:
        publisher = MagicMock()
        # create_topic should succeed silently (already exists)
        publisher.create_topic.side_effect = Exception("ALREADY_EXISTS")
        # publish returns a future
        future = MagicMock()
        future.result.return_value = "message-id-123"
        publisher.publish.return_value = future
        mock_cls.return_value = publisher
        yield publisher


@pytest.fixture
def pubsub_bus(mock_publisher):
    """Fresh CloudPubSubEventBus with mocked Pub/Sub client."""
    from autopilot.core.bus_pubsub import CloudPubSubEventBus

    return CloudPubSubEventBus(
        project_id="test-project",
        topic_name="test-events",
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is reset between tests."""
    reset_event_bus()
    yield
    reset_event_bus()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Protocol Compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestProtocolCompliance:
    def test_implements_protocol(self, pubsub_bus):
        """CloudPubSubEventBus satisfies EventBusProtocol ABC."""
        assert isinstance(pubsub_bus, EventBusProtocol)

    def test_has_all_protocol_methods(self, pubsub_bus):
        """All required protocol methods are implemented."""
        assert hasattr(pubsub_bus, "subscribe")
        assert hasattr(pubsub_bus, "unsubscribe")
        assert hasattr(pubsub_bus, "publish")
        assert hasattr(pubsub_bus, "history")
        assert hasattr(pubsub_bus, "replay")
        assert hasattr(pubsub_bus, "clear")
        assert hasattr(pubsub_bus, "subscription_count")
        assert hasattr(pubsub_bus, "stats")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hybrid Publish Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHybridPublish:
    @pytest.mark.asyncio
    async def test_publish_dispatches_locally_and_to_pubsub(
        self, pubsub_bus, mock_publisher
    ):
        """publish() sends to both local subscribers AND Cloud Pub/Sub."""
        received = []

        async def handler(msg):
            received.append(msg)

        pubsub_bus.subscribe("test.topic", handler)
        msg = await pubsub_bus.publish("test.topic", {"key": "val"}, sender="test")

        # Local delivery happened
        assert len(received) == 1
        assert received[0].topic == "test.topic"
        assert received[0].payload == {"key": "val"}

        # Pub/Sub publish was called
        mock_publisher.publish.assert_called_once()
        call_args = mock_publisher.publish.call_args
        assert call_args[0][0] == "projects/test-project/topics/test-events"

        # Returned message is correct
        assert isinstance(msg, AgentMessage)
        assert msg.topic == "test.topic"

    @pytest.mark.asyncio
    async def test_pubsub_error_does_not_break_local_delivery(
        self, pubsub_bus, mock_publisher
    ):
        """If Pub/Sub publish fails, local delivery still works."""
        mock_publisher.publish.side_effect = Exception("Network error")

        received = []

        async def handler(msg):
            received.append(msg)

        pubsub_bus.subscribe("test", handler)
        msg = await pubsub_bus.publish("test", {"data": 1})

        # Local delivery succeeded despite Pub/Sub failure
        assert len(received) == 1
        assert msg.topic == "test"

    @pytest.mark.asyncio
    async def test_publish_with_correlation_id(self, pubsub_bus, mock_publisher):
        """correlation_id is passed through to the message."""
        msg = await pubsub_bus.publish(
            "t", {"x": 1}, sender="s", correlation_id="cid-123"
        )
        assert msg.correlation_id == "cid-123"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Subscribe / Unsubscribe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSubscription:
    def test_subscribe_returns_handle(self, pubsub_bus):
        """subscribe() returns a Subscription handle."""
        handler = AsyncMock()
        sub = pubsub_bus.subscribe("topic", handler)
        assert isinstance(sub, Subscription)
        assert sub.topic_pattern == "topic"
        assert pubsub_bus.subscription_count == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self, pubsub_bus):
        """After unsubscribe, handler stops receiving."""
        received = []

        async def handler(msg):
            received.append(msg)

        sub = pubsub_bus.subscribe("event", handler)
        await pubsub_bus.publish("event")
        assert len(received) == 1

        result = pubsub_bus.unsubscribe(sub)
        assert result is True
        assert pubsub_bus.subscription_count == 0

        await pubsub_bus.publish("event")
        assert len(received) == 1  # No new message

    def test_unsubscribe_unknown_returns_false(self, pubsub_bus):
        """Unsubscribing an unknown subscription returns False."""
        fake = Subscription(id="nonexistent", topic_pattern="x")
        assert pubsub_bus.unsubscribe(fake) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Wildcard Topics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestWildcardTopics:
    @pytest.mark.asyncio
    async def test_star_wildcard(self, pubsub_bus):
        """'agent.*' matches 'agent.error' and 'agent.started'."""
        received = []

        async def handler(msg):
            received.append(msg.topic)

        pubsub_bus.subscribe("agent.*", handler)
        await pubsub_bus.publish("agent.error")
        await pubsub_bus.publish("agent.started")
        await pubsub_bus.publish("workflow.error")  # Should NOT match

        assert "agent.error" in received
        assert "agent.started" in received
        assert "workflow.error" not in received

    @pytest.mark.asyncio
    async def test_catch_all(self, pubsub_bus):
        """'*' matches everything."""
        received = []

        async def handler(msg):
            received.append(msg.topic)

        pubsub_bus.subscribe("*", handler)
        await pubsub_bus.publish("a")
        await pubsub_bus.publish("b")
        assert len(received) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dead-Letter Isolation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDeadLetterIsolation:
    @pytest.mark.asyncio
    async def test_handler_error_does_not_block_others(self, pubsub_bus):
        """A failing handler doesn't block other handlers."""
        success = []

        async def bad(msg):
            raise ValueError("broken")

        async def good(msg):
            success.append(msg)

        pubsub_bus.subscribe("event", bad)
        pubsub_bus.subscribe("event", good)
        await pubsub_bus.publish("event")

        assert len(success) == 1
        assert pubsub_bus.stats["errors"] == 1
        assert pubsub_bus.stats["delivered"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  History & Stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHistoryStats:
    @pytest.mark.asyncio
    async def test_history_ring_buffer(self, pubsub_bus):
        """history() returns recent messages."""
        await pubsub_bus.publish("events", {"n": 1})
        await pubsub_bus.publish("events", {"n": 2})
        await pubsub_bus.publish("events", {"n": 3})

        history = pubsub_bus.history("events")
        assert len(history) == 3
        assert history[0].payload["n"] == 3  # Newest first

    @pytest.mark.asyncio
    async def test_stats_tracking(self, pubsub_bus):
        """Stats track published, delivered, errors."""

        async def good(msg):
            pass

        async def bad(msg):
            raise RuntimeError("boom")

        pubsub_bus.subscribe("t", good)
        pubsub_bus.subscribe("t", bad)

        await pubsub_bus.publish("t")
        await pubsub_bus.publish("t")

        assert pubsub_bus.stats["published"] == 2
        assert pubsub_bus.stats["delivered"] == 2
        assert pubsub_bus.stats["errors"] == 2

    @pytest.mark.asyncio
    async def test_clear(self, pubsub_bus):
        """clear() removes subscriptions and history."""
        pubsub_bus.subscribe("a", AsyncMock())
        await pubsub_bus.publish("a")
        pubsub_bus.clear()

        assert pubsub_bus.subscription_count == 0
        assert pubsub_bus.history("a") == []
        assert pubsub_bus.stats == {"published": 0, "delivered": 0, "errors": 0}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  from_env()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFromEnv:
    def test_from_env_reads_project(self, mock_publisher):
        """from_env() reads GOOGLE_CLOUD_PROJECT."""
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "my-project"}):
            bus = CloudPubSubEventBus.from_env()
            assert bus._project_id == "my-project"
            assert bus._topic_name == "autopilot-events"

    def test_from_env_reads_custom_topic(self, mock_publisher):
        """from_env() reads EVENTBUS_PUBSUB_TOPIC."""
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        with patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "proj",
                "EVENTBUS_PUBSUB_TOPIC": "custom-topic",
            },
        ):
            bus = CloudPubSubEventBus.from_env()
            assert bus._topic_name == "custom-topic"

    def test_from_env_raises_without_project(self, mock_publisher):
        """from_env() raises if no project ID available."""
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        with patch.dict(os.environ, {}, clear=True):
            # Ensure GOOGLE_CLOUD_PROJECT and GCLOUD_PROJECT are not set
            os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
            os.environ.pop("GCLOUD_PROJECT", None)
            with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
                CloudPubSubEventBus.from_env()

    def test_repr(self, pubsub_bus):
        """repr() shows project and topic."""
        r = repr(pubsub_bus)
        assert "CloudPubSubEventBus" in r
        assert "test-project" in r
        assert "test-events" in r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Factory Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFactory:
    def test_create_event_bus_memory(self):
        """create_event_bus('memory') returns EventBus."""
        bus = create_event_bus("memory")
        assert isinstance(bus, EventBus)
        assert isinstance(bus, EventBusProtocol)

    def test_create_event_bus_default(self):
        """create_event_bus() defaults to memory."""
        bus = create_event_bus()
        assert isinstance(bus, EventBus)

    def test_create_event_bus_pubsub(self, mock_publisher):
        """create_event_bus('pubsub') returns CloudPubSubEventBus."""
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-proj"}):
            bus = create_event_bus("pubsub")
            assert isinstance(bus, CloudPubSubEventBus)
            assert isinstance(bus, EventBusProtocol)

    def test_create_event_bus_env_var(self, mock_publisher):
        """EVENTBUS_BACKEND env var selects backend."""
        from autopilot.core.bus_pubsub import CloudPubSubEventBus

        with patch.dict(
            os.environ,
            {
                "EVENTBUS_BACKEND": "pubsub",
                "GOOGLE_CLOUD_PROJECT": "env-proj",
            },
        ):
            bus = create_event_bus()
            assert isinstance(bus, CloudPubSubEventBus)

    def test_get_event_bus_returns_protocol(self):
        """get_event_bus() returns EventBusProtocol type."""
        bus = get_event_bus()
        assert isinstance(bus, EventBusProtocol)

    def test_get_event_bus_singleton(self):
        """get_event_bus() returns same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
