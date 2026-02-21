"""
Tests for AgentBus Events — Reactive Decoupling.

Covers:
  - FunctionalAgent ctx injection
  - SubscriberRegistry lifecycle (register, introspect, unregister)
  - publish_transaction_event step
  - TransactionEvent model
  - Telegram subscriber handler
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from autopilot.core.agent import FunctionalAgent
from autopilot.core.bus import AgentBus, AgentMessage, get_agent_bus, reset_agent_bus
from autopilot.core.context import AgentContext
from autopilot.core.subscribers import (
    SubscriberRegistry,
    get_subscriber_registry,
    reset_subscriber_registry,
)
from workflows.bank_to_ynab.models.events import TransactionEvent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset bus and subscriber registry between tests."""
    yield
    reset_agent_bus()
    reset_subscriber_registry()


@pytest.fixture
def bus():
    """Fresh AgentBus for each test."""
    reset_agent_bus()
    return get_agent_bus()


@pytest.fixture
def registry():
    """Fresh SubscriberRegistry for each test."""
    reset_subscriber_registry()
    return get_subscriber_registry()


@pytest.fixture
def sample_final_result_data():
    """Sample final_result_data from push_to_ynab step."""
    return {
        "payee": "Restaurante El Cielo",
        "amount": -50000,
        "date": "2026-02-18",
        "memo": "Compra con tarjeta terminada en 52e0",
        "budget_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "account_id": "f1e2d3c4-b5a6-7890-abcd-ef0987654321",
        "category_id": "c1d2e3f4-a5b6-7890-abcd-ef1122334455",
        "created_in_ynab": True,
        "ynab_transaction_id": "tx-mock-001",
        "is_successful": True,
        "match_confidence": "high",
        "match_reasoning": "Card suffix match",
        "category_reasoning": "Restaurant → Dining Out",
        "category_balance": {
            "category_name": "Dining Out",
            "budgeted": 500000,
            "activity": -320000,
            "balance": 180000,
            "is_overspent": False,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FunctionalAgent ctx Injection Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCtxInjection:
    """Tests that FunctionalAgent auto-injects ctx when declared."""

    @pytest.mark.asyncio
    async def test_ctx_injected_when_annotated(self):
        """FunctionalAgent injects AgentContext when param is `ctx: AgentContext`."""
        received_ctx = None

        async def step_with_ctx(ctx: AgentContext, **state) -> dict:
            nonlocal received_ctx
            received_ctx = ctx
            return {"got_ctx": True}

        agent = FunctionalAgent(step_with_ctx)
        ctx = AgentContext(pipeline_name="test")
        result = await agent.invoke(ctx, {"foo": "bar"})

        assert received_ctx is not None
        assert isinstance(received_ctx, AgentContext)
        assert result.get("got_ctx") is True

    @pytest.mark.asyncio
    async def test_ctx_not_injected_for_kwargs_only(self):
        """Functions with only **kwargs do NOT get ctx injected as a kwarg."""
        received_keys = None

        def step_without_ctx(**state) -> dict:
            nonlocal received_keys
            received_keys = set(state.keys())
            return {"ok": True}

        agent = FunctionalAgent(step_without_ctx)
        ctx = AgentContext(pipeline_name="test")
        ctx.update_state({"foo": "bar"})
        await agent.invoke(ctx, {"baz": 1})

        # ctx should NOT appear in state keys
        assert "ctx" not in received_keys

    @pytest.mark.asyncio
    async def test_ctx_injection_with_string_annotation(self):
        """FunctionalAgent handles string annotation 'AgentContext' from TYPE_CHECKING."""
        received_ctx = None

        # Simulate TYPE_CHECKING pattern: annotation is a string
        async def step_with_str_ctx(ctx: "AgentContext", **state) -> dict:
            nonlocal received_ctx
            received_ctx = ctx
            return {}

        agent = FunctionalAgent(step_with_str_ctx)
        ctx = AgentContext(pipeline_name="test")
        await agent.invoke(ctx, {})

        assert received_ctx is not None

    @pytest.mark.asyncio
    async def test_ctx_injection_preserves_other_params(self):
        """ctx injection doesn't interfere with normal parameter extraction."""
        async def step(ctx: AgentContext, payee: str, amount: float) -> dict:
            return {"payee": payee, "amount": amount, "has_ctx": ctx is not None}

        agent = FunctionalAgent(step)
        ctx = AgentContext(pipeline_name="test")
        result = await agent.invoke(ctx, {"payee": "Test", "amount": 100.0})

        assert result["payee"] == "Test"
        assert result["amount"] == 100.0
        assert result["has_ctx"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SubscriberRegistry Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSubscriberRegistry:
    """Tests for the platform SubscriberRegistry."""

    def test_register_creates_subscription(self, registry, bus):
        async def handler(msg):
            pass

        sub = registry.register("test.topic", handler, name="my_handler")
        assert sub is not None
        assert registry.count == 1

    def test_registered_introspection(self, registry, bus):
        async def handler(msg):
            pass

        registry.register("a.topic", handler, name="handler_a")
        registry.register("b.topic", handler, name="handler_b")

        entries = registry.registered
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert names == {"handler_a", "handler_b"}

    def test_unregister_all_clears(self, registry, bus):
        async def handler(msg):
            pass

        registry.register("test.*", handler, name="h1")
        registry.register("test.*", handler, name="h2")
        assert registry.count == 2

        registry.unregister_all()
        assert registry.count == 0

    @pytest.mark.asyncio
    async def test_registered_handler_receives_events(self, registry, bus):
        received = []

        async def handler(msg):
            received.append(msg)

        registry.register("test.event", handler, name="test_handler")
        await bus.publish("test.event", {"data": 42}, sender="test")

        assert len(received) == 1
        assert received[0].payload["data"] == 42

    @pytest.mark.asyncio
    async def test_unregistered_handler_stops_receiving(self, registry, bus):
        received = []

        async def handler(msg):
            received.append(msg)

        registry.register("test.event", handler, name="test_handler")
        await bus.publish("test.event", {"seq": 1}, sender="test")
        assert len(received) == 1

        registry.unregister_all()
        await bus.publish("test.event", {"seq": 2}, sender="test")
        assert len(received) == 1  # Still 1, no new events

    def test_singleton_returns_same_instance(self):
        a = get_subscriber_registry()
        b = get_subscriber_registry()
        assert a is b


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TransactionEvent Model Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTransactionEvent:
    """Tests for the TransactionEvent Pydantic model."""

    def test_from_pipeline_state(self, sample_final_result_data):
        event = TransactionEvent.from_pipeline_state(sample_final_result_data)
        assert event.payee == "Restaurante El Cielo"
        assert event.amount == -50000
        assert event.created_in_ynab is True
        assert event.category_balance is not None

    def test_from_pipeline_state_ignores_unknown_keys(self):
        data = {"payee": "Test", "unknown_field_xyz": 123}
        event = TransactionEvent.from_pipeline_state(data)
        assert event.payee == "Test"

    def test_model_dump_roundtrip(self, sample_final_result_data):
        event = TransactionEvent.from_pipeline_state(sample_final_result_data)
        dumped = event.model_dump()
        restored = TransactionEvent.model_validate(dumped)
        assert restored.payee == event.payee
        assert restored.amount == event.amount


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  publish_transaction_event Step Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPublishTransactionEvent:
    """Tests for the publish_transaction_event pipeline step."""

    @pytest.mark.asyncio
    async def test_publishes_to_bus(self, bus, sample_final_result_data):
        """The step publishes a transaction.created event to the AgentBus."""
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("transaction.created", handler)

        from workflows.bank_to_ynab.steps import publish_transaction_event

        agent = FunctionalAgent(publish_transaction_event)
        ctx = AgentContext(pipeline_name="bank_to_ynab")
        ctx.update_state({"final_result_data": sample_final_result_data})

        result = await agent.invoke(ctx, ctx.state)

        assert result.get("event_published") is True
        assert len(received) == 1
        assert received[0].payload["payee"] == "Restaurante El Cielo"
        assert received[0].topic == "transaction.created"

    @pytest.mark.asyncio
    async def test_skips_when_no_final_result(self, bus):
        """No event is published when final_result_data is empty."""
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("transaction.created", handler)

        from workflows.bank_to_ynab.steps import publish_transaction_event

        agent = FunctionalAgent(publish_transaction_event)
        ctx = AgentContext(pipeline_name="bank_to_ynab")

        result = await agent.invoke(ctx, {})

        assert result == {}
        assert len(received) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Telegram Subscriber Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTelegramSubscriber:
    """Tests for the reactive Telegram notification subscriber."""

    @pytest.mark.asyncio
    async def test_formats_notifier_context(self):
        """The subscriber formats the same context the old pipeline step produced."""
        from workflows.bank_to_ynab.subscribers.telegram_subscriber import (
            _format_notifier_context,
        )

        payload = {
            "payee": "Restaurante El Cielo",
            "amount": -50000,
            "category_balance": {
                "category_name": "Dining Out",
                "budgeted": 500000,
                "activity": -320000,
                "balance": 180000,
                "is_overspent": False,
            },
        }

        result = _format_notifier_context(payload)
        assert "final_result_data" in result
        assert "category_balance" in result
        assert "message" in result
        assert "Restaurante El Cielo" in result["final_result_data"]
        assert "Dining Out" in result["category_balance"]

    @pytest.mark.asyncio
    async def test_skips_empty_payload(self):
        """Subscriber does nothing on empty payload."""
        from workflows.bank_to_ynab.subscribers.telegram_subscriber import (
            on_transaction_created,
        )

        msg = AgentMessage(
            topic="transaction.created",
            payload={},
            sender="test",
        )

        # Should not raise
        await on_transaction_created(msg)

    @pytest.mark.asyncio
    async def test_invokes_notifier_agent(self, sample_final_result_data):
        """Subscriber runs the LLM notifier agent with correct state."""
        from workflows.bank_to_ynab.subscribers.telegram_subscriber import (
            on_transaction_created,
        )

        msg = AgentMessage(
            topic="transaction.created",
            payload=sample_final_result_data,
            sender="bank_to_ynab",
        )

        # Mock the ADKAgent.invoke to avoid actual LLM calls
        with patch(
            "autopilot.core.agent.ADKAgent"
        ) as MockADKAgent:
            mock_agent = AsyncMock()
            mock_agent.invoke = AsyncMock(return_value={})
            MockADKAgent.return_value = mock_agent

            await on_transaction_created(msg)

            # Verify the agent was instantiated and invoked
            MockADKAgent.assert_called_once()
            mock_agent.invoke.assert_called_once()

            # Check that the state passed to invoke contains formatted data
            call_args = mock_agent.invoke.call_args
            state = call_args[0][1]  # Second positional arg
            assert "final_result_data" in state
            assert "category_balance" in state
            assert "message" in state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  End-to-End: Bus → Subscriber Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBusSubscriberIntegration:
    """Integration: event published on bus triggers subscriber."""

    @pytest.mark.asyncio
    async def test_full_flow_bus_to_subscriber(
        self, bus, registry, sample_final_result_data
    ):
        """Publishing transaction.created on the bus triggers registered handler."""
        invoked = asyncio.Event()

        async def mock_handler(msg: AgentMessage):
            assert msg.topic == "transaction.created"
            assert msg.payload["payee"] == "Restaurante El Cielo"
            invoked.set()

        registry.register("transaction.created", mock_handler, name="test_handler")
        await bus.publish(
            "transaction.created",
            sample_final_result_data,
            sender="bank_to_ynab",
        )

        assert invoked.is_set()
