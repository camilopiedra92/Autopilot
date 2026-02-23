"""
Tests for autopilot.core — Edge-Native Agentic Primitives.

Covers:
  - AgentContext: creation, state management, child context, elapsed_ms
  - BaseAgent: subclassing, invoke() lifecycle wrapping
  - FunctionalAgent: sync/async function wrapping, state unpacking
  - Pipeline: sequential execution, state propagation, error handling
  - PipelineBuilder: auto-detection of step types, build validation
"""

import asyncio
import pytest
from typing import Any
from unittest.mock import MagicMock

from autopilot.core.context import AgentContext
from autopilot.core.agent import BaseAgent, FunctionalAgent, ADKAgent, functional_agent
from autopilot.core.pipeline import Pipeline, PipelineBuilder


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentContext Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentContext:
    def test_creation_defaults(self):
        ctx = AgentContext()
        assert ctx.execution_id  # UUID generated
        assert ctx.pipeline_name == ""
        assert ctx.state == {}
        assert ctx.metadata == {}
        assert ctx.logger is not None

    def test_creation_with_args(self):
        ctx = AgentContext(
            pipeline_name="test_pipeline",
            state={"key": "value"},
            metadata={"source": "test"},
        )
        assert ctx.pipeline_name == "test_pipeline"
        assert ctx.state == {"key": "value"}
        assert ctx.metadata == {"source": "test"}

    def test_get_and_update_state(self):
        ctx = AgentContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

        ctx.update_state({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

        # Updates are additive
        ctx.update_state({"c": 3})
        assert ctx.get("a") == 1
        assert ctx.get("c") == 3

    def test_elapsed_ms(self):
        ctx = AgentContext()
        # Should be > 0 but small
        assert ctx.elapsed_ms >= 0

    def test_for_step_shares_state(self):
        ctx = AgentContext(pipeline_name="parent")
        ctx.update_state({"shared": True})

        child = ctx.for_step("step_1")

        # Same execution_id
        assert child.execution_id == ctx.execution_id
        # Shared state (same reference)
        assert child.state is ctx.state
        assert child.get("shared") is True

        # Updating child state reflects in parent
        child.update_state({"from_child": True})
        assert ctx.get("from_child") is True

    @pytest.mark.asyncio
    async def test_publish_event(self):
        """ctx.publish() sends an event through the unified EventBus."""
        from autopilot.core.bus import get_event_bus, reset_event_bus

        reset_event_bus()
        bus = get_event_bus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("test_event", handler)
        ctx = AgentContext(pipeline_name="test")
        await ctx.publish("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0].topic == "test_event"
        assert received[0].payload["key"] == "value"
        assert received[0].payload["execution_id"] == ctx.execution_id
        assert received[0].sender == "test"
        reset_event_bus()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BaseAgent Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DoubleAgent(BaseAgent[dict, dict]):
    """Test agent that doubles a numeric 'value' in input."""

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        return {"value": input.get("value", 0) * 2}


class FailingAgent(BaseAgent[dict, dict]):
    """Test agent that always raises."""

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        raise RuntimeError("deliberate failure")


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_invoke_success(self):
        agent = DoubleAgent("doubler")

        ctx = AgentContext(pipeline_name="test")
        result = await agent.invoke(ctx, {"value": 5})

        assert result == {"value": 10}

    @pytest.mark.asyncio
    async def test_invoke_failure_raises(self):
        agent = FailingAgent("failer")

        ctx = AgentContext(pipeline_name="test")
        with pytest.raises(RuntimeError, match="deliberate failure"):
            await agent.invoke(ctx, {})

    def test_repr(self):
        agent = DoubleAgent("doubler", description="Doubles values")
        assert "DoubleAgent" in repr(agent)
        assert "doubler" in repr(agent)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FunctionalAgent Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFunctionalAgent:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def add_one(value: int) -> dict:
            return {"value": value + 1}

        agent = FunctionalAgent(add_one)
        assert agent.name == "add_one"

        ctx = AgentContext()
        result = await agent.invoke(ctx, {"value": 10})
        assert result == {"value": 11}

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def async_double(value: int) -> dict:
            return {"value": value * 2}

        agent = FunctionalAgent(async_double)

        ctx = AgentContext()
        result = await agent.invoke(ctx, {"value": 7})
        assert result == {"value": 14}

    @pytest.mark.asyncio
    async def test_pydantic_return(self):
        """FunctionalAgent should auto-dump Pydantic models."""
        from pydantic import BaseModel

        class Result(BaseModel):
            answer: str

        def make_result() -> Result:
            return Result(answer="hello")

        agent = FunctionalAgent(make_result)

        ctx = AgentContext()
        result = await agent.invoke(ctx, {})
        assert result == {"answer": "hello"}

    @pytest.mark.asyncio
    async def test_optional_pydantic_hydration(self):
        """FunctionalAgent should selectively auto-hydrate Optional[PydanticModel]."""
        from pydantic import BaseModel
        from typing import Optional

        class UserInfo(BaseModel):
            name: str

        def greet_user(user: Optional[UserInfo] = None) -> dict:
            if user:
                return {"message": f"Hello {user.name}"}
            return {"message": "Hello Guest"}

        agent = FunctionalAgent(greet_user)
        ctx = AgentContext()

        # Test 1: Hydrated correctly when input is provided
        result1 = await agent.invoke(ctx, {"user": {"name": "Alice"}})
        assert result1 == {"message": "Hello Alice"}

        # Test 2: Defaults to None when input is absent
        result2 = await agent.invoke(ctx, {})
        assert result2 == {"message": "Hello Guest"}

    @pytest.mark.asyncio
    async def test_pep604_union_hydration(self):
        """FunctionalAgent should auto-hydrate PEP 604 Unions (Type | None)."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str

        def greet_user(user: UserInfo | None = None) -> dict:
            if user:
                return {"message": f"Hello {user.name}"}
            return {"message": "Hello Guest"}

        agent = FunctionalAgent(greet_user)
        ctx = AgentContext()

        # Test 1: Hydrated correctly when input is provided
        result1 = await agent.invoke(ctx, {"user": {"name": "Bob"}})
        assert result1 == {"message": "Hello Bob"}

        # Test 2: Defaults to None when input is absent
        result2 = await agent.invoke(ctx, {})
        assert result2 == {"message": "Hello Guest"}

    @pytest.mark.asyncio
    async def test_strict_pydantic_hydration_raises_validation_error(self):
        """FunctionalAgent should strictly raise ValidationError if hydration fails, not swallow it."""
        from pydantic import BaseModel, ValidationError

        class StrictModel(BaseModel):
            required_field: str

        def strictly_typed_step(data: StrictModel) -> dict:
            return {"received": data.required_field}

        agent = FunctionalAgent(strictly_typed_step)
        ctx = AgentContext()

        # Passing an empty dict should fail fast violently at hydration
        with pytest.raises(ValidationError) as exc_info:
            await agent.invoke(ctx, {"data": {}})

        assert "required_field" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reads_from_ctx_state(self):
        """FunctionalAgent should read args from ctx.state if not in input."""

        def greet(name: str) -> dict:
            return {"greeting": f"Hello, {name}!"}

        agent = FunctionalAgent(greet)

        ctx = AgentContext(state={"name": "World"})
        result = await agent.invoke(ctx, {})
        assert result == {"greeting": "Hello, World!"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @functional_agent Decorator Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFunctionalAgentDecorator:
    def test_bare_decorator(self):
        @functional_agent
        def my_func(x: int) -> dict:
            return {"result": x}

        assert isinstance(my_func, FunctionalAgent)
        assert my_func.name == "my_func"

    def test_decorator_with_name(self):
        @functional_agent(name="custom_name")
        def my_func(x: int) -> dict:
            return {"result": x}

        assert isinstance(my_func, FunctionalAgent)
        assert my_func.name == "custom_name"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipeline:
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Pipeline should run steps in order and accumulate state."""

        def step_a(email_body: str) -> dict:
            return {"parsed": email_body.upper()}

        def step_b(parsed: str) -> dict:
            return {"final": f"Processed: {parsed}"}

        pipeline = Pipeline(
            "test",
            [
                FunctionalAgent(step_a),
                FunctionalAgent(step_b),
            ],
        )

        result = await pipeline.execute(initial_input={"email_body": "hello"})

        assert result.success is True
        assert result.state["parsed"] == "HELLO"
        assert result.state["final"] == "Processed: HELLO"
        assert result.steps_completed == ["step_a", "step_b"]
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Pipeline should propagate errors and set success=False."""

        def good_step() -> dict:
            return {"ok": True}

        agent_good = FunctionalAgent(good_step)
        agent_bad = FailingAgent("bad_step")

        pipeline = Pipeline("test", [agent_good, agent_bad])

        with pytest.raises(RuntimeError, match="deliberate failure"):
            await pipeline.execute()

    @pytest.mark.asyncio
    async def test_uses_provided_context(self):
        """Pipeline should use a provided AgentContext."""

        def echo_name() -> dict:
            return {"done": True}

        pipeline = Pipeline("test", [FunctionalAgent(echo_name)])

        ctx = AgentContext(
            execution_id="custom-id",
            pipeline_name="custom_pipeline",
        )
        result = await pipeline.execute(ctx, initial_input={"x": 1})

        assert result.execution_id == "custom-id"
        assert ctx.get("done") is True

    def test_repr(self):
        pipeline = Pipeline(
            "test",
            [
                FunctionalAgent(lambda: None, name="a"),
                FunctionalAgent(lambda: None, name="b"),
            ],
        )
        r = repr(pipeline)
        assert "test" in r
        assert "a" in r
        assert "b" in r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PipelineBuilder Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineBuilder:
    def test_add_base_agent(self):
        builder = PipelineBuilder("test")
        agent = DoubleAgent("doubler")
        builder.step(agent)
        pipeline = builder.build()
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0] is agent

    def test_add_function_auto_wraps(self):
        def my_func(x: int) -> dict:
            return {"result": x}

        pipeline = PipelineBuilder("test").step(my_func).build()
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], FunctionalAgent)
        assert pipeline.steps[0].name == "my_func"

    def test_add_adk_agent_auto_wraps(self):
        """An object with .name and .instruction should be detected as ADK."""
        mock_adk = MagicMock()
        mock_adk.name = "my_llm_agent"
        mock_adk.instruction = "You are a parser"
        mock_adk.description = "Parses things"

        pipeline = PipelineBuilder("test").step(mock_adk).build()
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], ADKAgent)

    def test_build_empty_raises(self):
        with pytest.raises(ValueError, match="no steps"):
            PipelineBuilder("empty").build()

    def test_invalid_step_type_raises(self):
        with pytest.raises(TypeError, match="Cannot add step"):
            PipelineBuilder("test").step(42)

    def test_fluent_api(self):
        def a():
            return {}

        def b():
            return {}

        pipeline = PipelineBuilder("fluent").step(a).step(b).build()
        assert pipeline.name == "fluent"
        assert len(pipeline.steps) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration: Full Pipeline E2E
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineE2E:
    """Simulates a mini bank_to_ynab-like pipeline end to end."""

    @pytest.mark.asyncio
    async def test_mini_pipeline(self):
        # Stage 1: Parse email (plain function)
        def parse_email(email_body: str) -> dict:
            return {
                "payee": "Amazon",
                "amount": -42.99,
                "card_suffix": "1234",
            }

        # Stage 2: Match account (plain function)
        def match_account(card_suffix: str) -> dict:
            accounts = {"1234": "Checking", "5678": "Savings"}
            return {
                "account_name": accounts.get(card_suffix, "Unknown"),
                "match_confidence": "high" if card_suffix in accounts else "low",
            }

        # Stage 3: Custom agent (subclass)
        class Categorizer(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                # Simulate LLM categorization
                return {"category": "Shopping", "category_reasoning": "Amazon purchase"}

        # Stage 4: Merge (plain function)
        def merge(payee: str, amount: float, account_name: str, category: str) -> dict:
            return {
                "transaction": {
                    "payee": payee,
                    "amount": amount,
                    "account": account_name,
                    "category": category,
                }
            }

        # Build pipeline
        pipeline = (
            PipelineBuilder("bank_to_ynab_mini")
            .step(parse_email)
            .step(match_account)
            .step(Categorizer("categorizer"))
            .step(merge)
            .build()
        )

        result = await pipeline.execute(
            initial_input={"email_body": "<html>Your Amazon purchase...</html>"}
        )

        assert result.success is True
        assert result.steps_completed == [
            "parse_email",
            "match_account",
            "categorizer",
            "merge",
        ]

        tx = result.state["transaction"]
        assert tx["payee"] == "Amazon"
        assert tx["amount"] == -42.99
        assert tx["account"] == "Checking"
        assert tx["category"] == "Shopping"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Adapter Tests — SequentialAgentAdapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


from autopilot.core.agent import (
    SequentialAgentAdapter,
    LoopAgentAdapter,
    ParallelAgentAdapter,
)
from autopilot.errors import MaxRetriesExceededError


class TestSequentialAgentAdapter:
    @pytest.mark.asyncio
    async def test_sequential_execution_and_state(self):
        """Children run in order and state accumulates."""

        class AddX(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {"x": input.get("x", 0) + 1}

        class AddY(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {"y": input.get("x", 0) * 10}

        seq = SequentialAgentAdapter("seq", children=[AddX("add_x"), AddY("add_y")])

        ctx = AgentContext(pipeline_name="test")
        result = await seq.invoke(ctx, {"x": 5})

        assert result["x"] == 6  # AddX: 5 + 1
        assert result["y"] == 60  # AddY: 6 * 10

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """If a child fails, the error bubbles up."""
        agent_ok = DoubleAgent("ok")
        agent_bad = FailingAgent("bad")

        seq = SequentialAgentAdapter("seq", children=[agent_ok, agent_bad])

        ctx = AgentContext(pipeline_name="test")
        with pytest.raises(RuntimeError, match="deliberate failure"):
            await seq.invoke(ctx, {"value": 1})

    def test_empty_children_raises(self):
        with pytest.raises(ValueError, match="at least one child"):
            SequentialAgentAdapter("empty", children=[])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Adapter Tests — LoopAgentAdapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLoopAgentAdapter:
    @pytest.mark.asyncio
    async def test_exits_on_condition(self):
        """Loop exits as soon as condition returns True."""
        call_count = 0

        class IncrementAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                nonlocal call_count
                call_count += 1
                return {"counter": input.get("counter", 0) + 1}

        loop = LoopAgentAdapter(
            "inc_loop",
            body=IncrementAgent("inc"),
            condition=lambda state: state.get("counter", 0) >= 3,
            max_iterations=10,
        )

        ctx = AgentContext(pipeline_name="test")
        result = await loop.invoke(ctx, {"counter": 0})

        assert result["counter"] == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_iterations_raises(self):
        """Raises MaxRetriesExceededError when condition never met."""

        class NoopAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {"still_bad": True}

        loop = LoopAgentAdapter(
            "doomed_loop",
            body=NoopAgent("noop"),
            condition=lambda state: False,  # never passes
            max_iterations=2,
        )

        ctx = AgentContext(pipeline_name="test")
        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await loop.invoke(ctx, {})

        assert exc_info.value.iterations == 2

    @pytest.mark.asyncio
    async def test_single_iteration_pass(self):
        """Condition met on first iteration — no retry needed."""

        class PassAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {"valid": True}

        loop = LoopAgentAdapter(
            "quick_loop",
            body=PassAgent("pass"),
            condition=lambda state: state.get("valid", False),
            max_iterations=5,
        )

        ctx = AgentContext(pipeline_name="test")
        result = await loop.invoke(ctx, {})

        assert result["valid"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Adapter Tests — ParallelAgentAdapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestParallelAgentAdapter:
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Branches run concurrently and results are merged."""
        import asyncio

        class SlowAgent(BaseAgent[dict, dict]):
            def __init__(self, key: str, value: Any, delay: float = 0.01):
                super().__init__(key)
                self._key = key
                self._value = value
                self._delay = delay

            async def run(self, ctx, input):
                await asyncio.sleep(self._delay)
                return {self._key: self._value}

        par = ParallelAgentAdapter(
            "fetch_all",
            branches=[
                SlowAgent("api1", "data_a", delay=0.02),
                SlowAgent("api2", "data_b", delay=0.02),
            ],
        )

        ctx = AgentContext(pipeline_name="test")

        import time

        start = time.monotonic()
        result = await par.invoke(ctx, {})
        elapsed = time.monotonic() - start

        assert result["api1"] == "data_a"
        assert result["api2"] == "data_b"
        # Should run concurrently: ~0.02s, not ~0.04s
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_error_in_branch(self):
        """If one branch fails, the whole parallel step fails."""
        agent_ok = DoubleAgent("ok")
        agent_bad = FailingAgent("bad")

        par = ParallelAgentAdapter("par", branches=[agent_ok, agent_bad])

        ctx = AgentContext(pipeline_name="test")
        with pytest.raises(RuntimeError, match="deliberate failure"):
            await par.invoke(ctx, {"value": 1})

    def test_empty_branches_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            ParallelAgentAdapter("empty", branches=[])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 PipelineBuilder Fluent API Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineBuilderV3:
    def test_loop_method(self):
        """Builder .loop() creates a LoopAgentAdapter step."""

        def my_body(x: int) -> dict:
            return {"x": x + 1}

        pipeline = (
            PipelineBuilder("test")
            .loop(
                my_body,
                condition=lambda s: s.get("x", 0) >= 3,
                max_iterations=5,
            )
            .build()
        )

        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], LoopAgentAdapter)
        assert "my_body" in pipeline.steps[0].name

    def test_parallel_method(self):
        """Builder .parallel() creates a ParallelAgentAdapter step."""

        def fetch_a() -> dict:
            return {"a": 1}

        def fetch_b() -> dict:
            return {"b": 2}

        pipeline = (
            PipelineBuilder("test").parallel(fetch_a, fetch_b, name="fetch_all").build()
        )

        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], ParallelAgentAdapter)
        assert pipeline.steps[0].name == "fetch_all"

    def test_chaining_all_methods(self):
        """All fluent methods can be chained together."""

        def a() -> dict:
            return {}

        def b() -> dict:
            return {}

        def c() -> dict:
            return {}

        pipeline = (
            PipelineBuilder("mixed")
            .step(a)
            .parallel(b, c, name="par")
            .loop(a, condition=lambda s: True, max_iterations=1)
            .build()
        )

        assert len(pipeline.steps) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Integration Test: Loop Retry Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLoopPipelineE2E:
    """Integration: pipeline with a loop step that retries until valid."""

    @pytest.mark.asyncio
    async def test_retry_until_valid(self):
        """Simulates: parse → validate (fails twice) → succeed on 3rd try."""
        attempt = {"n": 0}

        class FlakyParser(BaseAgent[dict, dict]):
            """Fails first 2 times, succeeds on 3rd."""

            async def run(self, ctx, input):
                attempt["n"] += 1
                if attempt["n"] < 3:
                    return {"parsed": None, "valid": False}
                return {"parsed": {"payee": "Amazon", "amount": -10.0}, "valid": True}

        def format_output(parsed: dict) -> dict:
            return {"formatted": f"Transaction: {parsed['payee']} ${parsed['amount']}"}

        pipeline = (
            PipelineBuilder("retry_pipeline")
            .loop(
                FlakyParser("parser"),
                condition=lambda s: s.get("valid", False),
                max_iterations=5,
                name="retry_parse",
            )
            .step(format_output)
            .build()
        )

        result = await pipeline.execute(initial_input={"raw_email": "..."})

        assert result.success is True
        assert result.state["valid"] is True
        assert "Amazon" in result.state["formatted"]
        assert attempt["n"] == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Phase 2 — DAG Orchestration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from autopilot.core.dag import DAGBuilder, DAGRunner
from autopilot.core.orchestrator import OrchestrationStrategy
from autopilot.errors import DAGCycleError, DAGDependencyError


class TestOrchestrationStrategy:
    """Tests for the OrchestrationStrategy enum."""

    def test_enum_values_exist(self):
        assert OrchestrationStrategy.SEQUENTIAL == "sequential"
        assert OrchestrationStrategy.DAG == "dag"
        assert OrchestrationStrategy.REACT == "react"
        assert OrchestrationStrategy.ROUTER == "router"

    def test_serializes_to_string(self):
        """Enum values should serialize cleanly."""
        assert OrchestrationStrategy.DAG.value == "dag"
        # str(Enum) includes class name + member name (uppercase)
        assert "DAG" in str(OrchestrationStrategy.DAG)


class TestDAGBuilder:
    """Tests for DAGBuilder validation and construction."""

    def test_single_node(self):
        agent = DoubleAgent("a")
        dag = DAGBuilder("test").node("a", agent).build()
        assert isinstance(dag, DAGRunner)

    def test_empty_dag_raises(self):
        with pytest.raises(ValueError, match="no nodes"):
            DAGBuilder("empty").build()

    def test_duplicate_node_raises(self):
        agent = DoubleAgent("a")
        with pytest.raises(ValueError, match="duplicate node name"):
            DAGBuilder("test").node("a", agent).node("a", agent)

    def test_dangling_dependency_raises(self):
        agent = DoubleAgent("a")
        with pytest.raises(DAGDependencyError, match="unknown node 'ghost'"):
            (DAGBuilder("test").node("a", agent, dependencies=["ghost"]).build())

    def test_cycle_detection(self):
        """A→B→A should raise DAGCycleError."""
        a, b = DoubleAgent("a"), DoubleAgent("b")
        with pytest.raises(DAGCycleError, match="cycle"):
            (
                DAGBuilder("test")
                .node("a", a, dependencies=["b"])
                .node("b", b, dependencies=["a"])
                .build()
            )

    def test_self_cycle_detection(self):
        """A→A should raise DAGCycleError."""
        a = DoubleAgent("a")
        with pytest.raises(DAGCycleError, match="cycle"):
            (DAGBuilder("test").node("a", a, dependencies=["a"]).build())

    def test_function_auto_wrap(self):
        """Plain functions should be auto-wrapped just like PipelineBuilder."""

        def my_func() -> dict:
            return {"x": 1}

        dag = DAGBuilder("test").node("fn", my_func).build()
        assert isinstance(dag, DAGRunner)

    def test_repr(self):
        builder = DAGBuilder("test").node("a", DoubleAgent("a"))
        assert "test" in repr(builder)
        assert "1" in repr(builder)  # 1 node


class TestDAGRunner:
    """Tests for DAGRunner execution engine."""

    @pytest.mark.asyncio
    async def test_single_node_execution(self):
        """DAG with one root node should execute and return state."""

        def produce() -> dict:
            return {"result": 42}

        dag = DAGBuilder("single").node("produce", produce).build()

        result = await dag.execute(initial_input={"seed": True})

        assert result.success is True
        assert result.state["result"] == 42
        assert result.state["seed"] is True
        assert result.steps_completed == ["produce"]
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        """A→B→C should execute in sequence, accumulating state."""

        class AddKey(BaseAgent[dict, dict]):
            def __init__(self, key: str, value: Any):
                super().__init__(key)
                self._key = key
                self._value = value

            async def run(self, ctx, input):
                return {self._key: self._value}

        dag = (
            DAGBuilder("chain")
            .node("a", AddKey("a", 1))
            .node("b", AddKey("b", 2), dependencies=["a"])
            .node("c", AddKey("c", 3), dependencies=["b"])
            .build()
        )

        result = await dag.execute()

        assert result.success is True
        assert result.state == {"a": 1, "b": 2, "c": 3}
        assert result.steps_completed == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_parallel_independent_nodes(self):
        """Two root nodes (no deps) should run concurrently."""

        class SlowAgent(BaseAgent[dict, dict]):
            def __init__(self, key: str, value: Any, delay: float = 0.02):
                super().__init__(key)
                self._key = key
                self._value = value
                self._delay = delay

            async def run(self, ctx, input):
                await asyncio.sleep(self._delay)
                return {self._key: self._value}

        dag = (
            DAGBuilder("parallel")
            .node("api1", SlowAgent("api1", "data_a", 0.02))
            .node("api2", SlowAgent("api2", "data_b", 0.02))
            .build()
        )

        import time as time_mod

        start = time_mod.monotonic()
        result = await dag.execute()
        elapsed = time_mod.monotonic() - start

        assert result.success is True
        assert result.state["api1"] == "data_a"
        assert result.state["api2"] == "data_b"
        # Should run concurrently: ~0.02s, not ~0.04s
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_diamond_pattern(self):
        """
        Classic diamond DAG:
          A → B → D
          A → C → D

        B and C should run in parallel after A.
        D should see outputs from both B and C.
        """

        class SetKey(BaseAgent[dict, dict]):
            def __init__(self, key: str):
                super().__init__(key)
                self._key = key

            async def run(self, ctx, input):
                return {self._key: f"{self._key}_done"}

        class MergeAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {"merged": f"{input.get('b', '?')}+{input.get('c', '?')}"}

        dag = (
            DAGBuilder("diamond")
            .node("a", SetKey("a"))
            .node("b", SetKey("b"), dependencies=["a"])
            .node("c", SetKey("c"), dependencies=["a"])
            .node("d", MergeAgent("merge"), dependencies=["b", "c"])
            .build()
        )

        result = await dag.execute()

        assert result.success is True
        assert result.state["a"] == "a_done"
        assert result.state["b"] == "b_done"
        assert result.state["c"] == "c_done"
        assert result.state["merged"] == "b_done+c_done"
        # Steps completed: a first, then b+c in parallel, then d
        assert "a" in result.steps_completed
        assert "b" in result.steps_completed
        assert "c" in result.steps_completed
        assert "d" in result.steps_completed

    @pytest.mark.asyncio
    async def test_state_accumulation_from_dependencies(self):
        """Each node should see the accumulated state from all upstream deps."""

        def root() -> dict:
            return {"root_val": 100}

        class ReaderAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                # This node should see root_val from its upstream dependency
                return {"downstream_val": input.get("root_val", 0) * 2}

        dag = (
            DAGBuilder("state_test")
            .node("root", root)
            .node("reader", ReaderAgent("reader"), dependencies=["root"])
            .build()
        )

        result = await dag.execute()

        assert result.state["root_val"] == 100
        assert result.state["downstream_val"] == 200

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """If any node fails, the DAG should fail and propagate the error."""

        def good() -> dict:
            return {"ok": True}

        dag = (
            DAGBuilder("error_test")
            .node("good", good)
            .node("bad", FailingAgent("bad"), dependencies=["good"])
            .build()
        )

        with pytest.raises(RuntimeError, match="deliberate failure"):
            await dag.execute()

    @pytest.mark.asyncio
    async def test_uses_provided_context(self):
        """DAGRunner should use a provided AgentContext."""

        def noop() -> dict:
            return {"done": True}

        dag = DAGBuilder("ctx_test").node("noop", noop).build()

        ctx = AgentContext(
            execution_id="custom-dag-id",
            pipeline_name="custom_dag",
        )
        result = await dag.execute(ctx, initial_input={"x": 1})

        assert result.execution_id == "custom-dag-id"
        assert ctx.get("done") is True

    def test_repr(self):
        dag = (
            DAGBuilder("repr_test")
            .node("a", DoubleAgent("a"))
            .node("b", DoubleAgent("b"), dependencies=["a"])
            .build()
        )
        r = repr(dag)
        assert "repr_test" in r
        assert "a" in r
        assert "b" in r


class TestBaseWorkflowStrategy:
    """Tests for BaseWorkflow V3 strategy support."""

    def test_default_strategy_is_sequential(self):
        """BaseWorkflow should default to SEQUENTIAL strategy."""
        from autopilot.base_workflow import BaseWorkflow
        from autopilot.models import WorkflowManifest, WorkflowResult

        class DummyWorkflow(BaseWorkflow):
            @property
            def manifest(self):
                return WorkflowManifest(
                    name="dummy",
                    description="test",
                    version="0.1.0",
                    triggers=[],
                )

            async def execute(self, trigger_data):
                return WorkflowResult(status="success", data={})

        wf = DummyWorkflow()
        assert wf.strategy == OrchestrationStrategy.SEQUENTIAL

    def test_build_dag_raises_by_default(self):
        """build_dag() should raise NotImplementedError if not overridden."""
        from autopilot.base_workflow import BaseWorkflow
        from autopilot.models import WorkflowManifest, WorkflowResult

        class DummyWorkflow(BaseWorkflow):
            @property
            def manifest(self):
                return WorkflowManifest(
                    name="dummy",
                    display_name="Dummy",
                    description="test",
                    version="0.1.0",
                    triggers=[],
                )

            async def execute(self, trigger_data):
                return WorkflowResult(status="success", data={})

        wf = DummyWorkflow()
        with pytest.raises(NotImplementedError, match="build_dag"):
            wf.build_dag()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Phase 3 — Session Service Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from autopilot.core.session import InMemorySessionService, Session
from autopilot.core.memory import InMemoryMemoryService, create_memory_service


class TestSessionStateDict:
    """Tests for ADK Session.state — plain dict operations, no wrappers."""

    def _make_session(self, initial_state=None):
        return Session(
            id="test-001",
            app_name="autopilot",
            user_id="test",
            state=dict(initial_state or {}),
        )

    def test_get_and_set(self):
        session = self._make_session()
        session.state["user_id"] = "u123"
        assert session.state["user_id"] == "u123"

    def test_get_missing_returns_default(self):
        session = self._make_session()
        assert session.state.get("missing") is None
        assert session.state.get("missing", "fallback") == "fallback"

    def test_delete_existing_key(self):
        session = self._make_session()
        session.state["key"] = "value"
        del session.state["key"]
        assert "key" not in session.state

    def test_clear_resets_all(self):
        session = self._make_session()
        session.state["a"] = 1
        session.state["b"] = 2
        session.state.clear()
        assert session.state == {}

    def test_initial_state(self):
        session = self._make_session({"preset": "value"})
        assert session.state["preset"] == "value"
        assert len(session.state) == 1

    def test_session_isolation(self):
        s1 = self._make_session()
        s2 = self._make_session()
        s1.state["only_in_s1"] = True
        assert "only_in_s1" not in s2.state

    def test_session_metadata(self):
        session = self._make_session()
        assert session.id == "test-001"
        assert session.app_name == "autopilot"
        assert session.user_id == "test"
        assert session.events == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Phase 3 — Memory Service Tests (ADK-native)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from google.adk.events.event import Event
from google.genai import types as genai_types
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry


def _make_event(text: str) -> Event:
    """Helper: create an ADK Event wrapping the given text."""
    return Event(
        author="agent",
        content=genai_types.Content(parts=[genai_types.Part(text=text)]),
    )


class TestInMemoryMemoryService:
    """Tests for ADK InMemoryMemoryService — keyword matching memory."""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        memory = InMemoryMemoryService()
        await memory.add_events_to_memory(
            app_name="test",
            user_id="u1",
            events=[_make_event("The user prefers dark mode")],
        )
        response = await memory.search_memory(
            app_name="test", user_id="u1", query="dark mode"
        )
        assert isinstance(response, SearchMemoryResponse)
        assert len(response.memories) >= 1

    @pytest.mark.asyncio
    async def test_search_returns_memory_entries(self):
        """search_memory should return MemoryEntry objects."""
        memory = InMemoryMemoryService()
        await memory.add_events_to_memory(
            app_name="test",
            user_id="u1",
            events=[_make_event("Python is great for AI")],
        )
        response = await memory.search_memory(
            app_name="test", user_id="u1", query="Python"
        )
        assert len(response.memories) >= 1
        entry = response.memories[0]
        assert isinstance(entry, MemoryEntry)
        assert entry.content is not None

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        memory = InMemoryMemoryService()
        response = await memory.search_memory(
            app_name="test", user_id="u1", query="anything"
        )
        assert isinstance(response, SearchMemoryResponse)
        assert response.memories == []

    @pytest.mark.asyncio
    async def test_multiple_events_searchable(self):
        memory = InMemoryMemoryService()
        events = [
            _make_event("The weather is sunny today"),
            _make_event("User prefers Python for data analysis"),
            _make_event("Python is a great programming language for AI"),
        ]
        await memory.add_events_to_memory(app_name="test", user_id="u1", events=events)
        response = await memory.search_memory(
            app_name="test", user_id="u1", query="Python"
        )
        assert len(response.memories) >= 1

    @pytest.mark.asyncio
    async def test_user_scoping(self):
        """Memories should be scoped by app_name/user_id."""
        memory = InMemoryMemoryService()
        await memory.add_events_to_memory(
            app_name="app_a", user_id="user1", events=[_make_event("Secret A")]
        )
        await memory.add_events_to_memory(
            app_name="app_a", user_id="user2", events=[_make_event("Secret B")]
        )
        # user1 should not see user2's memories
        response = await memory.search_memory(
            app_name="app_a", user_id="user1", query="Secret"
        )
        texts = []
        for m in response.memories:
            if m.content and m.content.parts:
                texts.extend(p.text for p in m.content.parts if p.text)
        # Should contain user1's data
        assert any("Secret A" in t for t in texts)

    @pytest.mark.asyncio
    async def test_session_based_add(self):
        """add_session_to_memory should also work for storing memories."""
        memory = InMemoryMemoryService()
        session = Session(
            id="sess-001",
            app_name="test",
            user_id="u1",
            events=[_make_event("Important information about coding")],
        )
        await memory.add_session_to_memory(session)
        response = await memory.search_memory(
            app_name="test", user_id="u1", query="coding"
        )
        assert len(response.memories) >= 1


class TestCreateMemoryService:
    """Tests for the create_memory_service() 12-Factor factory."""

    def test_default_is_in_memory(self, monkeypatch):
        monkeypatch.delenv("MEMORY_BACKEND", raising=False)
        from autopilot.core.memory import reset_memory_service

        reset_memory_service()
        svc = create_memory_service()
        assert isinstance(svc, InMemoryMemoryService)

    def test_explicit_memory_backend(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        svc = create_memory_service()
        assert isinstance(svc, InMemoryMemoryService)

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BACKEND", "redis")
        with pytest.raises(ValueError, match="Unknown MEMORY_BACKEND"):
            create_memory_service()

    def test_vertexai_without_engine_id_raises(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BACKEND", "vertexai")
        monkeypatch.delenv("MEMORY_AGENT_ENGINE_ID", raising=False)
        with pytest.raises(ValueError, match="MEMORY_AGENT_ENGINE_ID is required"):
            create_memory_service()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Phase 3 — AgentContext Session & Memory Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentContextPhase3:
    """Tests for AgentContext with ADK-native Session and Memory."""

    @pytest.mark.asyncio
    async def test_ensure_session_creates_session(self):
        """ensure_session() should create an ADK Session directly."""
        ctx = AgentContext(pipeline_name="test")
        assert ctx.session is None

        await ctx.ensure_session()

        assert ctx.session is not None
        assert isinstance(ctx.session, Session)
        assert ctx.session.app_name == "autopilot"
        assert ctx.session.user_id == "default"

    @pytest.mark.asyncio
    async def test_ensure_session_idempotent(self):
        """Calling ensure_session() twice should be a no-op on second call."""
        ctx = AgentContext(pipeline_name="test")
        await ctx.ensure_session()
        first_session = ctx.session
        await ctx.ensure_session()
        assert ctx.session is first_session  # Same object

    @pytest.mark.asyncio
    async def test_for_step_propagates_session_and_memory(self):
        """Child context should share session_service, session, and memory."""
        ctx = AgentContext(pipeline_name="test")
        await ctx.ensure_session()

        child = ctx.for_step("step_1")

        assert child.session_service is ctx.session_service
        assert child.session is ctx.session
        assert child.memory is ctx.memory

    @pytest.mark.asyncio
    async def test_auto_provisioned_session_service_and_memory(self):
        """Context without explicit services should auto-provision them."""
        ctx = AgentContext(pipeline_name="test")
        assert ctx.session_service is not None
        assert isinstance(ctx.session_service, InMemorySessionService)
        assert ctx.memory is not None
        assert isinstance(ctx.memory, InMemoryMemoryService)

    @pytest.mark.asyncio
    async def test_session_state_dict_through_context(self):
        """Session.state dict access should work — the ADK way."""
        ctx = AgentContext(pipeline_name="test")
        await ctx.ensure_session()

        ctx.session.state["user_tz"] = "EST"
        assert ctx.session.state["user_tz"] == "EST"
        del ctx.session.state["user_tz"]
        assert "user_tz" not in ctx.session.state

    @pytest.mark.asyncio
    async def test_injected_session_used_directly(self):
        """Injecting an ADK Session directly should work — no wrappers."""
        session = Session(
            id="injected-001",
            app_name="autopilot",
            user_id="test",
            state={"pre": "loaded"},
        )
        ctx = AgentContext(pipeline_name="test", session=session)

        assert ctx.session is session
        assert ctx.session.state["pre"] == "loaded"

    @pytest.mark.asyncio
    async def test_remember_and_recall(self):
        """Convenience methods remember() and recall() should use ADK memory API."""
        memory = InMemoryMemoryService()
        ctx = AgentContext(pipeline_name="test", memory=memory)

        # remember() stores via add_events_to_memory, returns None
        result = await ctx.remember("User prefers dark mode", {"source": "prefs"})
        assert result is None

        # recall() returns SearchMemoryResponse
        response = await ctx.recall("dark mode")
        assert isinstance(response, SearchMemoryResponse)
        assert len(response.memories) >= 1

    @pytest.mark.asyncio
    async def test_remember_with_auto_provisioned_memory(self):
        """remember() with auto-provisioned memory should work."""
        ctx = AgentContext(pipeline_name="test")
        result = await ctx.remember("Something important")
        assert result is None  # ADK add_events_to_memory returns None

    @pytest.mark.asyncio
    async def test_recall_with_auto_provisioned_memory(self):
        """recall() with auto-provisioned (empty) memory returns empty response."""
        ctx = AgentContext(pipeline_name="test")
        response = await ctx.recall("anything")
        assert isinstance(response, SearchMemoryResponse)
        assert response.memories == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Phase 3 — E2E: Cross-Execution Memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCrossExecutionMemory:
    """E2E: Agent A saves memory in run 1 → Agent B retrieves it in run 2."""

    @pytest.mark.asyncio
    async def test_cross_execution_memory_e2e(self):
        """
        Simulates two pipeline runs sharing an ADK MemoryService:
        - Run 1: An agent records text into memory via ctx.remember().
        - Run 2: A different agent queries memory via ctx.recall().

        Both pipelines use the same pipeline_name so memories are scoped
        to the same app_name in ADK's memory service.
        """
        # Shared memory service persists across runs
        shared_memory = InMemoryMemoryService()

        # ── Run 1: Agent saves text to memory ─────────────────────────
        class WriterAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                await ctx.remember(
                    f"Processed transaction for {input['payee']} of ${input['amount']}",
                    metadata={"agent": "writer", "workflow": "bank_to_ynab"},
                )
                return {"saved": True}

        writer_pipeline = Pipeline("run_1", [WriterAgent("writer")])

        ctx1 = AgentContext(pipeline_name="shared_app", memory=shared_memory)
        result1 = await writer_pipeline.execute(
            ctx1,
            initial_input={"payee": "Amazon", "amount": 42.99},
        )
        assert result1.success is True

        # ── Run 2: Different agent recalls from memory ────────────────
        class ReaderAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                response = await ctx.recall("Amazon transaction")
                memories = response.memories
                top_text = ""
                if memories and memories[0].content and memories[0].content.parts:
                    top_text = memories[0].content.parts[0].text or ""
                return {
                    "found_memories": len(memories),
                    "top_memory": top_text,
                }

        reader_pipeline = Pipeline("run_2", [ReaderAgent("reader")])

        ctx2 = AgentContext(pipeline_name="shared_app", memory=shared_memory)
        result2 = await reader_pipeline.execute(ctx2)

        assert result2.success is True
        assert result2.state["found_memories"] >= 1
        assert "Amazon" in result2.state["top_memory"]
        assert "$42.99" in result2.state["top_memory"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N2 — Structured Output Enforcement Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStructuredOutputEnforcement:
    """Tests for N2 — Structured Output Enforcement via Gemini native JSON mode."""

    @pytest.mark.asyncio
    async def test_adk_agent_structured_output_from_state(self):
        """ADKAgent.run() should re-validate through Pydantic when output_schema is set."""
        from pydantic import BaseModel
        from unittest.mock import AsyncMock, patch

        class ParsedEmail(BaseModel):
            payee: str
            amount: float
            is_successful: bool = True

        # Create a mock ADK agent with output_schema and output_key
        mock_adk_agent = MagicMock()
        mock_adk_agent.name = "email_parser"
        mock_adk_agent.description = "Parses emails"
        mock_adk_agent.output_key = "parsed_email"
        mock_adk_agent.output_schema = ParsedEmail

        agent = ADKAgent(mock_adk_agent)

        # Mock the ADKRunner to return a result with state containing the output
        mock_result = MagicMock()
        mock_result.state = {
            "parsed_email": {
                "payee": "Amazon",
                "amount": -42.99,
                "is_successful": True,
            }
        }
        mock_result.parsed_json = {}
        mock_result.final_text = ""

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch(
            "autopilot.core.adk_runner.get_adk_runner", return_value=mock_runner
        ):
            ctx = AgentContext(pipeline_name="test")
            result = await agent.run(ctx, {"message": "parse this email"})

        # Should return validated Pydantic output via state path
        assert "parsed_email" in result
        assert result["parsed_email"]["payee"] == "Amazon"
        assert result["parsed_email"]["amount"] == -42.99
        assert result["parsed_email"]["is_successful"] is True
        # Pydantic re-validation should exclude None fields
        assert None not in result["parsed_email"].values()

    @pytest.mark.asyncio
    async def test_adk_agent_fallback_for_schemaless(self):
        """ADKAgent.run() should fall back to parsed_json for schemaless agents."""
        from unittest.mock import AsyncMock, patch

        # Create a mock ADK agent WITHOUT output_schema
        mock_adk_agent = MagicMock()
        mock_adk_agent.name = "free_text_agent"
        mock_adk_agent.description = "Generates text"
        mock_adk_agent.output_key = None
        mock_adk_agent.output_schema = None

        agent = ADKAgent(mock_adk_agent)

        # Mock the ADKRunner to return a result with parsed_json
        mock_result = MagicMock()
        mock_result.state = {}
        mock_result.parsed_json = {"summary": "This is a summary"}
        mock_result.final_text = '{"summary": "This is a summary"}'

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch(
            "autopilot.core.adk_runner.get_adk_runner", return_value=mock_runner
        ):
            ctx = AgentContext(pipeline_name="test")
            result = await agent.run(ctx, {"message": "summarize this"})

        # Should fall back to parsed_json (no output_key, no output_schema)
        assert result == {"summary": "This is a summary"}

    def test_create_platform_agent_schema_sets_isolation(self):
        """create_platform_agent with output_schema should set agent isolation flags."""
        from pydantic import BaseModel
        from autopilot.agents.base import create_platform_agent
        from google.adk.agents import LlmAgent

        class SimpleOutput(BaseModel):
            value: str

        agent = create_platform_agent(
            name="isolated_agent",
            instruction="Produce structured output.",
            description="Produces structured output",
            output_schema=SimpleOutput,
            output_key="result",
        )

        # When output_schema is set, the agent should be an LlmAgent (no fallback)
        assert isinstance(agent, LlmAgent)
        assert agent.output_schema is SimpleOutput
        assert agent.output_key == "result"
        # Agent transfers should be disabled for isolation
        assert agent.disallow_transfer_to_parent is True
        assert agent.disallow_transfer_to_peers is True

    def test_create_platform_agent_no_schema_no_isolation(self):
        """create_platform_agent without output_schema should NOT set isolation flags."""
        from autopilot.agents.base import create_platform_agent
        from google.adk.agents import LlmAgent

        agent = create_platform_agent(
            name="free_agent",
            instruction="Do whatever you want.",
            description="A free agent",
        )

        assert isinstance(agent, LlmAgent)
        # No isolation when no output_schema
        assert agent.disallow_transfer_to_parent is False
        assert agent.disallow_transfer_to_peers is False
