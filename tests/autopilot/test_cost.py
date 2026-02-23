"""
Tests for autopilot.core.cost — Real-time Cost Tracking.

Covers:
  - CostSnapshot creation and serialization
  - CostTracker accumulation across multiple calls
  - Model pricing lookup (exact, prefix, fallback)
  - ContextVar isolation between executions
  - Budget guardrail (blocking and allowing)
  - Integration with after_model_logger callback
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from autopilot.core.cost import (
    CostSnapshot,
    CostTracker,
    get_cost_tracker,
    reset_cost_tracker,
    _lookup_pricing,
    _DEFAULT_PRICING,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CostSnapshot Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCostSnapshot:
    def test_default_values(self):
        snap = CostSnapshot()
        assert snap.prompt_tokens == 0
        assert snap.candidates_tokens == 0
        assert snap.cached_tokens == 0
        assert snap.thoughts_tokens == 0
        assert snap.total_tokens == 0
        assert snap.estimated_cost_usd == 0.0
        assert snap.llm_calls == 0
        assert snap.per_agent == {}

    def test_to_dict(self):
        snap = CostSnapshot(
            prompt_tokens=100,
            candidates_tokens=50,
            cached_tokens=20,
            total_tokens=150,
            estimated_cost_usd=0.00123456,
            llm_calls=2,
        )
        d = snap.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["candidates_tokens"] == 50
        assert d["cached_tokens"] == 20
        assert d["total_tokens"] == 150
        assert d["estimated_cost_usd"] == 0.00123456
        assert d["llm_calls"] == 2
        assert d["per_agent"] == {}

    def test_to_dict_nested_per_agent(self):
        child_snap = CostSnapshot(prompt_tokens=50, total_tokens=50, llm_calls=1)
        snap = CostSnapshot(
            prompt_tokens=50,
            total_tokens=50,
            llm_calls=1,
            per_agent={"child_agent": child_snap},
        )
        d = snap.to_dict()
        assert "child_agent" in d["per_agent"]
        assert d["per_agent"]["child_agent"]["prompt_tokens"] == 50

    def test_immutable(self):
        snap = CostSnapshot(prompt_tokens=100)
        with pytest.raises(AttributeError):
            snap.prompt_tokens = 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CostTracker Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_usage_metadata(
    prompt: int = 0,
    candidates: int = 0,
    cached: int = 0,
    thoughts: int = 0,
):
    """Create a mock GenerateContentResponseUsageMetadata."""
    mock = MagicMock()
    mock.prompt_token_count = prompt
    mock.candidates_token_count = candidates
    mock.cached_content_token_count = cached
    mock.thoughts_token_count = thoughts
    return mock


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        snap = tracker.snapshot()
        assert snap.llm_calls == 0
        assert snap.total_tokens == 0
        assert snap.estimated_cost_usd == 0.0

    def test_record_single_call(self):
        tracker = CostTracker()
        usage = _make_usage_metadata(prompt=100, candidates=50)
        tracker.record("test_agent", usage, "gemini-2.0-flash")

        snap = tracker.snapshot()
        assert snap.llm_calls == 1
        assert snap.prompt_tokens == 100
        assert snap.candidates_tokens == 50
        assert snap.total_tokens == 150
        assert snap.estimated_cost_usd > 0

    def test_record_multiple_calls_accumulates(self):
        tracker = CostTracker()
        usage1 = _make_usage_metadata(prompt=100, candidates=50)
        usage2 = _make_usage_metadata(prompt=200, candidates=100)

        tracker.record("agent_a", usage1, "gemini-2.0-flash")
        tracker.record("agent_b", usage2, "gemini-2.0-flash")

        snap = tracker.snapshot()
        assert snap.llm_calls == 2
        assert snap.prompt_tokens == 300
        assert snap.candidates_tokens == 150
        assert snap.total_tokens == 450

    def test_per_agent_breakdown(self):
        tracker = CostTracker()
        usage1 = _make_usage_metadata(prompt=100, candidates=50)
        usage2 = _make_usage_metadata(prompt=200, candidates=100)

        tracker.record("parser", usage1, "gemini-2.0-flash")
        tracker.record("categorizer", usage2, "gemini-2.0-flash")

        snap = tracker.snapshot()
        assert "parser" in snap.per_agent
        assert "categorizer" in snap.per_agent
        assert snap.per_agent["parser"].prompt_tokens == 100
        assert snap.per_agent["categorizer"].prompt_tokens == 200

    def test_cached_tokens_reduce_cost(self):
        tracker_no_cache = CostTracker()
        tracker_with_cache = CostTracker()

        # Same total prompt tokens, but some are cached
        usage_no_cache = _make_usage_metadata(prompt=1000, candidates=500)
        usage_cached = _make_usage_metadata(prompt=1000, candidates=500, cached=800)

        tracker_no_cache.record("a", usage_no_cache, "gemini-2.0-flash")
        tracker_with_cache.record("a", usage_cached, "gemini-2.0-flash")

        # Cached version should be cheaper
        assert (
            tracker_with_cache.snapshot().estimated_cost_usd
            < tracker_no_cache.snapshot().estimated_cost_usd
        )

    def test_record_none_usage_is_noop(self):
        tracker = CostTracker()
        tracker.record("agent", None, "gemini-2.0-flash")
        assert tracker.snapshot().llm_calls == 0

    def test_estimated_cost_property(self):
        tracker = CostTracker()
        usage = _make_usage_metadata(prompt=100, candidates=50)
        tracker.record("agent", usage, "gemini-2.0-flash")
        assert tracker.estimated_cost_usd > 0
        assert tracker.estimated_cost_usd == tracker.snapshot().estimated_cost_usd

    def test_llm_calls_property(self):
        tracker = CostTracker()
        assert tracker.llm_calls == 0
        usage = _make_usage_metadata(prompt=10, candidates=5)
        tracker.record("a", usage, "gemini-2.0-flash")
        tracker.record("b", usage, "gemini-2.0-flash")
        assert tracker.llm_calls == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model Pricing Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelPricing:
    def test_exact_match(self):
        rate = _lookup_pricing("gemini-2.0-flash")
        assert rate == _DEFAULT_PRICING["gemini-2.0-flash"]

    def test_prefix_match(self):
        # "gemini-2.0-flash-001" should match "gemini-2.0-flash" prefix
        rate = _lookup_pricing("gemini-2.0-flash-001")
        assert rate == _DEFAULT_PRICING["gemini-2.0-flash"]

    def test_preview_suffix(self):
        rate = _lookup_pricing("gemini-3-flash-preview")
        assert rate == _DEFAULT_PRICING["gemini-3-flash"]

    def test_fallback_for_unknown_model(self):
        rate = _lookup_pricing("some-unknown-model-xyz")
        # Should fallback to cheapest pricing
        assert rate == (0.0001, 0.0004)

    def test_pro_model_pricing(self):
        rate = _lookup_pricing("gemini-2.5-pro")
        assert rate == _DEFAULT_PRICING["gemini-2.5-pro"]
        # Pro should be more expensive than Flash
        flash_rate = _lookup_pricing("gemini-2.0-flash")
        assert rate[0] > flash_rate[0]
        assert rate[1] > flash_rate[1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ContextVar Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCostTrackerContextVar:
    def test_get_returns_same_instance(self):
        reset_cost_tracker()
        t1 = get_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is t2

    def test_reset_creates_new_instance(self):
        t1 = get_cost_tracker()
        t1.record("agent", _make_usage_metadata(prompt=100), "gemini-2.0-flash")
        assert t1.llm_calls == 1

        reset_cost_tracker()
        t2 = get_cost_tracker()
        assert t2 is not t1
        assert t2.llm_calls == 0

    @pytest.mark.asyncio
    async def test_isolation_between_tasks(self):
        """Concurrent tasks should have isolated cost trackers."""
        results = {}

        async def execution(name: str, tokens: int):
            reset_cost_tracker()
            tracker = get_cost_tracker()
            usage = _make_usage_metadata(prompt=tokens)
            tracker.record(name, usage, "gemini-2.0-flash")
            # Yield to allow concurrent execution
            await asyncio.sleep(0.01)
            snap = tracker.snapshot()
            results[name] = snap.prompt_tokens

        await asyncio.gather(
            execution("exec_a", 100),
            execution("exec_b", 500),
        )

        # Each execution should see only its own tokens
        assert results["exec_a"] == 100
        assert results["exec_b"] == 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Budget Guardrail Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBudgetGuardrail:
    def test_allows_when_under_budget(self):
        from autopilot.agents.callbacks import create_budget_guardrail

        reset_cost_tracker()
        guardrail = create_budget_guardrail(max_cost_usd=1.0)

        ctx = MagicMock()
        ctx.agent_name = "test_agent"
        req = MagicMock()

        result = guardrail(ctx, req)
        assert result is None  # Proceed

    def test_blocks_when_over_budget(self):
        from autopilot.agents.callbacks import create_budget_guardrail

        reset_cost_tracker()
        tracker = get_cost_tracker()

        # Record enough usage to exceed $0.001 budget
        usage = _make_usage_metadata(prompt=10000, candidates=5000)
        tracker.record("expensive_agent", usage, "gemini-2.5-pro")

        guardrail = create_budget_guardrail(max_cost_usd=0.001)

        ctx = MagicMock()
        ctx.agent_name = "next_agent"
        req = MagicMock()

        result = guardrail(ctx, req)
        assert result is not None  # Blocked
        assert result.error_code == "BUDGET_EXCEEDED"
        assert "Budget exceeded" in result.content.parts[0].text

    def test_exact_boundary(self):
        """At exactly the budget, should still block (>= comparison)."""
        from autopilot.agents.callbacks import create_budget_guardrail

        reset_cost_tracker()
        tracker = get_cost_tracker()

        # Record usage and set budget to exactly the cost
        usage = _make_usage_metadata(prompt=1000, candidates=500)
        tracker.record("agent", usage, "gemini-2.0-flash")
        exact_cost = tracker.estimated_cost_usd

        guardrail = create_budget_guardrail(max_cost_usd=exact_cost)

        ctx = MagicMock()
        ctx.agent_name = "next_agent"
        req = MagicMock()

        result = guardrail(ctx, req)
        assert result is not None  # Blocked at boundary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentContext Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentContextCost:
    def test_cost_property_returns_snapshot(self):
        from autopilot.core.context import AgentContext

        reset_cost_tracker()
        ctx = AgentContext(pipeline_name="test")
        snap = ctx.cost
        assert isinstance(snap, CostSnapshot)
        assert snap.llm_calls == 0

    def test_cost_reflects_tracker_updates(self):
        from autopilot.core.context import AgentContext

        reset_cost_tracker()
        tracker = get_cost_tracker()
        usage = _make_usage_metadata(prompt=500, candidates=200)
        tracker.record("agent", usage, "gemini-2.0-flash")

        ctx = AgentContext(pipeline_name="test")
        snap = ctx.cost
        assert snap.prompt_tokens == 500
        assert snap.candidates_tokens == 200
        assert snap.llm_calls == 1

    def test_cost_tracker_property(self):
        from autopilot.core.context import AgentContext

        reset_cost_tracker()
        ctx = AgentContext(pipeline_name="test")
        tracker = ctx.cost_tracker
        assert isinstance(tracker, CostTracker)
        assert tracker.llm_calls == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PipelineResult Cost Field Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineResultCost:
    def test_cost_field_default_none(self):
        from autopilot.models import PipelineResult

        result = PipelineResult(session_id="test", final_text="hello")
        assert result.cost is None

    def test_cost_field_with_snapshot(self):
        from autopilot.models import PipelineResult

        cost_data = CostSnapshot(
            prompt_tokens=100,
            candidates_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            llm_calls=1,
        ).to_dict()

        result = PipelineResult(
            session_id="test",
            final_text="hello",
            cost=cost_data,
        )
        assert result.cost is not None
        assert result.cost["total_tokens"] == 150
        assert result.cost["llm_calls"] == 1

    def test_cost_field_serialization(self):
        from autopilot.models import PipelineResult

        result = PipelineResult(
            session_id="test",
            final_text="hello",
            cost={"total_tokens": 100, "estimated_cost_usd": 0.001},
        )
        # Pydantic serialization round-trip
        dumped = result.model_dump()
        assert dumped["cost"]["total_tokens"] == 100
