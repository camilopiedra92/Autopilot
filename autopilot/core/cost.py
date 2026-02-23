"""
CostTracker — Real-time cost estimation for LLM executions.

Provides per-execution token usage accumulation and USD cost estimation
using an async-safe ContextVar (same pattern as ``pipeline_session_id``).

The ``after_model_callback`` extracts ``usage_metadata`` from each LLM
response and feeds it into the tracker.  ``AgentContext.cost`` exposes
a read-only ``CostSnapshot`` for budget-based routing decisions.

Pricing is configurable via environment variables or a static lookup
table for Gemini models.
"""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from google.genai import types


# ── Model Pricing (USD per 1K tokens) ────────────────────────────────
# Prices for Gemini models as of 2025.  Override at runtime via
# env var GEMINI_PRICING_JSON='{"gemini-2.0-flash": [0.0001, 0.0004]}'
# Format: model_prefix -> (input_per_1k, output_per_1k)

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # ── Flash models ─────────────────────────────────────────────────
    # Gemini 2.0 Flash: $0.10 / $0.40 per 1M tokens
    "gemini-2.0-flash": (0.0001, 0.0004),
    # Gemini 2.0 Flash-Lite: $0.075 / $0.30 per 1M tokens
    "gemini-2.0-flash-lite": (0.000075, 0.0003),
    # Gemini 2.5 Flash: $0.30 / $2.50 per 1M tokens
    "gemini-2.5-flash": (0.0003, 0.0025),
    # Gemini 2.5 Flash-Lite: $0.15 / $0.60 per 1M tokens
    "gemini-2.5-flash-lite": (0.00015, 0.0006),
    # Gemini 3 Flash Preview: $0.50 / $3.00 per 1M tokens
    "gemini-3-flash": (0.0005, 0.003),
    # ── Pro models ───────────────────────────────────────────────────
    # Gemini 2.5 Pro (<=200k): $1.25 / $10.00 per 1M tokens
    "gemini-2.5-pro": (0.00125, 0.01),
    # Gemini 3 Pro Preview (<=200k): $2.00 / $12.00 per 1M tokens
    "gemini-3-pro": (0.002, 0.012),
    # Gemini 3.1 Pro Preview (<=200k): $2.00 / $12.00 per 1M tokens
    "gemini-3.1-pro": (0.002, 0.012),
}


def _load_pricing() -> dict[str, tuple[float, float]]:
    """Load model pricing, merging env overrides with defaults."""
    pricing = dict(_DEFAULT_PRICING)

    env_json = os.environ.get("GEMINI_PRICING_JSON")
    if env_json:
        import json

        try:
            overrides = json.loads(env_json)
            for model, rates in overrides.items():
                if isinstance(rates, (list, tuple)) and len(rates) == 2:
                    pricing[model] = (float(rates[0]), float(rates[1]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # Silently ignore malformed pricing — use defaults

    return pricing


def _lookup_pricing(model: str) -> tuple[float, float]:
    """Find pricing for a model by prefix match.

    Tries exact match first, then progressively shorter prefixes.
    Falls back to Gemini Flash pricing if no match found.
    """
    pricing = _load_pricing()

    # Exact match
    if model in pricing:
        return pricing[model]

    # Prefix match (e.g. "gemini-3-flash-preview" → "gemini-3-flash")
    for prefix, rates in sorted(pricing.items(), key=lambda x: -len(x[0])):
        if model.startswith(prefix):
            return rates

    # Default fallback: cheapest model pricing
    return (0.0001, 0.0004)


# ── CostSnapshot — Immutable read-only view ──────────────────────────


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable snapshot of accumulated cost for a single execution.

    Attributes:
        prompt_tokens: Total input/prompt tokens across all LLM calls.
        candidates_tokens: Total output/candidate tokens across all calls.
        cached_tokens: Tokens served from context cache (reduced cost).
        thoughts_tokens: Tokens used for model "thinking" (if applicable).
        total_tokens: Sum of prompt + candidates tokens.
        estimated_cost_usd: Estimated total cost in USD.
        llm_calls: Number of LLM invocations in this execution.
        per_agent: Per-agent cost breakdown {agent_name: CostSnapshot}.
    """

    prompt_tokens: int = 0
    candidates_tokens: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_calls: int = 0
    per_agent: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "cached_tokens": self.cached_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "llm_calls": self.llm_calls,
            "per_agent": {
                name: snap.to_dict() if isinstance(snap, CostSnapshot) else snap
                for name, snap in self.per_agent.items()
            },
        }


# ── CostTracker — Mutable per-execution accumulator ─────────────────


class CostTracker:
    """Accumulates token usage and cost estimates for a single execution.

    Thread-safe via ContextVar isolation (one tracker per async context).
    """

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._candidates_tokens: int = 0
        self._cached_tokens: int = 0
        self._thoughts_tokens: int = 0
        self._estimated_cost_usd: float = 0.0
        self._llm_calls: int = 0
        self._per_agent: dict[str, CostTracker] = {}

    def record(
        self,
        agent_name: str,
        usage_metadata: Optional[types.GenerateContentResponseUsageMetadata],
        model: str = "",
    ) -> None:
        """Record token usage from a single LLM call.

        Args:
            agent_name: Name of the agent that made the call.
            usage_metadata: Token usage from ``LlmResponse.usage_metadata``.
            model: Model ID used (for pricing lookup).
        """
        if usage_metadata is None:
            return

        self._accumulate(usage_metadata, model)

        # Per-agent breakdown (flat — no further nesting)
        if agent_name not in self._per_agent:
            self._per_agent[agent_name] = CostTracker()
        self._per_agent[agent_name]._accumulate(usage_metadata, model)

    def _accumulate(
        self,
        usage_metadata: types.GenerateContentResponseUsageMetadata,
        model: str,
    ) -> None:
        """Internal: accumulate token counts and cost without per-agent delegation."""
        prompt = usage_metadata.prompt_token_count or 0
        candidates = usage_metadata.candidates_token_count or 0
        cached = usage_metadata.cached_content_token_count or 0
        thoughts = usage_metadata.thoughts_token_count or 0

        self._prompt_tokens += prompt
        self._candidates_tokens += candidates
        self._cached_tokens += cached
        self._thoughts_tokens += thoughts
        self._llm_calls += 1

        # Calculate cost for this call
        input_rate, output_rate = _lookup_pricing(model)
        # Cached tokens are typically billed at a reduced rate (often ~25%)
        effective_prompt = prompt - cached
        call_cost = (
            (effective_prompt * input_rate / 1000)
            + (cached * input_rate * 0.25 / 1000)
            + (candidates * output_rate / 1000)
        )
        self._estimated_cost_usd += call_cost

    def snapshot(self) -> CostSnapshot:
        """Create an immutable snapshot of the current accumulated cost."""
        return CostSnapshot(
            prompt_tokens=self._prompt_tokens,
            candidates_tokens=self._candidates_tokens,
            cached_tokens=self._cached_tokens,
            thoughts_tokens=self._thoughts_tokens,
            total_tokens=self._prompt_tokens + self._candidates_tokens,
            estimated_cost_usd=self._estimated_cost_usd,
            llm_calls=self._llm_calls,
            per_agent={
                name: tracker.snapshot() for name, tracker in self._per_agent.items()
            },
        )

    @property
    def estimated_cost_usd(self) -> float:
        """Current accumulated cost estimate."""
        return self._estimated_cost_usd

    @property
    def llm_calls(self) -> int:
        """Number of LLM calls recorded."""
        return self._llm_calls


# ── ContextVar singleton ─────────────────────────────────────────────

_cost_tracker: contextvars.ContextVar[CostTracker | None] = contextvars.ContextVar(
    "_cost_tracker", default=None
)


def get_cost_tracker() -> CostTracker:
    """Get or create the CostTracker for the current async context."""
    tracker = _cost_tracker.get(None)
    if tracker is None:
        tracker = CostTracker()
        _cost_tracker.set(tracker)
    return tracker


def reset_cost_tracker() -> contextvars.Token:
    """Reset the cost tracker for a new execution. Returns a reset token."""
    tracker = CostTracker()
    return _cost_tracker.set(tracker)
