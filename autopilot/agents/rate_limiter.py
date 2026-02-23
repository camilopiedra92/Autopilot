"""
Model Rate Limiter — Proactive QPM throttling for LLM model calls.

Implements a token-bucket rate limiter per model ID, injected as an ADK
``before_model_callback`` via ``create_platform_agent``. This is Layer 2
of the dual-layer rate limiting strategy:

  Layer 1 (Reactive):  ADK-native ``HttpRetryOptions`` — auto-retries 429s
  Layer 2 (Proactive):  This module — prevents 429s by throttling locally

The limiter uses **back-pressure** (``asyncio.sleep``) instead of rejection:
calls wait for their turn instead of failing, which is more resilient for
concurrent pipeline runs.

Configuration (12-Factor):
  ``MODEL_RATE_LIMIT_QPM`` env var — global QPM per model. Default ``0`` (disabled).
  Production: ``--set-env-vars MODEL_RATE_LIMIT_QPM=1500`` at deploy time.

Usage::

    # Automatic — injected by create_platform_agent when QPM > 0
    agent = create_platform_agent(name="parser", instruction="...")

    # Manual — for custom pipelines
    from autopilot.agents.rate_limiter import get_model_rate_limiter
    limiter = get_model_rate_limiter()
    if limiter:
        allowed = await limiter.acquire("gemini-3-flash-preview")
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Token Bucket
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class _TokenBucket:
    """Token bucket for a single model — refills at ``refill_rate`` tokens/sec."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def wait_time(self) -> float:
        """Seconds until next token is available. 0.0 if available now."""
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.refill_rate

    def try_acquire(self) -> bool:
        """Consume one token if available. Returns True if acquired."""
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ModelRateLimiter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ModelRateLimiter:
    """
    Per-model token-bucket rate limiter.

    Each model ID gets an independent bucket with ``qpm`` capacity.
    Thread-safe via ``asyncio.Lock`` per bucket (safe for single-worker
    asyncio on Cloud Run).

    Args:
        qpm: Queries per minute per model.
    """

    def __init__(self, qpm: int) -> None:
        if qpm <= 0:
            raise ValueError(
                "QPM must be positive; use get_model_rate_limiter() for disabled check"
            )
        self._qpm = qpm
        self._refill_rate = qpm / 60.0  # tokens per second
        self._buckets: dict[str, _TokenBucket] = {}

    def _get_bucket(self, model: str) -> _TokenBucket:
        """Get or create a bucket for the given model."""
        if model not in self._buckets:
            self._buckets[model] = _TokenBucket(
                capacity=self._qpm,
                refill_rate=self._refill_rate,
            )
        return self._buckets[model]

    async def acquire(self, model: str) -> float:
        """
        Acquire a token for the given model, sleeping if necessary.

        Returns:
            Seconds waited (0.0 if no wait was needed).
        """
        bucket = self._get_bucket(model)
        async with bucket.lock:
            wait = bucket.wait_time()
            if wait > 0:
                logger.warning(
                    "model_rate_limited",
                    model=model,
                    qpm=self._qpm,
                    wait_seconds=round(wait, 3),
                )
        if wait > 0:
            await asyncio.sleep(wait)
            # Re-acquire after sleep
            async with bucket.lock:
                bucket.try_acquire()
        else:
            async with bucket.lock:
                bucket.try_acquire()
        return wait

    @property
    def qpm(self) -> int:
        return self._qpm

    def __repr__(self) -> str:
        return f"<ModelRateLimiter qpm={self._qpm} models={list(self._buckets.keys())}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADK Callback Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_model_rate_limit_callback(
    limiter: ModelRateLimiter,
) -> callable:
    """
    Create an ADK ``before_model_callback`` that applies rate limiting.

    The callback uses back-pressure (``asyncio.sleep``) instead of rejection.
    It always returns ``None`` (never blocks the call) — it just delays it.

    Args:
        limiter: The ModelRateLimiter instance.

    Returns:
        A ``before_model_callback`` compatible with ADK ``LlmAgent``.
    """

    async def _rate_limit_before_model(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        # Extract model from the agent's config
        model = getattr(callback_context, "agent_name", "unknown")
        # Use the model from the LLM request config if available
        config = getattr(llm_request, "config", None)
        if config:
            model_id = getattr(config, "model", None)
            if model_id:
                model = str(model_id)

        waited = await limiter.acquire(model)
        if waited > 0:
            logger.info(
                "model_rate_limit_applied",
                agent=callback_context.agent_name,
                model=model,
                waited_seconds=round(waited, 3),
            )
        return None  # Always proceed after wait

    _rate_limit_before_model.__name__ = "model_rate_limit_callback"
    return _rate_limit_before_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_limiter: ModelRateLimiter | None = None
_callback: callable | None = None


def get_model_rate_limiter() -> ModelRateLimiter | None:
    """
    Get the global ModelRateLimiter singleton.

    Reads ``MODEL_RATE_LIMIT_QPM`` env var (default: ``0`` = disabled).
    Returns ``None`` when rate limiting is disabled.
    """
    global _limiter
    if _limiter is not None:
        return _limiter

    qpm = int(os.environ.get("MODEL_RATE_LIMIT_QPM", "0"))
    if qpm <= 0:
        return None

    _limiter = ModelRateLimiter(qpm=qpm)
    logger.info("model_rate_limiter_initialized", qpm=qpm)
    return _limiter


def get_model_rate_limit_callback() -> callable | None:
    """
    Get the global rate limit ``before_model_callback``.

    Returns ``None`` when rate limiting is disabled (``MODEL_RATE_LIMIT_QPM=0``).
    """
    global _callback
    if _callback is not None:
        return _callback

    limiter = get_model_rate_limiter()
    if limiter is None:
        return None

    _callback = create_model_rate_limit_callback(limiter)
    return _callback


def reset_model_rate_limiter() -> None:
    """Reset the global rate limiter. For testing only."""
    global _limiter, _callback
    _limiter = None
    _callback = None
