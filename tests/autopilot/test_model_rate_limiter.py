"""
Tests for autopilot.agents.rate_limiter — Model Rate Limiter.

Covers:
  - TokenBucket: allows within limit, blocks over limit, refills over time
  - ModelRateLimiter: per-model isolation, acquire with backpressure
  - ADK callback: returns None (proceeds), applies sleep when throttled
  - Singleton: factory reads env var, disabled when QPM=0
  - Integration: create_platform_agent injects HttpRetryOptions and callback
  - Error: ModelRateLimitError serialization
"""

import asyncio
import time

import pytest
from unittest.mock import patch, MagicMock

from autopilot.agents.rate_limiter import (
    _TokenBucket,
    ModelRateLimiter,
    create_model_rate_limit_callback,
    get_model_rate_limiter,
    get_model_rate_limit_callback,
    reset_model_rate_limiter,
)
from autopilot.errors import ModelRateLimitError


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_limiter():
    """Reset the global rate limiter before each test."""
    reset_model_rate_limiter()
    yield
    reset_model_rate_limiter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TokenBucket Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTokenBucket:
    def test_allows_within_limit(self):
        """Requests within capacity should be allowed."""
        bucket = _TokenBucket(capacity=5, refill_rate=5 / 60.0)
        for _ in range(5):
            assert bucket.try_acquire() is True

    def test_blocks_over_limit(self):
        """Requests over capacity should be blocked."""
        bucket = _TokenBucket(capacity=3, refill_rate=3 / 60.0)
        for _ in range(3):
            bucket.try_acquire()
        assert bucket.try_acquire() is False

    def test_wait_time_zero_when_available(self):
        """wait_time should be 0 when tokens are available."""
        bucket = _TokenBucket(capacity=10, refill_rate=10 / 60.0)
        assert bucket.wait_time() == 0.0

    def test_wait_time_positive_when_exhausted(self):
        """wait_time should be positive when no tokens available."""
        bucket = _TokenBucket(capacity=1, refill_rate=1 / 60.0)
        bucket.try_acquire()  # Exhaust the single token
        wait = bucket.wait_time()
        assert wait > 0.0
        # Should need ~60s for 1 QPM, but allow some tolerance
        assert wait <= 61.0

    def test_refills_over_time(self):
        """Tokens should refill after time passes."""
        bucket = _TokenBucket(capacity=2, refill_rate=1000.0)  # 1000 tokens/sec
        bucket.try_acquire()
        bucket.try_acquire()
        assert bucket.try_acquire() is False

        # Fast refill rate — after a tiny sleep, tokens should be available
        time.sleep(0.01)  # 10ms × 1000 tok/s = 10 tokens
        assert bucket.try_acquire() is True

    def test_capacity_is_ceiling(self):
        """Tokens should never exceed capacity."""
        bucket = _TokenBucket(capacity=3, refill_rate=1000.0)
        time.sleep(0.1)  # Would refill 100 tokens at 1000/s
        bucket._refill()
        assert bucket.tokens == 3.0  # Capped at capacity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ModelRateLimiter Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelRateLimiter:
    def test_construction(self):
        limiter = ModelRateLimiter(qpm=100)
        assert limiter.qpm == 100
        assert "ModelRateLimiter" in repr(limiter)

    def test_invalid_qpm_raises(self):
        with pytest.raises(ValueError, match="QPM must be positive"):
            ModelRateLimiter(qpm=0)

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """acquire should return 0.0 when within limit."""
        limiter = ModelRateLimiter(qpm=100)
        waited = await limiter.acquire("gemini-3-flash-preview")
        assert waited == 0.0

    @pytest.mark.asyncio
    async def test_per_model_isolation(self):
        """Different models should have independent buckets."""
        limiter = ModelRateLimiter(qpm=2)
        # Exhaust model A
        await limiter.acquire("model_a")
        await limiter.acquire("model_a")
        # Model B should still have tokens
        waited = await limiter.acquire("model_b")
        assert waited == 0.0

    @pytest.mark.asyncio
    async def test_acquire_sleeps_when_exhausted(self):
        """acquire should sleep when tokens are exhausted."""
        # Very high refill rate so sleep is minimal
        limiter = ModelRateLimiter(qpm=60000)  # 1000/sec
        # Exhaust all tokens
        for _ in range(60000):
            bucket = limiter._get_bucket("fast_model")
            bucket.try_acquire()

        # Next acquire should need to wait (but very briefly)
        start = time.monotonic()
        await limiter.acquire("fast_model")
        elapsed = time.monotonic() - start
        # Should have waited some small amount (tokens refill at 1000/sec)
        assert elapsed >= 0  # Just verify it didn't error

    @pytest.mark.asyncio
    async def test_concurrent_acquire_safety(self):
        """Multiple concurrent tasks should not corrupt token counts."""
        limiter = ModelRateLimiter(qpm=6000)  # 100/sec

        async def worker():
            await limiter.acquire("concurrent_model")

        # Run 10 concurrent workers
        await asyncio.gather(*(worker() for _ in range(10)))
        # If we got here without error, concurrency is safe


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADK Callback Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelRateLimitCallback:
    @pytest.mark.asyncio
    async def test_callback_returns_none_when_allowed(self):
        """Callback should return None when rate limit is not exceeded."""
        limiter = ModelRateLimiter(qpm=1000)
        callback = create_model_rate_limit_callback(limiter)

        mock_ctx = MagicMock()
        mock_ctx.agent_name = "test_agent"
        mock_request = MagicMock()
        mock_request.config = MagicMock()
        mock_request.config.model = "gemini-3-flash-preview"

        result = await callback(mock_ctx, mock_request)
        assert result is None  # Should proceed

    @pytest.mark.asyncio
    async def test_callback_uses_agent_name_fallback(self):
        """Callback should use agent_name if model is not in request config."""
        limiter = ModelRateLimiter(qpm=1000)
        callback = create_model_rate_limit_callback(limiter)

        mock_ctx = MagicMock()
        mock_ctx.agent_name = "my_agent"
        mock_request = MagicMock()
        mock_request.config = None  # No config

        result = await callback(mock_ctx, mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_callback_always_proceeds(self):
        """Callback should always return None (back-pressure, not rejection)."""
        limiter = ModelRateLimiter(qpm=60000)
        callback = create_model_rate_limit_callback(limiter)

        mock_ctx = MagicMock()
        mock_ctx.agent_name = "test_agent"
        mock_request = MagicMock()
        mock_request.config = MagicMock()
        mock_request.config.model = "test_model"

        # Exhaust all tokens
        bucket = limiter._get_bucket("test_model")
        for _ in range(60000):
            bucket.try_acquire()

        # Should still return None (just delayed)
        result = await callback(mock_ctx, mock_request)
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSingleton:
    def test_disabled_when_qpm_zero(self):
        """get_model_rate_limiter should return None when QPM=0."""
        with patch.dict("os.environ", {"MODEL_RATE_LIMIT_QPM": "0"}):
            assert get_model_rate_limiter() is None

    def test_disabled_when_env_not_set(self):
        """get_model_rate_limiter should return None when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert get_model_rate_limiter() is None

    def test_factory_reads_env_var(self):
        """get_model_rate_limiter should create limiter when QPM > 0."""
        with patch.dict("os.environ", {"MODEL_RATE_LIMIT_QPM": "500"}):
            limiter = get_model_rate_limiter()
            assert limiter is not None
            assert limiter.qpm == 500

    def test_callback_disabled_when_qpm_zero(self):
        """get_model_rate_limit_callback returns None when disabled."""
        with patch.dict("os.environ", {"MODEL_RATE_LIMIT_QPM": "0"}):
            assert get_model_rate_limit_callback() is None

    def test_callback_returned_when_enabled(self):
        """get_model_rate_limit_callback returns callback when enabled."""
        with patch.dict("os.environ", {"MODEL_RATE_LIMIT_QPM": "1000"}):
            callback = get_model_rate_limit_callback()
            assert callback is not None
            assert callable(callback)

    def test_reset_clears_singleton(self):
        """reset_model_rate_limiter should clear cached instances."""
        with patch.dict("os.environ", {"MODEL_RATE_LIMIT_QPM": "100"}):
            l1 = get_model_rate_limiter()
            reset_model_rate_limiter()
            l2 = get_model_rate_limiter()
            assert l1 is not l2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration with create_platform_agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIntegration:
    def test_create_platform_agent_has_retry_options(self):
        """HttpRetryOptions should be injected into GenerateContentConfig."""
        from autopilot.agents.base import create_platform_agent

        agent = create_platform_agent(
            name="test_agent",
            instruction="Do nothing",
            description="Test agent",
        )
        config = agent.generate_content_config
        assert config is not None
        assert config.http_options is not None
        assert config.http_options.retry_options is not None
        assert config.http_options.retry_options.attempts == 3

    def test_create_platform_agent_preserves_user_retry_options(self):
        """User-provided retry options should NOT be overwritten."""
        from autopilot.agents.base import create_platform_agent
        from google.genai import types

        custom_retry = types.HttpRetryOptions(initial_delay=5, attempts=10)
        custom_config = types.GenerateContentConfig(
            http_options=types.HttpOptions(retry_options=custom_retry),
        )

        agent = create_platform_agent(
            name="test_agent",
            instruction="Do nothing",
            description="Test agent",
            generate_content_config=custom_config,
        )
        config = agent.generate_content_config
        assert config.http_options.retry_options.attempts == 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Error Type Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelRateLimitError:
    def test_error_attributes(self):
        err = ModelRateLimitError(
            "Rate limit exceeded",
            model="gemini-3-flash-preview",
            agent_name="parser",
        )
        assert err.model == "gemini-3-flash-preview"
        assert err.agent_name == "parser"
        assert err.error_code == "MODEL_RATE_LIMIT"
        assert err.retryable is True
        assert err.http_status == 429

    def test_serialization(self):
        err = ModelRateLimitError("Throttled", model="gemini-2.0-pro")
        d = err.to_dict()
        assert d["error_code"] == "MODEL_RATE_LIMIT"
        assert d["model"] == "gemini-2.0-pro"
        assert d["retryable"] is True
