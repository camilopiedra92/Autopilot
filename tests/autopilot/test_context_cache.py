"""
Tests for autopilot.agents.context_cache — Per-Agent Context Caching.

Covers:
  - create_context_cache_config: default values, custom env vars
  - has_cache_context: unmarked vs marked agents
  - create_platform_agent: cache_context=True sets marker, False does not
  - ADKRunner integration: App wrapper used only when agent opts in
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from autopilot.agents.context_cache import (
    create_context_cache_config,
    has_cache_context,
    PLATFORM_CACHE_CONTEXT_ATTR,
)
from google.adk.agents.context_cache_config import ContextCacheConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  create_context_cache_config Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreateContextCacheConfig:
    def test_returns_config_with_defaults(self):
        """Should return ContextCacheConfig with default values."""
        with patch.dict("os.environ", {}, clear=True):
            config = create_context_cache_config()

        assert isinstance(config, ContextCacheConfig)
        assert config.min_tokens == 2048
        assert config.ttl_seconds == 1800
        assert config.cache_intervals == 10

    def test_reads_custom_env_vars(self):
        """Should read values from environment variables."""
        env = {
            "CONTEXT_CACHE_MIN_TOKENS": "4096",
            "CONTEXT_CACHE_TTL_SECONDS": "600",
            "CONTEXT_CACHE_INTERVALS": "5",
        }
        with patch.dict("os.environ", env, clear=True):
            config = create_context_cache_config()

        assert config.min_tokens == 4096
        assert config.ttl_seconds == 600
        assert config.cache_intervals == 5

    def test_partial_env_vars_use_defaults(self):
        """Should use defaults for unset env vars."""
        env = {"CONTEXT_CACHE_TTL_SECONDS": "900"}
        with patch.dict("os.environ", env, clear=True):
            config = create_context_cache_config()

        assert config.min_tokens == 2048  # default
        assert config.ttl_seconds == 900  # custom
        assert config.cache_intervals == 10  # default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  has_cache_context Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHasCacheContext:
    def test_returns_false_for_unmarked_agent(self):
        """Agents without the marker attribute should return False."""
        agent = MagicMock(spec=[])
        assert has_cache_context(agent) is False

    def test_returns_true_for_marked_agent(self):
        """Agents with the marker attribute set to True should return True."""
        agent = MagicMock(spec=[])
        setattr(agent, PLATFORM_CACHE_CONTEXT_ATTR, True)
        assert has_cache_context(agent) is True

    def test_returns_false_for_marker_set_to_false(self):
        """Agents with the marker attribute set to False should return False."""
        agent = MagicMock(spec=[])
        setattr(agent, PLATFORM_CACHE_CONTEXT_ATTR, False)
        assert has_cache_context(agent) is False

    def test_returns_false_for_none(self):
        """Agents with the marker attribute set to None should return False."""
        agent = MagicMock(spec=[])
        setattr(agent, PLATFORM_CACHE_CONTEXT_ATTR, None)
        assert has_cache_context(agent) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration with create_platform_agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCreatePlatformAgentCacheContext:
    def test_cache_context_true_sets_marker(self):
        """cache_context=True should set the marker attribute."""
        from autopilot.agents.base import create_platform_agent

        agent = create_platform_agent(
            name="cacheable_agent",
            instruction="Long instruction with lots of context...",
            description="A cacheable agent for testing.",
            cache_context=True,
        )
        assert has_cache_context(agent) is True

    def test_cache_context_false_no_marker(self):
        """cache_context=False (default) should NOT set the marker attribute."""
        from autopilot.agents.base import create_platform_agent

        agent = create_platform_agent(
            name="normal_agent",
            instruction="Short instruction.",
            description="A normal agent for testing.",
        )
        assert has_cache_context(agent) is False

    def test_cache_context_default_is_false(self):
        """Default cache_context should be False."""
        from autopilot.agents.base import create_platform_agent

        agent = create_platform_agent(
            name="default_agent",
            instruction="instruction",
            description="A default agent for testing.",
            cache_context=False,
        )
        assert has_cache_context(agent) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADKRunner Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestADKRunnerContextCache:
    @pytest.mark.asyncio
    async def test_uses_app_wrapper_when_cache_context_enabled(self):
        """ADKRunner should construct App when agent has cache_context=True."""
        from autopilot.core.adk_runner import ADKRunner

        runner = ADKRunner(app_name="test_app", user_id="test_user")

        mock_agent = MagicMock()
        mock_agent.name = "cached_agent"
        setattr(mock_agent, PLATFORM_CACHE_CONTEXT_ATTR, True)

        with (
            patch("autopilot.core.adk_runner.Runner") as mock_runner_cls,
            patch("autopilot.core.adk_runner.App") as mock_app_cls,
            patch(
                "autopilot.core.adk_runner.create_context_cache_config"
            ) as mock_create_config,
            patch("autopilot.core.adk_runner.get_event_bus"),
        ):
            mock_config = MagicMock(spec=ContextCacheConfig)
            mock_create_config.return_value = mock_config

            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = AsyncMock(return_value=iter([]))
            mock_runner_cls.return_value = mock_runner_instance

            # App should be constructed with context_cache_config
            mock_app_cls.return_value = MagicMock()

            try:
                await runner._run_adk_agent(
                    mock_agent, "test message", None, "stream_id"
                )
            except Exception:
                pass  # We only care about the Runner construction

            mock_app_cls.assert_called_once()
            call_kwargs = mock_app_cls.call_args
            assert call_kwargs.kwargs.get("context_cache_config") == mock_config
            assert call_kwargs.kwargs.get("name") == "test_app"

    @pytest.mark.asyncio
    async def test_does_not_use_app_when_cache_context_disabled(self):
        """ADKRunner should NOT construct App when agent has no cache_context."""
        from autopilot.core.adk_runner import ADKRunner

        runner = ADKRunner(app_name="test_app", user_id="test_user")

        mock_agent = MagicMock()
        mock_agent.name = "normal_agent"
        # No PLATFORM_CACHE_CONTEXT_ATTR set

        with (
            patch("autopilot.core.adk_runner.Runner") as mock_runner_cls,
            patch("autopilot.core.adk_runner.App") as mock_app_cls,
            patch("autopilot.core.adk_runner.get_event_bus"),
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_async = AsyncMock(return_value=iter([]))
            mock_runner_cls.return_value = mock_runner_instance

            try:
                await runner._run_adk_agent(
                    mock_agent, "test message", None, "stream_id"
                )
            except Exception:
                pass

            # App should NOT be constructed
            mock_app_cls.assert_not_called()

            # Runner should be constructed with app_name and agent directly
            mock_runner_cls.assert_called_once()
            call_kwargs = mock_runner_cls.call_args
            assert call_kwargs.kwargs.get("app_name") == "test_app"
            assert call_kwargs.kwargs.get("agent") == mock_agent
