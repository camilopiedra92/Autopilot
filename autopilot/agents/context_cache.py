"""
Context Cache Configuration — Per-Agent Gemini Context Caching.

Provides a factory for ADK's native ContextCacheConfig and utilities
for marking/detecting cache-enabled agents. Caching is per-agent opt-in
via ``create_platform_agent(cache_context=True)``.

Env vars (12-Factor, global defaults when an agent opts in):
  - CONTEXT_CACHE_MIN_TOKENS: Min tokens to trigger caching (default: 2048)
  - CONTEXT_CACHE_TTL_SECONDS: Cache TTL in seconds (default: 1800 = 30 min)
  - CONTEXT_CACHE_INTERVALS: Max uses before refresh (default: 10)
"""

import os
import warnings

import structlog
from google.adk.agents.context_cache_config import ContextCacheConfig

logger = structlog.get_logger(__name__)

# Marker attribute set on LlmAgent instances by create_platform_agent
PLATFORM_CACHE_CONTEXT_ATTR = "_platform_cache_context"


def has_cache_context(agent: object) -> bool:
    """Check if an agent is marked for context caching."""
    return getattr(agent, PLATFORM_CACHE_CONTEXT_ATTR, False) is True


def create_context_cache_config() -> ContextCacheConfig:
    """Create a ContextCacheConfig from environment variables.

    Returns:
        Configured ContextCacheConfig with values from env vars or defaults.
    """
    min_tokens = int(os.getenv("CONTEXT_CACHE_MIN_TOKENS", "2048"))
    ttl_seconds = int(os.getenv("CONTEXT_CACHE_TTL_SECONDS", "1800"))
    cache_intervals = int(os.getenv("CONTEXT_CACHE_INTERVALS", "10"))

    # Suppress ADK's EXPERIMENTAL UserWarning — we intentionally opt into this feature
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*EXPERIMENTAL.*ContextCacheConfig.*"
        )
        config = ContextCacheConfig(
            min_tokens=min_tokens,
            ttl_seconds=ttl_seconds,
            cache_intervals=cache_intervals,
        )

    logger.info(
        "context_cache_config_created",
        min_tokens=min_tokens,
        ttl_seconds=ttl_seconds,
        cache_intervals=cache_intervals,
    )

    return config
