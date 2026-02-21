"""
SessionService — Short-term state management for agent executions.

Provides a key-value store scoped to a single execution or conversation.
Agents use sessions to share ephemeral state (e.g. user preferences,
intermediate results) within a pipeline run.

Two implementations:
  - InMemorySessionService: Dict-backed, no external deps (dev/test).
  - Drop-in replacement for RedisSessionService in prod.

Aligned with Google ADK's SessionService contract.
"""

from __future__ import annotations

import abc
import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BaseSessionService — Abstract contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaseSessionService(abc.ABC):
    """
    Abstract contract for session state management.

    A session is a key-value namespace scoped to a pipeline execution.
    Implementations must provide CRUD operations and a snapshot method
    for serialization / debugging.
    """

    @abc.abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key. Returns *default* if missing."""
        ...

    @abc.abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Store a value under *key*."""
        ...

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove *key*. Returns True if the key existed, False otherwise."""
        ...

    @abc.abstractmethod
    async def clear(self) -> None:
        """Remove all keys from the session."""
        ...

    @abc.abstractmethod
    async def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the entire session state."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  InMemorySessionService — Dict-backed implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InMemorySessionService(BaseSessionService):
    """
    In-memory session store backed by a plain ``dict``.

    Ideal for development, testing, and single-process deployments.
    State is lost when the process exits — use ``RedisSessionService``
    (or similar) for durable, multi-process sessions.

    Thread-safety note: Python's GIL makes dict operations atomic for
    single keys.  For truly concurrent workloads, wrap with a lock.
    """

    def __init__(self, *, initial_state: dict[str, Any] | None = None):
        self._store: dict[str, Any] = dict(initial_state or {})
        self._created_at: datetime = datetime.now(timezone.utc)
        self._updated_at: datetime = self._created_at

    # ── CRUD ─────────────────────────────────────────────────────────

    async def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._updated_at = datetime.now(timezone.utc)

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            self._updated_at = datetime.now(timezone.utc)
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()
        self._updated_at = datetime.now(timezone.utc)

    async def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy — callers can't mutate internal state."""
        return dict(self._store)

    # ── Introspection ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of keys currently stored."""
        return len(self._store)

    def __repr__(self) -> str:
        return f"<InMemorySessionService keys={self.size}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RedisSessionService — Redis-backed implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RedisSessionService(BaseSessionService):
    """
    Redis-backed session store for distributed, multi-process deployments.

    Keys are prefixed with the session ID to isolate contexts.
    Dependencies: `redis` (redis-py async module).
    """

    def __init__(
        self, redis_url: str, session_id: str, prefix: str = "autopilot:session:"
    ):
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._prefix = f"{prefix}{session_id}:"
        self._session_id = session_id

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # ── CRUD ─────────────────────────────────────────────────────────

    async def get(self, key: str, default: Any = None) -> Any:
        value = await self._client.get(self._key(key))
        if value is None:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def set(self, key: str, value: Any) -> None:
        if not isinstance(value, str):
            value = json.dumps(value)
        await self._client.set(self._key(key), value)

    async def delete(self, key: str) -> bool:
        deleted = await self._client.delete(self._key(key))
        return deleted > 0

    async def clear(self) -> None:
        # Warning: This uses keys(), which is slow on large databases.
        # Alternatively, use scan_iter for safer operations in production.
        keys = await self._client.keys(f"{self._prefix}*")
        if keys:
            await self._client.delete(*keys)

    async def snapshot(self) -> dict[str, Any]:
        keys = await self._client.keys(f"{self._prefix}*")
        if not keys:
            return {}

        values = await self._client.mget(keys)
        # Strip prefix from keys
        prefix_len = len(self._prefix)

        result = {}
        for k, v in zip(keys, values):
            if v is not None:
                short_key = k[prefix_len:]
                try:
                    result[short_key] = json.loads(v)
                except json.JSONDecodeError:
                    result[short_key] = v
        return result

    # ── Introspection ────────────────────────────────────────────────

    async def size(self) -> int:
        """Number of keys currently stored. Returns size asynchronously."""
        keys = await self._client.keys(f"{self._prefix}*")
        return len(keys)

    def __repr__(self) -> str:
        return f"<RedisSessionService session_id={self._session_id}>"
