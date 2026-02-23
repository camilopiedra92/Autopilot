# Implementation Plan: Unified Event Bus Architecture

> **Status**: 📋 Ready for Review
> **Author**: Autopilot Platform
> **Date**: 2026-02-21
> **Scope**: Unify `PipelineEventBus` + `AgentBus` → single `EventBus` with distributed backend
> **Policy**: Zero deprecated code — old modules are deleted, not shimmed

## 1. Problem Statement

The platform has **two independent, in-memory event systems** that evolved organically:

| System             | Location                          | Purpose                            | Model                   |
| ------------------ | --------------------------------- | ---------------------------------- | ----------------------- |
| `PipelineEventBus` | `autopilot/services/event_bus.py` | SSE streaming of pipeline progress | 1:1 (Queue per session) |
| `AgentBus`         | `autopilot/core/bus.py`           | A2A pub/sub, event-driven triggers | 1:N (topic fanout)      |

### Why This Is a Problem

1. **Confusing API surface**: `ctx.emit()` vs `ctx.publish()` — developers must guess which to use
2. **Dead code path**: `PipelineEventBus` was built for SSE/dashboard — the system is now headless API, no consumer exists
3. **Duplicated concerns**: Both systems do event dispatch, timestamps, error isolation, logging
4. **No persistence**: Both are pure in-memory — events are lost on restart/scale-to-zero
5. **No cross-instance fanout**: If Cloud Run scales to N>1 instances, events are siloed
6. **No replay/DLQ**: If a handler fails, the event is gone — no retry possible
7. **Misalignment with ADK**: Google ADK's event model is channel-based (Session Events → Event Stream → Consumers), not dual-bus

### Current Dependency Map

**PipelineEventBus Producers** (3 call sites):

```
autopilot/agents/pipeline_runner.py   → emit() for pipeline_started, pipeline_completed, pipeline_error
autopilot/agents/callbacks.py         → _emit_event_async() for stage_started, stage_completed
autopilot/agents/tool_callbacks.py    → _emit_event_async() for tool_started, tool_completed
```

**PipelineEventBus consumed via**:

```
autopilot/core/context.py             → ctx.emit() convenience method
```

**AgentBus Producers**:

```
autopilot/api/webhooks.py             → bus.publish("email.received", ...)
workflows/bank_to_ynab/steps.py       → ctx.publish("transaction.created", ...)
```

**AgentBus Consumers**:

```
workflows/bank_to_ynab/workflow.py    → subscribes to "email.received"
workflows/bank_to_ynab/workflow.py    → subscribes to "transaction.created" (Telegram)
```

**Tests referencing PipelineEventBus** (mock `get_event_bus`):

```
tests/autopilot/test_tools.py         → 3 patches
tests/autopilot/test_bus.py           → 3 patches (ctx integration)
tests/autopilot/test_dsl.py           → 7 patches (_mock_event_bus helper)
tests/autopilot/test_core.py          → 5 patches
```

## 2. Target Architecture

### Design Principles

1. **Single Bus**: One `EventBus` replaces both systems
2. **ADK-Aligned**: Follow Google ADK's event model — typed events, channels, subscriptions
3. **Protocol-First**: Define an `EventBusProtocol` (ABC) so the bus is swappable (in-memory → Redis → Pub/Sub)
4. **Topic Namespacing**: Pipeline progress events become standard topics (`pipeline.*`, `stage.*`, `tool.*`)
5. **Backward Compatible**: `ctx.emit()` becomes an alias for `ctx.publish()` with auto-topic mapping
6. **Cloud-Native Ready**: The protocol is designed for distributed backends from day one
7. **Clean Deletion**: Old modules (`services/event_bus.py`) are fully deleted — no shims, no deprecated code

### New Topic Hierarchy

```
pipeline.started          ← was: PipelineEventBus emit("pipeline_started")
pipeline.completed        ← was: PipelineEventBus emit("pipeline_completed")
pipeline.error            ← was: PipelineEventBus emit("pipeline_error")
pipeline.timeout          ← was: PipelineEventBus stream timeout

stage.started             ← was: PipelineEventBus emit("stage_started")
stage.completed           ← was: PipelineEventBus emit("stage_completed")

tool.started              ← was: PipelineEventBus emit("tool_started")
tool.completed            ← was: PipelineEventBus emit("tool_completed")

email.received            ← unchanged (AgentBus)
transaction.created       ← unchanged (AgentBus)
agent.*                   ← unchanged (AgentBus)
```

### Architecture Diagram (Full Vision)

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                  EventBusProtocol (ABC)                      │
                    │                                                             │
                    │  publish() · subscribe() · unsubscribe() · history()        │
                    │  replay() · clear() · stats · subscription_count            │
                    └───────────────────────┬─────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
          ┌─────────▼──────────┐  ┌────────▼─────────┐  ┌─────────▼──────────┐
          │  InMemoryEventBus  │  │ RedisStreamBus   │  │  CloudPubSubBus    │
          │  (dev/test)        │  │ (production ✅)   │  │  (future/external) │
          │                    │  │                   │  │                    │
          │ • dict + deque     │  │ • Redis Streams   │  │ • GCP Pub/Sub      │
          │ • asyncio.gather   │  │ • Consumer Groups │  │ • Push/Pull subs   │
          │ • fnmatch routing  │  │ • At-least-once   │  │ • Fully managed    │
          │ • ring-buffer hist │  │ • Event replay    │  │ • Cross-region     │
          │ • OTel tracing     │  │ • TTL persistence │  │                    │
          │                    │  │ • DLQ support     │  │                    │
          └────────────────────┘  │ • Cross-instance  │  └────────────────────┘
                                  │ • OTel tracing    │
                                  └───────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              ┌─────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
              │ Producers   │        │ Subscribers  │        │ Middleware   │
              │             │        │              │        │              │
              │ Pipeline    │        │ Workflows    │        │ EventStore   │
              │ Runner      │        │ (email.*)    │        │ (audit log)  │
              │             │        │              │        │              │
              │ Callbacks   │        │ Notifiers    │        │ Metrics      │
              │ (stage.*)   │        │ (txn.*)      │        │ (Prometheus) │
              │             │        │              │        │              │
              │ Webhooks    │        │ SSE Bridge   │        │ Retry/DLQ    │
              │ (email.*)   │        │ (pipeline.*) │        │              │
              └─────────────┘        └──────────────┘        └──────────────┘
```

### Backend Selection Strategy

| Environment                | Backend               | Why                                       |
| -------------------------- | --------------------- | ----------------------------------------- |
| **Unit Tests**             | `InMemoryEventBus`    | Zero deps, instant, deterministic         |
| **Local Dev**              | `InMemoryEventBus`    | No infra needed, fast iteration           |
| **Production (Cloud Run)** | `RedisStreamEventBus` | Persistent, cross-instance, at-least-once |
| **Multi-Region (future)**  | `CloudPubSubEventBus` | Global fanout, fully managed              |

Selection is driven by environment config:

```python
# autopilot/core/bus.py — factory function
def create_event_bus(backend: str = "memory") -> EventBusProtocol:
    """Factory for creating the appropriate bus backend.

    Args:
        backend: One of "memory", "redis", "pubsub".
                 Defaults to EVENTBUS_BACKEND env var or "memory".
    """
    backend = os.getenv("EVENTBUS_BACKEND", backend)

    if backend == "redis":
        from autopilot.core.bus_redis import RedisStreamEventBus
        return RedisStreamEventBus.from_env()
    elif backend == "pubsub":
        from autopilot.core.bus_pubsub import CloudPubSubEventBus
        return CloudPubSubEventBus.from_env()
    else:
        return EventBus()  # InMemoryEventBus
```

## 3. Implementation Steps

### Phase 1: Define EventBus Protocol + Migrate AgentBus → EventBus

**Goal**: Establish the protocol (ABC), rename `AgentBus` → `EventBus`, add topic namespacing.

#### Step 1.1: Create `EventBusProtocol` ABC

**File**: `autopilot/core/bus.py` (modify in-place)

```python
from abc import ABC, abstractmethod

class EventBusProtocol(ABC):
    """
    Abstract Event Bus protocol — swappable backends.

    All event bus implementations MUST implement this interface.
    Aligned with Google ADK's event streaming model.

    Implementations:
      - EventBus (InMemory): dev/test — asyncio.Queue, zero deps
      - RedisStreamEventBus: production — Redis Streams, consumer groups
      - CloudPubSubEventBus: future — GCP Pub/Sub, fully managed
    """

    @abstractmethod
    def subscribe(self, topic_pattern: str, handler: MessageHandler) -> Subscription: ...

    @abstractmethod
    def unsubscribe(self, subscription: Subscription) -> bool: ...

    @abstractmethod
    async def publish(self, topic: str, payload: dict | None = None, *, sender: str = "", correlation_id: str | None = None) -> AgentMessage: ...

    @abstractmethod
    def history(self, topic: str, *, limit: int = 50) -> list[AgentMessage]: ...

    @abstractmethod
    async def replay(self, topic: str, *, since: str | None = None, handler: MessageHandler | None = None) -> list[AgentMessage]:
        """Replay persisted events for a topic.

        In-memory: replays from ring buffer history.
        Redis/Pub/Sub: replays from persistent stream with optional cursor.

        Args:
            topic: Topic pattern to replay.
            since: Optional ISO-8601 timestamp or stream ID to replay from.
            handler: Optional handler to invoke for each replayed event.

        Returns:
            List of replayed AgentMessage objects.
        """
        ...

    @abstractmethod
    def clear(self) -> None: ...

    @property
    @abstractmethod
    def subscription_count(self) -> int: ...

    @property
    @abstractmethod
    def stats(self) -> dict[str, int]: ...
```

**Changes**:

- Add `EventBusProtocol` ABC at the top of `bus.py`
- Make `AgentBus` implement `EventBusProtocol`
- Rename class `AgentBus` → `EventBus` (keep `AgentBus = EventBus` alias for backward compat)
- Rename `get_agent_bus()` → `get_event_bus()` (keep old name as alias)
- Rename `reset_agent_bus()` → `reset_event_bus()` (keep old name as alias)
- Add `replay()` method to `EventBus` (in-memory: replays from ring buffer)
- Add `create_event_bus()` factory function for backend selection

**Rationale**: The `AgentBus` already IS the good implementation. We formalize the interface, rename for clarity, and add `replay()` for future distributed backends.

#### Step 1.2: Add Event Middleware Support

**File**: `autopilot/core/bus.py` (add to `EventBus` class)

```python
EventMiddleware = Callable[[AgentMessage], Awaitable[AgentMessage | None]]

class EventBus(EventBusProtocol):
    def __init__(self, *, history_limit: int = 100) -> None:
        ...
        self._middleware: list[EventMiddleware] = []

    def use(self, middleware: EventMiddleware) -> None:
        """Register middleware that intercepts every published event.

        Middleware runs in order before dispatch. If middleware returns None,
        the event is suppressed (filtered). Otherwise the returned event
        (possibly modified) is dispatched.

        Use cases: event logging, metrics, persistence, filtering.
        """
        self._middleware.append(middleware)
```

**In `publish()`**, run middleware chain before dispatching:

```python
# Run middleware chain
for mw in self._middleware:
    result = await mw(msg)
    if result is None:
        return msg  # Filtered
    msg = result
```

This enables Phase 3's EventStore and Phase 4's Redis backend to be plugged in without modifying the bus core.

#### Step 1.3: Update `AgentContext` — Unify `emit()` and `publish()`

**File**: `autopilot/core/context.py`

```python
async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
    """
    Convenience alias: publish a platform observability event.

    Prefer ``ctx.publish(topic, data)`` directly with the canonical topic hierarchy.
    This method auto-maps legacy ``event_type`` slugs to canonical dot-separated
    topics (e.g. ``"stage_started"`` → ``"stage.started"``).
    """
    # Auto-map legacy event types → canonical topics
    topic = _LEGACY_TOPIC_MAP.get(event_type, f"platform.{event_type}")
    payload = {
        "execution_id": self.execution_id,
        "pipeline": self.pipeline_name,
        **(data or {}),
    }
    await self.bus.publish(topic, payload, sender=self.pipeline_name)

_LEGACY_TOPIC_MAP = {
    "pipeline_started": "pipeline.started",
    "pipeline_completed": "pipeline.completed",
    "pipeline_error": "pipeline.error",
    "stage_started": "stage.started",
    "stage_completed": "stage.completed",
    "tool_started": "tool.started",
    "tool_completed": "tool.completed",
    "step_started": "stage.started",
    "step_completed": "stage.completed",
}
```

**Remove**:

- Delete `from autopilot.services.event_bus import get_event_bus` import
- Delete `_stream_id` attribute (no longer needed)
- Clean `for_step()` to remove `_stream_id` propagation

#### Step 1.4: Migrate `PipelineRunner` → Use `EventBus`

**File**: `autopilot/agents/pipeline_runner.py`

**Before**:

```python
from autopilot.services.event_bus import get_event_bus
...
event_bus = get_event_bus()
await event_bus.emit(effective_stream_id, {"type": "pipeline_started", ...})
```

**After**:

```python
from autopilot.core.bus import get_event_bus
...
bus = get_event_bus()
await bus.publish("pipeline.started", {
    "session_id": session_id,
    "pipeline_name": getattr(pipeline, "name", "unknown"),
}, sender="pipeline_runner")
```

**Remove**:

- Delete `stream_session_id` parameter (no more per-session queues)
- Delete `effective_stream_id` logic
- Delete `await event_bus.end_stream(effective_stream_id)` in `finally`
- Replace all `event_bus.emit(...)` with `bus.publish(topic, payload)`

#### Step 1.5: Migrate Callbacks → Use `EventBus`

**File**: `autopilot/agents/callbacks.py`

**Replace** `_emit_event_async()` fire-and-forget pattern:

```python
def _emit_event_async(event: dict) -> None:
    """Publish an observability event to the unified EventBus."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _publish():
        try:
            from autopilot.core.bus import get_event_bus
            bus = get_event_bus()
            event_type = event.get("type", "unknown")
            topic = _CALLBACK_TOPIC_MAP.get(event_type, f"platform.{event_type}")
            await bus.publish(topic, event, sender="platform_callbacks")
        except Exception:
            pass  # Never let bus errors break the pipeline

    loop.create_task(_publish())

_CALLBACK_TOPIC_MAP = {
    "stage_started": "stage.started",
    "stage_completed": "stage.completed",
    "tool_started": "tool.started",
    "tool_completed": "tool.completed",
}
```

**Remove**:

- Delete `pipeline_session_id` ContextVar (no longer needed for session-scoped queues)
- Delete references to `autopilot.services.event_bus`

#### Step 1.6: Delete `autopilot/services/event_bus.py`

The entire file is dead code after migration. **Hard delete** — no shim, no deprecation wrapper.

- Delete `autopilot/services/event_bus.py` entirely
- Remove any re-exports from `autopilot/services/__init__.py`
- Update **all** import sites to use `autopilot.core.bus` directly:
  - `autopilot/agents/pipeline_runner.py` (done in Step 1.4)
  - `autopilot/agents/callbacks.py` (done in Step 1.5)
  - `autopilot/core/context.py` (done in Step 1.3)
  - All test files that mock `autopilot.core.context.get_event_bus` (done in Phase 2)

> **Policy**: We do NOT leave deprecated code in the codebase. If something is replaced, it is deleted.

### Phase 2: Update Tests

#### Step 2.1: Update test mocks

All tests that currently `patch("autopilot.core.context.get_event_bus")` need to be updated to mock the unified bus. Since `ctx.emit()` now delegates to `ctx.bus.publish()`, the mock target changes.

**Affected test files**:

- `tests/autopilot/test_core.py` — 5 patches
- `tests/autopilot/test_bus.py` — 3 patches (ctx integration section)
- `tests/autopilot/test_dsl.py` — 7 patches (`_mock_event_bus` helper)
- `tests/autopilot/test_tools.py` — 3 patches

**Pattern**: Replace `patch("autopilot.core.context.get_event_bus")` with `patch("autopilot.core.bus.get_event_bus")` or mock `ctx.bus` directly.

#### Step 2.2: Add new tests for unified behavior

**File**: `tests/autopilot/test_bus.py` (extend)

New test cases:

- `test_ctx_emit_publishes_to_bus`: Verify `ctx.emit("stage_started", data)` results in a `stage.started` event on the bus
- `test_legacy_topic_mapping`: Verify all legacy event types map to correct canonical topics
- `test_middleware_chain`: Verify middleware intercepts and can filter/modify events
- `test_event_bus_protocol_compliance`: Verify `EventBus` satisfies `EventBusProtocol` ABC
- `test_replay_from_history`: Verify `bus.replay("topic.*")` returns historical events
- `test_create_event_bus_factory`: Verify factory selects correct backend from env

#### Step 2.3: Run full test suite

```bash
pytest tests/ -x -v --tb=short
```

### Phase 3: Event Persistence — EventStore Middleware

**Goal**: Add an `EventStore` middleware that persists events for audit, replay, and debugging. This is the foundation for the distributed backend (Phase 4) — Redis Streams IS the event store.

#### Step 3.1: Define `EventStore` ABC

**File**: `autopilot/core/event_store.py` (new)

```python
from abc import ABC, abstractmethod
from autopilot.core.bus import AgentMessage

class EventStore(ABC):
    """Persistent event storage for audit, replay, and debugging.

    Implementations:
      - InMemoryEventStore: dev/test — bounded list
      - RedisEventStore: production — Redis Streams with TTL
    """

    @abstractmethod
    async def append(self, event: AgentMessage) -> None: ...

    @abstractmethod
    async def query(self, topic: str, *, limit: int = 100, since: str | None = None) -> list[AgentMessage]: ...

    @abstractmethod
    async def count(self, topic: str | None = None) -> int: ...

    @abstractmethod
    async def trim(self, max_age_hours: int = 72) -> int:
        """Remove events older than max_age_hours. Returns count removed."""
        ...
```

#### Step 3.2: Implement `InMemoryEventStore`

**File**: `autopilot/core/event_store.py`

```python
class InMemoryEventStore(EventStore):
    """Development-only event store backed by a bounded list."""

    def __init__(self, max_events: int = 10_000):
        self._events: list[AgentMessage] = []
        self._max = max_events

    async def append(self, event: AgentMessage) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]

    async def query(self, topic, *, limit=100, since=None):
        matching = [e for e in self._events if fnmatch.fnmatch(e.topic, topic)]
        if since:
            matching = [e for e in matching if e.timestamp > since]
        return matching[-limit:]

    async def count(self, topic=None):
        if topic is None:
            return len(self._events)
        return sum(1 for e in self._events if fnmatch.fnmatch(e.topic, topic))

    async def trim(self, max_age_hours=72):
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        before = len(self._events)
        self._events = [e for e in self._events if e.timestamp > cutoff]
        return before - len(self._events)
```

#### Step 3.3: Wire as Middleware

```python
def create_event_store_middleware(store: EventStore) -> EventMiddleware:
    async def persist(event: AgentMessage) -> AgentMessage:
        await store.append(event)
        return event  # Pass through — don't filter
    return persist

# Usage:
bus = get_event_bus()
store = InMemoryEventStore()
bus.use(create_event_store_middleware(store))
```

### Phase 4: Distributed Backend — RedisStreamEventBus 🚀

**Goal**: Implement a production-grade distributed event bus using Redis Streams. This is the cloud-native backend that solves cross-instance fanout, event persistence, at-least-once delivery, and replay.

#### Why Redis Streams (Not Cloud Pub/Sub)

| Factor               | Redis Streams                    | Cloud Pub/Sub                        |
| -------------------- | -------------------------------- | ------------------------------------ |
| **Latency**          | ~1ms (same-region Memorystore)   | ~50ms (network hop)                  |
| **Persistence**      | ✅ With TTL (configurable)       | ✅ 7-day retention                   |
| **Consumer Groups**  | ✅ Native (`XREADGROUP`)         | ✅ Subscriptions                     |
| **At-least-once**    | ✅ `XACK` pattern                | ✅ Native                            |
| **Per-message cost** | $0 (flat Memorystore pricing)    | $0.04/1M messages                    |
| **Event replay**     | ✅ `XRANGE` with ID/timestamp    | ❌ No native replay                  |
| **Already in infra** | ✅ `RedisSessionService` roadmap | ✅ Used for Gmail triggers           |
| **Complexity**       | Medium (redis-py async)          | Low (managed)                        |
| **Best for**         | Internal platform events         | External integrations (already used) |

**Verdict**: Redis Streams for internal events (low latency, replay, same infra as sessions). Cloud Pub/Sub stays for external triggers (Gmail webhooks — already working).

#### Step 4.1: Implement `RedisStreamEventBus`

**File**: `autopilot/core/bus_redis.py` (new)

```python
"""
RedisStreamEventBus — Distributed event bus backed by Redis Streams.

Uses Redis Streams for persistent, cross-instance event delivery with
consumer groups for at-least-once processing.

Architecture:
  - Each topic maps to a Redis Stream key: "eventbus:{topic}"
  - Each subscriber group maps to a Consumer Group: "eventbus:group:{name}"
  - Events are XADD'd to the stream and XREAD by consumer groups
  - Dead-letter: failed events are moved to "eventbus:dlq:{topic}"
  - History: XRANGE on the stream key
  - Replay: XRANGE from a specific ID/timestamp

Requirements:
  - redis[hiredis] >= 5.0 (async support)
  - Redis 6.2+ (XAUTOCLAIM for DLQ)
  - Cloud Memorystore (GCP) or ElastiCache (AWS) for production

Environment Variables:
  - REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
  - EVENTBUS_CONSUMER_GROUP: Consumer group name (default: "autopilot")
  - EVENTBUS_CONSUMER_NAME: Unique consumer name per instance (default: hostname)
  - EVENTBUS_STREAM_MAXLEN: Max events per stream (default: 10000)
  - EVENTBUS_DLQ_ENABLED: Enable dead-letter queue (default: true)
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import socket
import structlog
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from opentelemetry import trace

from autopilot.core.bus import (
    EventBusProtocol,
    AgentMessage,
    Subscription,
    MessageHandler,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class RedisStreamEventBus(EventBusProtocol):
    """
    Distributed event bus backed by Redis Streams.

    Features:
      - Cross-instance event fanout via Redis Pub/Sub notifications
      - Persistent event storage via Redis Streams (XADD/XRANGE)
      - Consumer groups for at-least-once delivery (XREADGROUP/XACK)
      - Dead-letter queue for failed handler invocations
      - Event replay from any point in the stream
      - Compatible with InMemoryEventBus API (same protocol)
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        consumer_group: str = "autopilot",
        consumer_name: str | None = None,
        stream_maxlen: int = 10_000,
        dlq_enabled: bool = True,
    ) -> None:
        self._redis = redis_client
        self._group = consumer_group
        self._consumer = consumer_name or f"{socket.gethostname()}-{uuid4().hex[:6]}"
        self._maxlen = stream_maxlen
        self._dlq_enabled = dlq_enabled

        # Local subscription registry (same process)
        self._subscriptions: dict[str, Subscription] = {}
        self._stats = {"published": 0, "delivered": 0, "errors": 0}
        self._listener_task: asyncio.Task | None = None
        self._running = False

    @classmethod
    def from_env(cls) -> RedisStreamEventBus:
        """Create from environment variables."""
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(url, decode_responses=True)
        return cls(
            redis_client=client,
            consumer_group=os.getenv("EVENTBUS_CONSUMER_GROUP", "autopilot"),
            consumer_name=os.getenv("EVENTBUS_CONSUMER_NAME"),
            stream_maxlen=int(os.getenv("EVENTBUS_STREAM_MAXLEN", "10000")),
            dlq_enabled=os.getenv("EVENTBUS_DLQ_ENABLED", "true").lower() == "true",
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background listener for cross-instance events."""
        if self._running:
            return
        self._running = True
        # Use Redis Pub/Sub for instant cross-instance notification
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("redis_bus_started", consumer=self._consumer, group=self._group)

    async def stop(self) -> None:
        """Stop the background listener gracefully."""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        await self._redis.aclose()
        logger.info("redis_bus_stopped", consumer=self._consumer)

    # ── Subscribe ────────────────────────────────────────────────────

    def subscribe(self, topic_pattern: str, handler: MessageHandler) -> Subscription:
        sub = Subscription(topic_pattern=topic_pattern, handler=handler)
        self._subscriptions[sub.id] = sub
        logger.debug("redis_bus_subscribed", sub_id=sub.id, topic=topic_pattern)
        return sub

    def unsubscribe(self, subscription: Subscription) -> bool:
        removed = self._subscriptions.pop(subscription.id, None)
        return removed is not None

    # ── Publish ──────────────────────────────────────────────────────

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        sender: str = "",
        correlation_id: str | None = None,
    ) -> AgentMessage:
        msg = AgentMessage(
            topic=topic,
            sender=sender,
            payload=payload or {},
            **({} if correlation_id is None else {"correlation_id": correlation_id}),
        )

        with tracer.start_as_current_span(
            "redis_bus.publish",
            attributes={"topic": topic, "sender": sender},
        ):
            # 1. Persist to Redis Stream
            stream_key = f"eventbus:{topic}"
            await self._redis.xadd(
                stream_key,
                {"data": msg.model_dump_json()},
                maxlen=self._maxlen,
            )

            # 2. Notify other instances via Redis Pub/Sub channel
            await self._redis.publish("eventbus:notify", topic)

            # 3. Dispatch to local subscribers (same-instance, zero latency)
            await self._dispatch_local(msg)

            self._stats["published"] += 1

        return msg

    # ── History & Replay ─────────────────────────────────────────────

    def history(self, topic: str, *, limit: int = 50) -> list[AgentMessage]:
        """Synchronous history — wraps async for compatibility."""
        # For sync contexts, return empty. Use replay() for full async access.
        return []

    async def replay(
        self,
        topic: str,
        *,
        since: str | None = None,
        handler: MessageHandler | None = None,
    ) -> list[AgentMessage]:
        """Replay events from Redis Stream.

        Args:
            topic: Exact topic (stream key).
            since: Redis stream ID to start from (e.g. "0-0" for all, or a timestamp-based ID).
            handler: Optional handler to invoke for each replayed event.

        Returns:
            List of replayed messages.
        """
        stream_key = f"eventbus:{topic}"
        start = since or "0-0"

        entries = await self._redis.xrange(stream_key, min=start, count=1000)
        messages = []

        for entry_id, data in entries:
            try:
                msg = AgentMessage.model_validate_json(data["data"])
                messages.append(msg)
                if handler:
                    await handler(msg)
            except Exception as exc:
                logger.warning("redis_bus_replay_error", entry_id=entry_id, error=str(exc))

        return messages

    # ── Introspection ────────────────────────────────────────────────

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def clear(self) -> None:
        self._subscriptions.clear()
        self._stats = {"published": 0, "delivered": 0, "errors": 0}

    # ── Internals ────────────────────────────────────────────────────

    async def _dispatch_local(self, msg: AgentMessage) -> None:
        """Dispatch to local (same-process) subscribers matching the topic."""
        matching = [
            sub.handler
            for sub in self._subscriptions.values()
            if sub.handler and fnmatch.fnmatch(msg.topic, sub.topic_pattern)
        ]

        if not matching:
            return

        results = await asyncio.gather(
            *(self._safe_invoke(h, msg) for h in matching),
            return_exceptions=True,
        )

        delivered = sum(1 for r in results if r is None)
        errors = sum(1 for r in results if r is not None)
        self._stats["delivered"] += delivered
        self._stats["errors"] += errors

        # DLQ: move failed events to dead-letter stream
        if errors > 0 and self._dlq_enabled:
            dlq_key = f"eventbus:dlq:{msg.topic}"
            await self._redis.xadd(
                dlq_key,
                {"data": msg.model_dump_json(), "errors": str(errors)},
                maxlen=1000,
            )

    async def _listen_loop(self) -> None:
        """Background loop: listen for cross-instance event notifications."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe("eventbus:notify")

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    topic = message["data"]
                    # Read latest unprocessed events from the stream
                    await self._consume_stream(topic)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe("eventbus:notify")
            await pubsub.aclose()

    async def _consume_stream(self, topic: str) -> None:
        """Consume new events from a Redis Stream via consumer group."""
        stream_key = f"eventbus:{topic}"

        # Ensure consumer group exists
        try:
            await self._redis.xgroup_create(
                stream_key, self._group, id="$", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise  # Group already exists — OK

        entries = await self._redis.xreadgroup(
            self._group,
            self._consumer,
            {stream_key: ">"},
            count=100,
            block=0,
        )

        for _stream, messages in entries:
            for msg_id, data in messages:
                try:
                    msg = AgentMessage.model_validate_json(data["data"])
                    await self._dispatch_local(msg)
                    await self._redis.xack(stream_key, self._group, msg_id)
                except Exception as exc:
                    logger.error(
                        "redis_bus_consume_error",
                        stream=stream_key,
                        msg_id=msg_id,
                        error=str(exc),
                    )

    @staticmethod
    async def _safe_invoke(handler: MessageHandler, msg: AgentMessage) -> None:
        """Invoke handler with dead-letter isolation."""
        with tracer.start_as_current_span(
            "redis_bus.handler",
            attributes={
                "topic": msg.topic,
                "handler": getattr(handler, "__name__", str(handler)),
            },
        ):
            try:
                await handler(msg)
            except Exception as exc:
                logger.error(
                    "redis_bus_handler_error",
                    topic=msg.topic,
                    error=str(exc),
                )
                raise
```

#### Step 4.2: Add Redis Dependencies

**File**: `pyproject.toml` (add to `[project.dependencies]`)

```
redis[hiredis]>=5.0
```

The `hiredis` extra provides a C-accelerated parser for ~10x faster Redis operations.

#### Step 4.3: Infrastructure — Cloud Memorystore

**Deployment**: Add Redis (Cloud Memorystore) to the Cloud Run deployment:

```bash
# Create Memorystore Redis instance (Basic tier, 1GB, same region as Cloud Run)
gcloud redis instances create autopilot-bus \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_7_0 \
  --tier=BASIC

# Get connection info
gcloud redis instances describe autopilot-bus --region=us-central1

# Add to Cloud Run env
gcloud run services update autopilot \
  --set-env-vars="REDIS_URL=redis://10.x.x.x:6379/0,EVENTBUS_BACKEND=redis"
```

> **Note**: Cloud Memorystore requires VPC Connector for Cloud Run access. This is a one-time setup.

#### Step 4.4: Startup Integration

**File**: `app.py` (lifespan hook)

```python
from contextlib import asynccontextmanager
from autopilot.core.bus import get_event_bus

@asynccontextmanager
async def lifespan(app):
    # Start distributed bus listener (if Redis backend)
    bus = get_event_bus()
    if hasattr(bus, "start"):
        await bus.start()
    yield
    if hasattr(bus, "stop"):
        await bus.stop()
```

#### Step 4.5: DLQ Monitoring Endpoint

**File**: `autopilot/api/system.py` (add)

```python
@router.get("/events/dlq")
async def list_dead_letters():
    """List events that failed handler processing."""
    bus = get_event_bus()
    if not isinstance(bus, RedisStreamEventBus):
        return {"backend": "memory", "dlq": []}

    # Scan for DLQ streams
    keys = await bus._redis.keys("eventbus:dlq:*")
    dlq = {}
    for key in keys:
        topic = key.replace("eventbus:dlq:", "")
        entries = await bus._redis.xrange(key, count=50)
        dlq[topic] = [json.loads(data["data"]) for _, data in entries]

    return {"backend": "redis", "dlq": dlq}
```

#### Step 4.6: Tests for RedisStreamEventBus

**File**: `tests/autopilot/test_bus_redis.py` (new)

```python
"""
Tests for RedisStreamEventBus — distributed event bus via Redis Streams.

Uses fakeredis for unit tests (no real Redis required).
Integration tests with real Redis are marked with @pytest.mark.integration.
"""

import pytest
from unittest.mock import AsyncMock
import fakeredis.aioredis

from autopilot.core.bus_redis import RedisStreamEventBus
from autopilot.core.bus import AgentMessage


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def redis_bus(redis_client):
    return RedisStreamEventBus(
        redis_client=redis_client,
        consumer_group="test",
        consumer_name="test-node",
    )


class TestRedisStreamPublish:
    @pytest.mark.asyncio
    async def test_publish_persists_to_stream(self, redis_bus, redis_client):
        await redis_bus.publish("test.event", {"key": "val"}, sender="test")
        entries = await redis_client.xrange("eventbus:test.event")
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_publish_dispatches_locally(self, redis_bus):
        received = []
        async def handler(msg): received.append(msg)
        redis_bus.subscribe("test.*", handler)
        await redis_bus.publish("test.event", {"n": 1})
        assert len(received) == 1


class TestRedisStreamReplay:
    @pytest.mark.asyncio
    async def test_replay_returns_persisted_events(self, redis_bus):
        await redis_bus.publish("audit.log", {"n": 1})
        await redis_bus.publish("audit.log", {"n": 2})
        events = await redis_bus.replay("audit.log")
        assert len(events) == 2
        assert events[0].payload["n"] == 1


class TestRedisStreamDLQ:
    @pytest.mark.asyncio
    async def test_failed_handler_writes_to_dlq(self, redis_bus, redis_client):
        async def bad(msg): raise ValueError("boom")
        redis_bus.subscribe("fail.*", bad)
        await redis_bus.publish("fail.event", {"x": 1})
        dlq = await redis_client.xrange("eventbus:dlq:fail.event")
        assert len(dlq) == 1
```

### Phase 5: Update Documentation

#### Step 5.1: Update ARCHITECTURE.md

Replace the dual-bus documentation with unified bus docs:

- Remove all references to `PipelineEventBus`
- Remove `services/event_bus.py` from any architecture diagrams
- Update `AgentContext` API docs (`emit()` → convenience alias for `publish()`)
- Document the topic hierarchy
- Document the middleware system
- Document `EventBusProtocol` for backend selection
- Document `RedisStreamEventBus` for production
- Add backend selection table (memory vs redis vs pubsub)
- Document DLQ monitoring endpoint
- Update deployment workflow with Redis setup

#### Step 5.2: Update Deployment Workflow

**File**: `.agent/workflows/deploy_to_cloud_run.md`

Add Redis/Memorystore setup to the deployment checklist.

## 4. Migration Checklist

| #    | Task                                                      | File(s)                             | Breaking?            | Phase |
| ---- | --------------------------------------------------------- | ----------------------------------- | -------------------- | ----- |
| 1.1  | Create `EventBusProtocol` ABC                             | `core/bus.py`                       | ❌                   | 1     |
| 1.2  | Rename `AgentBus` → `EventBus` + aliases                  | `core/bus.py`                       | ❌ (aliases)         | 1     |
| 1.3  | Add middleware support to `EventBus`                      | `core/bus.py`                       | ❌                   | 1     |
| 1.4  | Add `replay()` method                                     | `core/bus.py`                       | ❌                   | 1     |
| 1.5  | Add `create_event_bus()` factory                          | `core/bus.py`                       | ❌                   | 1     |
| 1.6  | Unify `ctx.emit()` → `ctx.publish()` delegation           | `core/context.py`                   | ❌ (backward compat) | 1     |
| 1.7  | Migrate `PipelineRunner` → `EventBus`                     | `agents/pipeline_runner.py`         | ❌                   | 1     |
| 1.8  | Migrate callbacks → `EventBus`                            | `agents/callbacks.py`               | ❌                   | 1     |
| 1.9  | Migrate tool callbacks → `EventBus`                       | `agents/tool_callbacks.py`          | ❌                   | 1     |
| 1.10 | **Delete** `services/event_bus.py` (hard delete, no shim) | `services/event_bus.py`             | ❌                   | 1     |
| 1.11 | Update `core/__init__.py` exports                         | `core/__init__.py`                  | ❌                   | 1     |
| 2.1  | Update test mocks (4 test files, ~18 patches)             | `tests/autopilot/test_*.py`         | ❌                   | 2     |
| 2.2  | Add new unified bus tests                                 | `tests/autopilot/test_bus.py`       | ❌                   | 2     |
| 2.3  | Run full test suite → green                               | —                                   | ❌                   | 2     |
| 3.1  | Create `EventStore` ABC + InMemory impl                   | `core/event_store.py`               | ❌                   | 3     |
| 3.2  | Wire event store as middleware                            | `core/bus.py`                       | ❌                   | 3     |
| 4.1  | Implement `RedisStreamEventBus`                           | `core/bus_redis.py`                 | ❌                   | 4     |
| 4.2  | Add `redis[hiredis]` dependency                           | `pyproject.toml`                    | ❌                   | 4     |
| 4.3  | Add `fakeredis` test dependency                           | `requirements-dev.txt`              | ❌                   | 4     |
| 4.4  | Startup/shutdown lifecycle integration                    | `app.py`                            | ❌                   | 4     |
| 4.5  | DLQ monitoring endpoint                                   | `api/system.py`                     | ❌                   | 4     |
| 4.6  | Redis bus tests (fakeredis)                               | `tests/autopilot/test_bus_redis.py` | ❌                   | 4     |
| 4.7  | Infrastructure: Cloud Memorystore setup                   | Terraform/gcloud                    | ❌                   | 4     |
| 5.1  | Update ARCHITECTURE.md                                    | `docs/ARCHITECTURE.md`              | ❌                   | 5     |
| 5.2  | Update deployment workflow                                | `.agent/workflows/`                 | ❌                   | 5     |

**Total**: 0 breaking changes across all phases. 0 deprecated code left behind.

## 5. Files Changed Summary

| Action     | File                                                               | Phase |
| ---------- | ------------------------------------------------------------------ | ----- |
| **MODIFY** | `autopilot/core/bus.py` — ABC, rename, middleware, replay, factory | 1     |
| **MODIFY** | `autopilot/core/context.py` — Unify emit() → publish()             | 1     |
| **MODIFY** | `autopilot/core/__init__.py` — Update exports                      | 1     |
| **MODIFY** | `autopilot/core/subscribers.py` — Update imports                   | 1     |
| **MODIFY** | `autopilot/agents/pipeline_runner.py` — Switch to EventBus         | 1     |
| **MODIFY** | `autopilot/agents/callbacks.py` — Remove ContextVar, use EventBus  | 1     |
| **MODIFY** | `autopilot/agents/tool_callbacks.py` — Update \_emit_event_async   | 1     |
| **DELETE** | `autopilot/services/event_bus.py` — Hard delete, no shim           | 1     |
| **MODIFY** | `tests/autopilot/test_bus.py` — Update mocks + new tests           | 2     |
| **MODIFY** | `tests/autopilot/test_core.py` — Update mocks                      | 2     |
| **MODIFY** | `tests/autopilot/test_dsl.py` — Update \_mock_event_bus helper     | 2     |
| **MODIFY** | `tests/autopilot/test_tools.py` — Update mocks                     | 2     |
| **CREATE** | `autopilot/core/event_store.py` — EventStore ABC + InMemory        | 3     |
| **CREATE** | `autopilot/core/bus_redis.py` — RedisStreamEventBus                | 4     |
| **CREATE** | `tests/autopilot/test_bus_redis.py` — Redis bus tests              | 4     |
| **MODIFY** | `pyproject.toml` — Add redis[hiredis] to `[project.dependencies]`  | 4     |
| **MODIFY** | `app.py` — Lifespan hooks for bus start/stop                       | 4     |
| **MODIFY** | `autopilot/api/system.py` — DLQ endpoint                           | 4     |
| **MODIFY** | `docs/ARCHITECTURE.md` — Full rewrite of event bus section         | 5     |
| **MODIFY** | `.agent/workflows/deploy_to_cloud_run.md` — Redis setup            | 5     |

## 6. ADK Alignment Notes

This design aligns with Google ADK's event architecture:

1. **Typed Events**: `AgentMessage` (Pydantic) mirrors ADK's `Event` model
2. **Channel Subscriptions**: Topic-based routing mirrors ADK's event channels
3. **Middleware Chain**: Similar to ADK's `before_model_callback`/`after_model_callback` composition pattern
4. **Protocol-First**: `EventBusProtocol` ABC follows ADK's service interfaces (`SessionService`, `ArtifactService`, etc.)
5. **Singleton + Injection**: `get_event_bus()` follows the same lazy singleton pattern as ADK's session services
6. **Backend Swappability**: Mirrors ADK's `InMemorySessionService` / `DatabaseSessionService` pattern exactly
7. **Consumer Groups**: Redis XREADGROUP mirrors ADK's event delivery guarantees

## 7. Risk Assessment

| Risk                                      | Mitigation                                                                               |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| Breaking existing tests                   | All imports updated in-place + backward-compatible class aliases (`AgentBus = EventBus`) |
| Performance regression (pub/sub vs queue) | InMemoryEventBus is functionally identical to current AgentBus — no regression           |
| Redis adds infra complexity               | Phase 4 is independent — system works fine with InMemory until ready                     |
| Redis connection failures in prod         | `RedisStreamEventBus` falls back to local dispatch if Redis is unreachable               |
| Losing `pipeline_session_id` ContextVar   | Replace with `sender` field in events — same correlation capability                      |
| Future SSE need                           | A subscriber on `pipeline.*` can bridge to any SSE/WebSocket transport                   |
| Cost of Cloud Memorystore                 | Basic 1GB instance ≈ $35/mo — justified by persistence + cross-instance                  |

## 8. Execution Order

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
 (core)     (tests)    (persist)   (redis)     (docs)

  1 PR        same PR    can be      separate    same PR
              as P1      separate    PR          as P4
```

1. ✅ **Phase 1** (Steps 1.1–1.11): Core unification — **do first**
2. ✅ **Phase 2** (Steps 2.1–2.3): Test migration — **same PR as Phase 1**
3. ✅ **Phase 3** (Steps 3.1–3.2): Event persistence middleware — **can ship independently**
4. ✅ **Phase 4** (Steps 4.1–4.7): Redis Streams distributed backend — **separate PR, requires infra**
5. ✅ **Phase 5** (Steps 5.1–5.2): Documentation update — **same PR as Phase 4**
