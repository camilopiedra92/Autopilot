"""
Pipeline Event Bus — asyncio.Queue-based event streaming for SSE.

Provides real-time pipeline stage events to SSE streaming endpoints.
Each pipeline invocation creates a session with its own asyncio.Queue.
Events are emitted from agent callbacks and consumed by the SSE generator.

Usage:
    # In the endpoint:
    queue = event_bus.create_session(session_id)
    async for event in event_bus.stream(session_id):
        yield event

    # In agent callbacks:
    await event_bus.emit(session_id, {"stage": "email_parser", "status": "completed", ...})
"""

import asyncio
import json
import structlog
from datetime import datetime, timezone
from typing import AsyncGenerator

logger = structlog.get_logger(__name__)

# Sentinel object to signal end of stream
_STREAM_END = object()


class PipelineEventBus:
    """
    Async event bus for pipeline stage streaming.

    Thread-safe session management with per-session asyncio.Queues.
    Supports multiple concurrent pipeline invocations.
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str) -> asyncio.Queue:
        """
        Create a new event queue for a pipeline session.

        Args:
            session_id: Unique pipeline session identifier.

        Returns:
            The asyncio.Queue for this session.
        """
        async with self._lock:
            if session_id in self._queues:
                logger.warning("event_bus_session_exists", session_id=session_id)
            queue: asyncio.Queue = asyncio.Queue()
            self._queues[session_id] = queue
            logger.debug("event_bus_session_created", session_id=session_id)
            return queue

    async def emit(self, session_id: str, event: dict) -> None:
        """
        Emit an event to the session's queue.

        Args:
            session_id: Pipeline session identifier.
            event: Event data dictionary. Must be JSON-serializable.
        """
        queue = self._queues.get(session_id)
        if queue is None:
            logger.debug(
                "event_bus_no_session",
                session_id=session_id,
                event_type=event.get("type", "unknown"),
            )
            return

        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        await queue.put(event)
        logger.debug(
            "event_bus_emitted",
            session_id=session_id,
            event_type=event.get("type", "unknown"),
            stage=event.get("stage", ""),
        )

    async def end_stream(self, session_id: str) -> None:
        """
        Signal the end of the event stream for a session.

        Args:
            session_id: Pipeline session identifier.
        """
        queue = self._queues.get(session_id)
        if queue is not None:
            await queue.put(_STREAM_END)
            logger.debug("event_bus_stream_ended", session_id=session_id)

    async def stream(
        self, session_id: str, timeout: float = 120.0
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields events from the session's queue.

        Yields events until the stream end sentinel is received or timeout.

        Args:
            session_id: Pipeline session identifier.
            timeout: Maximum seconds to wait for each event (default 120s).

        Yields:
            Event dictionaries from the pipeline.
        """
        queue = self._queues.get(session_id)
        if queue is None:
            logger.warning("event_bus_stream_no_session", session_id=session_id)
            return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "event_bus_stream_timeout",
                        session_id=session_id,
                        timeout=timeout,
                    )
                    yield {
                        "type": "timeout",
                        "message": "Stream timed out waiting for events.",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    break

                if event is _STREAM_END:
                    break

                yield event
        finally:
            await self.cleanup_session(session_id)

    async def cleanup_session(self, session_id: str) -> None:
        """
        Remove a session's queue from the bus.

        Args:
            session_id: Pipeline session identifier.
        """
        async with self._lock:
            self._queues.pop(session_id, None)
            logger.debug("event_bus_session_cleaned", session_id=session_id)


# ── Singleton ─────────────────────────────────────────────────────────

_event_bus: PipelineEventBus | None = None


def get_event_bus() -> PipelineEventBus:
    """Get or create the global PipelineEventBus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = PipelineEventBus()
    return _event_bus
