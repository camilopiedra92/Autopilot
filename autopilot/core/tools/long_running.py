"""
LongRunningTool — Platform wrapper for ADK's LongRunningFunctionTool.

Provides support for tools that initiate asynchronous operations:
  - Batch processing (e.g., create 50 YNAB transactions)
  - Approval flows (e.g., wait for manager approval)
  - External system callbacks (e.g., webhook completion)

The tool function returns an initial status immediately. The agent can
then check back on subsequent turns via the ADK session's pending
function calls mechanism.

ADK Pattern::

    # In ADK, you wrap a function:
    tool = LongRunningFunctionTool(func=create_batch)

    # The agent uses it, gets an initial response, and can ask
    # "is the batch done yet?" on the next turn.

This module adds:
  - Platform-level tracking of long-running operations
  - Integration with the ToolRegistry for discoverability
  - Conversion to ADK-compatible LongRunningFunctionTool

Usage::

    from autopilot.core.tools.long_running import long_running_tool, get_operation_tracker

    @long_running_tool(tags=["finance", "batch"])
    def create_batch_transactions(transactions: list[dict]) -> dict:
        \"\"\"Create multiple transactions in batch.\"\"\"
        operation_id = start_batch_processing(transactions)
        return {"status": "pending", "operation_id": operation_id}

    # Track operations
    tracker = get_operation_tracker()
    status = tracker.get_status("op-123")
"""

from __future__ import annotations

import inspect
import time
import structlog
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from uuid import uuid4

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OperationStatus — Tracking long-running operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class OperationStatus:
    """
    Tracks the status of a long-running tool operation.

    Attributes:
        operation_id: Unique identifier for this operation.
        tool_name: The tool that initiated the operation.
        status: Current status ("pending", "running", "completed", "failed").
        result: Final result (populated when status is "completed").
        error: Error message (populated when status is "failed").
        created_at: Timestamp when the operation was created.
        updated_at: Timestamp of the last status update.
        metadata: Additional context (e.g., progress percentage).
    """

    operation_id: str
    tool_name: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if the operation has reached a terminal state."""
        return self.status in ("completed", "failed")

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the operation was created."""
        return time.monotonic() - self.created_at


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OperationTracker — Track all long-running operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OperationTracker:
    """
    In-memory tracker for long-running tool operations.

    Provides lifecycle management for async operations:
      - Create operations with auto-generated IDs
      - Update status (pending → running → completed/failed)
      - Query active and completed operations
      - Cleanup stale entries

    Thread-safety: safe for single-process asyncio (like Cloud Run).
    """

    def __init__(self) -> None:
        self._operations: dict[str, OperationStatus] = {}

    def create(
        self,
        tool_name: str,
        *,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationStatus:
        """
        Create a new operation entry.

        Args:
            tool_name: The tool initiating the operation.
            operation_id: Optional ID (auto-generated if not provided).
            metadata: Optional initial metadata.

        Returns:
            The created OperationStatus.
        """
        op_id = operation_id or f"op-{uuid4().hex[:12]}"
        op = OperationStatus(
            operation_id=op_id,
            tool_name=tool_name,
            metadata=metadata or {},
        )
        self._operations[op_id] = op
        logger.info(
            "operation_created",
            operation_id=op_id,
            tool=tool_name,
        )
        return op

    def update(
        self,
        operation_id: str,
        *,
        status: str | None = None,
        result: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationStatus | None:
        """
        Update an existing operation.

        Args:
            operation_id: The operation to update.
            status: New status value.
            result: Final result (for completed operations).
            error: Error message (for failed operations).
            metadata: Additional metadata to merge.

        Returns:
            The updated OperationStatus, or None if not found.
        """
        op = self._operations.get(operation_id)
        if op is None:
            logger.warning("operation_not_found", operation_id=operation_id)
            return None

        if status:
            op.status = status
        if result is not None:
            op.result = result
        if error is not None:
            op.error = error
        if metadata:
            op.metadata.update(metadata)
        op.updated_at = time.monotonic()

        logger.info(
            "operation_updated",
            operation_id=operation_id,
            status=op.status,
        )
        return op

    def get_status(self, operation_id: str) -> OperationStatus | None:
        """Get the current status of an operation."""
        return self._operations.get(operation_id)

    def list_active(self) -> list[OperationStatus]:
        """List all non-terminal operations."""
        return [op for op in self._operations.values() if not op.is_terminal]

    def list_by_tool(self, tool_name: str) -> list[OperationStatus]:
        """List all operations for a given tool."""
        return [op for op in self._operations.values() if op.tool_name == tool_name]

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        """
        Remove completed/failed operations older than max_age_seconds.

        Returns:
            Number of operations removed.
        """
        now = time.monotonic()
        to_remove = [
            op_id
            for op_id, op in self._operations.items()
            if op.is_terminal and (now - op.updated_at) > max_age_seconds
        ]
        for op_id in to_remove:
            del self._operations[op_id]

        if to_remove:
            logger.info("operations_cleaned_up", count=len(to_remove))
        return len(to_remove)

    def clear(self) -> None:
        """Remove all tracked operations."""
        self._operations.clear()

    def __len__(self) -> int:
        return len(self._operations)

    def __repr__(self) -> str:
        active = len(self.list_active())
        return f"<OperationTracker total={len(self)} active={active}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_tracker: OperationTracker | None = None


def get_operation_tracker() -> OperationTracker:
    """Process-global singleton accessor for the OperationTracker."""
    global _tracker
    if _tracker is None:
        _tracker = OperationTracker()
    return _tracker


def reset_operation_tracker() -> None:
    """Reset the global tracker. For testing only."""
    global _tracker
    _tracker = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LongRunningTool — ADK-compatible wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LongRunningTool:
    """
    Wraps a function as a long-running tool compatible with Google ADK.

    The wrapped function should return an initial status dict with at minimum
    a ``status`` field. The platform's OperationTracker automatically tracks
    the operation lifecycle.

    Converts to ADK's ``LongRunningFunctionTool`` via ``to_adk_tool()``.

    Attributes:
        func: The underlying function.
        name: Tool name (defaults to func.__name__).
        description: Tool description (defaults to docstring).
        tags: Searchable tags for the ToolRegistry.
    """

    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").split("\n")[0]
        self.tags = tags or []
        self._is_async = inspect.iscoroutinefunction(func)

    def to_adk_tool(self) -> Any:
        """
        Convert to an ADK ``LongRunningFunctionTool``.

        Returns:
            An instance of ``google.adk.tools.LongRunningFunctionTool``.

        Raises:
            ImportError: If ADK is not installed.
        """
        try:
            from google.adk.tools import LongRunningFunctionTool
        except ImportError as e:
            raise ImportError(
                "google-adk is required for LongRunningFunctionTool. "
                "Install with: pip install google-adk"
            ) from e

        return LongRunningFunctionTool(func=self.func)

    def register(self) -> None:
        """Register this tool in both the ToolRegistry and OperationTracker."""
        from autopilot.core.tools.registry import get_tool_registry

        registry = get_tool_registry()
        registry.register(
            self.func,
            name=self.name,
            description=f"[Long-Running] {self.description}",
            tags=self.tags + ["long_running"],
            source="long_running",
        )

    def __repr__(self) -> str:
        return f"<LongRunningTool name={self.name!r}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @long_running_tool decorator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def long_running_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> LongRunningTool | Callable[..., LongRunningTool]:
    """
    Decorator to create a LongRunningTool and register it.

    Usage::

        @long_running_tool(tags=["finance", "batch"])
        def create_batch_transactions(transactions: list[dict]) -> dict:
            \"\"\"Create multiple transactions in batch.\"\"\"
            op_id = start_processing(transactions)
            return {"status": "pending", "operation_id": op_id}

        @long_running_tool
        def approve_expense(expense_id: str) -> dict:
            ...

    Returns:
        A LongRunningTool instance (auto-registered in the ToolRegistry).
    """

    def decorator(fn: Callable) -> LongRunningTool:
        tool = LongRunningTool(fn, name=name, description=description, tags=tags)
        tool.register()
        return tool

    if func is not None:
        return decorator(func)
    return decorator
