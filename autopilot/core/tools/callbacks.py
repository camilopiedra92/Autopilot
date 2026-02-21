"""
Tool Lifecycle Callbacks — before/after hooks for every tool invocation.

Mirrors Google ADK's ``before_tool_callback`` / ``after_tool_callback``
pattern, providing platform-level interception points around every tool call.

This enables:
  - Pre-execution validation (guardrails at the tool level)
  - Rate limiting per tool
  - Granular audit logging
  - Auth injection / credential checking
  - Post-execution transforms (skip summarization, result caching)

Usage::

    from autopilot.core.tools.callbacks import (
        ToolCallbackManager, get_callback_manager,
    )

    manager = get_callback_manager()

    # Register a before-hook
    @manager.before
    async def check_auth(tool_name: str, args: dict, context: dict) -> dict | None:
        if not context.get("api_key"):
            return {"error": "Missing API key", "blocked": True}
        return None  # Allow execution

    # Register an after-hook
    @manager.after
    async def audit_log(tool_name: str, args: dict, result: Any, context: dict) -> Any:
        logger.info("tool_executed", tool=tool_name, args=args)
        return result  # Pass through unchanged

Architecture:
  - ``BeforeToolCallback``: Called before a tool runs. Return None to proceed,
    or a dict to short-circuit with that result.
  - ``AfterToolCallback``: Called after a tool runs. Return the (possibly
    modified) result.
  - ``ToolCallbackManager``: Manages ordered lists of callbacks and runs them.

Note: This complements ADK's native callbacks. When tools run through ADK's
``Runner``, ADK's own callbacks fire. These platform callbacks fire for ALL
tool invocations (including non-ADK pipeline tools).
"""

from __future__ import annotations

import time
import structlog
from typing import Any, Callable, Protocol, runtime_checkable

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Callback Protocols
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@runtime_checkable
class BeforeToolCallback(Protocol):
    """
    Called before a tool executes.

    Args:
        tool_name: Name of the tool about to execute.
        args: Arguments that will be passed to the tool.
        context: Mutable context dict (state, session data, etc.).

    Returns:
        None to allow execution, or a dict to short-circuit with that result.
    """

    async def __call__(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class AfterToolCallback(Protocol):
    """
    Called after a tool executes.

    Args:
        tool_name: Name of the tool that just executed.
        args: Arguments that were passed to the tool.
        result: The tool's return value.
        context: Mutable context dict (state, session data, etc.).

    Returns:
        The (possibly modified) result to return to the caller.
    """

    async def __call__(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        context: dict[str, Any],
    ) -> Any: ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolCallbackManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolCallbackManager:
    """
    Manages ordered lists of before/after tool callbacks.

    Callbacks are executed in registration order. Before callbacks can
    short-circuit execution by returning a non-None result. After callbacks
    receive the result and can modify it before it is returned to the caller.

    Thread-safety: safe for single-process asyncio (like Cloud Run).
    """

    def __init__(self) -> None:
        self._before: list[BeforeToolCallback] = []
        self._after: list[AfterToolCallback] = []
        self._tool_filters: dict[str, list[str]] = {}  # callback_id -> tool_names

    # ── Registration ─────────────────────────────────────────────────

    def register_before(
        self,
        callback: BeforeToolCallback,
        *,
        tools: list[str] | None = None,
    ) -> None:
        """
        Register a before-tool callback.

        Args:
            callback: The callback to register.
            tools: Optional list of tool names this callback applies to.
                   If None, applies to ALL tools.
        """
        self._before.append(callback)
        if tools:
            self._tool_filters[id(callback)] = tools
        logger.debug(
            "before_callback_registered",
            callback=getattr(callback, "__name__", repr(callback)),
            tools=tools or "all",
        )

    def register_after(
        self,
        callback: AfterToolCallback,
        *,
        tools: list[str] | None = None,
    ) -> None:
        """
        Register an after-tool callback.

        Args:
            callback: The callback to register.
            tools: Optional list of tool names this callback applies to.
                   If None, applies to ALL tools.
        """
        self._after.append(callback)
        if tools:
            self._tool_filters[id(callback)] = tools
        logger.debug(
            "after_callback_registered",
            callback=getattr(callback, "__name__", repr(callback)),
            tools=tools or "all",
        )

    # ── Decorator syntax ─────────────────────────────────────────────

    def before(
        self,
        func: Callable | None = None,
        *,
        tools: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a before-tool callback."""
        def decorator(fn: Callable) -> Callable:
            self.register_before(fn, tools=tools)
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def after(
        self,
        func: Callable | None = None,
        *,
        tools: list[str] | None = None,
    ) -> Callable:
        """Decorator to register an after-tool callback."""
        def decorator(fn: Callable) -> Callable:
            self.register_after(fn, tools=tools)
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    # ── Execution ────────────────────────────────────────────────────

    def _applies_to(self, callback: Callable, tool_name: str) -> bool:
        """Check if a callback applies to the given tool."""
        filter_list = self._tool_filters.get(id(callback))
        if filter_list is None:
            return True  # No filter = applies to all
        return tool_name in filter_list

    async def run_before(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Execute all before callbacks in order.

        Returns:
            None if all callbacks allow execution, or a dict result
            from the first callback that short-circuits.
        """
        for callback in self._before:
            if not self._applies_to(callback, tool_name):
                continue

            try:
                result = await callback(tool_name, args, context)
                if result is not None:
                    logger.info(
                        "tool_blocked_by_before_callback",
                        tool=tool_name,
                        callback=getattr(callback, "__name__", "unknown"),
                        result_keys=list(result.keys()) if isinstance(result, dict) else None,
                    )
                    return result
            except Exception as exc:
                logger.error(
                    "before_callback_error",
                    tool=tool_name,
                    callback=getattr(callback, "__name__", "unknown"),
                    error=str(exc),
                )
                # Don't block execution on callback errors
                continue

        return None

    async def run_after(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        context: dict[str, Any],
    ) -> Any:
        """
        Execute all after callbacks in order.

        Each callback receives the result from the previous one (chain).

        Returns:
            The final (possibly modified) result.
        """
        current = result
        for callback in self._after:
            if not self._applies_to(callback, tool_name):
                continue

            try:
                current = await callback(tool_name, args, current, context)
            except Exception as exc:
                logger.error(
                    "after_callback_error",
                    tool=tool_name,
                    callback=getattr(callback, "__name__", "unknown"),
                    error=str(exc),
                )
                # Pass through unchanged on callback error

        return current

    # ── Introspection ────────────────────────────────────────────────

    @property
    def before_count(self) -> int:
        return len(self._before)

    @property
    def after_count(self) -> int:
        return len(self._after)

    def clear(self) -> None:
        """Remove all registered callbacks."""
        self._before.clear()
        self._after.clear()
        self._tool_filters.clear()

    def __repr__(self) -> str:
        return f"<ToolCallbackManager before={self.before_count} after={self.after_count}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_manager: ToolCallbackManager | None = None


def get_callback_manager() -> ToolCallbackManager:
    """Process-global singleton accessor for the ToolCallbackManager."""
    global _manager
    if _manager is None:
        _manager = ToolCallbackManager()
    return _manager


def reset_callback_manager() -> None:
    """Reset the global callback manager. For testing only."""
    global _manager
    _manager = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Built-in Callbacks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def audit_log_callback(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    context: dict[str, Any],
) -> Any:
    """
    Built-in after-callback that logs every tool invocation.

    Emits a structured log with tool name, argument keys (not values
    for security), execution duration, and result type.
    """
    logger.info(
        "tool_audit_log",
        tool=tool_name,
        arg_keys=list(args.keys()),
        result_type=type(result).__name__,
        execution_id=context.get("execution_id", "unknown"),
    )
    return result


def create_rate_limit_callback(
    max_calls: int = 10,
    window_seconds: float = 60.0,
) -> BeforeToolCallback:
    """
    Factory for a before-callback that enforces per-tool rate limiting.

    Args:
        max_calls: Maximum number of calls allowed within the window.
        window_seconds: Time window in seconds.

    Returns:
        A before-callback that blocks execution when the limit is exceeded.
    """
    call_log: dict[str, list[float]] = {}

    async def rate_limit_callback(
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        timestamps = call_log.setdefault(tool_name, [])

        # Prune old timestamps
        cutoff = now - window_seconds
        timestamps[:] = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= max_calls:
            logger.warning(
                "tool_rate_limited",
                tool=tool_name,
                max_calls=max_calls,
                window_seconds=window_seconds,
            )
            return {
                "error": f"Rate limit exceeded for '{tool_name}': "
                         f"{max_calls} calls per {window_seconds}s",
                "blocked": True,
                "retryable": True,
            }

        timestamps.append(now)
        return None

    rate_limit_callback.__name__ = "rate_limit_callback"
    return rate_limit_callback


async def auth_check_callback(
    tool_name: str,
    args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Built-in before-callback that checks for required credentials.

    Looks for a ``required_auth`` key in the context dict. If present,
    verifies that the required credential keys exist in the context.

    Context keys:
        required_auth: list[str] — credential keys required for this tool
        (the credential values themselves should also be in context)
    """
    required = context.get("required_auth")
    if not required:
        return None  # No auth requirement

    missing = [key for key in required if not context.get(key)]
    if missing:
        logger.warning(
            "tool_auth_missing",
            tool=tool_name,
            missing_keys=missing,
        )
        return {
            "error": f"Missing credentials for '{tool_name}': {missing}",
            "blocked": True,
            "auth_required": True,
        }

    return None
"""
Description: Tool lifecycle callbacks for before/after hook interception around every tool call, enabling rate limiting, audit logging, auth checking, and result transformation.
"""
