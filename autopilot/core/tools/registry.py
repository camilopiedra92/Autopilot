"""
ToolRegistry — Centralized tool registry for the Edge-Native AI Platform.

Every tool in the platform is a plain Python function (sync or async) that
gets registered here.  The registry extracts metadata (name, description,
parameter schema) and can convert tools into ADK ``FunctionTool`` instances
ready for injection into any ``LlmAgent``.

Key concepts:
  - ``@tool`` decorator: Registers a function and captures metadata.
  - ``ToolInfo``: Pydantic model describing a registered tool.
  - ``ToolRegistry``: Singleton store for all platform tools.
  - ``to_adk_tools()``: Batch-converts to ``google.adk.tools.FunctionTool``.

Usage::

    from autopilot.core.tools import tool, get_tool_registry

    @tool(tags=["ynab", "finance"])
    async def create_transaction(budget_id: str, payee: str, amount: float) -> dict:
        \"\"\"Create a YNAB transaction.\"\"\"
        ...

    # Later — give all registered tools to an LlmAgent:
    agent = LlmAgent(tools=get_tool_registry().to_adk_tools())
"""

import inspect
import functools
import structlog
from typing import Callable, Sequence

from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolInfo — Metadata model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolInfo(BaseModel):
    """Metadata for a registered tool, suitable for dashboards and listing APIs."""

    name: str
    description: str = ""
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter name → type annotation as string.",
    )
    tags: list[str] = Field(default_factory=list)
    source: str = Field(
        default="manual",
        description="Origin: 'decorator', 'connector', 'mcp', or 'manual'.",
    )
    is_async: bool = False
    requires_context: bool = Field(
        default=False,
        description="True if the tool accepts a ToolContext parameter (auto-detected).",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolRegistry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolRegistry:
    """
    Global, process-level registry of reusable tools.

    Mirrors the ``ConnectorRegistry`` pattern: a singleton dictionary
    of callables keyed by unique name.  Each entry also carries a
    ``ToolInfo`` with rich metadata for discoverability.

    Thread-safety: safe for single-process asyncio (like Cloud Run).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._info: dict[str, ToolInfo] = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> None:
        """
        Register a callable as a platform tool.

        Args:
            func: The function to register (sync or async).
            name: Override name (defaults to ``func.__name__``).
            description: Override description (defaults to docstring first line).
            tags: Searchable tags for filtering.
            source: Origin indicator for auditing.

        Raises:
            autopilot.errors.ToolRegistryError: If duplicate name detected.
        """
        from autopilot.errors import ToolRegistryError

        resolved_name = name or func.__name__
        if resolved_name in self._tools:
            raise ToolRegistryError(
                f"Tool '{resolved_name}' is already registered. "
                f"Use a unique name or unregister first.",
                tool_name=resolved_name,
            )

        # Extract metadata from function signature
        sig = inspect.signature(func)
        params: dict[str, str] = {}
        _requires_context = False
        for pname, param in sig.parameters.items():
            # Detect ToolContext parameter (ADK auto-injects when present)
            annotation = param.annotation
            ann_name = getattr(annotation, "__name__", str(annotation))
            if pname == "tool_context" or ann_name == "ToolContext":
                _requires_context = True
                continue  # Don't include in user-facing params
            if annotation is inspect.Parameter.empty:
                params[pname] = "Any"
            else:
                params[pname] = ann_name

        resolved_desc = description or _extract_docstring(func)

        info = ToolInfo(
            name=resolved_name,
            description=resolved_desc,
            parameters=params,
            tags=tags or [],
            source=source,
            is_async=inspect.iscoroutinefunction(func),
            requires_context=_requires_context,
        )

        self._tools[resolved_name] = func
        self._info[resolved_name] = info

        logger.info(
            "tool_registered",
            name=resolved_name,
            source=source,
            params=list(params.keys()),
            tags=info.tags,
        )

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)
        self._info.pop(name, None)

    # ── Lookup ───────────────────────────────────────────────────────

    def get(self, name: str) -> Callable:
        """
        Get a tool callable by name.

        Raises:
            autopilot.errors.ToolRegistryError: If tool not found.
        """
        from autopilot.errors import ToolRegistryError

        if name not in self._tools:
            available = list(self._tools.keys())
            raise ToolRegistryError(
                f"Tool '{name}' not found. Available: {available}",
                tool_name=name,
            )
        return self._tools[name]

    def get_info(self, name: str) -> ToolInfo:
        """Get ToolInfo by name. Raises ToolRegistryError if not found."""
        from autopilot.errors import ToolRegistryError

        if name not in self._info:
            raise ToolRegistryError(
                f"Tool '{name}' not found.",
                tool_name=name,
            )
        return self._info[name]

    def list_all(self) -> list[ToolInfo]:
        """List all registered tools with metadata."""
        return list(self._info.values())

    def by_tag(self, tag: str) -> list[ToolInfo]:
        """Filter tools that contain the given tag."""
        return [info for info in self._info.values() if tag in info.tags]

    # ── ADK Conversion ───────────────────────────────────────────────

    def to_adk_tools(
        self,
        *,
        names: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list:
        """
        Convert registered tools to ADK ``FunctionTool`` instances.

        Tools that accept a ``tool_context: ToolContext`` parameter are
        automatically detected. ADK injects ``ToolContext`` at runtime
        when it sees the parameter in the function signature.

        For long-running tools (tagged with 'long_running'), uses
        ``LongRunningFunctionTool`` instead of ``FunctionTool``.

        Args:
            names: If provided, only include these tool names.
            tags: If provided, only include tools with at least one matching tag.

        Returns:
            List of ADK-compatible tools ready for ``LlmAgent(tools=...)``.
        """
        from google.adk.tools import FunctionTool

        selected = self._select(names=names, tags=tags)
        adk_tools = []
        for name in selected:
            func = self._tools[name]
            info = self._info[name]

            # Use LongRunningFunctionTool for long-running operations
            if "long_running" in info.tags:
                try:
                    from google.adk.tools import LongRunningFunctionTool

                    adk_tools.append(LongRunningFunctionTool(func=func))
                except ImportError:
                    # Fall back to regular FunctionTool
                    adk_tools.append(FunctionTool(func=func))
            else:
                # FunctionTool — ADK auto-injects ToolContext if the
                # function signature includes tool_context: ToolContext
                adk_tools.append(FunctionTool(func=func))

        logger.info(
            "tools_converted_to_adk",
            count=len(adk_tools),
            names=selected,
            context_aware=[n for n in selected if self._info[n].requires_context],
        )
        return adk_tools

    # ── Internal ─────────────────────────────────────────────────────

    def _select(
        self,
        *,
        names: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[str]:
        """Return tool names matching filters. No filter = all tools.

        For ``names`` lookups, applies **lazy connector resolution**: if a
        requested name follows the ``connector.method`` pattern and is not
        yet registered, the registry auto-discovers the connector and
        exposes its tools before retrying.
        """
        if names is not None:
            resolved: list[str] = []
            for n in names:
                if n in self._tools:
                    resolved.append(n)
                elif "." in n:
                    # Lazy connector tool resolution
                    self._try_auto_register_connector(n)
                    if n in self._tools:
                        resolved.append(n)
                    else:
                        logger.warning("tool_not_found_after_auto_register", name=n)
                else:
                    logger.warning("tool_not_found", name=n)
            return resolved

        if tags is not None:
            tag_set = set(tags)
            return [
                name for name, info in self._info.items() if tag_set & set(info.tags)
            ]

        return list(self._tools.keys())

    def _try_auto_register_connector(self, tool_name: str) -> None:
        """Lazy-register connector tools when first referenced.

        Detects the ``connector.method`` naming convention, fetches the
        connector from the ``ConnectorRegistry``, and calls
        ``expose_connector_tools()`` to register all its methods.

        This is idempotent — if tools for the connector are already
        registered (detected via ``connector:name`` tag), it's a no-op.
        """
        connector_name = tool_name.split(".", 1)[0]

        # Skip if already registered (idempotent)
        if self.by_tag(f"connector:{connector_name}"):
            return

        try:
            from autopilot.connectors import get_connector_registry
            from autopilot.core.tools.connector_bridge import expose_connector_tools

            connector = get_connector_registry().get(connector_name)
            expose_connector_tools(connector)
            logger.info(
                "connector_tools_auto_registered",
                connector=connector_name,
                trigger=tool_name,
            )
        except KeyError:
            logger.debug(
                "connector_auto_register_skipped",
                connector=connector_name,
                reason=f"Connector '{connector_name}' not in registry",
            )
        except Exception as e:
            logger.warning(
                "connector_auto_register_failed",
                connector=connector_name,
                error=str(e),
            )

    @property
    def names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self._tools)}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Process-global singleton accessor for the ToolRegistry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_tool_registry() -> None:
    """Reset the global registry. For testing only."""
    global _registry
    _registry = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @tool decorator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """
    Decorator that registers a function as a platform tool.

    Can be used bare or with arguments::

        @tool
        def my_tool(x: int) -> dict:
            \"\"\"Does something.\"\"\"
            ...

        @tool(tags=["finance"], name="custom_name")
        async def my_async_tool(budget_id: str) -> dict:
            ...

    The decorated function is unchanged — it can still be called normally.
    Registration happens at import time into the global ToolRegistry.
    """

    def decorator(fn: Callable) -> Callable:
        registry = get_tool_registry()
        registry.register(
            fn, name=name, description=description, tags=tags, source="decorator"
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        # Preserve async nature
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)

            async_wrapper._tool_info = registry.get_info(name or fn.__name__)
            return async_wrapper

        wrapper._tool_info = registry.get_info(name or fn.__name__)
        return wrapper

    if func is not None:
        # Bare @tool usage
        return decorator(func)

    # @tool(...) usage
    return decorator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_docstring(func: Callable) -> str:
    """Extract the first line of a function's docstring, or empty string."""
    doc = inspect.getdoc(func)
    if not doc:
        return ""
    return doc.strip().split("\n")[0]
