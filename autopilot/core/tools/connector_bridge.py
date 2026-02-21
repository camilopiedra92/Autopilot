"""
Connector-as-Tool Bridge — Auto-expose connector methods as platform tools.

Introspects ``BaseConnector`` subclasses (specifically their ``.client``
attribute) and registers each public async method as a tool in the
``ToolRegistry``.  This allows any LLM agent to "see" and invoke
connector operations (e.g. ``ynab.create_transaction``,
``gmail.get_unread_emails``) without manual wiring.

Usage::

    from autopilot.connectors import get_connector_registry
    from autopilot.core.tools import expose_connector_tools, get_tool_registry

    ynab = get_connector_registry().get("ynab")
    expose_connector_tools(ynab)

    # Now the tool registry has ynab.get_accounts, ynab.create_transaction, etc.
    tools = get_tool_registry().to_adk_tools(tags=["connector:ynab"])
"""

import inspect
import structlog
from typing import Any, Sequence

from autopilot.connectors.base_connector import BaseConnector
from autopilot.core.tools.registry import get_tool_registry

logger = structlog.get_logger(__name__)

# Methods that take ToolContext as a parameter name
_TOOL_CONTEXT_PARAM = "tool_context"

# Lifecycle methods that should never be exposed as tools
_EXCLUDED_METHODS = frozenset(
    {
        "setup",
        "teardown",
        "health_check",
        "get_info",
        "close",
        # Dunder and private methods are filtered by the leading-underscore check
    }
)


def expose_connector_tools(
    connector: BaseConnector,
    *,
    methods: Sequence[str] | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    """
    Register public async methods from a connector's client as platform tools.

    Introspects the connector's ``.client`` attribute (if present) or the
    connector itself.  For each qualifying method, creates a namespaced tool
    entry: ``<connector_name>.<method_name>``.

    Args:
        connector: A ``BaseConnector`` instance.
        methods: If provided, only expose these method names. Otherwise all
                 qualifying methods are exposed.
        tags: Additional tags to apply to all tools from this connector.

    Returns:
        List of tool names that were registered.
    """
    registry = get_tool_registry()
    connector_name = connector.name

    # Determine the target object to introspect
    target = getattr(connector, "client", connector)
    base_tags = [f"connector:{connector_name}"] + (tags or [])

    registered: list[str] = []

    for attr_name in dir(target):
        # Skip private/dunder methods
        if attr_name.startswith("_"):
            continue

        # Skip lifecycle hooks
        if attr_name in _EXCLUDED_METHODS:
            continue

        # Filter to specific methods if requested
        if methods and attr_name not in methods:
            continue

        member = getattr(target, attr_name, None)

        # Only expose callable methods (prefer async, accept sync)
        if not callable(member) or not inspect.ismethod(member):
            continue

        tool_name = f"{connector_name}.{attr_name}"

        # Extract description from docstring
        doc = inspect.getdoc(member) or ""
        description = (
            doc.strip().split("\n")[0] if doc else f"{connector_name} {attr_name}"
        )

        # Detect if the method accepts ToolContext
        sig = inspect.signature(member)
        _has_tool_context = _TOOL_CONTEXT_PARAM in sig.parameters

        try:
            # Create a closure that captures the bound method
            bound_method = member

            if inspect.iscoroutinefunction(member):

                async def _tool_wrapper(*args, _fn=bound_method, **kwargs):
                    return await _fn(*args, **kwargs)
            else:

                def _tool_wrapper(*args, _fn=bound_method, **kwargs):
                    return _fn(*args, **kwargs)

            # Preserve original metadata for ADK FunctionTool introspection
            _tool_wrapper.__name__ = tool_name.replace(".", "_")
            _tool_wrapper.__qualname__ = tool_name
            _tool_wrapper.__doc__ = doc
            _tool_wrapper.__wrapped__ = member

            # Copy signature AND annotations from the original method.
            # ADK's declaration builder uses these to resolve parameter types.
            _tool_wrapper.__signature__ = inspect.signature(member)
            try:
                _tool_wrapper.__annotations__ = inspect.get_annotations(
                    member, eval_str=True
                )
            except Exception:
                _tool_wrapper.__annotations__ = getattr(member, "__annotations__", {})

            # Set __module__ so ADK resolves types from the correct namespace
            _tool_wrapper.__module__ = getattr(member, "__module__", __name__)

            registry.register(
                _tool_wrapper,
                name=tool_name,
                description=description,
                tags=base_tags,
                source="connector",
            )
            registered.append(tool_name)

        except Exception as e:
            logger.warning(
                "connector_tool_registration_failed",
                connector=connector_name,
                method=attr_name,
                error=str(e),
            )

    logger.info(
        "connector_tools_exposed",
        connector=connector_name,
        tools=registered,
        count=len(registered),
    )
    return registered


def register_all_connector_tools(
    connector_registry: Any | None = None,
    *,
    tags: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Register tools from ALL connectors in the ConnectorRegistry.

    Args:
        connector_registry: Optional ``ConnectorRegistry``. If not provided,
                            uses the global singleton.
        tags: Additional tags applied to all connector tools.

    Returns:
        Dict mapping connector name → list of registered tool names.
    """
    if connector_registry is None:
        from autopilot.connectors import get_connector_registry

        connector_registry = get_connector_registry()

    result: dict[str, list[str]] = {}

    for connector_info in connector_registry.list_all():
        connector = connector_registry.get(connector_info.name)
        try:
            registered = expose_connector_tools(connector, tags=tags)
            result[connector_info.name] = registered
        except Exception as e:
            logger.warning(
                "connector_tool_registration_skipped",
                connector=connector_info.name,
                error=str(e),
                reason="Client initialization or introspection failed",
            )
            result[connector_info.name] = []

    logger.info(
        "all_connector_tools_registered",
        connectors=list(result.keys()),
        total_tools=sum(len(v) for v in result.values()),
    )
    return result
