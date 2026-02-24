"""
Tool Ecosystem — V3 Phase 4+: Centralized tool registry, bridges, and lifecycle.

Provides:
  - ToolRegistry: Global registry of reusable tools for LLM agents
  - @tool: Decorator for auto-registering functions as platform tools
  - ToolInfo: Metadata model for registered tools
  - expose_connector_tools: Bridge that converts Connector methods into tools
  - MCPBridge: Wraps ADK McpToolset for external MCP server integration
  - ToolCallbackManager: Before/after lifecycle hooks for tool invocations
  - ToolAuthConfig/ToolAuthManager: Credential management for tools
  - LongRunningTool: Wrapper for async/long-running operations
"""

from autopilot.core.tools.registry import (
    ToolInfo,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
    tool,
)
from autopilot.core.tools.connector_bridge import (
    expose_connector_tools,
    register_all_connector_tools,
)
from autopilot.core.tools.mcp import MCPBridge, MCPRegistry, get_mcp_registry
from autopilot.core.tools.callbacks import (
    ToolCallbackManager,
    get_callback_manager,
    reset_callback_manager,
    audit_log_callback,
    create_rate_limit_callback,
    auth_check_callback,
)
from autopilot.core.tools.auth import (
    ToolAuthConfig,
    AuthCredential,
    ToolAuthManager,
    get_auth_manager,
    reset_auth_manager,
)
from autopilot.core.tools.long_running import (
    LongRunningTool,
    OperationStatus,
    OperationTracker,
    get_operation_tracker,
    reset_operation_tracker,
    long_running_tool,
)

from autopilot.core.tools.search import search_web

__all__ = [
    # Registry
    "ToolInfo",
    "ToolRegistry",
    "get_tool_registry",
    "reset_tool_registry",
    "tool",
    # Connector Bridge
    "expose_connector_tools",
    "register_all_connector_tools",
    # MCP
    "MCPBridge",
    "MCPRegistry",
    "get_mcp_registry",
    # Callbacks (V3 Phase 7)
    "ToolCallbackManager",
    "get_callback_manager",
    "reset_callback_manager",
    "audit_log_callback",
    "create_rate_limit_callback",
    "auth_check_callback",
    # Auth (V3 Phase 7)
    "ToolAuthConfig",
    "AuthCredential",
    "ToolAuthManager",
    "get_auth_manager",
    "reset_auth_manager",
    # Long-Running (V3 Phase 7)
    "LongRunningTool",
    "OperationStatus",
    "OperationTracker",
    "get_operation_tracker",
    "reset_operation_tracker",
    "long_running_tool",
    "search_web",
]
