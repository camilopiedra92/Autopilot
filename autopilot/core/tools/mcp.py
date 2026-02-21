"""
MCP Bridge — Platform wrapper for ADK's McpToolset.

Provides a registry for managing multiple MCP server connections so that
agents can discover and use tools from external MCP servers (e.g.
``brave-search``, ``notion``, ``pinecone``) through a uniform platform
interface.

The heavy lifting is done by ADK's native ``McpToolset``.  This module
simply provides lifecycle management and a platform-level registry.

Usage::

    from autopilot.core.tools.mcp import MCPBridge, MCPRegistry

    # Register an MCP server
    bridge = MCPBridge(
        server_name="brave_search",
        command="npx",
        args=["-y", "@anthropic/mcp-brave-search"],
        env={"BRAVE_API_KEY": "..."},
    )

    mcp_registry = MCPRegistry()
    mcp_registry.register(bridge)

    # Get tools for an agent
    toolsets = mcp_registry.get_all_toolsets()
    agent = LlmAgent(tools=toolsets)
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import Any

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MCPBridge — Single MCP server connection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MCPBridge:
    """
    Configuration and factory for connecting to an external MCP server.

    Wraps the ADK ``McpToolset`` + ``StdioConnectionParams`` into a
    platform-native dataclass.  Call ``get_toolset()`` to obtain an
    ADK-compatible toolset ready for injection into an ``LlmAgent``.

    Attributes:
        server_name: Human-readable identifier (e.g. "brave_search").
        command: Shell command to start the MCP server (e.g. "npx").
        args: Arguments for the command (e.g. ["-y", "@brave/mcp-server"]).
        env: Environment variables to pass to the server process.
        timeout: Connection timeout in seconds.
    """

    server_name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    def get_toolset(self) -> Any:
        """
        Create and return an ADK ``McpToolset`` configured for this server.

        Returns:
            A ``google.adk.tools.mcp_tool.McpToolset`` instance.

        Raises:
            autopilot.errors.MCPBridgeError: If ADK MCP dependencies are unavailable.
        """
        from autopilot.errors import MCPBridgeError

        try:
            from google.adk.tools.mcp_tool import McpToolset
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                StdioConnectionParams,
            )
            from mcp import StdioServerParameters
        except ImportError as e:
            raise MCPBridgeError(
                "MCP dependencies not installed. Install with: "
                "pip install 'google-adk[mcp]' mcp",
                server_name=self.server_name,
            ) from e

        connection_params = StdioConnectionParams(
            server_params=StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env if self.env else None,
            ),
            timeout=self.timeout,
        )

        toolset = McpToolset(connection_params=connection_params)

        logger.info(
            "mcp_toolset_created",
            server=self.server_name,
            command=self.command,
            args=self.args,
        )
        return toolset

    def __repr__(self) -> str:
        return f"<MCPBridge server={self.server_name!r} command={self.command!r}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MCPRegistry — Multi-server management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MCPRegistry:
    """
    Registry for managing multiple MCP server connections.

    Provides a central place to configure all external MCP servers
    the platform can connect to, and batch-retrieve their toolsets.
    """

    def __init__(self) -> None:
        self._bridges: dict[str, MCPBridge] = {}

    def register(self, bridge: MCPBridge) -> None:
        """Register an MCP bridge by its server name."""
        self._bridges[bridge.server_name] = bridge
        logger.info("mcp_bridge_registered", server=bridge.server_name)

    def get(self, name: str) -> MCPBridge:
        """
        Get an MCP bridge by server name.

        Raises:
            autopilot.errors.MCPBridgeError: If server not found.
        """
        from autopilot.errors import MCPBridgeError

        if name not in self._bridges:
            available = list(self._bridges.keys())
            raise MCPBridgeError(
                f"MCP server '{name}' not registered. Available: {available}",
                server_name=name,
            )
        return self._bridges[name]

    def get_toolset(self, name: str) -> Any:
        """Get a ready-to-use ``McpToolset`` for the named server."""
        return self.get(name).get_toolset()

    def get_all_toolsets(self) -> list:
        """Get ``McpToolset`` instances for all registered servers."""
        return [bridge.get_toolset() for bridge in self._bridges.values()]

    def list_servers(self) -> list[str]:
        """List registered server names."""
        return list(self._bridges.keys())

    def __len__(self) -> int:
        return len(self._bridges)

    def __contains__(self, name: str) -> bool:
        return name in self._bridges

    def __repr__(self) -> str:
        return f"<MCPRegistry servers={list(self._bridges.keys())}>"
