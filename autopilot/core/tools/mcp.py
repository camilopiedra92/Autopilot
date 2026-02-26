"""
MCP Bridge — Platform wrapper for ADK's McpToolset.

Provides a registry for managing multiple MCP server connections so that
agents can discover and use tools from external MCP servers (e.g.
``brave-search``, ``notion``, ``home-assistant``) through a uniform
platform interface.

Supports two connection modes:

- **Stdio** — Local MCP servers spawned as child processes (e.g. npx).
- **SSE** — Remote MCP servers over HTTP Server-Sent Events (e.g. Home Assistant).

The heavy lifting is done by ADK's native ``McpToolset``.  This module
simply provides lifecycle management and a platform-level registry.

Usage::

    from autopilot.core.tools.mcp import MCPBridge, MCPRegistry

    # Stdio-based MCP server (e.g. Brave Search)
    stdio_bridge = MCPBridge(
        server_name="brave_search",
        command="npx",
        args=["-y", "@anthropic/mcp-brave-search"],
        env={"BRAVE_API_KEY": "..."},
    )

    # SSE-based MCP server (e.g. Home Assistant)
    ha_bridge = MCPBridge(
        server_name="homeassistant",
        url="http://192.168.1.100:8123/mcp/sse",
        headers={"Authorization": "Bearer YOUR_HASS_TOKEN"},
    )

    mcp_registry = MCPRegistry()
    mcp_registry.register(stdio_bridge)
    mcp_registry.register(ha_bridge)

    # Get tools for an agent
    toolsets = mcp_registry.get_all_toolsets()
    agent = LlmAgent(tools=toolsets)
"""

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

    Supports three connection modes:

    **Stdio mode** (local subprocess)::

        MCPBridge(
            server_name="brave_search",
            command="npx",
            args=["-y", "@brave/mcp-server"],
        )

    **SSE mode** (remote HTTP SSE stream)::

        MCPBridge(
            server_name="github",
            url="https://mcp.example.com/sse",
            transport="sse",
        )

    **Streamable HTTP mode** (remote HTTP — default for url)::

        MCPBridge(
            server_name="homeassistant",
            url="https://ha.example.com/api/mcp",
            headers={"Authorization": "Bearer TOKEN"},
        )

    Attributes:
        server_name: Human-readable identifier (e.g. "brave_search").
        command: Shell command to start the MCP server (Stdio mode).
        args: Arguments for the command (Stdio mode).
        env: Environment variables to pass to the server process (Stdio mode).
        url: Endpoint URL for remote MCP servers (SSE or HTTP mode).
        headers: HTTP headers for authentication (SSE/HTTP mode).
        transport: URL transport type: ``"http"`` (Streamable HTTP, default)
            or ``"sse"`` (Server-Sent Events).
        tool_filter: Optional list of tool names to expose. If set, only
            these tools are available to the agent (ADK best practice).
        tool_name_prefix: Optional prefix for tool names to avoid conflicts
            when multiple MCP servers are registered (ADK best practice).
        timeout: Connection timeout in seconds.
    """

    server_name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    transport: str = "http"  # "http" (Streamable HTTP) or "sse"
    tool_filter: list[str] | None = None
    tool_name_prefix: str = ""
    timeout: int = 30

    @property
    def mode(self) -> str:
        """Return the connection mode: 'http', 'sse', or 'stdio'."""
        if self.url:
            return self.transport
        return "stdio"

    def get_toolset(self) -> Any:
        """
        Create and return an ADK ``McpToolset`` configured for this server.

        Automatically selects the connection mode:
        - If ``url`` is set and ``transport="http"`` → ``StreamableHTTPConnectionParams``
        - If ``url`` is set and ``transport="sse"`` → ``SseConnectionParams``
        - If ``command`` is set → ``StdioConnectionParams``

        Returns:
            A ``google.adk.tools.mcp_tool.McpToolset`` instance.

        Raises:
            autopilot.errors.MCPBridgeError: If configuration is invalid or
                ADK MCP dependencies are unavailable.
        """
        from autopilot.errors import MCPBridgeError

        if self.url and self.command:
            raise MCPBridgeError(
                f"MCPBridge '{self.server_name}' has both 'url' and 'command' set. "
                "Use 'url' for remote mode OR 'command' for Stdio mode, not both.",
                server_name=self.server_name,
            )
        if not self.url and not self.command:
            raise MCPBridgeError(
                f"MCPBridge '{self.server_name}' has neither 'url' nor 'command' set. "
                "Provide 'url' for remote mode or 'command' for Stdio mode.",
                server_name=self.server_name,
            )

        if self.url:
            if self.transport == "sse":
                return self._create_sse_toolset()
            return self._create_http_toolset()
        return self._create_stdio_toolset()

    def _create_http_toolset(self) -> Any:
        """Create an McpToolset using Streamable HTTP connection."""
        from autopilot.errors import MCPBridgeError

        try:
            from google.adk.tools.mcp_tool import McpToolset
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                StreamableHTTPConnectionParams,
            )
        except ImportError as e:
            raise MCPBridgeError(
                "MCP dependencies not installed. Install with: "
                "pip install 'google-adk[mcp]' mcp",
                server_name=self.server_name,
            ) from e

        connection_params = StreamableHTTPConnectionParams(
            url=self.url,
            headers=self.headers if self.headers else None,
            timeout=self.timeout,
        )

        kwargs: dict[str, Any] = {"connection_params": connection_params}
        if self.tool_filter:
            kwargs["tool_filter"] = self.tool_filter
        if self.tool_name_prefix:
            kwargs["tool_name_prefix"] = self.tool_name_prefix

        toolset = McpToolset(**kwargs)

        logger.info(
            "mcp_http_toolset_created",
            server=self.server_name,
            url=self.url,
            tool_filter=self.tool_filter,
            tool_name_prefix=self.tool_name_prefix or None,
        )
        return toolset

    def _create_sse_toolset(self) -> Any:
        """Create an McpToolset using SSE connection (remote HTTP server)."""
        from autopilot.errors import MCPBridgeError

        try:
            from google.adk.tools.mcp_tool import McpToolset
            from google.adk.tools.mcp_tool.mcp_session_manager import (
                SseConnectionParams,
            )
        except ImportError as e:
            raise MCPBridgeError(
                "MCP dependencies not installed. Install with: "
                "pip install 'google-adk[mcp]' mcp",
                server_name=self.server_name,
            ) from e

        connection_params = SseConnectionParams(
            url=self.url,
            headers=self.headers if self.headers else None,
            timeout=self.timeout,
        )

        kwargs: dict[str, Any] = {"connection_params": connection_params}
        if self.tool_filter:
            kwargs["tool_filter"] = self.tool_filter
        if self.tool_name_prefix:
            kwargs["tool_name_prefix"] = self.tool_name_prefix

        toolset = McpToolset(**kwargs)

        logger.info(
            "mcp_sse_toolset_created",
            server=self.server_name,
            url=self.url,
            tool_filter=self.tool_filter,
            tool_name_prefix=self.tool_name_prefix or None,
        )
        return toolset

    def _create_stdio_toolset(self) -> Any:
        """Create an McpToolset using Stdio connection (local subprocess)."""
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

        kwargs: dict[str, Any] = {"connection_params": connection_params}
        if self.tool_filter:
            kwargs["tool_filter"] = self.tool_filter
        if self.tool_name_prefix:
            kwargs["tool_name_prefix"] = self.tool_name_prefix

        toolset = McpToolset(**kwargs)

        logger.info(
            "mcp_stdio_toolset_created",
            server=self.server_name,
            command=self.command,
            args=self.args,
            tool_filter=self.tool_filter,
            tool_name_prefix=self.tool_name_prefix or None,
        )
        return toolset

    def __repr__(self) -> str:
        if self.url:
            return f"<MCPBridge server={self.server_name!r} url={self.url!r}>"
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton + Auto-Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_registry: MCPRegistry | None = None


def get_mcp_registry() -> MCPRegistry:
    """Process-global singleton accessor for the MCPRegistry.

    On first call, auto-discovers MCP servers from environment variables
    (same pattern as ``get_connector_registry``).
    """
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
        _auto_register_mcp_servers(_registry)
    return _registry


def reset_mcp_registry() -> None:
    """Reset the global registry. For testing only."""
    global _registry
    _registry = None


def _auto_register_mcp_servers(registry: MCPRegistry) -> None:
    """Discover MCP servers from environment variables and register them.

    Currently supported:
      - ``HASS_URL`` + ``HASS_TOKEN`` → Home Assistant MCP via Streamable HTTP.
      - Optional ``CF_ACCESS_CLIENT_ID`` + ``CF_ACCESS_CLIENT_SECRET`` for
        Cloudflare Tunnel authentication.

    New MCP servers can be added here as env-var pairs are defined.
    """
    import os

    # ── Home Assistant ────────────────────────────────────────────
    hass_url = os.environ.get("HASS_URL", "")
    hass_token = os.environ.get("HASS_TOKEN", "")

    if hass_url and hass_token:
        headers: dict[str, str] = {"Authorization": f"Bearer {hass_token}"}

        # Cloudflare Tunnel — Service Token auth (optional)
        cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
        cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
        if cf_client_id and cf_client_secret:
            headers["CF-Access-Client-Id"] = cf_client_id
            headers["CF-Access-Client-Secret"] = cf_client_secret

        registry.register(
            MCPBridge(
                server_name="homeassistant",
                url=f"{hass_url.rstrip('/')}/api/mcp",
                headers=headers,
                tool_name_prefix="ha_",
            )
        )
        logger.info(
            "mcp_auto_registered",
            server="homeassistant",
            url=hass_url,
            cf_tunnel=bool(cf_client_id),
        )
    else:
        logger.debug(
            "mcp_auto_register_skipped",
            server="homeassistant",
            reason="HASS_URL or HASS_TOKEN not set",
        )
