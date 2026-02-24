"""
Tests for MCPBridge connection modes.

Validates the tri-mode MCPBridge:
  - Streamable HTTP mode (url + transport="http") — default for URL-based
  - SSE mode (url + transport="sse") for legacy SSE MCP servers
  - Stdio mode (command → StdioConnectionParams) for local subprocess
  - Validation rejects ambiguous or empty configurations
  - tool_filter / tool_name_prefix are passed to McpToolset
"""

import pytest
from unittest.mock import MagicMock, patch

from autopilot.core.tools.mcp import MCPBridge, MCPRegistry
from autopilot.errors import MCPBridgeError


# ── Mode Detection ───────────────────────────────────────────────────


def test_mode_http_when_url_set():
    """Default transport for URL-based bridges is 'http' (Streamable HTTP)."""
    bridge = MCPBridge(server_name="ha", url="https://ha.example.com/api/mcp")
    assert bridge.mode == "http"


def test_mode_sse_when_url_and_transport_sse():
    bridge = MCPBridge(
        server_name="github", url="https://mcp.example.com/sse", transport="sse"
    )
    assert bridge.mode == "sse"


def test_mode_stdio_when_command_set():
    bridge = MCPBridge(server_name="brave", command="npx")
    assert bridge.mode == "stdio"


def test_mode_stdio_when_nothing_set():
    """Default mode is stdio when neither url nor command are set."""
    bridge = MCPBridge(server_name="empty")
    assert bridge.mode == "stdio"


# ── Validation ───────────────────────────────────────────────────────


def test_both_url_and_command_raises():
    bridge = MCPBridge(
        server_name="bad",
        url="https://ha.example.com/api/mcp",
        command="npx",
    )
    with pytest.raises(MCPBridgeError, match="both 'url' and 'command'"):
        bridge.get_toolset()


def test_neither_url_nor_command_raises():
    bridge = MCPBridge(server_name="empty")
    with pytest.raises(MCPBridgeError, match="neither 'url' nor 'command'"):
        bridge.get_toolset()


# ── Streamable HTTP Toolset Creation (default for URL) ───────────────


@patch("autopilot.core.tools.mcp.MCPBridge._create_http_toolset")
def test_get_toolset_dispatches_to_http_by_default(mock_http):
    """URL-based bridges dispatch to Streamable HTTP by default."""
    mock_http.return_value = MagicMock()
    bridge = MCPBridge(
        server_name="homeassistant",
        url="https://ha.example.com/api/mcp",
        headers={"Authorization": "Bearer test-token"},
    )

    result = bridge.get_toolset()

    mock_http.assert_called_once()
    assert result is mock_http.return_value


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StreamableHTTPConnectionParams",
)
def test_http_toolset_passes_url_and_headers(mock_http_params, mock_toolset):
    bridge = MCPBridge(
        server_name="homeassistant",
        url="https://ha.example.com/api/mcp",
        headers={"Authorization": "Bearer my-token"},
        timeout=60,
    )

    bridge.get_toolset()

    mock_http_params.assert_called_once_with(
        url="https://ha.example.com/api/mcp",
        headers={"Authorization": "Bearer my-token"},
        timeout=60,
    )
    mock_toolset.assert_called_once_with(
        connection_params=mock_http_params.return_value
    )


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StreamableHTTPConnectionParams",
)
def test_http_toolset_no_headers_passes_none(mock_http_params, mock_toolset):
    bridge = MCPBridge(
        server_name="test_http",
        url="https://api.example.com/mcp",
    )

    bridge.get_toolset()

    mock_http_params.assert_called_once_with(
        url="https://api.example.com/mcp",
        headers=None,
        timeout=30,
    )


# ── SSE Toolset Creation ─────────────────────────────────────────────


@patch("autopilot.core.tools.mcp.MCPBridge._create_sse_toolset")
def test_get_toolset_dispatches_to_sse_when_transport_sse(mock_sse):
    mock_sse.return_value = MagicMock()
    bridge = MCPBridge(
        server_name="github",
        url="https://mcp.example.com/sse",
        transport="sse",
    )

    result = bridge.get_toolset()

    mock_sse.assert_called_once()
    assert result is mock_sse.return_value


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.SseConnectionParams",
)
def test_sse_toolset_passes_url_and_headers(mock_sse_params, mock_toolset):
    bridge = MCPBridge(
        server_name="legacy_sse",
        url="http://localhost:3000/sse",
        transport="sse",
        headers={"Authorization": "Bearer sse-token"},
        timeout=60,
    )

    bridge.get_toolset()

    mock_sse_params.assert_called_once_with(
        url="http://localhost:3000/sse",
        headers={"Authorization": "Bearer sse-token"},
        timeout=60,
    )


# ── Stdio Toolset Creation (Regression) ──────────────────────────────


@patch("autopilot.core.tools.mcp.MCPBridge._create_stdio_toolset")
def test_get_toolset_dispatches_to_stdio(mock_stdio):
    mock_stdio.return_value = MagicMock()
    bridge = MCPBridge(
        server_name="brave_search",
        command="npx",
        args=["-y", "@brave/mcp-server"],
    )

    result = bridge.get_toolset()

    mock_stdio.assert_called_once()
    assert result is mock_stdio.return_value


@patch("mcp.StdioServerParameters")
@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StdioConnectionParams",
)
def test_stdio_toolset_passes_command_and_args(
    mock_stdio_params, mock_toolset, mock_server_params
):
    bridge = MCPBridge(
        server_name="test_stdio",
        command="npx",
        args=["-y", "some-server"],
        env={"API_KEY": "secret"},
        timeout=45,
    )

    bridge.get_toolset()

    mock_server_params.assert_called_once_with(
        command="npx",
        args=["-y", "some-server"],
        env={"API_KEY": "secret"},
    )
    mock_stdio_params.assert_called_once_with(
        server_params=mock_server_params.return_value,
        timeout=45,
    )


# ── __repr__ ─────────────────────────────────────────────────────────


def test_repr_url_mode():
    bridge = MCPBridge(server_name="ha", url="https://ha.example.com/api/mcp")
    assert "url=" in repr(bridge)
    assert "ha" in repr(bridge)


def test_repr_stdio_mode():
    bridge = MCPBridge(server_name="brave", command="npx")
    assert "command=" in repr(bridge)
    assert "brave" in repr(bridge)


# ── tool_filter & tool_name_prefix (ADK Best Practices) ─────────────


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StreamableHTTPConnectionParams",
)
def test_http_tool_filter_passed_to_toolset(mock_http_params, mock_toolset):
    bridge = MCPBridge(
        server_name="ha",
        url="https://ha.example.com/api/mcp",
        tool_filter=["entity_action", "call_service"],
    )

    bridge.get_toolset()

    mock_toolset.assert_called_once_with(
        connection_params=mock_http_params.return_value,
        tool_filter=["entity_action", "call_service"],
    )


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StreamableHTTPConnectionParams",
)
def test_http_tool_name_prefix_passed_to_toolset(mock_http_params, mock_toolset):
    bridge = MCPBridge(
        server_name="ha",
        url="https://ha.example.com/api/mcp",
        tool_name_prefix="ha_",
    )

    bridge.get_toolset()

    mock_toolset.assert_called_once_with(
        connection_params=mock_http_params.return_value,
        tool_name_prefix="ha_",
    )


@patch("mcp.StdioServerParameters")
@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StdioConnectionParams",
)
def test_stdio_tool_filter_and_prefix(mock_stdio_params, mock_toolset, _):
    bridge = MCPBridge(
        server_name="test",
        command="npx",
        tool_filter=["read_file"],
        tool_name_prefix="fs_",
    )

    bridge.get_toolset()

    mock_toolset.assert_called_once_with(
        connection_params=mock_stdio_params.return_value,
        tool_filter=["read_file"],
        tool_name_prefix="fs_",
    )


@patch("google.adk.tools.mcp_tool.McpToolset")
@patch(
    "google.adk.tools.mcp_tool.mcp_session_manager.StreamableHTTPConnectionParams",
)
def test_no_filter_or_prefix_not_passed(mock_http_params, mock_toolset):
    """When tool_filter/tool_name_prefix are empty, they must not be passed."""
    bridge = MCPBridge(server_name="ha", url="https://ha.example.com/api/mcp")

    bridge.get_toolset()

    mock_toolset.assert_called_once_with(
        connection_params=mock_http_params.return_value,
    )


# ── MCPRegistry ──────────────────────────────────────────────────────


def test_registry_with_mixed_bridges():
    """Registry works with all three connection modes."""
    registry = MCPRegistry()

    stdio = MCPBridge(server_name="brave", command="npx")
    http = MCPBridge(server_name="ha", url="https://ha.example.com/api/mcp")
    sse = MCPBridge(
        server_name="legacy", url="http://localhost:3000/sse", transport="sse"
    )

    registry.register(stdio)
    registry.register(http)
    registry.register(sse)

    assert len(registry) == 3
    assert registry.get("brave").mode == "stdio"
    assert registry.get("ha").mode == "http"
    assert registry.get("legacy").mode == "sse"


def test_registry_list_servers():
    registry = MCPRegistry()
    registry.register(MCPBridge(server_name="a", command="npx"))
    registry.register(MCPBridge(server_name="b", url="https://example.com/mcp"))

    assert set(registry.list_servers()) == {"a", "b"}
