"""
Tests for autopilot.core.tools — V3 Phase 4: Tool Ecosystem.

Covers:
  - ToolRegistry: register, get, list, duplicate detection, unregister
  - @tool decorator: auto-registration, metadata extraction, bare and parameterized
  - to_adk_tools(): conversion to ADK FunctionTool
  - by_tag(): tag-based filtering
  - Connector-as-Tool bridge: mock connector → auto-registered tools
  - MCPBridge: construction and get_toolset() behavior
  - MCPRegistry: multi-server management
  - Error types: ToolRegistryError, MCPBridgeError
  - AgentContext integration: ctx.tools property
"""

import pytest
from unittest.mock import patch, MagicMock

from autopilot.core.tools.registry import (
    ToolInfo,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
    tool,
)
from autopilot.core.tools.mcp import MCPBridge, MCPRegistry
from autopilot.errors import ToolRegistryError, MCPBridgeError


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the global tool registry before each test."""
    reset_tool_registry()
    yield
    reset_tool_registry()


@pytest.fixture
def registry():
    """Fresh, isolated ToolRegistry instance."""
    return ToolRegistry()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolInfo Model Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolInfo:
    def test_create_minimal(self):
        info = ToolInfo(name="my_tool")
        assert info.name == "my_tool"
        assert info.description == ""
        assert info.parameters == {}
        assert info.tags == []
        assert info.source == "manual"
        assert info.is_async is False

    def test_create_full(self):
        info = ToolInfo(
            name="ynab_create",
            description="Creates a YNAB transaction",
            parameters={"budget_id": "str", "amount": "float"},
            tags=["ynab", "finance"],
            source="connector",
            is_async=True,
        )
        assert info.name == "ynab_create"
        assert info.tags == ["ynab", "finance"]
        assert info.is_async is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolRegistry Core Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolRegistry:
    def test_register_and_get(self, registry: ToolRegistry):
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        registry.register(greet)
        assert "greet" in registry
        assert len(registry) == 1

        fn = registry.get("greet")
        assert fn("Alice") == "Hello, Alice!"

    def test_register_with_custom_name(self, registry: ToolRegistry):
        def my_func():
            pass

        registry.register(my_func, name="custom_tool")
        assert "custom_tool" in registry
        assert "my_func" not in registry

    def test_register_duplicate_raises(self, registry: ToolRegistry):
        def tool_a():
            pass

        registry.register(tool_a, name="dupe")

        def tool_b():
            pass

        with pytest.raises(ToolRegistryError, match="already registered"):
            registry.register(tool_b, name="dupe")

    def test_get_nonexistent_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolRegistryError, match="not found"):
            registry.get("nonexistent")

    def test_unregister(self, registry: ToolRegistry):
        def temp_tool():
            pass

        registry.register(temp_tool)
        assert "temp_tool" in registry

        registry.unregister("temp_tool")
        assert "temp_tool" not in registry
        assert len(registry) == 0

    def test_unregister_nonexistent_is_noop(self, registry: ToolRegistry):
        registry.unregister("ghost")  # Should not raise

    def test_list_all(self, registry: ToolRegistry):
        def tool_a():
            """Tool A."""
            pass

        def tool_b(x: int) -> int:
            """Tool B."""
            return x

        registry.register(tool_a, tags=["group1"])
        registry.register(tool_b, tags=["group2"])

        infos = registry.list_all()
        assert len(infos) == 2
        names = {i.name for i in infos}
        assert names == {"tool_a", "tool_b"}

    def test_get_info(self, registry: ToolRegistry):
        def my_tool(x: int, y: str) -> dict:
            """My great tool."""
            return {"result": x}

        registry.register(my_tool, tags=["test"])

        info = registry.get_info("my_tool")
        assert info.name == "my_tool"
        assert info.description == "My great tool."
        assert "x" in info.parameters
        assert "y" in info.parameters
        assert info.parameters["x"] == "int"
        assert info.parameters["y"] == "str"
        assert info.tags == ["test"]
        assert info.source == "manual"

    def test_get_info_nonexistent_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolRegistryError, match="not found"):
            registry.get_info("ghost")

    def test_by_tag(self, registry: ToolRegistry):
        def finance_tool():
            pass

        def search_tool():
            pass

        def multi_tool():
            pass

        registry.register(finance_tool, tags=["finance", "ynab"])
        registry.register(search_tool, tags=["search"])
        registry.register(multi_tool, tags=["finance", "search"])

        finance = registry.by_tag("finance")
        assert len(finance) == 2
        assert {i.name for i in finance} == {"finance_tool", "multi_tool"}

        search = registry.by_tag("search")
        assert len(search) == 2

    def test_async_function_metadata(self, registry: ToolRegistry):
        async def async_tool(query: str) -> dict:
            """Search for something."""
            return {"results": []}

        registry.register(async_tool)
        info = registry.get_info("async_tool")
        assert info.is_async is True

    def test_names_property(self, registry: ToolRegistry):
        def a():
            pass

        def b():
            pass

        registry.register(a)
        registry.register(b)
        assert set(registry.names) == {"a", "b"}

    def test_repr(self, registry: ToolRegistry):
        assert "ToolRegistry" in repr(registry)
        assert "0" in repr(registry)

    def test_to_adk_tools(self, registry: ToolRegistry):
        """to_adk_tools() returns FunctionTool instances."""

        def my_tool(x: int) -> str:
            """A tool for testing."""
            return str(x)

        registry.register(my_tool)

        with patch("google.adk.tools.FunctionTool") as MockFT:
            MockFT.side_effect = lambda func: MagicMock(name=f"FT({func.__name__})")

            adk_tools = registry.to_adk_tools()

            assert len(adk_tools) == 1
            MockFT.assert_called_once()

    def test_to_adk_tools_filter_by_names(self, registry: ToolRegistry):
        def a():
            pass

        def b():
            pass

        registry.register(a)
        registry.register(b)

        with patch("google.adk.tools.FunctionTool") as MockFT:
            MockFT.side_effect = lambda func: MagicMock()
            adk_tools = registry.to_adk_tools(names=["a"])
            assert len(adk_tools) == 1

    def test_to_adk_tools_filter_by_tags(self, registry: ToolRegistry):
        def finance():
            pass

        def search():
            pass

        registry.register(finance, tags=["finance"])
        registry.register(search, tags=["search"])

        with patch("google.adk.tools.FunctionTool") as MockFT:
            MockFT.side_effect = lambda func: MagicMock()
            adk_tools = registry.to_adk_tools(tags=["finance"])
            assert len(adk_tools) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  @tool Decorator Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolDecorator:
    def test_bare_decorator(self):
        @tool
        def my_simple_tool(x: int) -> dict:
            """Doubles the input."""
            return {"result": x * 2}

        # Function still works
        assert my_simple_tool(5) == {"result": 10}

        # Registered in global registry
        registry = get_tool_registry()
        assert "my_simple_tool" in registry
        info = registry.get_info("my_simple_tool")
        assert info.description == "Doubles the input."
        assert info.source == "decorator"

    def test_decorator_with_args(self):
        @tool(name="custom_name", tags=["finance"])
        def my_func(amount: float) -> dict:
            """Process amount."""
            return {"amount": amount}

        registry = get_tool_registry()
        assert "custom_name" in registry
        info = registry.get_info("custom_name")
        assert info.tags == ["finance"]

    def test_async_decorator(self):
        @tool(tags=["async"])
        async def async_processor(data: str) -> dict:
            """Process data asynchronously."""
            return {"processed": data}

        registry = get_tool_registry()
        info = registry.get_info("async_processor")
        assert info.is_async is True

    @pytest.mark.asyncio
    async def test_async_function_still_callable(self):
        @tool
        async def async_tool(n: int) -> dict:
            """Async tool."""
            return {"squared": n**2}

        result = await async_tool(4)
        assert result == {"squared": 16}

    def test_has_tool_info_attribute(self):
        @tool
        def annotated_tool(x: int) -> int:
            """Has info."""
            return x

        assert hasattr(annotated_tool, "_tool_info")
        assert annotated_tool._tool_info.name == "annotated_tool"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSingleton:
    def test_get_tool_registry_returns_same_instance(self):
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_tool_registry()
        reset_tool_registry()
        r2 = get_tool_registry()
        assert r1 is not r2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Connector-as-Tool Bridge Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MockClient:
    """Simulates a connector's client with multiple methods."""

    async def get_accounts(self, budget_id: str) -> list:
        """Fetch accounts for a budget."""
        return [{"id": "acc_1", "name": "Checking"}]

    async def create_transaction(self, budget_id: str, amount: float) -> dict:
        """Create a new transaction."""
        return {"id": "tx_1"}

    async def close(self):
        """Close connection — should be excluded."""
        pass

    def _private_method(self):
        """Should be excluded."""
        pass


class MockConnector:
    """Simulates a BaseConnector with a .client attribute."""

    @property
    def name(self) -> str:
        return "mock_service"

    @property
    def icon(self) -> str:
        return "🔧"

    @property
    def description(self) -> str:
        return "Mock connector for tests"

    def __init__(self):
        self._client = MockClient()

    @property
    def client(self):
        return self._client


class TestConnectorBridge:
    def test_expose_connector_tools(self):
        from autopilot.core.tools.connector_bridge import expose_connector_tools

        connector = MockConnector()
        registry = get_tool_registry()

        registered = expose_connector_tools(connector)

        # Should register public async methods except lifecycle hooks
        assert "mock_service.get_accounts" in registered
        assert "mock_service.create_transaction" in registered

        # close and _private_method should be excluded
        assert "mock_service.close" not in registered
        assert "mock_service._private_method" not in registered

        # Verify in registry
        assert "mock_service.get_accounts" in registry
        info = registry.get_info("mock_service.get_accounts")
        assert "connector:mock_service" in info.tags
        assert info.source == "connector"

    def test_expose_specific_methods(self):
        from autopilot.core.tools.connector_bridge import expose_connector_tools

        connector = MockConnector()
        registered = expose_connector_tools(connector, methods=["get_accounts"])

        assert "mock_service.get_accounts" in registered
        assert "mock_service.create_transaction" not in registered

    def test_expose_with_extra_tags(self):
        from autopilot.core.tools.connector_bridge import expose_connector_tools

        connector = MockConnector()
        expose_connector_tools(connector, tags=["finance", "extra"])

        registry = get_tool_registry()
        info = registry.get_info("mock_service.get_accounts")
        assert "connector:mock_service" in info.tags
        assert "finance" in info.tags
        assert "extra" in info.tags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MCP Bridge Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMCPBridge:
    def test_creation(self):
        bridge = MCPBridge(
            server_name="brave_search",
            command="npx",
            args=["-y", "@brave/mcp-search"],
            env={"API_KEY": "test"},
            timeout=15,
        )
        assert bridge.server_name == "brave_search"
        assert bridge.command == "npx"
        assert bridge.args == ["-y", "@brave/mcp-search"]
        assert bridge.timeout == 15

    def test_repr(self):
        bridge = MCPBridge(server_name="test", command="npx")
        assert "MCPBridge" in repr(bridge)
        assert "test" in repr(bridge)

    def test_get_toolset_missing_deps_raises(self):
        """Should raise MCPBridgeError if MCP deps are not installed."""
        bridge = MCPBridge(server_name="test", command="npx")

        with patch.dict(
            "sys.modules",
            {
                "google.adk.tools.mcp_tool": None,
            },
        ):
            with pytest.raises(MCPBridgeError, match="MCP dependencies"):
                bridge.get_toolset()


class TestMCPRegistry:
    def test_register_and_get(self):
        mcp_reg = MCPRegistry()
        bridge = MCPBridge(server_name="notion", command="npx")

        mcp_reg.register(bridge)
        assert "notion" in mcp_reg
        assert len(mcp_reg) == 1
        assert mcp_reg.get("notion") is bridge

    def test_get_nonexistent_raises(self):
        mcp_reg = MCPRegistry()
        with pytest.raises(MCPBridgeError, match="not registered"):
            mcp_reg.get("ghost")

    def test_list_servers(self):
        mcp_reg = MCPRegistry()
        mcp_reg.register(MCPBridge(server_name="a", command="npx"))
        mcp_reg.register(MCPBridge(server_name="b", command="npx"))

        assert set(mcp_reg.list_servers()) == {"a", "b"}

    def test_repr(self):
        mcp_reg = MCPRegistry()
        assert "MCPRegistry" in repr(mcp_reg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Error Type Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorTypes:
    def test_tool_registry_error(self):
        err = ToolRegistryError("duplicate", tool_name="my_tool")
        assert err.tool_name == "my_tool"
        assert err.error_code == "TOOL_REGISTRY_ERROR"
        assert err.retryable is False
        assert err.http_status == 422

        d = err.to_dict()
        assert d["tool_name"] == "my_tool"
        assert d["error_code"] == "TOOL_REGISTRY_ERROR"

    def test_mcp_bridge_error(self):
        err = MCPBridgeError("connection failed", server_name="brave")
        assert err.server_name == "brave"
        assert err.error_code == "MCP_BRIDGE_ERROR"
        assert err.retryable is True
        assert err.http_status == 502

        d = err.to_dict()
        assert d["server_name"] == "brave"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AgentContext Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentContextTools:
    def test_context_has_tools_property(self):
        from autopilot.core.context import AgentContext

        ctx = AgentContext(pipeline_name="test")
        assert ctx.tools is not None
        assert isinstance(ctx.tools, ToolRegistry)

    def test_context_tools_returns_singleton(self):
        from autopilot.core.context import AgentContext

        ctx = AgentContext()
        global_reg = get_tool_registry()
        assert ctx.tools is global_reg

    def test_context_tools_access_registered_tool(self):
        from autopilot.core.context import AgentContext

        # Register a tool
        def helper(x: int) -> int:
            return x + 1

        registry = get_tool_registry()
        registry.register(helper)

        ctx = AgentContext()
        fn = ctx.tools.get("helper")
        assert fn(10) == 11
