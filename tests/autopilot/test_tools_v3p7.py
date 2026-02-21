"""
Tests for V3 Phase 7 — Tool Ecosystem ADK Upgrade.

Covers:
  - ToolCallbackManager: before/after hooks, tool filtering, chaining, error handling
  - ToolAuthConfig/ToolAuthManager: credential resolution, request/provide flows
  - LongRunningTool: OperationTracker lifecycle, decorator, ADK conversion
  - ToolContext detection: requires_context in ToolInfo
  - Error types: ToolCallbackError, ToolAuthError
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

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
    OperationTracker,
    LongRunningTool,
    get_operation_tracker,
    reset_operation_tracker,
)
from autopilot.core.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
)
from autopilot.errors import ToolCallbackError, ToolAuthError


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset all tool singletons between tests."""
    reset_callback_manager()
    reset_auth_manager()
    reset_operation_tracker()
    reset_tool_registry()
    yield
    reset_callback_manager()
    reset_auth_manager()
    reset_operation_tracker()
    reset_tool_registry()


@pytest.fixture
def callback_mgr():
    return ToolCallbackManager()


@pytest.fixture
def auth_mgr():
    return ToolAuthManager()


@pytest.fixture
def tracker():
    return OperationTracker()


@pytest.fixture
def registry():
    return ToolRegistry()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolCallbackManager Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolCallbackManager:

    async def test_before_allows_execution(self, callback_mgr: ToolCallbackManager):
        """Before callback returning None allows tool to proceed."""
        async def allow(tool_name, args, context):
            return None

        callback_mgr.register_before(allow)
        result = await callback_mgr.run_before("test_tool", {}, {})
        assert result is None

    async def test_before_blocks_execution(self, callback_mgr: ToolCallbackManager):
        """Before callback returning dict blocks tool execution."""
        async def block(tool_name, args, context):
            return {"error": "blocked", "blocked": True}

        callback_mgr.register_before(block)
        result = await callback_mgr.run_before("test_tool", {}, {})
        assert result == {"error": "blocked", "blocked": True}

    async def test_before_first_block_wins(self, callback_mgr: ToolCallbackManager):
        """First blocking callback short-circuits the chain."""
        call_order = []

        async def first(tool_name, args, context):
            call_order.append("first")
            return {"blocked_by": "first"}

        async def second(tool_name, args, context):
            call_order.append("second")
            return None

        callback_mgr.register_before(first)
        callback_mgr.register_before(second)

        result = await callback_mgr.run_before("test_tool", {}, {})
        assert result == {"blocked_by": "first"}
        assert call_order == ["first"]  # Second never called

    async def test_after_chains_results(self, callback_mgr: ToolCallbackManager):
        """After callbacks receive result from previous callback."""
        async def double(tool_name, args, result, context):
            return result * 2

        async def add_ten(tool_name, args, result, context):
            return result + 10

        callback_mgr.register_after(double)
        callback_mgr.register_after(add_ten)

        result = await callback_mgr.run_after("test_tool", {}, 5, {})
        assert result == 20  # (5 * 2) + 10

    async def test_tool_specific_filtering(self, callback_mgr: ToolCallbackManager):
        """Callbacks with tool filters only run for matching tools."""
        calls = []

        async def finance_only(tool_name, args, context):
            calls.append(tool_name)
            return None

        callback_mgr.register_before(finance_only, tools=["ynab.create"])

        await callback_mgr.run_before("ynab.create", {}, {})
        await callback_mgr.run_before("gmail.send", {}, {})

        assert calls == ["ynab.create"]  # gmail.send was filtered out

    async def test_before_error_does_not_block(self, callback_mgr: ToolCallbackManager):
        """Exception in before callback is swallowed; execution continues."""
        async def broken(tool_name, args, context):
            raise RuntimeError("broken callback")

        callback_mgr.register_before(broken)
        result = await callback_mgr.run_before("test_tool", {}, {})
        assert result is None  # Execution allowed despite error

    async def test_decorator_syntax_before(self, callback_mgr: ToolCallbackManager):
        """@manager.before decorator registers the callback."""
        @callback_mgr.before
        async def my_hook(tool_name, args, context):
            return None

        assert callback_mgr.before_count == 1

    async def test_decorator_syntax_after_with_tools(self, callback_mgr: ToolCallbackManager):
        """@manager.after(tools=[...]) decorator with tool filtering."""
        @callback_mgr.after(tools=["search"])
        async def my_hook(tool_name, args, result, context):
            return result

        assert callback_mgr.after_count == 1

    async def test_clear(self, callback_mgr: ToolCallbackManager):
        """clear() removes all registered callbacks."""
        async def h1(tool_name, args, context):
            return None

        async def h2(tool_name, args, result, context):
            return result

        callback_mgr.register_before(h1)
        callback_mgr.register_after(h2)
        assert callback_mgr.before_count == 1
        assert callback_mgr.after_count == 1

        callback_mgr.clear()
        assert callback_mgr.before_count == 0
        assert callback_mgr.after_count == 0

    async def test_repr(self, callback_mgr: ToolCallbackManager):
        assert "before=0" in repr(callback_mgr)
        assert "after=0" in repr(callback_mgr)

    def test_singleton(self):
        mgr1 = get_callback_manager()
        mgr2 = get_callback_manager()
        assert mgr1 is mgr2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Built-in Callback Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuiltinCallbacks:

    async def test_audit_log_passes_through(self):
        """audit_log_callback returns the result unchanged."""
        result = await audit_log_callback("test", {"a": 1}, {"data": "ok"}, {})
        assert result == {"data": "ok"}

    async def test_rate_limit_allows_under_limit(self):
        limiter = create_rate_limit_callback(max_calls=3, window_seconds=60)
        for _ in range(3):
            result = await limiter("tool", {}, {})
            assert result is None  # Allowed

    async def test_rate_limit_blocks_over_limit(self):
        limiter = create_rate_limit_callback(max_calls=2, window_seconds=60)
        await limiter("tool", {}, {})
        await limiter("tool", {}, {})
        result = await limiter("tool", {}, {})
        assert result is not None
        assert result["blocked"] is True
        assert "Rate limit" in result["error"]

    async def test_rate_limit_per_tool(self):
        """Rate limits are tracked per-tool independently."""
        limiter = create_rate_limit_callback(max_calls=1, window_seconds=60)
        assert await limiter("tool_a", {}, {}) is None
        assert await limiter("tool_b", {}, {}) is None
        # tool_a is now over limit, but tool_b still has room
        assert await limiter("tool_a", {}, {}) is not None
        assert await limiter("tool_b", {}, {}) is not None

    async def test_auth_check_passes_no_requirement(self):
        """No required_auth in context → allow."""
        result = await auth_check_callback("tool", {}, {})
        assert result is None

    async def test_auth_check_passes_with_creds(self):
        """All required credentials present → allow."""
        ctx = {"required_auth": ["api_key"], "api_key": "secret123"}
        result = await auth_check_callback("tool", {}, ctx)
        assert result is None

    async def test_auth_check_blocks_missing_creds(self):
        """Missing required credentials → block."""
        ctx = {"required_auth": ["api_key"]}
        result = await auth_check_callback("tool", {}, ctx)
        assert result is not None
        assert result["blocked"] is True
        assert "api_key" in str(result["error"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolAuthManager Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolAuthManager:

    def test_register_config(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="ynab.create", auth_type="api_key", credential_key="YNAB_TOKEN")
        auth_mgr.register(config)
        assert auth_mgr.get_config("ynab.create") is config

    def test_get_config_not_registered(self, auth_mgr: ToolAuthManager):
        assert auth_mgr.get_config("nonexistent") is None

    def test_resolve_from_env(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="ynab.create", credential_key="TEST_YNAB_KEY")
        auth_mgr.register(config)

        with patch.dict(os.environ, {"TEST_YNAB_KEY": "env-secret"}):
            cred = auth_mgr.get_credential("ynab.create")

        assert cred is not None
        assert cred.token == "env-secret"
        assert cred.metadata.get("source") == "env"

    def test_resolve_from_state(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="search.api", credential_key="SEARCH_KEY")
        auth_mgr.register(config)

        cred = auth_mgr.get_credential("search.api", state={"SEARCH_KEY": "state-secret"})
        assert cred is not None
        assert cred.token == "state-secret"

    def test_state_takes_priority_over_env(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="tool", credential_key="KEY")
        auth_mgr.register(config)

        with patch.dict(os.environ, {"KEY": "from-env"}):
            cred = auth_mgr.get_credential("tool", state={"KEY": "from-state"})

        assert cred is not None
        assert cred.token == "from-state"

    def test_cache_hit(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="tool", credential_key="KEY")
        auth_mgr.register(config)

        # First call resolves from state
        cred1 = auth_mgr.get_credential("tool", state={"KEY": "cached"})
        # Second call should hit cache (even without state)
        cred2 = auth_mgr.get_credential("tool")
        assert cred1 is cred2

    def test_no_credential_available(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="tool", credential_key="MISSING_KEY")
        auth_mgr.register(config)
        cred = auth_mgr.get_credential("tool")
        assert cred is None

    def test_request_credential(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(
            tool_name="tool",
            auth_type="oauth2",
            credential_key="OAUTH_TOKEN",
            scopes=("read", "write"),
        )
        auth_mgr.register(config)

        result = auth_mgr.request_credential("tool")
        assert result["status"] == "auth_required"
        assert result["auth_type"] == "oauth2"
        assert "tool" in auth_mgr.pending_requests

    def test_provide_credential(self, auth_mgr: ToolAuthManager):
        config = ToolAuthConfig(tool_name="tool", credential_key="KEY")
        auth_mgr.register(config)

        auth_mgr.request_credential("tool")
        assert "tool" in auth_mgr.pending_requests

        cred = auth_mgr.provide_credential("tool", "user-provided-token")
        assert cred.token == "user-provided-token"
        assert auth_mgr.has_credential("tool")
        assert "tool" not in auth_mgr.pending_requests

    def test_request_no_config(self, auth_mgr: ToolAuthManager):
        result = auth_mgr.request_credential("unknown")
        assert result["status"] == "error"

    def test_list_configs(self, auth_mgr: ToolAuthManager):
        auth_mgr.register(ToolAuthConfig(tool_name="a"))
        auth_mgr.register(ToolAuthConfig(tool_name="b"))
        assert len(auth_mgr.list_configs()) == 2

    def test_clear(self, auth_mgr: ToolAuthManager):
        auth_mgr.register(ToolAuthConfig(tool_name="a"))
        auth_mgr.request_credential("a")
        auth_mgr.clear()
        assert len(auth_mgr.list_configs()) == 0
        assert len(auth_mgr.pending_requests) == 0

    def test_repr(self, auth_mgr: ToolAuthManager):
        assert "configs=0" in repr(auth_mgr)

    def test_singleton(self):
        mgr1 = get_auth_manager()
        mgr2 = get_auth_manager()
        assert mgr1 is mgr2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AuthCredential Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAuthCredential:

    def test_valid_credential(self):
        cred = AuthCredential(auth_type="api_key", token="secret")
        assert cred.is_valid is True

    def test_invalid_empty_token(self):
        cred = AuthCredential(auth_type="api_key", token="")
        assert cred.is_valid is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OperationTracker Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOperationTracker:

    def test_create_operation(self, tracker: OperationTracker):
        op = tracker.create("batch_tool")
        assert op.tool_name == "batch_tool"
        assert op.status == "pending"
        assert op.operation_id.startswith("op-")
        assert len(tracker) == 1

    def test_create_with_custom_id(self, tracker: OperationTracker):
        op = tracker.create("tool", operation_id="custom-123")
        assert op.operation_id == "custom-123"

    def test_update_operation(self, tracker: OperationTracker):
        op = tracker.create("tool")
        tracker.update(op.operation_id, status="running")
        assert tracker.get_status(op.operation_id).status == "running"

    def test_complete_operation(self, tracker: OperationTracker):
        op = tracker.create("tool")
        tracker.update(op.operation_id, status="completed", result={"data": "done"})
        updated = tracker.get_status(op.operation_id)
        assert updated.is_terminal is True
        assert updated.result == {"data": "done"}

    def test_fail_operation(self, tracker: OperationTracker):
        op = tracker.create("tool")
        tracker.update(op.operation_id, status="failed", error="timeout")
        updated = tracker.get_status(op.operation_id)
        assert updated.is_terminal is True
        assert updated.error == "timeout"

    def test_update_nonexistent_returns_none(self, tracker: OperationTracker):
        assert tracker.update("nonexistent") is None

    def test_list_active(self, tracker: OperationTracker):
        op1 = tracker.create("tool_a")
        op2 = tracker.create("tool_b")
        tracker.update(op1.operation_id, status="completed")

        active = tracker.list_active()
        assert len(active) == 1
        assert active[0].operation_id == op2.operation_id

    def test_list_by_tool(self, tracker: OperationTracker):
        tracker.create("tool_a")
        tracker.create("tool_a")
        tracker.create("tool_b")
        assert len(tracker.list_by_tool("tool_a")) == 2
        assert len(tracker.list_by_tool("tool_b")) == 1

    def test_cleanup(self, tracker: OperationTracker):
        op = tracker.create("tool")
        tracker.update(op.operation_id, status="completed")
        # Force the updated_at to be old enough
        tracker._operations[op.operation_id].updated_at -= 7200
        removed = tracker.cleanup(max_age_seconds=3600)
        assert removed == 1
        assert len(tracker) == 0

    def test_clear(self, tracker: OperationTracker):
        tracker.create("tool")
        tracker.create("tool")
        tracker.clear()
        assert len(tracker) == 0

    def test_repr(self, tracker: OperationTracker):
        assert "total=0" in repr(tracker)

    def test_singleton(self):
        t1 = get_operation_tracker()
        t2 = get_operation_tracker()
        assert t1 is t2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolContext Detection Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToolContextDetection:

    def test_regular_function_no_context(self, registry: ToolRegistry):
        """Regular function should NOT have requires_context."""
        def simple_tool(x: int, y: str) -> dict:
            """Simple tool."""
            return {}

        registry.register(simple_tool)
        info = registry.get_info("simple_tool")
        assert info.requires_context is False
        assert "tool_context" not in info.parameters

    def test_function_with_tool_context_param(self, registry: ToolRegistry):
        """Function with 'tool_context' param should be detected."""
        def context_tool(x: int, tool_context) -> dict:
            """Uses context."""
            return {}

        registry.register(context_tool)
        info = registry.get_info("context_tool")
        assert info.requires_context is True
        # tool_context should NOT appear in user-facing params
        assert "tool_context" not in info.parameters
        assert "x" in info.parameters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LongRunningTool Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLongRunningTool:

    def test_create_tool(self):
        def batch_process(items: list) -> dict:
            """Process items in batch."""
            return {"status": "pending"}

        tool = LongRunningTool(batch_process, tags=["batch"])
        assert tool.name == "batch_process"
        assert "batch" in tool.tags
        assert "Process items in batch" in tool.description

    def test_repr(self):
        def my_tool():
            pass

        tool = LongRunningTool(my_tool)
        assert "my_tool" in repr(tool)

    def test_register_adds_to_registry(self):
        def my_lr_tool(x: int) -> dict:
            """A long-running tool."""
            return {}

        tool = LongRunningTool(my_lr_tool, name="test_lr_tool")
        tool.register()

        registry = get_tool_registry()
        info = registry.get_info("test_lr_tool")
        assert "long_running" in info.tags
        assert info.source == "long_running"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Error Type Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNewErrorTypes:

    def test_tool_callback_error(self):
        err = ToolCallbackError(
            "Callback failed",
            callback_name="rate_limit",
            tool_name="search",
        )
        d = err.to_dict()
        assert d["error_code"] == "TOOL_CALLBACK_ERROR"
        assert d["callback_name"] == "rate_limit"
        assert d["tool_name"] == "search"
        assert err.retryable is False

    def test_tool_auth_error(self):
        err = ToolAuthError(
            "Missing credentials",
            tool_name="ynab.create",
        )
        d = err.to_dict()
        assert d["error_code"] == "TOOL_AUTH_ERROR"
        assert d["tool_name"] == "ynab.create"
        assert err.http_status == 401
