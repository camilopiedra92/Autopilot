"""
A2A Protocol Server Tests — Validates agent card discovery, message/send,
task lifecycle, and error handling.

Tests use FastAPI TestClient (same pattern as test_api_v1.py).
"""

import json
import os
import pytest
from unittest.mock import MagicMock, AsyncMock

pytest.importorskip("a2a", reason="a2a-sdk not installed")

os.environ["API_KEY_SECRET"] = "test-secret-123"

from fastapi.testclient import TestClient

from app import app
from autopilot.api.a2a.agent_card import build_agent_card
from autopilot.api.a2a.request_handler import (
    AutopilotA2ARequestHandler,
    _map_status,
    _extract_workflow_request,
)
from autopilot.errors import A2AWorkflowNotFoundError, A2ATaskNotFoundError
from autopilot.models import (
    RunStatus,
    TriggerType,
    WorkflowInfo,
    WorkflowManifest,
    TriggerConfig,
)

from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskState,
    TextPart,
)

client = TestClient(app)

HEADERS = {"X-API-Key": "test-secret-123"}


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_mock_registry(workflows=None):
    """Create a mock WorkflowRegistry with optional workflows."""
    registry = MagicMock()

    if workflows is None:
        info = WorkflowInfo(
            name="test_flow",
            display_name="Test Flow",
            version="1.0.0",
            enabled=True,
            description="A test workflow",
            triggers=[TriggerConfig(type=TriggerType.WEBHOOK, path="/test")],
            icon="⚡",
            color="#000",
            tags=["test", "demo"],
        )
        registry.list_all.return_value = [info]
        registry.count = 1
    else:
        registry.list_all.return_value = workflows
        registry.count = len(workflows)

    return registry


def _make_mock_workflow(
    name="test_flow",
    status=RunStatus.SUCCESS,
    result=None,
):
    """Create a mock workflow with a run() method."""
    wf = MagicMock()
    wf.manifest = WorkflowManifest(name=name, display_name="Test Flow", version="1.0.0")

    run = MagicMock()
    run.id = "run-abc123"
    run.status = status
    run.result = result or {"msg": "done"}
    run.error = None if status == RunStatus.SUCCESS else "Something failed"
    run.duration_ms = 42.5

    wf.run = AsyncMock(return_value=run)
    return wf


def _make_send_params(workflow_id: str, **payload):
    """Create A2A MessageSendParams for a workflow."""
    data = {"workflow": workflow_id, **payload}
    return MessageSendParams(
        message=Message(
            messageId="msg-001",
            role=Role.user,
            parts=[Part(root=TextPart(text=json.dumps(data)))],
        )
    )


# ── Agent Card Tests ─────────────────────────────────────────────────


def test_build_agent_card_with_workflows():
    """AgentCard includes workflows as skills."""
    registry = _make_mock_registry()
    card = build_agent_card(registry)

    assert card.name == "AutoPilot"
    assert card.version
    assert len(card.skills) == 1
    assert card.skills[0].id == "test_flow"
    assert card.skills[0].name == "Test Flow"
    assert card.skills[0].description == "A test workflow"
    assert "test" in card.skills[0].tags
    assert "demo" in card.skills[0].tags


def test_build_agent_card_empty_registry():
    """AgentCard with no workflows has empty skills list."""
    registry = _make_mock_registry(workflows=[])
    card = build_agent_card(registry)

    assert card.name == "AutoPilot"
    assert card.skills == []
    assert card.capabilities.streaming is True


def test_build_agent_card_capabilities():
    """AgentCard advertises streaming."""
    registry = _make_mock_registry()
    card = build_agent_card(registry)

    assert card.capabilities.streaming is True
    assert card.capabilities.push_notifications is False


def test_build_agent_card_skips_disabled_workflows():
    """Disabled workflows are excluded from skills."""
    info_enabled = WorkflowInfo(
        name="enabled_flow",
        display_name="Enabled",
        version="1.0.0",
        enabled=True,
        description="Active",
        triggers=[],
        icon="⚡",
        color="#000",
        tags=[],
    )
    info_disabled = WorkflowInfo(
        name="disabled_flow",
        display_name="Disabled",
        version="1.0.0",
        enabled=False,
        description="Inactive",
        triggers=[],
        icon="⚡",
        color="#000",
        tags=[],
    )
    registry = _make_mock_registry(workflows=[info_enabled, info_disabled])
    card = build_agent_card(registry)

    assert len(card.skills) == 1
    assert card.skills[0].id == "enabled_flow"


# ── Task State Mapping Tests ─────────────────────────────────────────


def test_status_mapping_success():
    assert _map_status(RunStatus.SUCCESS) == TaskState.completed


def test_status_mapping_failed():
    assert _map_status(RunStatus.FAILED) == TaskState.failed


def test_status_mapping_running():
    assert _map_status(RunStatus.RUNNING) == TaskState.working


def test_status_mapping_pending():
    assert _map_status(RunStatus.PENDING) == TaskState.submitted


def test_status_mapping_skipped():
    assert _map_status(RunStatus.SKIPPED) == TaskState.completed


# ── Message Extraction Tests ─────────────────────────────────────────


def test_extract_workflow_request_valid():
    """Valid JSON TextPart with workflow key is extracted correctly."""
    params = _make_send_params("bank_to_ynab", body="test email")
    wf_id, data = _extract_workflow_request(params)

    assert wf_id == "bank_to_ynab"
    assert data == {"body": "test email"}


def test_extract_workflow_request_missing_workflow():
    """Missing 'workflow' key raises A2AWorkflowNotFoundError."""
    params = MessageSendParams(
        message=Message(
            messageId="msg-002",
            role=Role.user,
            parts=[Part(root=TextPart(text=json.dumps({"body": "no workflow"})))],
        )
    )
    with pytest.raises(A2AWorkflowNotFoundError):
        _extract_workflow_request(params)


def test_extract_workflow_request_invalid_json():
    """Non-JSON TextPart raises A2AWorkflowNotFoundError."""
    params = MessageSendParams(
        message=Message(
            messageId="msg-003",
            role=Role.user,
            parts=[Part(root=TextPart(text="not json"))],
        )
    )
    with pytest.raises(A2AWorkflowNotFoundError):
        _extract_workflow_request(params)


# ── Request Handler Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_message_send_success():
    """message/send executes workflow and returns completed Task."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow()
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("test_flow")

    task = await handler.on_message_send(params)

    assert task.status.state == TaskState.completed
    assert task.id
    assert task.context_id
    assert task.metadata["workflow_id"] == "test_flow"
    assert task.metadata["run_id"] == "run-abc123"
    assert task.metadata["duration_ms"] == 42.5
    assert task.artifacts is not None
    assert len(task.artifacts) == 1

    # Verify workflow.run was called correctly
    wf.run.assert_called_once_with(TriggerType.MANUAL, {})


@pytest.mark.asyncio
async def test_on_message_send_with_payload():
    """message/send passes trigger data to the workflow."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow()
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("test_flow", body="email text", auto_create=True)

    await handler.on_message_send(params)

    wf.run.assert_called_once_with(
        TriggerType.MANUAL, {"body": "email text", "auto_create": True}
    )


@pytest.mark.asyncio
async def test_on_message_send_workflow_not_found():
    """message/send with unknown workflow raises A2AWorkflowNotFoundError."""
    registry = _make_mock_registry()
    registry.get.return_value = None

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("nonexistent")

    with pytest.raises(A2AWorkflowNotFoundError):
        await handler.on_message_send(params)


@pytest.mark.asyncio
async def test_on_message_send_failed_workflow():
    """message/send with failed workflow returns Task with failed state."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow(status=RunStatus.FAILED, result={})
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("test_flow")

    task = await handler.on_message_send(params)

    assert task.status.state == TaskState.failed


@pytest.mark.asyncio
async def test_on_get_task_found():
    """tasks/get returns a previously stored task."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow()
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("test_flow")

    # Execute to store the task
    task = await handler.on_message_send(params)

    # Retrieve it
    from a2a.types import TaskQueryParams

    retrieved = await handler.on_get_task(TaskQueryParams(id=task.id))
    assert retrieved is not None
    assert retrieved.id == task.id
    assert retrieved.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_on_get_task_not_found():
    """tasks/get raises A2ATaskNotFoundError for unknown task ID."""
    registry = _make_mock_registry()
    handler = AutopilotA2ARequestHandler(registry)

    from a2a.types import TaskQueryParams

    with pytest.raises(A2ATaskNotFoundError):
        await handler.on_get_task(TaskQueryParams(id="nonexistent"))


@pytest.mark.asyncio
async def test_on_cancel_task_unsupported():
    """tasks/cancel raises ServerError wrapping UnsupportedOperationError."""
    registry = _make_mock_registry()
    handler = AutopilotA2ARequestHandler(registry)

    from a2a.server.request_handlers.request_handler import ServerError
    from a2a.types import TaskIdParams

    with pytest.raises(ServerError):
        await handler.on_cancel_task(TaskIdParams(id="any"))


@pytest.mark.asyncio
async def test_on_message_send_stream_lifecycle():
    """message/stream yields correct lifecycle events."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow()
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)
    params = _make_send_params("test_flow")

    from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent

    events = []
    async for event in handler.on_message_send_stream(params):
        events.append(event)

    # Expected: submitted, working, artifact, completed (final)
    assert len(events) == 4

    # First: submitted
    assert isinstance(events[0], TaskStatusUpdateEvent)
    assert events[0].status.state == TaskState.submitted
    assert events[0].final is False

    # Second: working
    assert isinstance(events[1], TaskStatusUpdateEvent)
    assert events[1].status.state == TaskState.working
    assert events[1].final is False

    # Third: artifact
    assert isinstance(events[2], TaskArtifactUpdateEvent)

    # Fourth: completed (final)
    assert isinstance(events[3], TaskStatusUpdateEvent)
    assert events[3].status.state == TaskState.completed
    assert events[3].final is True


@pytest.mark.asyncio
async def test_task_store_ring_buffer():
    """Task store evicts oldest tasks when over capacity."""
    registry = _make_mock_registry()
    wf = _make_mock_workflow()
    registry.get.return_value = wf

    handler = AutopilotA2ARequestHandler(registry)

    # Override max size for testing
    import autopilot.api.a2a.request_handler as rh

    original = rh._MAX_TASK_STORE_SIZE
    rh._MAX_TASK_STORE_SIZE = 3

    try:
        task_ids = []
        for i in range(5):
            params = _make_send_params("test_flow")
            task = await handler.on_message_send(params)
            task_ids.append(task.id)

        # Only last 3 should remain
        assert len(handler._tasks) == 3
        # First 2 should be evicted
        assert task_ids[0] not in handler._tasks
        assert task_ids[1] not in handler._tasks
        # Last 3 should be present
        assert task_ids[2] in handler._tasks
        assert task_ids[3] in handler._tasks
        assert task_ids[4] in handler._tasks
    finally:
        rh._MAX_TASK_STORE_SIZE = original


# ── Integration: Agent Card Endpoint ─────────────────────────────────


def test_agent_card_endpoint():
    """GET /.well-known/agent-card.json returns valid agent card."""
    with TestClient(app) as tc:
        response = tc.get("/.well-known/agent-card.json")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "AutoPilot"
        assert "version" in data
        assert "capabilities" in data
        assert "skills" in data
        assert isinstance(data["skills"], list)


def test_agent_card_unauthenticated():
    """Agent card endpoint does NOT require API key (A2A spec)."""
    with TestClient(app) as tc:
        response = tc.get("/.well-known/agent-card.json")
        assert response.status_code == 200
