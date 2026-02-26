"""
V1 API Route tests — Unified endpoint verification.

Tests the unified /api/v1/* endpoints using FastAPI TestClient.
These tests cover basic API auth, enriched workflow listing,
trigger endpoint, and run listing.

For comprehensive endpoint coverage (HITL, copilot, events, SSE),
see test_api_v1_endpoints.py and test_api_v1_integration.py.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from autopilot.core.run_log import InMemoryRunLogService, reset_run_log_service
from autopilot.models import WorkflowManifest, WorkflowInfo, TriggerConfig, TriggerType


API_KEY = "test-v1-key"


@pytest.fixture(autouse=True)
def _reset():
    reset_run_log_service()
    yield
    reset_run_log_service()


def _make_app():
    """Create a fresh app with test API key (same pattern as test_api_v1_endpoints.py)."""
    import importlib
    import os

    os.environ["API_KEY_SECRET"] = API_KEY

    import app as app_module

    importlib.reload(app_module)
    return app_module.app


def _make_manifest():
    return WorkflowManifest(
        name="test_flow",
        display_name="Test Flow",
        description="A test workflow",
        version="1.0.0",
        triggers=[TriggerConfig(type=TriggerType.WEBHOOK, path="/test")],
    )


def _make_workflow(manifest=None):
    wf = MagicMock()
    wf.manifest = manifest or _make_manifest()
    wf._workflow_dir = "/tmp/test_flow"
    wf.recent_runs = []
    return wf


def _make_info(manifest=None):
    m = manifest or _make_manifest()
    return WorkflowInfo(
        name=m.name,
        display_name=m.display_name,
        description=m.description,
        version=m.version,
        icon=m.icon,
        color=m.color,
        enabled=m.enabled,
        triggers=m.triggers,
        tags=m.tags,
    )


# ── Auth Tests ─────────────────────────────────────────────────────────


def test_missing_api_key():
    """GET /api/v1/workflows requires X-API-Key header."""
    app = _make_app()
    client = TestClient(app)
    response = client.get("/api/v1/workflows")
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_invalid_api_key():
    """GET /api/v1/workflows rejects wrong X-API-Key."""
    app = _make_app()
    client = TestClient(app)
    response = client.get("/api/v1/workflows", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


# ── Workflow CRUD Tests ────────────────────────────────────────────────


@patch("autopilot.api.v1.routes.get_run_log_service")
@patch("autopilot.api.v1.routes.get_registry")
def test_list_workflows(mock_get_registry, mock_run_log):
    """GET /api/v1/workflows returns enriched workflow list."""
    mock_get_registry.return_value.list_all.return_value = [_make_info()]
    mock_get_registry.return_value.get.return_value = _make_workflow()

    svc = InMemoryRunLogService()
    mock_run_log.return_value = svc

    app = _make_app()
    client = TestClient(app)
    response = client.get("/api/v1/workflows", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert "workflows" in data
    assert data["total"] == 1
    assert data["workflows"][0]["id"] == "test_flow"


# ── Execute / Trigger Tests ───────────────────────────────────────────


@patch("autopilot.api.v1.routes.get_event_bus")
@patch("autopilot.api.v1.routes.get_registry")
def test_trigger_workflow(mock_get_registry, mock_bus):
    """POST /api/v1/workflows/{id}/trigger dispatches via EventBus."""
    mock_get_registry.return_value.get.return_value = _make_workflow()

    bus = AsyncMock()
    bus.publish = AsyncMock()
    mock_bus.return_value = bus

    app = _make_app()
    client = TestClient(app)
    response = client.post(
        "/api/v1/workflows/test_flow/trigger",
        headers={"X-API-Key": API_KEY},
        json={"payload": {"foo": "bar"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "dispatched"
    assert data["workflow_id"] == "test_flow"

    bus.publish.assert_called_once()


# ── Run List Tests ────────────────────────────────────────────────────


@patch("autopilot.api.v1.routes.get_run_log_service")
@patch("autopilot.api.v1.routes.get_registry")
def test_list_workflow_runs(mock_get_registry, mock_run_log):
    """GET /api/v1/workflows/{id}/runs returns paginated runs from RunLogService."""
    mock_get_registry.return_value.get.return_value = _make_workflow()

    svc = InMemoryRunLogService()
    mock_run_log.return_value = svc

    app = _make_app()
    client = TestClient(app)
    response = client.get(
        "/api/v1/workflows/test_flow/runs", headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == "test_flow"
    assert isinstance(data["runs"], list)
