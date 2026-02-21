import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

import os

# Ensure the mock secret is set for tests
os.environ["API_KEY_SECRET"] = "test-secret-123"

from app import app
from autopilot.models import WorkflowManifest, WorkflowInfo, TriggerConfig, TriggerType

client = TestClient(app)

HEADERS = {"X-API-Key": "test-secret-123"}


@pytest.fixture
def mock_registry():
    with patch("autopilot.api.v1.routes.get_registry") as mock_get_registry:
        registry = MagicMock()
        mock_get_registry.return_value = registry

        info = WorkflowInfo(
            name="test_flow",
            display_name="Test Flow",
            version="1.0.0",
            enabled=True,
            description="A test workflow",
            triggers=[TriggerConfig(type=TriggerType.WEBHOOK, path="/test")],
            icon="⚡",
            color="#000",
            tags=["test"],
        )
        registry.list_all.return_value = [info]
        registry.count = 1

        wf = MagicMock()
        wf.manifest = WorkflowManifest(
            name="test_flow", display_name="Test Flow", version="1.0.0"
        )
        wf.recent_runs = []
        registry.get.return_value = wf

        yield registry


@pytest.fixture
def mock_router_svc():
    with patch("autopilot.api.v1.routes.get_router") as mock_get_router:
        router = MagicMock()
        mock_get_router.return_value = router

        # Mock run response
        run = MagicMock()
        run.id = "run-123"
        run.workflow_id = "test_flow"
        from autopilot.models import RunStatus

        run.status = RunStatus.SUCCESS
        run.result = {"msg": "done"}
        run.error = None

        router.route_manual = AsyncMock(return_value=run)

        yield router


def test_list_workflows(mock_registry):
    response = client.get("/api/v1/workflows", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "workflows" in data
    assert data["total"] == 1
    assert data["workflows"][0]["id"] == "test_flow"


def test_missing_api_key():
    response = client.get("/api/v1/workflows")
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_invalid_api_key():
    response = client.get("/api/v1/workflows", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]


def test_execute_workflow(mock_router_svc):
    payload = {"foo": "bar"}
    response = client.post(
        "/api/v1/workflows/test_flow/execute", headers=HEADERS, json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["run_id"] == "run-123"

    mock_router_svc.route_manual.assert_called_once_with("test_flow", payload)


def test_get_workflow_runs(mock_registry):
    response = client.get("/api/v1/workflows/test_flow/runs", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == "test_flow"
    assert isinstance(data["runs"], list)
