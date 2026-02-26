"""Tests for V1 API endpoints.

Tests use FastAPI TestClient with mocked platform services.
Follows the same patterns as tests/test_api_v1.py:
  - _make_app() creates a fresh app with test API key
  - _make_manifest() / _make_workflow() / _make_info() create test fixtures
  - @patch decorators mock platform singletons (registry, run_log, event_bus)
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from autopilot.core.run_log import InMemoryRunLogService, reset_run_log_service
from autopilot.models import WorkflowManifest, WorkflowInfo, TriggerConfig, TriggerType


API_KEY = "test-v1-endpoints-key"


@pytest.fixture(autouse=True)
def _reset():
    reset_run_log_service()
    yield
    reset_run_log_service()


def _make_app():
    """Create a fresh app with mocked registry."""
    import importlib
    import os

    os.environ["API_KEY_SECRET"] = API_KEY

    import app as app_module

    importlib.reload(app_module)
    return app_module.app


def _make_manifest():
    return WorkflowManifest(
        name="test_workflow",
        display_name="Test Workflow",
        description="A test workflow",
        version="1.0.0",
        triggers=[TriggerConfig(type=TriggerType.MANUAL)],
    )


def _make_workflow(manifest=None):
    wf = MagicMock()
    wf.manifest = manifest or _make_manifest()
    wf._workflow_dir = "/tmp/test_workflow"
    wf.recent_runs = []
    wf.total_runs = 0
    wf.success_rate = 0.0
    wf.last_run = None
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


class TestWorkflows:
    """Tests for /api/v1/workflows endpoints."""

    def test_list_requires_api_key(self):
        """Verify 401 when X-API-Key header is missing."""
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/workflows")
        assert r.status_code == 401

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_list_workflows(self, mock_run_log, mock_registry):
        """Verify list returns enriched workflow data."""
        mock_registry.return_value.list_all.return_value = [_make_info()]
        mock_registry.return_value.get.return_value = _make_workflow()
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["workflows"][0]["id"] == "test_workflow"

    @patch("autopilot.api.v1.routes.get_registry")
    def test_workflow_not_found(self, mock_registry):
        """Verify 404 with WorkflowNotFoundError."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestPipeline:
    """Tests for /api/v1/workflows/{id}/pipeline endpoint."""

    @patch("autopilot.api.v1.routes.get_registry")
    def test_pipeline_not_found(self, mock_registry):
        """Verify 404 for non-existent workflow pipeline."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent/pipeline",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404

    @patch("autopilot.api.v1.routes.get_registry")
    def test_pipeline_empty_workflow(self, mock_registry):
        """Verify empty pipeline graph when no pipeline.yaml exists."""
        mock_registry.return_value.get.return_value = _make_workflow()

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/test_workflow/pipeline",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["strategy"] == "SEQUENTIAL"
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["layers"] == []


class TestRuns:
    """Tests for /api/v1/workflows/{id}/runs endpoints."""

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_list_runs_empty(self, mock_run_log, mock_registry):
        """Verify empty run list for workflow with no history."""
        mock_registry.return_value.get.return_value = _make_workflow()
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/test_workflow/runs",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["runs"] == []
        assert data["meta"]["next_cursor"] is None
        assert data["stats"]["total"] == 0

    @patch("autopilot.api.v1.routes.get_registry")
    def test_runs_workflow_not_found(self, mock_registry):
        """Verify 404 for runs on non-existent workflow."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent/runs",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestHealth:
    """Tests for /api/v1/health endpoint."""

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_health(self, mock_bus, mock_registry):
        """Verify health endpoint returns healthy status."""
        mock_registry.return_value.list_all.return_value = [_make_info()]
        mock_bus.return_value.stats = {"published": 0, "delivered": 0, "errors": 0}

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/health",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["workflows"]["total"] == 1


class TestEvents:
    """Tests for /api/v1/events endpoint."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_events_empty(self, mock_bus):
        """Verify empty events when bus has no history."""
        mock_bus.return_value._history = {}

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/events",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["events"] == []
        assert data["total"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 2 — HITL, Trigger, Copilot Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHITLPendingRuns:
    """Tests for GET /api/v1/runs/pending-action."""

    def test_requires_api_key(self):
        """Verify 401 when X-API-Key header is missing."""
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/runs/pending-action")
        assert r.status_code == 401

    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_empty_pending_list(self, mock_run_log):
        """Verify empty list when no runs are paused."""
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/runs/pending-action",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["pending"] == []
        assert data["total"] == 0

    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_paused_runs_returned(self, mock_run_log):
        """Verify paused runs are returned as PendingRunItem projections."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        paused_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RS.PAUSED,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )

        svc = AsyncMock()
        svc.get_pending_runs = AsyncMock(return_value=[paused_run])
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/runs/pending-action",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["pending"][0]["run_id"] == "run_abc123"
        assert data["pending"][0]["workflow_id"] == "test_workflow"
        assert data["pending"][0]["status"] == "paused"


class TestHITLResume:
    """Tests for POST /api/v1/workflows/{id}/runs/{run_id}/resume."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_success(self, mock_registry, mock_run_log, mock_bus):
        """Verify successful resume dispatches event to EventBus."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        mock_registry.return_value.get.return_value = _make_workflow()

        paused_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RS.PAUSED,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=paused_run)
        mock_run_log.return_value = svc

        bus = AsyncMock()
        bus.publish = AsyncMock()
        mock_bus.return_value = bus

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"approved": True}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "dispatched"
        assert data["run_id"] == "run_abc123"
        assert data["workflow_id"] == "test_workflow"

        # Verify EventBus was called
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "api.hitl_resumed"

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_not_paused_409(self, mock_registry, mock_run_log):
        """Verify 409 Conflict when run is not PAUSED."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        mock_registry.return_value.get.return_value = _make_workflow()

        running_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RS.RUNNING,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=running_run)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/resume",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 409

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_run_not_found_404(self, mock_registry, mock_run_log):
        """Verify 404 when run_id does not exist."""
        mock_registry.return_value.get.return_value = _make_workflow()

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=None)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/nonexistent/resume",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404

    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_workflow_not_found_404(self, mock_registry):
        """Verify 404 when workflow_id does not exist."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/nonexistent/runs/run_abc/resume",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestManualTrigger:
    """Tests for POST /api/v1/workflows/{id}/trigger."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_success(self, mock_registry, mock_bus):
        """Verify successful trigger dispatches event to EventBus."""
        mock_registry.return_value.get.return_value = _make_workflow()

        bus = AsyncMock()
        bus.publish = AsyncMock()
        mock_bus.return_value = bus

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/trigger",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"key": "value"}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "dispatched"
        assert data["workflow_id"] == "test_workflow"
        assert data["trigger_type"] == "MANUAL"

        # Verify EventBus was called
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "api.workflow_triggered"

    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_disabled_workflow(self, mock_registry):
        """Verify error when workflow is disabled."""
        manifest = _make_manifest()
        manifest.enabled = False
        wf = _make_workflow(manifest=manifest)
        mock_registry.return_value.get.return_value = wf

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/trigger",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 500  # APIError base → 500

    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_workflow_not_found_404(self, mock_registry):
        """Verify 404 when workflow_id does not exist."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/nonexistent/trigger",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestCopilot:
    """Tests for POST /api/v1/copilot/ask."""

    def test_requires_api_key(self):
        """Verify 401 when X-API-Key header is missing."""
        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            json={"query": "What workflows failed?"},
        )
        assert r.status_code == 401

    @patch("autopilot.agents.copilot.get_event_bus")
    @patch("autopilot.agents.copilot.get_run_log_service")
    @patch("autopilot.agents.copilot.get_registry")
    def test_copilot_structured_response(self, mock_registry, mock_run_log, mock_bus):
        """Verify copilot returns structured CopilotResponse."""
        mock_registry.return_value.list_all.return_value = [_make_info()]

        svc = AsyncMock()
        svc.get_stats = AsyncMock(return_value={"total": 5, "successful": 4})
        svc.list_runs = AsyncMock(return_value=([], None))
        mock_run_log.return_value = svc

        mock_bus.return_value._history = {}

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "Show me stats for all workflows"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "tools_used" in data
        assert "iterations" in data
        assert isinstance(data["tools_used"], list)
        assert data["iterations"] > 0

    def test_copilot_min_query_length(self):
        """Verify 422 when query is too short (min_length=3)."""
        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "hi"},
        )
        assert r.status_code == 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 3 — Cancel, Delete, Stats, Toggle, Filters, Versioning tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCancelRun:
    """Tests for POST /api/v1/workflows/{id}/runs/{run_id}/cancel."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_cancel_running_success(self, mock_registry, mock_run_log, mock_bus):
        """Verify successful cancel of a RUNNING run."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        mock_registry.return_value.get.return_value = _make_workflow()

        running_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RS.RUNNING,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=running_run)
        svc.save_run = AsyncMock()
        mock_run_log.return_value = svc

        bus = AsyncMock()
        bus.publish = AsyncMock()
        mock_bus.return_value = bus

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/cancel",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"] == "run_abc123"
        assert data["status"] == "cancelled"

        # Verify EventBus was called
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "api.run_cancelled"

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_cancel_completed_409(self, mock_registry, mock_run_log):
        """Verify 409 when trying to cancel a SUCCESS run."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        mock_registry.return_value.get.return_value = _make_workflow()

        completed_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RS.SUCCESS,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=completed_run)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/cancel",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 409

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_cancel_run_not_found_404(self, mock_registry, mock_run_log):
        """Verify 404 when run_id does not exist."""
        mock_registry.return_value.get.return_value = _make_workflow()

        svc = AsyncMock()
        svc.get_run = AsyncMock(return_value=None)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/nonexistent/cancel",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestDeleteRun:
    """Tests for DELETE /api/v1/workflows/{id}/runs/{run_id}."""

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_delete_success(self, mock_registry, mock_run_log):
        """Verify successful deletion returns 200."""
        mock_registry.return_value.get.return_value = _make_workflow()

        svc = AsyncMock()
        svc.delete_run = AsyncMock(return_value=True)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.delete(
            "/api/v1/workflows/test_workflow/runs/run_abc123",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["deleted"] is True
        assert data["run_id"] == "run_abc123"

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_delete_not_found_404(self, mock_registry, mock_run_log):
        """Verify 404 when run does not exist."""
        mock_registry.return_value.get.return_value = _make_workflow()

        svc = AsyncMock()
        svc.delete_run = AsyncMock(return_value=False)
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.delete(
            "/api/v1/workflows/test_workflow/runs/nonexistent",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestPlatformStats:
    """Tests for GET /api/v1/stats."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_stats_response(self, mock_registry, mock_run_log, mock_bus):
        """Verify stats returns aggregated data."""
        mock_registry.return_value.list_all.return_value = [_make_info()]

        svc = AsyncMock()
        svc.get_stats = AsyncMock(return_value={"total": 10, "successful": 8})
        mock_run_log.return_value = svc

        mock_bus.return_value.stats = {"published": 50, "delivered": 49, "errors": 1}

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/stats",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_workflows"] == 1
        assert data["total_runs"] == 10
        assert data["total_successful"] == 8
        assert data["total_failed"] == 2
        assert data["global_success_rate"] == 80.0
        assert data["top_workflow"] == "test_workflow"
        assert len(data["workflows"]) == 1
        assert data["bus_stats"]["published"] == 50

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_stats_empty_platform(self, mock_registry, mock_run_log, mock_bus):
        """Verify stats with no workflows returns zero values."""
        mock_registry.return_value.list_all.return_value = []
        mock_bus.return_value.stats = {"published": 0, "delivered": 0, "errors": 0}

        svc = AsyncMock()
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/stats",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_workflows"] == 0
        assert data["total_runs"] == 0
        assert data["global_success_rate"] == 0.0


class TestWorkflowToggle:
    """Tests for PATCH /api/v1/workflows/{id}."""

    @patch("autopilot.api.v1.routes.get_registry")
    def test_disable_workflow(self, mock_registry):
        """Verify workflow can be disabled via PATCH."""
        wf = _make_workflow()
        mock_registry.return_value.get.return_value = wf

        app = _make_app()
        client = TestClient(app)
        r = client.patch(
            "/api/v1/workflows/test_workflow",
            headers={"X-API-Key": API_KEY},
            json={"enabled": False},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is False
        assert data["workflow_id"] == "test_workflow"
        assert data["status"] == "updated"
        # Verify the manifest was actually updated
        assert wf.manifest.enabled is False

    @patch("autopilot.api.v1.routes.get_registry")
    def test_enable_workflow(self, mock_registry):
        """Verify workflow can be re-enabled via PATCH."""
        manifest = _make_manifest()
        manifest.enabled = False
        wf = _make_workflow(manifest=manifest)
        mock_registry.return_value.get.return_value = wf

        app = _make_app()
        client = TestClient(app)
        r = client.patch(
            "/api/v1/workflows/test_workflow",
            headers={"X-API-Key": API_KEY},
            json={"enabled": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert wf.manifest.enabled is True

    @patch("autopilot.api.v1.routes.get_registry")
    def test_toggle_not_found_404(self, mock_registry):
        """Verify 404 for non-existent workflow."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.patch(
            "/api/v1/workflows/nonexistent",
            headers={"X-API-Key": API_KEY},
            json={"enabled": False},
        )
        assert r.status_code == 404


class TestRunFilters:
    """Tests for GET /api/v1/workflows/{id}/runs with status/since filters."""

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_filter_by_status(self, mock_registry, mock_run_log):
        """Verify status filter returns only matching runs."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone

        mock_registry.return_value.get.return_value = _make_workflow()

        runs = [
            WorkflowRun(
                id="run_1",
                workflow_id="test_workflow",
                status=RS.SUCCESS,
                trigger_type=TriggerType.MANUAL,
                started_at=datetime.now(timezone.utc),
            ),
            WorkflowRun(
                id="run_2",
                workflow_id="test_workflow",
                status=RS.FAILED,
                trigger_type=TriggerType.MANUAL,
                started_at=datetime.now(timezone.utc),
            ),
        ]

        svc = AsyncMock()
        svc.list_runs = AsyncMock(return_value=(runs, None))
        svc.get_stats = AsyncMock(return_value={"total": 2, "successful": 1})
        mock_run_log.return_value = svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/test_workflow/runs?status=failed",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == "run_2"

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_filter_by_since(self, mock_registry, mock_run_log):
        """Verify since filter returns only runs after the given time."""
        from autopilot.models import WorkflowRun, RunStatus as RS
        from datetime import datetime, timezone, timedelta

        mock_registry.return_value.get.return_value = _make_workflow()

        now = datetime.now(timezone.utc)
        runs = [
            WorkflowRun(
                id="run_old",
                workflow_id="test_workflow",
                status=RS.SUCCESS,
                trigger_type=TriggerType.MANUAL,
                started_at=now - timedelta(days=2),
            ),
            WorkflowRun(
                id="run_new",
                workflow_id="test_workflow",
                status=RS.SUCCESS,
                trigger_type=TriggerType.MANUAL,
                started_at=now,
            ),
        ]

        svc = AsyncMock()
        svc.list_runs = AsyncMock(return_value=(runs, None))
        svc.get_stats = AsyncMock(return_value={"total": 2, "successful": 2})
        mock_run_log.return_value = svc

        since = (now - timedelta(hours=1)).isoformat()
        app = _make_app()
        client = TestClient(app)
        r = client.get(
            f"/api/v1/workflows/test_workflow/runs?since={since}",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == "run_new"


class TestAPIVersioning:
    """Tests for API versioning headers on /api/v1/* responses."""

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_version_headers_present(self, mock_bus, mock_registry):
        """Verify API-Version and X-API-Docs headers are present."""
        mock_registry.return_value.list_all.return_value = [_make_info()]
        mock_bus.return_value.stats = {"published": 0, "delivered": 0, "errors": 0}

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/health",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        assert r.headers.get("API-Version") == "v1"
        assert r.headers.get("X-API-Docs") == "/docs"


class TestOpenAPIExport:
    """Tests for GET /api/v1/openapi.json."""

    def test_openapi_returns_spec(self):
        """Verify /api/v1/openapi.json returns a valid filtered spec."""
        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/openapi.json",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert "paths" in data
        assert "info" in data
        assert data["info"]["title"] == "AutoPilot API v1"
        # All paths should be /api/v1/*
        for path in data["paths"]:
            assert path.startswith("/api/v1"), f"Non-v1 path found: {path}"
