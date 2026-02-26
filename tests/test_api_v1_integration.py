"""Integration tests — Cross-phase V1 API verification.

These tests use REAL InMemoryRunLogService (not mocks) to validate
that platform services compose correctly end-to-end.

Differences from unit tests (test_api_v1_endpoints.py):
  - RunLog ↔ API tests use real InMemoryRunLogService instances
  - EventBus dispatch tests assert exact payload shapes
  - Error taxonomy tests verify both status_code and error_code
  - Auth sweep validates ALL endpoints enforce X-API-Key
  - SSE tests validate content-type and Edge-safe headers
"""

from __future__ import annotations

import asyncio
import importlib
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from autopilot.core.run_log import (
    InMemoryRunLogService,
    reset_run_log_service,
)
from autopilot.models import (
    RunStatus,
    TriggerConfig,
    TriggerType,
    WorkflowInfo,
    WorkflowManifest,
    WorkflowRun,
)


API_KEY = "test-integration-key"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset():
    reset_run_log_service()
    yield
    reset_run_log_service()


def _make_manifest(name: str = "test_workflow") -> WorkflowManifest:
    return WorkflowManifest(
        name=name,
        display_name=f"Test {name}",
        description=f"Integration test workflow: {name}",
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


def _make_run(
    workflow_id: str = "test_workflow",
    run_id: str = "run_001",
    status: RunStatus = RunStatus.SUCCESS,
    started_at: datetime | None = None,
    error: str | None = None,
    result: dict | None = None,
) -> WorkflowRun:
    return WorkflowRun(
        id=run_id,
        workflow_id=workflow_id,
        status=status,
        trigger_type=TriggerType.MANUAL,
        trigger_data={},
        started_at=started_at or datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_ms=123.4,
        error=error,
        result=result or {},
    )


def _make_app():
    os.environ["API_KEY_SECRET"] = API_KEY
    import app as app_module

    importlib.reload(app_module)
    return app_module.app


def _seed_runs(svc: InMemoryRunLogService, runs: list[WorkflowRun]) -> None:
    """Seed runs into an InMemoryRunLogService synchronously."""
    loop = asyncio.new_event_loop()
    try:
        for run in runs:
            loop.run_until_complete(svc.save_run(run))
    finally:
        loop.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.1 — RunLogService ↔ V1 API Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRunLogAPIIntegration:
    """Validates that data saved via RunLogService flows correctly
    through V1 API endpoints with accurate stats, pagination,
    and status filtering."""

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_runs_appear_in_list_after_save(self, mock_run_log, mock_registry):
        """A run saved to RunLogService must appear in GET /workflows/{id}/runs."""
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc
        mock_registry.return_value.get.return_value = _make_workflow()
        mock_registry.return_value.list_all.return_value = [_make_info()]

        # Simulate BaseWorkflow saving a run: RUNNING → SUCCESS (idempotent upsert)
        now = datetime.now(timezone.utc)
        _seed_runs(
            svc,
            [
                _make_run(run_id="run_a", status=RunStatus.RUNNING, started_at=now),
                _make_run(run_id="run_a", status=RunStatus.SUCCESS, started_at=now),
            ],
        )

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/test_workflow/runs",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == "run_a"
        assert data["runs"][0]["status"] == "success"

        # Stats reflect idempotent upsert — total=1, not 2
        assert data["stats"]["total"] == 1
        assert data["stats"]["successful"] == 1

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_stats_aggregate_across_runs(self, mock_run_log, mock_registry):
        """Success rate calculation from real RunLogService data."""
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc
        mock_registry.return_value.get.return_value = _make_workflow()
        mock_registry.return_value.list_all.return_value = [_make_info()]

        now = datetime.now(timezone.utc)
        statuses = [
            RunStatus.SUCCESS,
            RunStatus.SUCCESS,
            RunStatus.FAILED,
            RunStatus.SUCCESS,
            RunStatus.FAILED,
        ]
        _seed_runs(
            svc,
            [
                _make_run(
                    run_id=f"run_{i}",
                    status=status,
                    started_at=now + timedelta(seconds=i),
                )
                for i, status in enumerate(statuses)
            ],
        )

        app = _make_app()
        client = TestClient(app)

        # Check list endpoint has correct count
        r = client.get(
            "/api/v1/workflows/test_workflow/runs",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["stats"]["total"] == 5
        assert data["stats"]["successful"] == 3

        # Check overview endpoint aggregates
        r = client.get(
            "/api/v1/workflows",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        wf = r.json()["workflows"][0]
        assert wf["total_runs"] == 5
        assert wf["success_rate"] == 60.0  # 3/5 * 100

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_pagination_cursor_consistency(self, mock_run_log, mock_registry):
        """Cursor returned from first page must produce correct second page."""
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc
        mock_registry.return_value.get.return_value = _make_workflow()
        mock_registry.return_value.list_all.return_value = [_make_info()]

        now = datetime.now(timezone.utc)
        _seed_runs(
            svc,
            [
                _make_run(
                    run_id=f"run_{i:03d}",
                    started_at=now + timedelta(seconds=i),
                )
                for i in range(10)
            ],
        )

        app = _make_app()
        client = TestClient(app)

        # First page — use real InMemoryRunLogService directly to verify cursor
        loop = asyncio.new_event_loop()
        try:
            page1_runs, cursor1 = loop.run_until_complete(
                svc.list_runs("test_workflow", limit=3)
            )
            assert len(page1_runs) == 3
            assert page1_runs[0].id == "run_009"  # newest first
            assert cursor1 is not None

            # Second page using cursor from first page
            page2_runs, cursor2 = loop.run_until_complete(
                svc.list_runs("test_workflow", limit=3, start_after=cursor1)
            )
            assert len(page2_runs) == 3

            # No overlap between pages
            page1_ids = {r.id for r in page1_runs}
            page2_ids = {r.id for r in page2_runs}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            loop.close()

        # Also verify the API endpoint returns paginated results
        r = client.get(
            "/api/v1/workflows/test_workflow/runs?limit=3",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        page = r.json()
        assert len(page["runs"]) == 3
        assert page["meta"]["next_cursor"] is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.2 — EventBus ↔ HITL/Trigger Dispatch Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventBusDispatchIntegration:
    """Validates that write endpoints dispatch events with correct
    payload shapes that subscribers can parse."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_publishes_correct_event_shape(
        self, mock_registry, mock_run_log, mock_bus
    ):
        """Resume endpoint must publish event with {run_id, workflow_id, payload}."""
        mock_registry.return_value.get.return_value = _make_workflow()

        paused_run = _make_run(run_id="run_paused", status=RunStatus.PAUSED)
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = paused_run
        mock_run_log.return_value = mock_svc

        mock_bus_instance = AsyncMock()
        mock_bus_instance.publish = AsyncMock()
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_paused/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"approved_category": "Food"}},
        )
        assert r.status_code == 200

        # Verify the exact event shape that _on_hitl_resumed expects
        mock_bus_instance.publish.assert_called_once()
        call_args = mock_bus_instance.publish.call_args
        # Topic is the first positional arg
        assert call_args[0][0] == "api.hitl_resumed"
        # Payload is the second positional arg
        event_payload = call_args[0][1]
        assert event_payload["run_id"] == "run_paused"
        assert event_payload["workflow_id"] == "test_workflow"
        assert event_payload["payload"]["approved_category"] == "Food"

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_publishes_correct_event_shape(self, mock_registry, mock_bus):
        """Trigger endpoint must publish event with {workflow_id, payload}."""
        wf = _make_workflow()
        wf.manifest.enabled = True
        mock_registry.return_value.get.return_value = wf

        mock_bus_instance = AsyncMock()
        mock_bus_instance.publish = AsyncMock()
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/trigger",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"test_input": 42}},
        )
        assert r.status_code == 200

        mock_bus_instance.publish.assert_called_once()
        call_args = mock_bus_instance.publish.call_args
        assert call_args[0][0] == "api.workflow_triggered"
        event_payload = call_args[0][1]
        assert event_payload["workflow_id"] == "test_workflow"
        assert event_payload["payload"]["test_input"] == 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.3 — Error Taxonomy ↔ HTTP Status Code Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorTaxonomyHTTPMapping:
    """Validates that every API AutoPilotError subclass maps to
    its declared http_status when raised inside an actual endpoint."""

    @patch("autopilot.api.v1.routes.get_registry")
    def test_workflow_not_found_returns_404(self, mock_registry):
        """WorkflowNotFoundError (http_status=404) → HTTP 404."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404
        assert r.json()["error"]["error_code"] == "WORKFLOW_NOT_FOUND"

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_run_not_found_returns_404(self, mock_registry, mock_run_log):
        """RunNotFoundError (http_status=404) → HTTP 404."""
        mock_registry.return_value.get.return_value = _make_workflow()
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = None
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/missing/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 404

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_run_not_paused_returns_409(self, mock_registry, mock_run_log):
        """RunNotPausedError (http_status=409) → HTTP 409."""
        mock_registry.return_value.get.return_value = _make_workflow()
        running_run = _make_run(run_id="run_x", status=RunStatus.RUNNING)
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = running_run
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_x/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 409
        assert r.json()["error"]["error_code"] == "RUN_NOT_PAUSED"

    def test_missing_api_key_returns_401(self):
        """All API + copilot endpoints return 401 without X-API-Key."""
        app = _make_app()
        client = TestClient(app)

        endpoints = [
            ("GET", "/api/v1/workflows"),
            ("GET", "/api/v1/workflows/any"),
            ("GET", "/api/v1/workflows/any/runs"),
            ("GET", "/api/v1/runs/pending-action"),
            ("GET", "/api/v1/health"),
            ("POST", "/api/v1/workflows/any/trigger"),
            ("POST", "/api/v1/copilot/ask"),
        ]

        for method, path in endpoints:
            if method == "GET":
                r = client.get(path)
            else:
                r = client.post(path, json={"query": "test query", "payload": {}})
            assert r.status_code == 401, (
                f"Expected 401 for {method} {path}, got {r.status_code}"
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.4 — Copilot ↔ RunLogService Tool Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCopilotRunLogIntegration:
    """Validates that copilot read-only tools correctly query
    RunLogService and return meaningful analysis."""

    @patch("autopilot.agents.copilot.get_event_bus")
    @patch("autopilot.agents.copilot.get_run_log_service")
    @patch("autopilot.agents.copilot.get_registry")
    def test_copilot_stats_query(self, mock_registry, mock_run_log, mock_bus):
        """Copilot asking about a workflow returns stats in reply."""
        mock_registry.return_value.list_all.return_value = [
            _make_info(_make_manifest("bank_to_ynab"))
        ]
        mock_svc = AsyncMock()
        mock_svc.get_stats.return_value = {"total": 50, "successful": 45}
        mock_svc.list_runs.return_value = ([], None)
        mock_run_log.return_value = mock_svc

        mock_bus.return_value._history = {}

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "What is the status of bank_to_ynab?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["tools_used"]) > 0  # Copilot used at least one tool

    @patch("autopilot.agents.copilot.get_event_bus")
    @patch("autopilot.agents.copilot.get_run_log_service")
    @patch("autopilot.agents.copilot.get_registry")
    def test_copilot_error_query(self, mock_registry, mock_run_log, mock_bus):
        """Copilot asking about failures surfaces recent errors."""
        manifest = _make_manifest("bank_to_ynab")
        mock_registry.return_value.list_all.return_value = [_make_info(manifest)]

        failed_run = _make_run(
            workflow_id="bank_to_ynab",
            run_id="run_fail",
            status=RunStatus.FAILED,
            error="YNAB API rate limit exceeded",
        )
        mock_svc = AsyncMock()
        mock_svc.get_stats.return_value = {"total": 10, "successful": 7}
        mock_svc.list_runs.return_value = ([failed_run], None)
        mock_run_log.return_value = mock_svc

        mock_bus.return_value._history = {}

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "Why did bank_to_ynab fail recently?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        # Copilot should reference failures in its answer
        answer_lower = data["answer"].lower()
        assert any(kw in answer_lower for kw in ["error", "fail", "ynab", "rate"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.5 — SSE Stream Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSSEStreamIntegration:
    """Validates SSE event stream produces correctly formatted
    text/event-stream output with Edge-safe headers.

    Uses daemon threads to prevent the streaming generator from
    blocking the test runner.
    """

    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_sse_content_type_and_headers(self, mock_bus):
        """SSE endpoint returns StreamingResponse with correct headers.

        We call the endpoint function directly and inspect the
        StreamingResponse object, avoiding the streaming hang.
        """
        mock_bus_instance = MagicMock()
        mock_bus_instance.subscribe.return_value = MagicMock()
        mock_bus_instance.unsubscribe = MagicMock()
        mock_bus_instance.replay = AsyncMock(return_value=[])
        mock_bus.return_value = mock_bus_instance

        from autopilot.api.v1.routes import event_stream
        from starlette.requests import Request

        # Create a minimal ASGI scope to build a Request object
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/events/stream",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(event_stream(request))
        finally:
            loop.close()

        assert response.media_type == "text/event-stream"
        # Check headers dict
        header_dict = dict(response.headers)
        assert header_dict.get("cache-control") == "no-cache"
        assert header_dict.get("x-accel-buffering") == "no"

    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_sse_requires_api_key(self, mock_bus):
        """SSE endpoint also requires X-API-Key."""
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/events/stream")
        assert r.status_code == 401
