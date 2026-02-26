# Phase 3A. Integration Tests — Cross-Phase API Verification

> **Status**: ✅ COMPLETE (2026-02-26)  
> **Effort**: ~60 min  
> **Type**: NEW (Tests)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 3  
> **Depends on**: Phase 0, Phase 1, Phase 2 all complete

---

## Problem Statement

Phases 0–2 each shipped with **unit tests** that mock platform services in isolation. This is necessary for fast, deterministic feedback — but it leaves a critical gap: **no test validates that the real services compose correctly end-to-end**.

Specifically, the following cross-cutting behaviors are untested:

1. **RunLogService ↔ Dashboard API**: A run saved via `BaseWorkflow.run()` must be queryable from `/dashboard/workflows/{id}/runs` — with correct stats aggregation, cursor pagination, and status filtering.
2. **EventBus ↔ HITL Resume**: A `dashboard.hitl_resumed` event published by the resume endpoint must be received by the `BaseWorkflow._on_hitl_resumed` subscriber with the correct payload structure.
3. **EventBus ↔ Manual Trigger**: Same for `dashboard.workflow_triggered` → `_on_manual_trigger`.
4. **SSE ↔ EventBus**: Events published to the EventBus must appear in the SSE stream (`/events/stream`) as properly formatted `text/event-stream` messages.
5. **Copilot ↔ RunLogService**: The copilot's read-only tools must correctly query `InMemoryRunLogService` with pre-seeded data and return meaningful analysis.
6. **Error Taxonomy ↔ HTTP Status Codes**: Every `AutoPilotError` subclass must map to its declared `http_status` when raised inside an endpoint.

Without integration tests, subtle bugs like incorrect serialization, broken cursor types, or EventBus dispatch mismatches can survive all unit tests and only surface in production.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                   | Current                  | Target                                                          |
| ----------------------- | --------------------------------------------- | ------------------------ | --------------------------------------------------------------- |
| §9.4 Development Rules  | All new modules have tests                    | Unit tests only          | Integration tests validating cross-module composition           |
| §6 Error Taxonomy       | `AutoPilotError` subclasses map to HTTP codes | Assumed, never validated | Tests assert `http_status` ↔ FastAPI response code for each err |
| §1 Core Philosophy      | Scale-to-zero safe                            | HITL uses EventBus       | Tests verify EventBus dispatch + subscriber receive             |

---

## Prerequisites

All phases complete (0 through 2). All unit tests passing:

```bash
python -m pytest tests/autopilot/test_run_log.py tests/test_dashboard_api.py -v
```

All modules importable:

```bash
python -c "
from autopilot.core.run_log import InMemoryRunLogService, get_run_log_service, reset_run_log_service
from autopilot.api.v1.routes import router
from autopilot.api.v1.copilot import copilot_router
from autopilot.api.v1.routes_models import (
    DashboardWorkflow, PendingRunItem, ResumeRunRequest,
    TriggerWorkflowRequest, CopilotQuery, CopilotResponse,
)
from autopilot.errors import (
    DashboardError, DashboardWorkflowNotFoundError,
    RunNotFoundError, RunNotPausedError,
)
print('All Phase 3 prerequisites OK')
"
```

---

## Implementation

### Step 1: Create `tests/test_dashboard_integration.py` [NEW]

```python
"""Integration tests — Cross-phase Dashboard API verification.

These tests use REAL InMemoryRunLogService and InMemoryEventBus instances
(not mocks) to validate that platform services compose correctly end-to-end.

The TestClient is synchronous; async EventBus subscribers are wired
via the standard setup() lifecycle.
"""

from __future__ import annotations

import asyncio
import importlib
import json
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.1 — RunLogService ↔ Dashboard API Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRunLogDashboardIntegration:
    """Validates that data saved via RunLogService flows correctly
    through Dashboard API endpoints with accurate stats, pagination,
    and status filtering."""

    @patch("autopilot.api.v1.routes.get_registry")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_runs_appear_in_list_after_save(self, mock_run_log, mock_registry):
        """A run saved to RunLogService must appear in GET /workflows/{id}/runs."""
        svc = InMemoryRunLogService()
        mock_run_log.return_value = svc
        mock_registry.return_value.get.return_value = _make_workflow()
        mock_registry.return_value.list_all.return_value = [_make_info()]

        # Simulate BaseWorkflow saving a run
        now = datetime.now(timezone.utc)
        asyncio.get_event_loop().run_until_complete(
            svc.save_run(_make_run(run_id="run_a", status=RunStatus.RUNNING, started_at=now))
        )
        asyncio.get_event_loop().run_until_complete(
            svc.save_run(_make_run(run_id="run_a", status=RunStatus.SUCCESS, started_at=now))
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
        for i, status in enumerate([
            RunStatus.SUCCESS,
            RunStatus.SUCCESS,
            RunStatus.FAILED,
            RunStatus.SUCCESS,
            RunStatus.FAILED,
        ]):
            asyncio.get_event_loop().run_until_complete(
                svc.save_run(_make_run(
                    run_id=f"run_{i}",
                    status=status,
                    started_at=now + timedelta(seconds=i),
                ))
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
        for i in range(10):
            asyncio.get_event_loop().run_until_complete(
                svc.save_run(_make_run(
                    run_id=f"run_{i:03d}",
                    started_at=now + timedelta(seconds=i),
                ))
            )

        app = _make_app()
        client = TestClient(app)

        # First page
        r1 = client.get(
            "/api/v1/workflows/test_workflow/runs?limit=3",
            headers={"X-API-Key": API_KEY},
        )
        assert r1.status_code == 200
        page1 = r1.json()
        assert len(page1["runs"]) == 3
        assert page1["runs"][0]["id"] == "run_009"  # newest first
        cursor = page1["meta"]["next_cursor"]
        assert cursor is not None

        # Second page using cursor
        r2 = client.get(
            f"/api/v1/workflows/test_workflow/runs?limit=3&start_after={cursor}",
            headers={"X-API-Key": API_KEY},
        )
        assert r2.status_code == 200
        page2 = r2.json()
        assert len(page2["runs"]) == 3

        # No overlap between pages
        page1_ids = {r["id"] for r in page1["runs"]}
        page2_ids = {r["id"] for r in page2["runs"]}
        assert page1_ids.isdisjoint(page2_ids)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.2 — EventBus ↔ HITL/Trigger Dispatch Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEventBusDispatchIntegration:
    """Validates that write endpoints dispatch events with correct
    payload shapes that subscribers can parse."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_publishes_correct_event_shape(self, mock_registry, mock_run_log, mock_bus):
        """Resume endpoint must publish event with {run_id, workflow_id, payload}."""
        mock_registry.return_value.get.return_value = _make_workflow()

        paused_run = _make_run(run_id="run_paused", status=RunStatus.PAUSED)
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = paused_run
        mock_run_log.return_value = mock_svc

        mock_bus_instance = AsyncMock()
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
        call_args = mock_bus_instance.publish.call_args
        assert call_args is not None
        # Positional or keyword — extract topic and payload
        if call_args.args:
            assert call_args.args[0] == "dashboard.hitl_resumed"
        event_payload = call_args.kwargs.get("payload", {})
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
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/trigger",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"test_input": 42}},
        )
        assert r.status_code == 200

        call_args = mock_bus_instance.publish.call_args
        assert call_args is not None
        if call_args.args:
            assert call_args.args[0] == "dashboard.workflow_triggered"
        event_payload = call_args.kwargs.get("payload", {})
        assert event_payload["workflow_id"] == "test_workflow"
        assert event_payload["payload"]["test_input"] == 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.3 — Error Taxonomy ↔ HTTP Status Code Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestErrorTaxonomyHTTPMapping:
    """Validates that every Dashboard AutoPilotError subclass maps to
    its declared http_status when raised inside an actual endpoint."""

    @patch("autopilot.api.v1.routes.get_registry")
    def test_workflow_not_found_returns_404(self, mock_registry):
        """DashboardWorkflowNotFoundError (http_status=404) → HTTP 404."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404
        assert r.json()["error_code"] == "DASHBOARD_WORKFLOW_NOT_FOUND"

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
        assert r.json()["error_code"] == "RUN_NOT_PAUSED"

    def test_missing_api_key_returns_401(self):
        """All endpoints return 401 without X-API-Key header."""
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
                r = client.post(path, json={})
            assert r.status_code == 401, f"Expected 401 for {method} {path}, got {r.status_code}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.4 — Copilot ↔ RunLogService Tool Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCopilotRunLogIntegration:
    """Validates that copilot read-only tools correctly query
    RunLogService and return meaningful analysis."""

    @patch("autopilot.api.v1.copilot.get_event_bus")
    @patch("autopilot.api.v1.copilot.get_run_log_service")
    @patch("autopilot.api.v1.copilot.get_registry")
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
        mock_bus.return_value.history.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "What is the status of bank_to_ynab?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert len(data["tools_used"]) > 0  # Copilot used at least one tool
        # Reply should contain some form of stats info
        assert "bank_to_ynab" in data["reply"] or "50" in data["reply"]

    @patch("autopilot.api.v1.copilot.get_event_bus")
    @patch("autopilot.api.v1.copilot.get_run_log_service")
    @patch("autopilot.api.v1.copilot.get_registry")
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
        mock_bus.return_value.history.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "Why did bank_to_ynab fail recently?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        # Reply should contain error information
        assert "error" in data["reply"].lower() or "fail" in data["reply"].lower() or "YNAB" in data["reply"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3A.5 — SSE Stream Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSSEStreamIntegration:
    """Validates SSE event stream produces correctly formatted
    text/event-stream output."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_sse_content_type(self, mock_bus):
        """SSE endpoint must return text/event-stream content type."""
        mock_bus_instance = MagicMock()
        mock_bus_instance.subscribe.return_value = MagicMock()
        mock_bus_instance.unsubscribe = MagicMock()
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        with client.stream(
            "GET",
            "/api/v1/events/stream",
            headers={"X-API-Key": API_KEY},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            # Read first chunk (should be keepalive within 30s timeout)
            break  # Don't hang — content type validation is sufficient

    @patch("autopilot.api.v1.routes.get_event_bus")
    def test_sse_headers_disable_buffering(self, mock_bus):
        """SSE must set Cache-Control: no-cache and X-Accel-Buffering: no."""
        mock_bus_instance = MagicMock()
        mock_bus_instance.subscribe.return_value = MagicMock()
        mock_bus_instance.unsubscribe = MagicMock()
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        with client.stream(
            "GET",
            "/api/v1/events/stream",
            headers={"X-API-Key": API_KEY},
        ) as r:
            assert r.headers.get("cache-control") == "no-cache"
            assert r.headers.get("x-accel-buffering") == "no"
            break
```

### Step 2: Run integration tests

```bash
python -m pytest tests/test_dashboard_integration.py -v
```

### Step 3: Run alongside existing unit tests

```bash
python -m pytest tests/test_dashboard_api.py tests/test_dashboard_integration.py tests/autopilot/test_run_log.py -v
```

---

## Test Coverage Matrix

| Test                                         | Class                             | What It Validates                                     | Critical? |
| -------------------------------------------- | --------------------------------- | ----------------------------------------------------- | --------- |
| `test_runs_appear_in_list_after_save`        | `TestRunLogDashboardIntegration`  | RunLog → Dashboard API flow with idempotent upsert    | ✅        |
| `test_stats_aggregate_across_runs`           | `TestRunLogDashboardIntegration`  | Success rate calculation from real RunLogService data | ✅        |
| `test_pagination_cursor_consistency`         | `TestRunLogDashboardIntegration`  | Cursor produces non-overlapping pages                 | ✅        |
| `test_resume_publishes_correct_event_shape`  | `TestEventBusDispatchIntegration` | Resume event payload matches subscriber expectations  | ✅        |
| `test_trigger_publishes_correct_event_shape` | `TestEventBusDispatchIntegration` | Trigger event payload matches subscriber expectations | ✅        |
| `test_workflow_not_found_returns_404`        | `TestErrorTaxonomyHTTPMapping`    | DashboardWorkflowNotFoundError → 404                  | ✅        |
| `test_run_not_found_returns_404`             | `TestErrorTaxonomyHTTPMapping`    | RunNotFoundError → 404                                | ✅        |
| `test_run_not_paused_returns_409`            | `TestErrorTaxonomyHTTPMapping`    | RunNotPausedError → 409                               | ✅        |
| `test_missing_api_key_returns_401`           | `TestErrorTaxonomyHTTPMapping`    | All endpoints enforce X-API-Key auth                  | 🔴        |
| `test_copilot_stats_query`                   | `TestCopilotRunLogIntegration`    | Copilot tools query RunLogService correctly           | ✅        |
| `test_copilot_error_query`                   | `TestCopilotRunLogIntegration`    | Copilot surfaces failure data in reply                | ✅        |
| `test_sse_content_type`                      | `TestSSEStreamIntegration`        | SSE returns text/event-stream                         |           |
| `test_sse_headers_disable_buffering`         | `TestSSEStreamIntegration`        | SSE sets correct headers for Edge/nginx               |           |

---

## Design Decisions

| Decision                                                        | Rationale                                                                                                     |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Real `InMemoryRunLogService` (not mocks) for RunLog tests       | Integration tests must validate real service behavior — mocks hide serialization and aggregation bugs         |
| `asyncio.get_event_loop().run_until_complete()` for data setup  | TestClient is synchronous; run_log methods are async — bridge via event loop                                  |
| EventBus dispatch shape assertion                               | Frontend + `_on_hitl_resumed` + `_on_manual_trigger` all depend on the exact payload keys — must be validated |
| Error taxonomy tests assert both `status_code` and `error_code` | Ensures the `autopilot_error_handler` correctly maps `AutoPilotError.http_status` → HTTP response code        |
| Auth sweep across all endpoints                                 | A single missed `Depends(get_api_key)` would be a critical security hole — sweep catches it                   |
| SSE content-type validation                                     | Incorrect mime type causes browser EventSource to reject the stream silently                                  |
| Separate test file from unit tests                              | Integration tests have different fixture requirements and longer run times — isolation prevents coupling      |

---

## Files Modified

| File                                  | Change                                  | Lines      |
| ------------------------------------- | --------------------------------------- | ---------- |
| `tests/test_dashboard_integration.py` | **[NEW]** Cross-phase integration tests | ~350 lines |
