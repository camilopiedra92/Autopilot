# Phase 1E. Unit Tests — Dashboard API Verification

> **Status**: ✅ COMPLETED  
> **Completed**: 2026-02-26  
> **Effort**: ~30 min  
> **Type**: NEW (Tests)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 1E  
> **Depends on**: Phase 1A (models), Phase 1B (router), Phase 1C (mounting)

---

## Problem Statement

The Dashboard API introduces 9 new endpoints, each composing data from multiple platform services (`WorkflowRegistry`, `RunLogService`, `ArtifactService`, `EventBus`). Without tests:

- Endpoint auth enforcement is unverified.
- Model serialization correctness is assumed.
- Error taxonomy integration (`DashboardWorkflowNotFoundError` → 404) is untested.
- Regressions from future pipeline.yaml schema changes would be silent.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                   | Current                     | Target                                    |
| ----------------------- | --------------------------------------------- | --------------------------- | ----------------------------------------- |
| §9.4 Development Rules  | All new modules have tests                    | No dashboard tests          | Test file with coverage for all endpoints |
| §6 Error Taxonomy       | `AutoPilotError` subclasses map to HTTP codes | Untested for dashboard      | Tests verify 404 for not-found errors     |
| Convention              | Copy test patterns from `test_api_v1.py`      | Existing patterns available | Same `TestClient` + mock pattern          |

---

## Implementation

### Step 1: Create `tests/test_dashboard_api.py` [NEW]

Create this file with the **complete** contents below.

```python
"""Tests for Dashboard API endpoints.

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


API_KEY = "test-dashboard-key"


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


class TestDashboardWorkflows:
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
        """Verify 404 with DashboardWorkflowNotFoundError."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/workflows/nonexistent",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 404


class TestDashboardPipeline:
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


class TestDashboardRuns:
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


class TestDashboardHealth:
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


class TestDashboardEvents:
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
```

### Step 2: Run tests

```bash
python -m pytest tests/test_dashboard_api.py -v
```

### Step 3: Run regression tests

Verify that existing platform tests still pass after the route mounting changes:

```bash
python -m pytest tests/test_api_v1.py -v
```

### Step 4: Full regression

```bash
python -m pytest tests/ -v --ignore=tests/autopilot/test_btc_strategy.py
```

### Step 5: Import smoke tests

```bash
python -c "from autopilot.api.v1.routes_models import DashboardWorkflow, PipelineGraph, RunTrace; print('Models OK')"
python -c "from autopilot.api.v1.routes import router; print('Router OK')"
python -c "from autopilot.api.v1.routes import router; print(f'{len(router.routes)} routes')"
```

---

## Test Coverage Matrix

| Test                           | Class                    | What It Validates                                 | Critical? |
| ------------------------------ | ------------------------ | ------------------------------------------------- | --------- |
| `test_list_requires_api_key`   | `TestDashboardWorkflows` | 401 without X-API-Key header                      | 🔴        |
| `test_list_workflows`          | `TestDashboardWorkflows` | 200 with enriched workflow data                   | ✅        |
| `test_workflow_not_found`      | `TestDashboardWorkflows` | 404 via `DashboardWorkflowNotFoundError`          | ✅        |
| `test_pipeline_not_found`      | `TestDashboardPipeline`  | 404 for non-existent workflow's pipeline          | ✅        |
| `test_pipeline_empty_workflow` | `TestDashboardPipeline`  | Empty graph for workflows without `pipeline.yaml` | ✅        |
| `test_list_runs_empty`         | `TestDashboardRuns`      | Empty run list + null cursor + zero stats         |           |
| `test_runs_workflow_not_found` | `TestDashboardRuns`      | 404 when listing runs for non-existent workflow   | ✅        |
| `test_health`                  | `TestDashboardHealth`    | Healthy status with workflow count and bus stats  | ✅        |
| `test_events_empty`            | `TestDashboardEvents`    | Empty events when EventBus has no history         |           |

---

## Design Decisions

| Decision                                                | Rationale                                                                                                |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| TestClient pattern (not `httpx.AsyncClient`)            | Matches existing `test_api_v1.py` patterns; synchronous tests are simpler                                |
| `@patch` on dashboard module, not global singletons     | Isolates mocks to dashboard scope; doesn't affect other test modules                                     |
| Auth test only on `list_workflows` (not every endpoint) | Auth is enforced at the V1 router level via `Depends`; testing one endpoint proves the dependency works  |
| `_make_app()` with `importlib.reload`                   | Ensures fresh route registration per test; prevents state leaking between tests                          |
| No SSE stream test                                      | SSE requires async generator testing with custom timeout handling; deferred to integration tests         |
| No artifact loading test                                | `get_run_trace` depends on `ArtifactService` which is mocked differently per backend; covered in Phase 3 |

---

## Files Modified

| File                          | Change                        | Lines      |
| ----------------------------- | ----------------------------- | ---------- |
| `tests/test_dashboard_api.py` | **[NEW]** Complete test suite | ~200 lines |
