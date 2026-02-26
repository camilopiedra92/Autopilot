# Phase 2E. Unit Tests — Phase 2 API Verification

> **Status**: ✅ COMPLETED  
> **Effort**: ~40 min  
> **Type**: MODIFY (Tests)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 2  
> **Depends on**: Phase 2A-2D (all Phase 2 endpoints and models)

---

## Problem Statement

Phase 2 introduces 4 new endpoints with complex validation logic:

- HITL pending runs listing (read, cross-workflow)
- HITL resume (write, state validation, EventBus dispatch)
- Manual workflow trigger (write, enabled check, EventBus dispatch)
- Copilot ask (ReAct agent execution)

Without tests:

- State validation (`PAUSED` check on resume) is unverified.
- EventBus dispatch behavior is assumed.
- Error taxonomy integration (409 Conflict for `RunNotPausedError`) is untested.
- Copilot tool routing logic is uncovered.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                   | Current                           | Target                                 |
| ----------------------- | --------------------------------------------- | --------------------------------- | -------------------------------------- |
| §9.4 Development Rules  | All new modules have tests                    | No Phase 2 tests                  | Test file covering all 4 new endpoints |
| §6 Error Taxonomy       | `AutoPilotError` subclasses map to HTTP codes | `RunNotPausedError` untested      | Tests verify 409 for not-paused resume |
| Convention              | Copy test patterns from Phase 1E              | Phase 1 test patterns established | Same `TestClient` + mock pattern       |

---

## Implementation

### Step 1: Add Phase 2 tests to `tests/test_dashboard_api.py` [MODIFY]

Append the following test classes to the **existing** `tests/test_dashboard_api.py`:

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Phase 2 Tests — HITL, Trigger, Copilot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from autopilot.models import RunStatus, WorkflowRun
from datetime import datetime, timezone


class TestHITLPendingRuns:
    """Tests for GET /api/v1/runs/pending-action."""

    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_pending_runs_empty(self, mock_run_log):
        """Verify empty list when no runs are paused."""
        mock_svc = AsyncMock()
        mock_svc.get_pending_runs.return_value = []
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/runs/pending-action",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["runs"] == []
        assert data["total"] == 0

    @patch("autopilot.api.v1.routes.get_run_log_service")
    def test_pending_runs_with_paused(self, mock_run_log):
        """Verify paused runs are returned as PendingRunItem projections."""
        paused_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RunStatus.PAUSED,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
            result={"__steps_completed__": ["step_a", "step_b"]},
        )
        mock_svc = AsyncMock()
        mock_svc.get_pending_runs.return_value = [paused_run]
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.get(
            "/api/v1/runs/pending-action",
            headers={"X-API-Key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["runs"][0]["run_id"] == "run_abc123"
        assert data["runs"][0]["status"] == "paused"
        assert data["runs"][0]["paused_step"] == "step_b"

    def test_pending_runs_requires_auth(self):
        """Verify 401 without X-API-Key header."""
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/v1/runs/pending-action")
        assert r.status_code == 401


class TestHITLResume:
    """Tests for POST /api/v1/workflows/{id}/runs/{run_id}/resume."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_success(self, mock_registry, mock_run_log, mock_bus):
        """Verify successful resume dispatches event and returns 200."""
        mock_registry.return_value.get.return_value = _make_workflow()

        paused_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RunStatus.PAUSED,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = paused_run
        mock_run_log.return_value = mock_svc

        mock_bus_instance = AsyncMock()
        mock_bus.return_value = mock_bus_instance

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {"approved": True}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "resuming"
        assert data["run_id"] == "run_abc123"
        assert data["workflow_id"] == "test_workflow"

        # Verify EventBus publish was called
        mock_bus_instance.publish.assert_called_once()

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_not_paused(self, mock_registry, mock_run_log):
        """Verify 409 when run is not in PAUSED state."""
        mock_registry.return_value.get.return_value = _make_workflow()

        running_run = WorkflowRun(
            id="run_abc123",
            workflow_id="test_workflow",
            status=RunStatus.RUNNING,
            trigger_type=TriggerType.MANUAL,
            started_at=datetime.now(timezone.utc),
        )
        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = running_run
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/run_abc123/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 409

    @patch("autopilot.api.v1.routes.get_run_log_service")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_run_not_found(self, mock_registry, mock_run_log):
        """Verify 404 when run doesn't exist."""
        mock_registry.return_value.get.return_value = _make_workflow()

        mock_svc = AsyncMock()
        mock_svc.get_run.return_value = None
        mock_run_log.return_value = mock_svc

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/runs/nonexistent/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 404

    @patch("autopilot.api.v1.routes.get_registry")
    def test_resume_workflow_not_found(self, mock_registry):
        """Verify 404 when workflow doesn't exist."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/nonexistent/runs/run_abc/resume",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 404


class TestManualTrigger:
    """Tests for POST /api/v1/workflows/{id}/trigger."""

    @patch("autopilot.api.v1.routes.get_event_bus")
    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_success(self, mock_registry, mock_bus):
        """Verify successful trigger dispatches event and returns 200."""
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
            json={"payload": {"test_key": "test_value"}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "dispatched"
        assert data["workflow_id"] == "test_workflow"

        # Verify EventBus publish was called
        mock_bus_instance.publish.assert_called_once()

    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_disabled_workflow(self, mock_registry):
        """Verify error when workflow is disabled."""
        wf = _make_workflow()
        wf.manifest.enabled = False
        mock_registry.return_value.get.return_value = wf

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/test_workflow/trigger",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 500  # DashboardError maps to 500

    @patch("autopilot.api.v1.routes.get_registry")
    def test_trigger_workflow_not_found(self, mock_registry):
        """Verify 404 for non-existent workflow."""
        mock_registry.return_value.get.return_value = None
        mock_registry.return_value.list_all.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/workflows/nonexistent/trigger",
            headers={"X-API-Key": API_KEY},
            json={"payload": {}},
        )
        assert r.status_code == 404


class TestCopilot:
    """Tests for POST /api/v1/copilot/ask."""

    @patch("autopilot.api.v1.copilot.get_event_bus")
    @patch("autopilot.api.v1.copilot.get_run_log_service")
    @patch("autopilot.api.v1.copilot.get_registry")
    def test_copilot_ask(self, mock_registry, mock_run_log, mock_bus):
        """Verify copilot returns a structured response."""
        mock_registry.return_value.list_all.return_value = [_make_info()]

        mock_svc = AsyncMock()
        mock_svc.get_stats.return_value = {"total": 10, "successful": 8}
        mock_svc.list_runs.return_value = ([], None)
        mock_run_log.return_value = mock_svc

        mock_bus.return_value._history = {}
        mock_bus.return_value.history.return_value = []

        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "What is the platform overview?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert "tools_used" in data
        assert "iterations" in data

    def test_copilot_query_too_short(self):
        """Verify 422 when query is too short (min_length=3)."""
        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            headers={"X-API-Key": API_KEY},
            json={"query": "hi"},
        )
        assert r.status_code == 422

    def test_copilot_requires_auth(self):
        """Verify 401 without X-API-Key header."""
        app = _make_app()
        client = TestClient(app)
        r = client.post(
            "/api/v1/copilot/ask",
            json={"query": "What is the platform overview?"},
        )
        assert r.status_code == 401
```

### Step 2: Run Phase 2 tests

```bash
python -m pytest tests/test_dashboard_api.py -v -k "Phase2 or HITL or Trigger or Copilot"
```

### Step 3: Run all dashboard tests

```bash
python -m pytest tests/test_dashboard_api.py -v
```

### Step 4: Run regression

```bash
python -m pytest tests/ -v --ignore=tests/autopilot/test_btc_strategy.py
```

### Step 5: Import smoke tests

```bash
python -c "
from autopilot.api.v1.routes_models import (
    PendingRunItem, ResumeRunRequest, ResumeRunResponse,
    TriggerWorkflowRequest, TriggerWorkflowResponse,
    CopilotQuery, CopilotResponse, CopilotToolCall,
)
print('Phase 2 Models OK')
"
python -c "from autopilot.api.v1.copilot import copilot_router; print('Copilot Router OK')"
python -c "from autopilot.errors import RunNotPausedError; print('Errors OK')"
python -c "from autopilot.api.v1.routes import router; print(f'{len(router.routes)} total routes')"
```

---

## Test Coverage Matrix

| Test                              | Class                 | What It Validates                                     | Critical? |
| --------------------------------- | --------------------- | ----------------------------------------------------- | --------- |
| `test_pending_runs_empty`         | `TestHITLPendingRuns` | Empty list when no runs are paused                    |           |
| `test_pending_runs_with_paused`   | `TestHITLPendingRuns` | PendingRunItem projection with paused_step extraction | ✅        |
| `test_pending_runs_requires_auth` | `TestHITLPendingRuns` | 401 without X-API-Key                                 | 🔴        |
| `test_resume_success`             | `TestHITLResume`      | 200 + EventBus publish on valid resume                | ✅        |
| `test_resume_not_paused`          | `TestHITLResume`      | 409 via `RunNotPausedError`                           | ✅        |
| `test_resume_run_not_found`       | `TestHITLResume`      | 404 via `RunNotFoundError`                            | ✅        |
| `test_resume_workflow_not_found`  | `TestHITLResume`      | 404 via `DashboardWorkflowNotFoundError`              | ✅        |
| `test_trigger_success`            | `TestManualTrigger`   | 200 + EventBus publish on valid trigger               | ✅        |
| `test_trigger_disabled_workflow`  | `TestManualTrigger`   | Error when workflow is disabled                       | ✅        |
| `test_trigger_workflow_not_found` | `TestManualTrigger`   | 404 for non-existent workflow                         | ✅        |
| `test_copilot_ask`                | `TestCopilot`         | Structured CopilotResponse with tools_used            | ✅        |
| `test_copilot_query_too_short`    | `TestCopilot`         | 422 for `min_length=3` Pydantic validation            |           |
| `test_copilot_requires_auth`      | `TestCopilot`         | 401 without X-API-Key (via parent router)             | 🔴        |

---

## Design Decisions

| Decision                                           | Rationale                                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Tests appended to existing `test_dashboard_api.py` | Single test file for all dashboard tests; shares fixtures and `_make_app()` factory                    |
| `AsyncMock` for `RunLogService` in HITL tests      | `get_pending_runs()` and `get_run()` are async methods — require `AsyncMock`                           |
| EventBus publish assertion on resume/trigger       | Verifies the critical dispatch happened (the actual workflow execution is EventBus consumer's concern) |
| No test for EventBus consumer (`_on_hitl_resumed`) | Consumer is a BaseWorkflow method; tested separately in workflow-level integration tests               |
| Copilot mock on `copilot` module, not `dashboard`  | Copilot tools import services at the module level; mocks must target `autopilot.api.v1.copilot`        |
| `test_copilot_query_too_short` validates Pydantic  | Ensures the `min_length=3` constraint on `CopilotQuery.query` is enforced                              |

---

## Files Modified

| File                          | Change                                  | Lines      |
| ----------------------------- | --------------------------------------- | ---------- |
| `tests/test_dashboard_api.py` | Add 4 test classes with 13 test methods | ~250 lines |
