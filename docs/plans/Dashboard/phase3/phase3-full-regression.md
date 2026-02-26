# Phase 3D. Full Regression & Smoke Tests — Dashboard Completeness Gate

> **Status**: ✅ COMPLETE (2026-02-26)  
> **Effort**: ~20 min  
> **Type**: VERIFICATION (Quality Gate)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 3  
> **Depends on**: Phase 3A (Integration Tests), Phase 3B (Deployment), Phase 3C (SSE/Edge)

---

## Problem Statement

The Dashboard spans 4 phases (0–2 implementation, 3 verification), 8+ new files, and 4 modified platform files. Before declaring the Dashboard feature complete, a **comprehensive regression gate** must confirm that:

1. **No existing platform tests broke** — Dashboard changes to `BaseWorkflow.setup()`, `BaseWorkflow.run()`, `errors.py`, and `models.py` could silently break non-dashboard workflows.
2. **All new modules import cleanly** — No circular imports, missing dependencies, or syntax errors.
3. **All new endpoints respond** — Every route in the dashboard router returns a valid response (not 500) when called with correct auth and expected input.
4. **Dockerfile builds** — The explicit COPY allowlist in the Dockerfile includes all new files.

This is the **final quality gate** before the Dashboard is declared production-ready.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                              | Current         | Target                                       |
| ----------------------- | ---------------------------------------- | --------------- | -------------------------------------------- |
| §9.4 Development Rules  | All new modules have tests               | Phase-by-phase  | Full-suite regression validates all together |
| §10.4 Deployment        | Dockerfile explicit COPY — no `COPY . .` | Assumed correct | Docker build succeeds with all new files     |
| Convention              | Platform changes must not regress        | Assumed safe    | Existing test suites pass after all changes  |

---

## Verification Steps

### Step 1: Import Smoke Tests

Verify every new module imports without error. This catches circular imports, missing `__init__.py` exports, and syntax errors.

```bash
python -c "
# Phase 0 — Platform Durability
from autopilot.models import RunStatus
assert hasattr(RunStatus, 'PAUSED'), 'PAUSED missing from RunStatus'

from autopilot.core.run_log import (
    RunLogProtocol,
    InMemoryRunLogService,
    FirestoreRunLogService,
    create_run_log_service,
    get_run_log_service,
    reset_run_log_service,
)

from autopilot.errors import (
    RunLogError,
    DashboardError,
    DashboardWorkflowNotFoundError,
    RunNotFoundError,
    RunNotPausedError,
)

# Phase 1 — Dashboard API
from autopilot.api.v1.routes_models import (
    PipelineNode, PipelineEdge, PipelineGraph,
    DashboardWorkflow, TokenMetrics,
    RunStepTrace, RunTrace,
    PaginationMeta, PaginatedRuns,
    AgentCardResponse, EventItem,
)

from autopilot.api.v1.routes import router

# Phase 2 — Super Agentic Endpoints
from autopilot.api.v1.routes_models import (
    PendingRunItem, ResumeRunRequest, ResumeRunResponse,
    TriggerWorkflowRequest, TriggerWorkflowResponse,
    CopilotQuery, CopilotResponse, CopilotToolCall,
)

from autopilot.api.v1.copilot import copilot_router

# Verify route counts
dashboard_routes = len(router.routes)
copilot_routes = len(copilot_router.routes)
print(f'✅ All imports OK — Dashboard: {dashboard_routes} routes, Copilot: {copilot_routes} routes')
"
```

### Step 2: Full Unit Test Suite

Run **all** unit tests across all phases:

```bash
# Phase 0 — RunLogService
python -m pytest tests/autopilot/test_run_log.py -v

# Phases 1–2 — Dashboard API endpoints
python -m pytest tests/test_dashboard_api.py -v

# Phase 3 — Integration tests
python -m pytest tests/test_dashboard_integration.py -v
```

### Step 3: Platform Regression

Run **all** existing platform tests to confirm Dashboard changes didn't break anything:

```bash
python -m pytest tests/ -v \
  --ignore=tests/autopilot/test_btc_strategy.py \
  --tb=short \
  -q
```

**Critical tests to watch**:

| Test File                     | Why                                                     |
| ----------------------------- | ------------------------------------------------------- |
| `tests/test_base_workflow.py` | `BaseWorkflow.setup()` and `.run()` were modified in P0 |
| `tests/test_api_v1.py`        | `routes.py` was modified to mount dashboard router      |
| `tests/test_pipeline.py`      | HITL pause/resume touches Pipeline state                |

### Step 4: Dockerfile Build Verification

Ensure the Docker image builds successfully with all new files:

```bash
docker build -t autopilot:dashboard-test .
```

**Verify these files are included** (check Dockerfile COPY allowlist):

| File                                   | Phase | Type   |
| -------------------------------------- | ----- | ------ |
| `autopilot/core/run_log.py`            | 0A    | NEW    |
| `autopilot/api/v1/dashboard_models.py` | 1A    | NEW    |
| `autopilot/api/v1/dashboard.py`        | 1B    | NEW    |
| `autopilot/api/v1/copilot.py`          | 2D    | NEW    |
| `autopilot/errors.py`                  | 0B    | MODIFY |
| `autopilot/models.py`                  | 0-PRE | MODIFY |
| `autopilot/base_workflow.py`           | 0C    | MODIFY |
| `autopilot/api/v1/routes.py`           | 1C    | MODIFY |

If the Dockerfile uses explicit COPY lines (not `COPY . .`), ensure new files are covered:

```dockerfile
# Verify these lines exist (or are covered by a directory COPY):
COPY autopilot/core/run_log.py autopilot/core/run_log.py
COPY autopilot/api/v1/dashboard_models.py autopilot/api/v1/dashboard_models.py
COPY autopilot/api/v1/dashboard.py autopilot/api/v1/dashboard.py
COPY autopilot/api/v1/copilot.py autopilot/api/v1/copilot.py
```

### Step 5: Container Startup Smoke Test

```bash
# Start container locally and verify it boots
docker run --rm -e API_KEY_SECRET=test -p 8080:8080 autopilot:dashboard-test &
sleep 3

# Health check
curl -s -H "X-API-Key: test" http://localhost:8080/api/v1/health | jq .

# Cleanup
docker stop $(docker ps -q --filter ancestor=autopilot:dashboard-test)
```

---

## Final Regression Checklist

| #   | Category        | Check                                           | Pass? |
| --- | --------------- | ----------------------------------------------- | ----- |
| 1   | **Imports**     | All new modules import without error            | [ ]   |
| 2   | **Imports**     | `RunStatus.PAUSED` exists                       | [ ]   |
| 3   | **Imports**     | Dashboard router has expected route count       | [ ]   |
| 4   | **Unit Tests**  | `test_run_log.py` — all passing                 | [ ]   |
| 5   | **Unit Tests**  | `test_dashboard_api.py` — all passing           | [ ]   |
| 6   | **Integration** | `test_dashboard_integration.py` — all passing   | [ ]   |
| 7   | **Regression**  | `test_base_workflow.py` — no regressions        | [ ]   |
| 8   | **Regression**  | `test_api_v1.py` — no regressions               | [ ]   |
| 9   | **Regression**  | Full `tests/` suite — all passing               | [ ]   |
| 10  | **Docker**      | Dockerfile builds successfully                  | [ ]   |
| 11  | **Docker**      | Container starts and responds on `/health`      | [ ]   |
| 12  | **Deployment**  | Cloud Run deployment succeeds (Phase 3B)        | [ ]   |
| 13  | **SSE**         | SSE 5-min disconnect fires correctly (Phase 3C) | [ ]   |

---

## Completion Criteria

The Dashboard feature is **complete** when:

- [x] All Phase 0 items done (RunLogService, errors, BaseWorkflow integration)
- [x] All Phase 1 items done (Dashboard API, response models, SSE, route mounting)
- [x] All Phase 2 items done (HITL, Manual Trigger, Copilot)
- [ ] All Phase 3A integration tests pass
- [ ] All Phase 3B deployment checks pass
- [ ] All Phase 3C SSE/Edge checks pass
- [ ] All Phase 3D regression checks pass (this document)

---

## Summary of All Dashboard Files

| Action     | File                                   | Phase | Lines |
| ---------- | -------------------------------------- | ----- | ----- |
| **NEW**    | `autopilot/core/run_log.py`            | 0A    | ~400  |
| **MODIFY** | `autopilot/errors.py`                  | 0B    | ~25   |
| **MODIFY** | `autopilot/models.py`                  | 0-PRE | ~2    |
| **MODIFY** | `autopilot/base_workflow.py`           | 0C/2B | ~80   |
| **NEW**    | `autopilot/api/v1/dashboard_models.py` | 1A/2A | ~250  |
| **NEW**    | `autopilot/api/v1/dashboard.py`        | 1B/2B | ~350  |
| **MODIFY** | `autopilot/api/v1/routes.py`           | 1C/2D | ~5    |
| **NEW**    | `autopilot/api/v1/copilot.py`          | 2D    | ~300  |
| **NEW**    | `tests/autopilot/test_run_log.py`      | 0E    | ~100  |
| **NEW**    | `tests/test_dashboard_api.py`          | 1E/2E | ~350  |
| **NEW**    | `tests/test_dashboard_integration.py`  | 3A    | ~350  |
| **TOTAL**  |                                        |       | ~2200 |

---

## Design Decisions

| Decision                                      | Rationale                                                                                     |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Import smoke test as first gate               | Fastest possible validation — catches 80% of integration issues in <1 second                  |
| Separate integration tests from unit tests    | Different fixture requirements, longer run times, different failure semantics                 |
| Dockerfile build as explicit check            | Explicit COPY allowlist means every new file must be added — easy to miss                     |
| Container startup smoke test                  | Health endpoint validates that all services initialize without error                          |
| Checklist format (not automated script)       | Some checks require human judgment (e.g., Cloud Trace inspection); checklist is more flexible |
| Full `tests/` regression (not just dashboard) | `BaseWorkflow` changes affect ALL workflows — must prove nothing broke                        |
