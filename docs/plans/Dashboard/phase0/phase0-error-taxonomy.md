# Phase 0B. Error Taxonomy — RunLog & Dashboard Errors

> **Status**: ✅ COMPLETED  
> **Effort**: ~10 min  
> **Type**: ADJUST (Platform Infrastructure)  
> **Parent**: [dashboard-implementation.md](./dashboard-implementation.md) § Phase 0B  
> **Depends on**: None

---

## Problem Statement

The error taxonomy in `autopilot/errors.py` has no error classes for the RunLog service or the Dashboard API. Structured errors are needed for:

- `RunLogService` failures (Firestore unavailable, serialization errors)
- Dashboard-specific lookup failures (workflow not found, run not found)

Without dedicated error classes, these would either use generic `Exception` (losing structured logging) or overload existing error types (violating the layered error hierarchy).

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                              | Current                            | Target                                  |
| ----------------------- | -------------------------------------------------------- | ---------------------------------- | --------------------------------------- |
| §9.3 Error Taxonomy     | Every error has `retryable`, `error_code`, `http_status` | No RunLog/Dashboard errors         | 4 new typed error classes               |
| §9.3 Error Taxonomy     | Hierarchy mirrors platform layers                        | Error hierarchy stops at A2A layer | Extended with RunLog + Dashboard layers |

---

## Implementation

### Step 1: Add error classes to `autopilot/errors.py`

**File**: `autopilot/errors.py`

#### 1a. Add to `__all__` (line 56-60)

**Current code** (line 56-60):

```python
    # A2A Protocol Layer (Phase 8)
    "A2AError",
    "A2AWorkflowNotFoundError",
    "A2ATaskNotFoundError",
]
```

**Replace with**:

```python
    # A2A Protocol Layer (Phase 8)
    "A2AError",
    "A2AWorkflowNotFoundError",
    "A2ATaskNotFoundError",
    # Run Log Layer (Dashboard Phase 0)
    "RunLogError",
    # Dashboard Layer (Dashboard Phase 0)
    "DashboardError",
    "DashboardWorkflowNotFoundError",
    "RunNotFoundError",
]
```

#### 1b. Add error classes at end of file (after line 511)

**Append** the following after the last line of the file:

```python


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Run Log Layer — Errors from durable run history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RunLogError(AutoPilotError):
    """Base for all run log service errors."""

    error_code = "RUN_LOG_ERROR"
    http_status = 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dashboard Layer — Errors from dashboard API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DashboardError(AutoPilotError):
    """Base for all dashboard API errors."""

    error_code = "DASHBOARD_ERROR"
    http_status = 500


class DashboardWorkflowNotFoundError(DashboardError):
    """Requested workflow not found in registry."""

    error_code = "DASHBOARD_WORKFLOW_NOT_FOUND"
    http_status = 404


class RunNotFoundError(DashboardError):
    """Specific run ID not found in the run log."""

    error_code = "RUN_NOT_FOUND"
    http_status = 404
```

### Step 2: Verify

```bash
python -c "from autopilot.errors import RunLogError, DashboardError, DashboardWorkflowNotFoundError, RunNotFoundError; print('OK')"
```

---

## Design Decisions

| Decision                                                                              | Rationale                                                                   |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `RunLogError` inherits directly from `AutoPilotError`                                 | RunLog is a platform service (like session/memory), not a dashboard feature |
| `DashboardError` inherits from `AutoPilotError`                                       | Dashboard is a separate layer, not a child of Pipeline or Agent             |
| `RunNotFoundError` inherits from `DashboardError`                                     | Run lookups happen through the Dashboard API, not the RunLog directly       |
| Both `DashboardWorkflowNotFoundError` and `RunNotFoundError` have `http_status = 404` | Standard REST — missing resources map to 404                                |

---

## Files Modified

| File                  | Change                                 | Lines     |
| --------------------- | -------------------------------------- | --------- |
| `autopilot/errors.py` | Add 4 error classes + update `__all__` | ~40 lines |
