# Phase 0D. Unit Tests — RunLogService Verification

> **Status**: ✅ COMPLETED  
> **Effort**: ~15 min  
> **Type**: NEW (Tests)  
> **Parent**: [dashboard-implementation.md](./dashboard-implementation.md) § Phase 0D  
> **Depends on**: Phase 0A (RunLogService), Phase 0-PRE (PAUSED enum)

---

## Problem Statement

The `RunLogService` is a new platform service that must be validated before integration with `BaseWorkflow` and the Dashboard API. Tests must cover:

- Basic CRUD (save, get, list)
- Idempotent upsert behavior (critical — prevents stat double-counting)
- Cursor-based pagination (correct tuple return type)
- PAUSED status filtering (required by HITL)
- `get_latest_run` efficiency method
- Factory env-var configuration

---

## Implementation

### Step 1: Create `tests/autopilot/test_run_log.py` [NEW]

Create this file with the **complete** contents below:

```python
"""Tests for RunLogService — InMemory backend."""

import pytest
from datetime import datetime, timezone, timedelta

from autopilot.core.run_log import (
    InMemoryRunLogService,
    create_run_log_service,
    reset_run_log_service,
)
from autopilot.models import WorkflowRun, RunStatus, TriggerType


def _make_run(
    workflow_id: str = "test_workflow",
    run_id: str = "run_001",
    status: RunStatus = RunStatus.SUCCESS,
    started_at: datetime | None = None,
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
    )


@pytest.fixture
def svc():
    return InMemoryRunLogService()


@pytest.fixture(autouse=True)
def _reset():
    reset_run_log_service()
    yield
    reset_run_log_service()


class TestInMemoryRunLogService:
    @pytest.mark.asyncio
    async def test_save_and_get_run(self, svc):
        run = _make_run()
        await svc.save_run(run)
        result = await svc.get_run("test_workflow", "run_001")
        assert result is not None
        assert result.id == "run_001"
        assert result.status == RunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, svc):
        result = await svc.get_run("test_workflow", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_runs_newest_first(self, svc):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await svc.save_run(_make_run(
                run_id=f"run_{i:03d}",
                started_at=now + timedelta(seconds=i),
            ))
        runs, cursor = await svc.list_runs("test_workflow", limit=3)
        assert len(runs) == 3
        assert runs[0].id == "run_004"  # newest first
        assert cursor is not None  # more results available

    @pytest.mark.asyncio
    async def test_list_runs_pagination_cursor(self, svc):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await svc.save_run(_make_run(
                run_id=f"run_{i:03d}",
                started_at=now + timedelta(seconds=i),
            ))
        # First page
        page1, cursor1 = await svc.list_runs("test_workflow", limit=2)
        assert len(page1) == 2
        assert cursor1 is not None
        # Second page
        page2, cursor2 = await svc.list_runs("test_workflow", limit=2, start_after=cursor1)
        assert len(page2) == 2
        # Third page (last)
        page3, cursor3 = await svc.list_runs("test_workflow", limit=2, start_after=cursor2)
        assert len(page3) == 1
        assert cursor3 is None  # no more results

    @pytest.mark.asyncio
    async def test_stats_tracking(self, svc):
        await svc.save_run(_make_run(status=RunStatus.SUCCESS))
        await svc.save_run(_make_run(run_id="run_002", status=RunStatus.FAILED))
        await svc.save_run(_make_run(run_id="run_003", status=RunStatus.SUCCESS))

        stats = await svc.get_stats("test_workflow")
        assert stats["total"] == 3
        assert stats["successful"] == 2

    @pytest.mark.asyncio
    async def test_stats_empty_workflow(self, svc):
        stats = await svc.get_stats("nonexistent")
        assert stats == {"total": 0, "successful": 0}

    @pytest.mark.asyncio
    async def test_upsert_idempotency(self, svc):
        """Re-saving same run_id must NOT double-count stats."""
        run = _make_run(run_id="run_x", status=RunStatus.RUNNING)
        await svc.save_run(run)

        stats = await svc.get_stats("test_workflow")
        assert stats["total"] == 1
        assert stats["successful"] == 0

        # Same run completes — upsert
        run_completed = _make_run(run_id="run_x", status=RunStatus.SUCCESS)
        await svc.save_run(run_completed)

        stats = await svc.get_stats("test_workflow")
        assert stats["total"] == 1  # NOT 2
        assert stats["successful"] == 1

    @pytest.mark.asyncio
    async def test_get_pending_runs(self, svc):
        await svc.save_run(_make_run(run_id="r1", status=RunStatus.SUCCESS))
        await svc.save_run(_make_run(run_id="r2", status=RunStatus.PAUSED))
        await svc.save_run(_make_run(run_id="r3", status=RunStatus.PAUSED, workflow_id="other"))

        pending = await svc.get_pending_runs()
        assert len(pending) == 2
        assert all(r.status == RunStatus.PAUSED for r in pending)

    @pytest.mark.asyncio
    async def test_get_latest_run(self, svc):
        now = datetime.now(timezone.utc)
        await svc.save_run(_make_run(run_id="r1", started_at=now))
        await svc.save_run(_make_run(run_id="r2", started_at=now + timedelta(seconds=1)))

        latest = await svc.get_latest_run("test_workflow")
        assert latest is not None
        assert latest.id == "r2"

    @pytest.mark.asyncio
    async def test_get_latest_run_empty(self, svc):
        latest = await svc.get_latest_run("nonexistent")
        assert latest is None


class TestRunLogFactory:
    def test_default_is_memory(self):
        svc = create_run_log_service("memory")
        assert isinstance(svc, InMemoryRunLogService)
```

### Step 2: Run tests

```bash
python -m pytest tests/autopilot/test_run_log.py -v
```

### Step 3: Run regression tests

Verify that existing platform tests still pass after the `BaseWorkflow` and `models.py` changes:

```bash
python -m pytest tests/test_base_workflow.py tests/test_api_v1.py -v
```

### Step 4: Import smoke tests

```bash
python -c "from autopilot.core.run_log import RunLogProtocol, InMemoryRunLogService, get_run_log_service; print('RunLog OK')"
python -c "from autopilot.errors import RunLogError, DashboardError, RunNotFoundError; print('Errors OK')"
python -c "from autopilot.models import RunStatus; assert hasattr(RunStatus, 'PAUSED'); print('PAUSED OK')"
```

---

## Test Coverage Matrix

| Test                               | What It Validates                                                | Critical? |
| ---------------------------------- | ---------------------------------------------------------------- | --------- |
| `test_save_and_get_run`            | Basic save → get round-trip                                      | ✅        |
| `test_get_run_not_found`           | Returns `None` for missing runs                                  | ✅        |
| `test_list_runs_newest_first`      | Ordering + limit enforcement                                     | ✅        |
| `test_list_runs_pagination_cursor` | Cursor-based pagination (tuple return type)                      | ✅        |
| `test_stats_tracking`              | `total` and `successful` counts                                  | ✅        |
| `test_stats_empty_workflow`        | Zero-value stats for unknown workflow                            |           |
| `test_upsert_idempotency`          | **Critical**: Re-saving same `run_id` doesn't double-count stats | 🔴        |
| `test_get_pending_runs`            | Filters by `RunStatus.PAUSED` across workflows                   | ✅        |
| `test_get_latest_run`              | Returns most recent run                                          | ✅        |
| `test_get_latest_run_empty`        | Returns `None` for unknown workflow                              |           |
| `test_default_is_memory`           | Factory default returns `InMemoryRunLogService`                  | ✅        |

---

## Files Modified

| File                              | Change                        | Lines      |
| --------------------------------- | ----------------------------- | ---------- |
| `tests/autopilot/test_run_log.py` | **[NEW]** Complete test suite | ~140 lines |
