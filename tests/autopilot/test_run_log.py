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
            await svc.save_run(
                _make_run(
                    run_id=f"run_{i:03d}",
                    started_at=now + timedelta(seconds=i),
                )
            )
        runs, cursor = await svc.list_runs("test_workflow", limit=3)
        assert len(runs) == 3
        assert runs[0].id == "run_004"  # newest first
        assert cursor is not None  # more results available

    @pytest.mark.asyncio
    async def test_list_runs_pagination_cursor(self, svc):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await svc.save_run(
                _make_run(
                    run_id=f"run_{i:03d}",
                    started_at=now + timedelta(seconds=i),
                )
            )
        # First page
        page1, cursor1 = await svc.list_runs("test_workflow", limit=2)
        assert len(page1) == 2
        assert cursor1 is not None
        # Second page
        page2, cursor2 = await svc.list_runs(
            "test_workflow", limit=2, start_after=cursor1
        )
        assert len(page2) == 2
        # Third page (last)
        page3, cursor3 = await svc.list_runs(
            "test_workflow", limit=2, start_after=cursor2
        )
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
        await svc.save_run(
            _make_run(run_id="r3", status=RunStatus.PAUSED, workflow_id="other")
        )

        pending = await svc.get_pending_runs()
        assert len(pending) == 2
        assert all(r.status == RunStatus.PAUSED for r in pending)

    @pytest.mark.asyncio
    async def test_get_latest_run(self, svc):
        now = datetime.now(timezone.utc)
        await svc.save_run(_make_run(run_id="r1", started_at=now))
        await svc.save_run(
            _make_run(run_id="r2", started_at=now + timedelta(seconds=1))
        )

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
