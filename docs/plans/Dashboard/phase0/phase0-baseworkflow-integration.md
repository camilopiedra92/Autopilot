# Phase 0C. BaseWorkflow Integration — Durable Run Persistence

> **Status**: ✅ COMPLETED  
> **Effort**: ~20 min  
> **Type**: ADJUST (Platform Integration)  
> **Parent**: [dashboard-implementation.md](./dashboard-implementation.md) § Phase 0C  
> **Depends on**: Phase 0A (RunLogService must exist)

---

## Problem Statement

`BaseWorkflow` maintains run history **only in `self._runs`** (a Python list). Two problems:

1. **Cold start**: When Cloud Run scales up a new instance, `self._runs` is empty. `total_runs`, `success_rate`, and `last_run` all reset to zero.
2. **No persistence**: Completed runs are never written to durable storage. The future Dashboard API cannot query historical run data.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                   | Current                   | Target                                                  |
| ----------------------- | --------------------------------------------- | ------------------------- | ------------------------------------------------------- |
| §1 Core Philosophy      | Scale-to-zero safe                            | Run stats lost on restart | Stats hydrated from RunLogService on cold start         |
| §1 Core Philosophy      | No `asyncio.create_task` in ephemeral compute | N/A                       | Inline `await` (fire-and-forget pattern via try/except) |
| §9.4 Development Rules  | Errors logged, never swallowed silently       | N/A                       | `logger.warning` on persist failure                     |

---

## Implementation

### Step 1: Modify `BaseWorkflow.setup()` — Cold-start hydration

**File**: `autopilot/base_workflow.py`  
**Line**: 224-226

**Current code**:

```python
    async def setup(self) -> None:
        """Called once when the workflow is registered. Override for initialization."""
        pass
```

**Replace with**:

```python
    async def setup(self) -> None:
        """Lifecycle hook — called once after registration.

        Hydrates run stats from durable backend on cold start.
        Override in subclasses for workflow-specific setup (call super!).
        """
        try:
            from autopilot.core.run_log import get_run_log_service

            svc = get_run_log_service()
            stats = await svc.get_stats(self.manifest.name)
            self._total_runs = stats.get("total", 0)
            self._successful_runs = stats.get("successful", 0)

            # Load most recent run for last_run property (uses efficient get_latest_run)
            latest = await svc.get_latest_run(self.manifest.name)
            if latest:
                self._runs = [latest]
            logger.debug(
                "run_log_hydrated",
                workflow=self.manifest.name,
                total_runs=self._total_runs,
                successful_runs=self._successful_runs,
            )
        except Exception:
            logger.debug("run_log_hydration_skipped", workflow=self.manifest.name)
```

**Key design choices**:

- Uses `get_latest_run()` (single-doc fetch) instead of `list_runs(limit=1)` (which returns a tuple and requires destructuring)
- `try/except` wraps everything — hydration failure must **never** block workflow registration
- Subclasses that override `setup()` must call `await super().setup()` (documented in docstring)

---

### Step 2: Modify `BaseWorkflow.run()` — Durable persistence

**File**: `autopilot/base_workflow.py`  
**Line**: 411-428 (inside the `finally` block)

**Current code** (the full `finally` block):

```python
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            run.duration_ms = round(elapsed_ms, 2)
            run.completed_at = datetime.now(timezone.utc)
            self._total_runs += 1

            # Keep last 100 runs in memory
            self._runs.append(run)
            if len(self._runs) > 100:
                self._runs = self._runs[-100:]

            logger.info(
                "workflow_run_completed",
                workflow=self.manifest.name,
                run_id=run_id,
                status=run.status.value,
                duration_ms=run.duration_ms,
            )
```

**Add this block** after `self._runs = self._runs[-100:]` (line 420) and before `logger.info(...)` (line 422):

```python
            # Persist to durable backend (fire-and-forget — errors logged, not raised)
            try:
                from autopilot.core.run_log import get_run_log_service
                await get_run_log_service().save_run(run)
            except Exception as exc:
                logger.warning(
                    "run_log_persist_failed",
                    run_id=run.id,
                    workflow=self.manifest.name,
                    error=str(exc),
                )
```

**Why this placement**:

- After in-memory tracking (backward compatible — in-memory always works)
- Before the final log message (so the log accurately reflects persistence status)
- Inside the `finally` block (persists both successful and failed runs)

**Why inline `await`** instead of fire-and-forget `asyncio.create_task`:

- ARCHITECTURE.md §1 forbids `asyncio.create_task` in ephemeral compute
- Cloud Run may terminate the container after the response — background tasks risk losing data
- The Firestore write is fast (~10-50ms) and won't meaningfully delay the response

**Idempotency**: `save_run` is called once per lifecycle in the `finally` block. However, if a future change adds an earlier save (e.g., at `RUNNING` status), the idempotent upsert in `RunLogService` handles it — stats are adjusted, not double-counted.

### Step 3: Verify

```bash
python -m pytest tests/test_base_workflow.py tests/test_api_v1.py -v
```

---

## Files Modified

| File                         | Change                                       | Lines     |
| ---------------------------- | -------------------------------------------- | --------- |
| `autopilot/base_workflow.py` | Replace `setup()` with hydration logic       | ~20 lines |
| `autopilot/base_workflow.py` | Add persistence block in `run()`'s `finally` | ~10 lines |
