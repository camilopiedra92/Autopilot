# Phase 0A. RunLogService — Durable Workflow Run History

> **Status**: ✅ COMPLETED  
> **Effort**: ~45 min  
> **Type**: NEW (Platform Infrastructure)  
> **Parent**: [dashboard-implementation.md](./dashboard-implementation.md) § Phase 0A  
> **Depends on**: Phase 0-PRE (`PAUSED` in `RunStatus`)

---

## Problem Statement

`BaseWorkflow` tracks run history **only in-memory** (`self._runs: list[WorkflowRun]`). When Cloud Run scales to zero, all run history is lost. The dashboard requires durable, queryable run history that survives container restarts.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                            | Current                          | Target                                      |
| ----------------------- | -------------------------------------- | -------------------------------- | ------------------------------------------- |
| §1 Core Philosophy      | Scale-to-zero safe                     | In-memory only — lost on restart | Firestore-backed with 12-Factor config      |
| §3.4 Session Service    | Backend-swappable services via env var | N/A                              | `RUN_LOG_BACKEND` (memory/firestore)        |
| §9.1 Observability      | OTel spans on I/O                      | No run log at all                | OTel spans on all methods                   |
| §9.4 Development Rules  | Protocol → Implementation pattern      | N/A                              | `RunLogProtocol` ABC → InMemory / Firestore |

---

## Prerequisites

### Phase 0-PRE: Add `PAUSED` to `RunStatus`

**File**: `autopilot/models.py`  
**Line**: 109-114

The `get_pending_runs()` method filters by `RunStatus.PAUSED`, which doesn't exist yet.

**Current code** (line 109-114):

```python
class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
```

**Replace with**:

```python
class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"  # HITL — waiting for human intervention
```

**Safety**: The `RunLogService` is brand new — no existing run data in any backend. Zero migration needed.

**Verify**:

```bash
python -c "from autopilot.models import RunStatus; assert hasattr(RunStatus, 'PAUSED'); print('PAUSED OK')"
```

---

## Implementation

### Step 1: Create `autopilot/core/run_log.py` [NEW]

Create this file with the **complete** contents below. This is the entire file — do not add anything extra.

```python
"""
RunLogService — Durable workflow run history.

Persists WorkflowRun objects across container restarts using a
swappable backend (12-Factor, RUN_LOG_BACKEND env var).

Backends:
  - "memory"    → InMemoryRunLogService (dev/test, zero deps)
  - "firestore" → FirestoreRunLogService (production, survives scale-to-zero)

Firestore document hierarchy::

    autopilot_runs/{workflow_id}               → { total: N, successful: N }
    └── runs/{run_id}                          → WorkflowRun.model_dump()

Usage::

    from autopilot.core.run_log import get_run_log_service

    svc = get_run_log_service()
    await svc.save_run(run)
    runs, cursor = await svc.list_runs("bank_to_ynab", limit=20)
    stats = await svc.get_stats("bank_to_ynab")
"""

from __future__ import annotations

import abc
import os
from collections import defaultdict
from typing import Any

import structlog
from opentelemetry import trace

from autopilot.models import RunStatus, WorkflowRun

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_ROOT_COLLECTION = "autopilot_runs"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Protocol — Abstract contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RunLogProtocol(abc.ABC):
    """Abstract contract for run log backends.

    All implementations must provide these async methods.
    Backend selection is config-driven via RUN_LOG_BACKEND env var.
    """

    @abc.abstractmethod
    async def save_run(self, run: WorkflowRun) -> None:
        """Persist a WorkflowRun. Must be idempotent (same run_id = upsert)."""
        ...

    @abc.abstractmethod
    async def list_runs(
        self, workflow_id: str, *, limit: int = 50, start_after: Any | None = None
    ) -> tuple[list[WorkflowRun], Any | None]:
        """List recent runs for a workflow, newest first.

        Returns:
            Tuple of (runs, next_cursor). `next_cursor` is an opaque token to be
            passed as `start_after` in the next call. None if no more results.
        """
        ...

    @abc.abstractmethod
    async def get_run(
        self, workflow_id: str, run_id: str
    ) -> WorkflowRun | None:
        """Get a specific run by ID. Returns None if not found."""
        ...

    @abc.abstractmethod
    async def get_stats(self, workflow_id: str) -> dict[str, int]:
        """Get aggregate stats: {"total": N, "successful": N}."""
        ...

    @abc.abstractmethod
    async def get_pending_runs(self) -> list[WorkflowRun]:
        """Get all runs globally that are currently in PAUSED/HITL state."""
        ...

    @abc.abstractmethod
    async def get_latest_run(self, workflow_id: str) -> WorkflowRun | None:
        """Get most recent run for a workflow. More efficient than list_runs(limit=1)."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  InMemoryRunLogService — Dev/Test backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InMemoryRunLogService(RunLogProtocol):
    """Dict-backed run log. Zero dependencies. For dev and test.

    Uses {run_id: WorkflowRun} dict for O(1) lookup and idempotent upsert.
    Stats are adjusted (not blindly incremented) on re-saves to prevent
    double-counting when BaseWorkflow.run() saves at both start and end.
    """

    def __init__(self) -> None:
        # workflow_id → {run_id → WorkflowRun}
        self._runs: dict[str, dict[str, WorkflowRun]] = defaultdict(dict)
        self._stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )

    async def save_run(self, run: WorkflowRun) -> None:
        with tracer.start_as_current_span(
            "run_log.save", attributes={"workflow_id": run.workflow_id, "run_id": run.id}
        ):
            wf_runs = self._runs[run.workflow_id]
            existing = wf_runs.get(run.id)

            if existing is None:
                # New run — increment total
                self._stats[run.workflow_id]["total"] += 1
                if run.status == RunStatus.SUCCESS:
                    self._stats[run.workflow_id]["successful"] += 1
            else:
                # Upsert — adjust successful count if status changed
                was_success = existing.status == RunStatus.SUCCESS
                is_success = run.status == RunStatus.SUCCESS
                if is_success and not was_success:
                    self._stats[run.workflow_id]["successful"] += 1
                elif was_success and not is_success:
                    self._stats[run.workflow_id]["successful"] -= 1

            wf_runs[run.id] = run

            # Cap at 200 entries (evict oldest by started_at)
            if len(wf_runs) > 200:
                oldest_id = min(wf_runs, key=lambda k: wf_runs[k].started_at)
                del wf_runs[oldest_id]

    async def list_runs(
        self, workflow_id: str, *, limit: int = 50, start_after: int | None = None
    ) -> tuple[list[WorkflowRun], int | None]:
        with tracer.start_as_current_span(
            "run_log.list", attributes={"workflow_id": workflow_id, "limit": limit}
        ):
            runs = list(self._runs.get(workflow_id, {}).values())
            # Newest first
            runs.sort(key=lambda r: r.started_at, reverse=True)

            start_idx = start_after or 0
            end_idx = start_idx + limit

            chunk = runs[start_idx:end_idx]
            next_cursor = end_idx if end_idx < len(runs) else None

            return chunk, next_cursor

    async def get_run(
        self, workflow_id: str, run_id: str
    ) -> WorkflowRun | None:
        with tracer.start_as_current_span(
            "run_log.get", attributes={"workflow_id": workflow_id, "run_id": run_id}
        ):
            return self._runs.get(workflow_id, {}).get(run_id)

    async def get_stats(self, workflow_id: str) -> dict[str, int]:
        with tracer.start_as_current_span(
            "run_log.stats", attributes={"workflow_id": workflow_id}
        ):
            return dict(self._stats.get(workflow_id, {"total": 0, "successful": 0}))

    async def get_pending_runs(self) -> list[WorkflowRun]:
        with tracer.start_as_current_span("run_log.get_pending_runs"):
            pending = []
            for wf_runs in self._runs.values():
                for run in wf_runs.values():
                    if run.status == RunStatus.PAUSED:
                        pending.append(run)
            return sorted(pending, key=lambda r: r.started_at, reverse=True)

    async def get_latest_run(self, workflow_id: str) -> WorkflowRun | None:
        with tracer.start_as_current_span(
            "run_log.get_latest", attributes={"workflow_id": workflow_id}
        ):
            wf_runs = self._runs.get(workflow_id, {})
            if not wf_runs:
                return None
            return max(wf_runs.values(), key=lambda r: r.started_at)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._runs.values())
        return f"InMemoryRunLogService(runs={total})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FirestoreRunLogService — Production backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FirestoreRunLogService(RunLogProtocol):
    """Firestore-backed run log for production Cloud Run deployments.

    Document hierarchy::

        autopilot_runs/{workflow_id}           → { total: N, successful: N }
        └── runs/{run_id}                      → WorkflowRun serialized

    Same patterns as FirestoreSessionService:
      - google.cloud.firestore.AsyncClient
      - from_env() factory for zero-config on Cloud Run
      - Transactional stats updates for atomicity

    Critical: save_run is IDEMPOTENT. Re-saving the same run_id adjusts
    stats (not double-counts) by checking if the doc already exists within
    the same transaction.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        root_collection: str = _ROOT_COLLECTION,
    ) -> None:
        from google.cloud import firestore

        self.client = firestore.AsyncClient(project=project)
        self.root_collection = root_collection

    @classmethod
    def from_env(cls) -> "FirestoreRunLogService":
        """Create from environment — zero-config on Cloud Run.

        Reads GOOGLE_CLOUD_PROJECT (auto-set on Cloud Run).
        """
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        return cls(project=project)

    # ── References ───────────────────────────────────────────────────

    def _workflow_ref(self, workflow_id: str):
        """Ref to the workflow stats document."""
        return self.client.collection(self.root_collection).document(workflow_id)

    def _run_ref(self, workflow_id: str, run_id: str):
        """Ref to a specific run document."""
        return (
            self._workflow_ref(workflow_id)
            .collection("runs")
            .document(run_id)
        )

    def _runs_collection(self, workflow_id: str):
        """Ref to the runs subcollection."""
        return self._workflow_ref(workflow_id).collection("runs")

    # ── Protocol implementation ──────────────────────────────────────

    async def save_run(self, run: WorkflowRun) -> None:
        with tracer.start_as_current_span(
            "run_log.save", attributes={"workflow_id": run.workflow_id, "run_id": run.id}
        ):
            from google.cloud import firestore as fs

            run_data = run.model_dump(mode="json")
            run_ref = self._run_ref(run.workflow_id, run.id)
            wf_ref = self._workflow_ref(run.workflow_id)

            @fs.async_transactional
            async def _save_with_stats(transaction):
                # 1. Check if run already exists (idempotent upsert)
                existing_doc = await run_ref.get(transaction=transaction)
                existing_status = None
                if existing_doc.exists:
                    existing_status = existing_doc.to_dict().get("status")

                # 2. Save the run document
                transaction.set(run_ref, run_data)

                # 3. Conditionally update stats
                stats_doc = await wf_ref.get(transaction=transaction)
                current = stats_doc.to_dict() if stats_doc.exists else {"total": 0, "successful": 0}

                if existing_status is None:
                    # New run — increment total
                    current["total"] = current.get("total", 0) + 1
                    if run.status.value == "success":
                        current["successful"] = current.get("successful", 0) + 1
                else:
                    # Upsert — adjust successful count only if status changed
                    was_success = existing_status == "success"
                    is_success = run.status.value == "success"
                    if is_success and not was_success:
                        current["successful"] = current.get("successful", 0) + 1
                    elif was_success and not is_success:
                        current["successful"] = max(0, current.get("successful", 0) - 1)

                transaction.set(wf_ref, current)

            transaction = self.client.transaction()
            await _save_with_stats(transaction)

            logger.debug(
                "run_log_saved",
                workflow_id=run.workflow_id,
                run_id=run.id,
                status=run.status.value,
            )

    async def list_runs(
        self, workflow_id: str, *, limit: int = 50, start_after: str | None = None
    ) -> tuple[list[WorkflowRun], str | None]:
        with tracer.start_as_current_span(
            "run_log.list", attributes={"workflow_id": workflow_id, "limit": limit}
        ):
            query = (
                self._runs_collection(workflow_id)
                .order_by("started_at", direction="DESCENDING")
                .limit(limit)
            )

            if start_after:
                from datetime import datetime, timezone
                try:
                    dt = datetime.fromisoformat(start_after)
                    query = query.start_after({"started_at": dt})
                except Exception:
                    pass

            runs: list[WorkflowRun] = []
            last_timestamp = None

            async for doc in query.stream():
                try:
                    data = doc.to_dict()
                    runs.append(WorkflowRun.model_validate(data))
                    last_timestamp = data.get("started_at")
                except Exception as exc:
                    logger.warning("run_log_parse_error", doc_id=doc.id, error=str(exc))

            next_cursor = last_timestamp if len(runs) == limit else None
            return runs, next_cursor

    async def get_run(
        self, workflow_id: str, run_id: str
    ) -> WorkflowRun | None:
        with tracer.start_as_current_span(
            "run_log.get", attributes={"workflow_id": workflow_id, "run_id": run_id}
        ):
            doc = await self._run_ref(workflow_id, run_id).get()
            if not doc.exists:
                return None
            return WorkflowRun.model_validate(doc.to_dict())

    async def get_stats(self, workflow_id: str) -> dict[str, int]:
        with tracer.start_as_current_span(
            "run_log.stats", attributes={"workflow_id": workflow_id}
        ):
            doc = await self._workflow_ref(workflow_id).get()
            if not doc.exists:
                return {"total": 0, "successful": 0}
            data = doc.to_dict()
            return {
                "total": data.get("total", 0),
                "successful": data.get("successful", 0),
            }

    async def get_pending_runs(self) -> list[WorkflowRun]:
        with tracer.start_as_current_span("run_log.get_pending_runs"):
            query = self.client.collection_group("runs").where(
                field_path="status", op_string="==", value="paused"
            ).order_by("started_at", direction="DESCENDING")

            runs: list[WorkflowRun] = []
            async for doc in query.stream():
                try:
                    runs.append(WorkflowRun.model_validate(doc.to_dict()))
                except Exception as exc:
                    logger.warning("run_log_parse_error", doc_id=doc.id, error=str(exc))
            return runs

    async def get_latest_run(self, workflow_id: str) -> WorkflowRun | None:
        with tracer.start_as_current_span(
            "run_log.get_latest", attributes={"workflow_id": workflow_id}
        ):
            query = (
                self._runs_collection(workflow_id)
                .order_by("started_at", direction="DESCENDING")
                .limit(1)
            )
            async for doc in query.stream():
                return WorkflowRun.model_validate(doc.to_dict())
            return None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        self.client.close()

    def __repr__(self) -> str:
        return (
            f"FirestoreRunLogService("
            f"project={self.client.project!r}, "
            f"collection={self.root_collection!r})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Factory + Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_run_log_service(backend: str | None = None) -> RunLogProtocol:
    """Factory for creating the appropriate run log backend.

    Backend selection follows 12-Factor App (Factor III: Config):
      - "memory" (default): In-memory for dev/test
      - "firestore": Firestore for production (durable)

    Args:
        backend: Override backend choice. Defaults to RUN_LOG_BACKEND
                 env var, falling back to "memory".
    """
    backend = backend or os.getenv("RUN_LOG_BACKEND", "memory")
    logger.info("run_log_backend_selected", backend=backend)

    if backend == "firestore":
        return FirestoreRunLogService.from_env()

    return InMemoryRunLogService()


_run_log: RunLogProtocol | None = None


def get_run_log_service() -> RunLogProtocol:
    """Get or create the global RunLogService singleton."""
    global _run_log
    if _run_log is None:
        _run_log = create_run_log_service()
    return _run_log


def reset_run_log_service() -> None:
    """Reset the singleton. For tests."""
    global _run_log
    _run_log = None
```

### Step 2: Verify compilation

```bash
python -c "from autopilot.core.run_log import RunLogProtocol, InMemoryRunLogService, get_run_log_service; print('OK')"
```

---

## Design Decisions

| Decision                                               | Rationale                                                                                                                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dict-backed (`{run_id: run}`) over list-append         | O(1) lookup + idempotent upsert prevents stat double-counting                                                                                                  |
| OTel spans on InMemory backend                         | Observability parity with Firestore — dev traces match prod                                                                                                    |
| `get_latest_run()` as separate method                  | Dedicated Firestore query `.limit(1)` is cheaper than `list_runs(limit=1)` which returns a tuple                                                               |
| Idempotent `save_run` with conditional stat adjustment | `BaseWorkflow.run()` calls `save_run` twice per lifecycle (RUNNING → final status). Naively incrementing `total` each time would produce `total=2` for one run |
| Ring-buffer cap at 200                                 | Prevents unbounded memory growth in InMemory backend. Evicts oldest, not newest                                                                                |
| Module-level singleton with `reset_run_log_service()`  | Same pattern as `session_service` — singleton for production, resettable for isolated tests                                                                    |

---

## Files Modified

| File                        | Change                                  | Lines      |
| --------------------------- | --------------------------------------- | ---------- |
| `autopilot/models.py`       | Add `PAUSED = "paused"` to `RunStatus`  | 1 line     |
| `autopilot/core/run_log.py` | **[NEW]** Complete RunLogService module | ~350 lines |
