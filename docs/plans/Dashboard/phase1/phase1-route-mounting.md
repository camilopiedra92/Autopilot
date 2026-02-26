# Phase 1C. Route Mounting & Configuration

> **Status**: ✅ COMPLETED  
> **Completed**: 2026-02-26  
> **Effort**: ~10 min  
> **Type**: ADJUST (API Layer + Config)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 1C & 1D  
> **Depends on**: Phase 1B (Dashboard Router)

---

## Problem Statement

The dashboard router exists as a standalone `APIRouter` object but is not mounted into the application. Additionally, `.dockerignore` and `.gitignore` need minor updates for any future frontend build artifacts (even though the frontend is external, these prevent accidental inclusion).

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                     | Current                      | Target                              |
| ----------------------- | ------------------------------- | ---------------------------- | ----------------------------------- |
| §1 Core Philosophy      | X-API-Key auth on all endpoints | V1 router has `Depends`      | Dashboard inherits from V1 router   |
| §10.4 Dockerfile        | Explicit COPY allowlist         | No dashboard paths in ignore | Add `dashboard/` to `.dockerignore` |

---

## Implementation

### Step 1: Modify `autopilot/api/v1/routes.py` — Mount dashboard router

**File**: `autopilot/api/v1/routes.py`  
**Line**: 12-18

**Current code** (lines 12-18):

```python
from autopilot.api.security import get_api_key
from autopilot.registry import get_registry
from autopilot.router import get_router

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_api_key)])
```

**Replace with**:

```python
from autopilot.api.security import get_api_key
from autopilot.api.v1.routes import router
from autopilot.registry import get_registry
from autopilot.router import get_router

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_api_key)])
# router merged into v1 router
```

**Key points**:

- `router` is added as a sub-router of the V1 router.
- It **inherits** the `X-API-Key` dependency from the parent router — no need to add auth again.
- Dashboard endpoints are accessible at `/api/v1/*`.
- The import is alphabetically sorted per Python convention.

### Step 2: Update `.dockerignore`

**File**: `.dockerignore`

**Append** the following line (if not already present):

```
dashboard/
```

**Why**: If a future external dashboard frontend is developed in the repo root, this prevents Docker from copying it into the backend container image.

### Step 3: Update `.gitignore`

**File**: `.gitignore`

**Append** the following lines (if not already present):

```
# Dashboard frontend (external — tracked in its own repo)
dashboard/node_modules/
dashboard/dist/
```

### Step 4: Verify

```bash
# Verify route count increased (existing 5 routes + 9 dashboard routes = 14+)
python -c "from autopilot.api.v1.routes import router; print(f'{len(router.routes)} routes')"

# Verify dashboard endpoints are accessible
python -c "
from autopilot.api.v1.routes import router
paths = [r.path for r in router.routes if hasattr(r, 'path')]
dashboard_paths = [p for p in paths if '/dashboard' in p]
print(f'{len(dashboard_paths)} dashboard routes: {dashboard_paths}')
"
```

---

## Design Decisions

| Decision                                                  | Rationale                                                               |
| --------------------------------------------------------- | ----------------------------------------------------------------------- |
| Mount via `include_router` (not a separate `app.mount()`) | Keeps all API routes under `/api/v1` with shared auth dependency        |
| Auth inherited from parent router                         | Single `Depends(get_api_key)` on the V1 router protects all sub-routers |
| `.dockerignore` entry for `dashboard/`                    | Prevents future frontend code from bloating backend image               |
| `.gitignore` entries for `node_modules/` and `dist/`      | Standard JS/TS ignore patterns for any frontend tooling                 |

---

## Files Modified

| File                         | Change                                              | Lines   |
| ---------------------------- | --------------------------------------------------- | ------- |
| `autopilot/api/v1/routes.py` | Add import + `include_router(router)`     | 2 lines |
| `.dockerignore`              | Add `dashboard/`                                    | 1 line  |
| `.gitignore`                 | Add `dashboard/node_modules/` and `dashboard/dist/` | 3 lines |
