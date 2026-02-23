# A5. Dockerfile — Multi-Stage Leak & Worker Optimization

> **Status**: 🔲 PENDING
> **Effort**: ~30 min
> **Type**: ADJUST (Architectural Hygiene)
> **Parent**: [Implementation Plan](./Implementation%20Plan) § A5

---

## Problem Statement

The Dockerfile has **two distinct issues**:

### Issue 1: Multi-Stage COPY Leak

The runtime stage uses `COPY . .` (line 42), which copies **everything** from the build context into the final image. While `.dockerignore` filters _some_ files, the current approach is **defense-in-depth-deficient**:

- **What `.dockerignore` currently excludes**: `.env`, `credentials.json`, `token.json`, `venv/`, `__pycache__/`, `tests/`, `.git/`, `.github/`, `.DS_Store`, `.vscode/`, `.idea/`, `*.md` (except `README.md`), `.agent/`, `.gemini/`, `.ruff_cache/`, `.pytest_cache/`
- **What still leaks into the runtime image**:
  - `docs/` — full documentation directory (plans, architecture docs)
  - `scripts/` — E2E test scripts (not needed at runtime)
  - `pyproject.toml` — build metadata (already consumed in builder stage)
  - `ruff.toml` — linter config
  - `.env.example` — example env file
  - `autopilot.egg-info/` — build artifact (should be caught by `.dockerignore`)
  - `README.md` — explicitly preserved but not needed at runtime
  - `workflows/*/tests/` — workflow-level test directories

**Risk**: Even though secrets _are_ protected by `.dockerignore`, relying on an _exclusion_ list is fragile — any new file added to the repo root automatically enters the image unless explicitly excluded. The edge pattern is **explicit inclusion** (allowlist > denylist).

### Issue 2: Hardcoded `--workers 2`

```dockerfile
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

**Problems**:

1. Cloud Run instances have `--cpu=1` (1 vCPU, per `ci.yml` line 173). Multiple Uvicorn **workers** (processes) on a single vCPU provide **zero** parallelism for CPU-bound code (GIL) and **waste memory** by duplicating the Python interpreter + app state per process.
2. The app is **I/O-bound** (async FastAPI + `httpx` + external API calls). A single `asyncio` event loop can handle Cloud Run's `--concurrency=80` without process-level parallelism.
3. The worker count is hardcoded — no way to tune per environment without rebuilding the image.
4. Uvicorn natively reads `WEB_CONCURRENCY` env var for worker count, but the hardcoded `--workers 2` overrides it.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                  | Current                                 | Target                             |
| ----------------------- | -------------------------------------------- | --------------------------------------- | ---------------------------------- |
| §1 Core Philosophy      | Edge-First, lightweight, no monolithic state | `COPY . .` copies non-runtime files     | Explicit COPY of only runtime code |
| §9.4 Development Rules  | No legacy code path                          | Worker count ignores Cloud Run topology | Dynamic via `WEB_CONCURRENCY`      |
| §10.3 CD                | Edge-native constraints                      | Hardcoded process topology              | Env-driven, Cloud Run-aware        |

---

## Implementation Plan

### Step 1: Explicit COPY in Runtime Stage

**Replace** `COPY . .` with explicit COPY directives for exactly what the runtime needs:

```dockerfile
# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.13.2-slim AS runtime

WORKDIR /app

# ... (system deps, copy site-packages from builder) ...

# Copy ONLY runtime-required application code (allowlist pattern)
COPY app.py ./
COPY autopilot/ ./autopilot/
COPY workflows/ ./workflows/
```

**What this achieves**:

- ✅ `app.py` — the single entry point
- ✅ `autopilot/` — the platform package
- ✅ `workflows/` — business logic (manifests, pipelines, agents, steps, data)
- ❌ `docs/`, `scripts/`, `tests/`, `pyproject.toml`, `ruff.toml`, `README.md`, `.env.example` — excluded by omission

**Critical consideration**: `workflows/*/tests/` directories will be copied since they're children of `workflows/`. This is acceptable because:

1. They add negligible image size (Python test files are tiny)
2. Excluding them would require a complex multi-step COPY or a post-COPY `RUN rm -rf`
3. The `.dockerignore` already excludes the platform `tests/` directory
4. No security risk — test files contain no secrets

### Step 2: Harden `.dockerignore` (Belt + Suspenders)

Even though Step 1 makes `.dockerignore` less critical, maintain it as defense-in-depth:

```dockerignore
# ── Documentation & plans (not needed at runtime) ───────────────────
docs/
scripts/

# ── Build artifacts ──────────────────────────────────────────────────
*.egg-info/

# ── Config files (consumed only during build or CI) ─────────────────
.env.example
ruff.toml
```

**Add these to the existing `.dockerignore`** — they currently leak through.

### Step 3: Fix Uvicorn Worker Strategy

**Replace** the hardcoded `--workers 2` CMD with a Cloud Run-optimized strategy:

```dockerfile
# ── Start ────────────────────────────────────────────────────────────
# Cloud Run sets PORT env var; uvicorn listens on 0.0.0.0:$PORT
#
# Worker strategy (Cloud Run with --cpu=1):
#   - Default: 1 worker (single async event loop handles concurrency=80)
#   - Tunable: set WEB_CONCURRENCY env var to override
#   - Uvicorn natively reads WEB_CONCURRENCY when --workers is omitted
#
# Why 1 worker is optimal:
#   - App is I/O-bound (async FastAPI + httpx + external APIs)
#   - Single asyncio event loop can saturate 1 vCPU for I/O workloads
#   - Multiple workers duplicate memory with zero CPU parallelism (GIL)
#   - Cloud Run handles horizontal scaling (more instances, not more workers)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Key changes**:

1. **Removed `--workers 2`** — Uvicorn defaults to 1 worker, which is optimal for `--cpu=1` Cloud Run.
2. **`WEB_CONCURRENCY` support** — Uvicorn natively reads this env var. If future scaling requires multiple workers (e.g., upgrading to `--cpu=2`), set `WEB_CONCURRENCY=2` in Cloud Run env without rebuilding the image.
3. **No Gunicorn needed** — Uvicorn's built-in process manager handles multi-worker mode if `WEB_CONCURRENCY` is set. Adding Gunicorn adds a dependency and cold-start latency for no benefit on single-vCPU deployments.

### Step 4: Update CI/CD Flags (Optional Enhancement)

In `.github/workflows/ci.yml`, the deploy step can optionally set `WEB_CONCURRENCY` to make the configuration explicit:

```yaml
--set-env-vars=PYTHONUNBUFFERED=1,...,WEB_CONCURRENCY=1
```

This is optional since Uvicorn defaults to 1 worker, but makes the architecture decision **visible** in the deployment configuration.

### Step 5: Validate

1. **Local build test** — Build the image and verify only `app.py`, `autopilot/`, and `workflows/` are present:

   ```bash
   docker build -t autopilot-test .
   docker run --rm autopilot-test ls -la /app/
   # Should show: app.py, autopilot/, workflows/ — nothing else
   ```

2. **Image size comparison** — Compare before/after image sizes:

   ```bash
   docker images autopilot-test
   ```

3. **Startup test** — Verify the app starts correctly with 1 worker:

   ```bash
   docker run --rm -p 8080:8080 \
     -e GOOGLE_API_KEY=test \
     -e YNAB_ACCESS_TOKEN=test \
     autopilot-test
   # Should see: "Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)"
   # Should NOT see: "Started parent process" (no multi-process manager)
   ```

4. **WEB_CONCURRENCY override test** — Verify the env var works:
   ```bash
   docker run --rm -e WEB_CONCURRENCY=2 autopilot-test \
     uvicorn app:app --host 0.0.0.0 --port 8080
   # Should see: "Started parent process" + "Started server process" x2
   ```

---

## Files Modified

| File            | Change                                                              | Lines    |
| --------------- | ------------------------------------------------------------------- | -------- |
| `Dockerfile`    | Replace `COPY . .` with explicit COPYs; remove `--workers 2`        | ~5 lines |
| `.dockerignore` | Add `docs/`, `scripts/`, `*.egg-info/`, `ruff.toml`, `.env.example` | ~5 lines |

## Files NOT Modified

| File             | Reason                                                |
| ---------------- | ----------------------------------------------------- |
| `app.py`         | No changes needed — worker config is Dockerfile-level |
| `ci.yml`         | Optional: `WEB_CONCURRENCY=1` for explicitness        |
| `pyproject.toml` | No new dependencies                                   |

---

## Risk Assessment

| Risk                                             | Likelihood | Impact                 | Mitigation                                                       |
| ------------------------------------------------ | ---------- | ---------------------- | ---------------------------------------------------------------- |
| Missing a runtime-required file in explicit COPY | Low        | High (app won't start) | Step 5 validation; CI smoke test catches it                      |
| `workflows/*/tests/` in image                    | None       | Negligible             | Tests are tiny, no secrets, no runtime impact                    |
| Single worker can't handle load                  | Very Low   | Medium                 | `WEB_CONCURRENCY` env var override; Cloud Run horizontal scaling |
| Breaking existing CD pipeline                    | Very Low   | Medium                 | Only Dockerfile changes; CI/CD workflow unchanged                |

---

## Decision Record

| Decision                                                  | Rationale                                                                                        |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Explicit COPY (allowlist) over `.dockerignore` (denylist) | Allowlist is defense-in-depth: new files are excluded by default                                 |
| 1 worker default                                          | Cloud Run `--cpu=1` + I/O-bound async app = 1 event loop is optimal                              |
| `WEB_CONCURRENCY` over `--workers` flag                   | Env-driven config > rebuild-required config; Uvicorn native support                              |
| No Gunicorn                                               | Unnecessary process manager for single-worker; adds cold-start latency                           |
| Keep `.dockerignore` additions                            | Belt-and-suspenders: protects `docker build` commands that might use `COPY . .` in future stages |
