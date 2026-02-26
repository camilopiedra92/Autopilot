# Phase 3B. Deployment Verification — Cloud Run Validation

> **Status**: ✅ COMPLETE (2026-02-26)  
> **Effort**: ~30 min  
> **Type**: VERIFICATION (Ops)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 3  
> **Depends on**: Phase 3A (Integration Tests pass), Phases 0–2 all complete

---

## Problem Statement

All Dashboard code has been tested locally with `InMemory*` backends. Before the Dashboard API is production-ready, it must be validated on **Cloud Run with production backends** (Firestore, Pub/Sub, GCS) to confirm:

1. **Scale-to-zero correctness**: Cold-start run log hydration (`BaseWorkflow.setup()`) works against real Firestore — stats and latest run are correctly loaded.
2. **Firestore integration**: `FirestoreRunLogService` CRUD operations match the `InMemoryRunLogService` behavior that all tests rely on.
3. **Pub/Sub EventBus**: HITL resume and manual trigger events survive Pub/Sub message delivery (not just in-memory `asyncio.Queue`).
4. **GCS artifacts**: Run trace endpoint (`/workflows/{id}/runs/{run_id}`) loads artifact JSON from GCS correctly.
5. **CORS**: If `API_CORS_ORIGINS` is set, only listed origins get CORS headers. If unset, no CORS middleware is mounted.
6. **12-Factor env var propagation**: All backend selections (`RUN_LOG_BACKEND`, `EVENTBUS_BACKEND`, `ARTIFACT_BACKEND`, `SESSION_BACKEND`) resolve correctly from Cloud Run env vars.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                         | Current                 | Target                                    |
| ----------------------- | --------------------------------------------------- | ----------------------- | ----------------------------------------- |
| §1 Core Philosophy      | Scale-to-zero: no state lost on container recycle   | Tested in-memory only   | Validated via deploy + scale cycle        |
| §1 Core Philosophy      | 12-Factor config — env vars drive backend selection | Assumed correct         | Verified on Cloud Run with prod env vars  |
| §1 Core Philosophy      | CORS disabled by default                            | Implemented, not tested | Verified by hitting deployed endpoint     |
| §10.4 Deployment        | Dockerfile explicit COPY allowlist                  | Assumed correct         | Container builds and starts without error |
| §5 Observability        | OTel spans on all endpoints                         | Implemented             | Verified spans appear in Cloud Trace      |

---

## Prerequisites

- All local tests passing: `python -m pytest tests/ -v --ignore=tests/autopilot/test_btc_strategy.py`
- `/deploy_to_cloud_run` workflow available and functional.
- GCP project with Firestore, Pub/Sub, and GCS configured.

---

## Verification Steps

### Step 1: Deploy to Cloud Run

Use the existing deployment workflow:

```bash
/deploy_to_cloud_run
```

Ensure the following env vars are set in the deploy command:

```bash
gcloud run deploy autopilot \
  --set-env-vars "RUN_LOG_BACKEND=firestore" \
  --set-env-vars "EVENTBUS_BACKEND=pubsub" \
  --set-env-vars "ARTIFACT_BACKEND=gcs" \
  --set-env-vars "SESSION_BACKEND=firestore" \
  --set-env-vars "API_KEY_SECRET=<your-api-key>"
```

### Step 2: Cold-Start Validation

After deployment, verify the container starts cleanly and all services initialize:

```bash
# Health check — must return 200 with workflow count
curl -s -H "X-API-Key: $API_KEY" https://<service-url>/api/v1/health | jq .
```

**Expected response**:

```json
{
  "status": "healthy",
  "workflows": {
    "total": 2,
    "enabled": 2
  },
  "bus": {
    "published": 0,
    "delivered": 0,
    "errors": 0
  }
}
```

### Step 3: Scale-to-Zero + Re-Hydration

Force a scale-to-zero event and verify re-hydration:

```bash
# 1. Trigger a workflow run to create durable data in Firestore
curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"test": true}}' \
  https://<service-url>/api/v1/workflows/bank_to_ynab/trigger | jq .

# 2. Wait ~15 minutes for Cloud Run to scale to zero (min-instances=0)
# Or force scale-down via: gcloud run services update autopilot --min-instances=0

# 3. Hit health endpoint to trigger cold start
curl -s -H "X-API-Key: $API_KEY" https://<service-url>/api/v1/health | jq .

# 4. Verify the run from step 1 is visible (hydrated from Firestore)
curl -s -H "X-API-Key: $API_KEY" \
  https://<service-url>/api/v1/workflows/bank_to_ynab/runs?limit=5 | jq '.stats'
```

**Success criteria**: The `stats.total` value after re-hydration must be ≥ 1, proving `FirestoreRunLogService.get_stats()` works on cold start.

### Step 4: CORS Verification

```bash
# Case 1: No CORS headers when API_CORS_ORIGINS is NOT set
curl -s -D- -o /dev/null \
  -H "Origin: https://evil.com" \
  -H "X-API-Key: $API_KEY" \
  https://<service-url>/api/v1/health

# Expected: NO 'Access-Control-Allow-Origin' header in response

# Case 2: If API_CORS_ORIGINS is explicitly set, only allowed origins get CORS
# (Re-deploy with: --set-env-vars "API_CORS_ORIGINS=https://dashboard.example.com")
curl -s -D- -o /dev/null \
  -H "Origin: https://dashboard.example.com" \
  -H "X-API-Key: $API_KEY" \
  https://<service-url>/api/v1/health

# Expected: 'Access-Control-Allow-Origin: https://dashboard.example.com' header present
```

### Step 5: Firestore Document Structure Verification

After at least one workflow run, verify the Firestore document hierarchy matches the spec:

```bash
# Check via gcloud or Firebase console:
# autopilot_runs/{workflow_id} → { total: N, successful: N }
# autopilot_runs/{workflow_id}/runs/{run_id} → WorkflowRun serialized
```

### Step 6: OpenTelemetry Span Verification

```bash
# Hit a traced endpoint
curl -s -H "X-API-Key: $API_KEY" \
  https://<service-url>/api/v1/workflows | jq .total

# Check Cloud Trace for spans:
# 1. Go to GCP Console → Trace → Trace list
# 2. Filter by service_name = "autopilot"
# 3. Verify spans: "dashboard.list_workflows", "run_log.stats"
```

### Step 7: Copilot Endpoint Smoke Test

```bash
curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the platform overview?"}' \
  https://<service-url>/api/v1/copilot/ask | jq .
```

**Success criteria**: Response contains `reply` and `tools_used` fields. The copilot should return stats about registered workflows.

---

## Verification Checklist

| #   | Check                                          | Pass? |
| --- | ---------------------------------------------- | ----- |
| 1   | Container builds and starts on Cloud Run       | [ ]   |
| 2   | Health endpoint returns 200 with workflow data | [ ]   |
| 3   | List workflows returns enriched data           | [ ]   |
| 4   | Firestore run log stores and retrieves runs    | [ ]   |
| 5   | Scale-to-zero → re-hydration preserves stats   | [ ]   |
| 6   | CORS absent when `API_CORS_ORIGINS` unset      | [ ]   |
| 7   | Cloud Trace shows OTel spans                   | [ ]   |
| 8   | Copilot responds with platform analysis        | [ ]   |
| 9   | Trigger endpoint dispatches via Pub/Sub        | [ ]   |

---

## Design Decisions

| Decision                                       | Rationale                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Manual verification (not automated CI)         | Cloud Run tests require real GCP infra + secrets — not viable in CI without significant setup    |
| Scale-to-zero test via natural timeout         | Simulating scale-to-zero artificially is fragile; natural 15-min timeout tests the real path     |
| CORS test with `curl -D-` headers              | Validates the actual HTTP response headers, not internal middleware state                        |
| Firestore hierarchy checked via console/gcloud | Direct document inspection confirms schema matches spec — API endpoints are secondary validation |
| Copilot smoke test (not comprehensive)         | Full copilot behavior is validated in integration tests; deployment only confirms connectivity   |

---

## Files Modified

| File | Change                              | Lines |
| ---- | ----------------------------------- | ----- |
| N/A  | Verification-only — no code changes | 0     |
