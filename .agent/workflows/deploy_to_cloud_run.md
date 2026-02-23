---
description: How to safely deploy the Autopilot backend to Google Cloud Run (Tier-1 Edge Architecture)
---

# Deploy to Cloud Run: Edge Architecture Standard

This workflow defines the Tier-1 standard for deploying the Autopilot backend to Google Cloud Run. It enforces edge-native principles: scale-to-zero, stateless compute, strict infrastructure-as-code, and deterministic secret injection.

> [!CAUTION]
> **NEVER** deploy via the Google Cloud Console UI. The UI strips volume mounts (like the Gmail OAuth tokens) unless explicitly re-added manually, causing silent failures in production. All deployments must be declarative and code-driven.

## 1. Architectural Inversions (No Technical Debt)

To maintain a world-class, edge-native architecture, we strictly adhere to the following invariants:

1. **Stateless Compute**: The container is fully ephemeral and assumes it can be destroyed at any millisecond. All state belongs in external, distributed stores (Redis/Chroma) or is passed natively in the execution context (`AgentContext`).
2. **Ephemeral File-Based Auth**: Third-party OAuth tokens (e.g., Gmail's `credentials.json` and `token.json`) **must not** be baked into the Docker image or read randomly from disk. They MUST be dynamically mounted at runtime from Google Secret Manager into memory-backed volume mounts.
3. **Scale-to-Zero Default**: To minimize cost and enforce pure event-driven patterns, `--min-instances=0` is mandatory.
4. **No In-Process Background Tasks**: The codebase contains **zero** background loops or `asyncio.create_task` renewal hacks. All time-sensitive renewals are delegated to Google Cloud Scheduler (`ping-bank-to-ynab`), which pings `POST /gmail/watch/renew` daily to keep the Gmail watch alive across scale-to-zero cycles.
5. **Auto-Recovery (Watch Stealing)**: Gmail allows only one active push notification webhook per developer account. The production connector will automatically intercept lock errors (typically caused by local testing) and forcefully steal back exclusivity to prevent downtime.

## 2. CI/CD Pipeline (Source of Truth)

The primary and **only recommended** way to deploy to production is via Git push to the `master` branch, triggering the GitHub Actions workflow (`.github/workflows/ci.yml`).

This guarantees:

1. Automated unit tests pass before the image is built.
2. Buildx builds a highly-optimized, cached Docker image directly pushed to Artifact Registry.
3. Secrets, environment variables, and workload identity settings are deterministically injected every time.

If you add a new Secret Manager variable, you **must register it** in `.github/workflows/ci.yml` within the `deploy-cloudrun` step flags.

## 3. Break-Glass / Manual Deployment (CLI)

If CI/CD is degraded, or you must execute an emergency rollback or hotfix from your local machine, use this exact command.

This command mirrors the CI/CD pipeline and adheres mathematically to our Tier-1 standards.

### Prerequisites

Authenticate and lock your context to the correct project to prevent deploying to the wrong environment:

```bash
gcloud auth login
gcloud config set project antigravity-bank-ynab
```

### The Deployment Command

// turbo-all

```bash
gcloud run deploy bank-to-ynab \
  --image=us-central1-docker.pkg.dev/antigravity-bank-ynab/bank-to-ynab/bank-to-ynab:latest \
  --region=us-central1 \
  --project=antigravity-bank-ynab \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=80 \
  --timeout=120s \
  --set-env-vars=PYTHONUNBUFFERED=1,GOOGLE_CLOUD_PROJECT=antigravity-bank-ynab,GCP_PUBSUB_TOPIC=projects/antigravity-bank-ynab/topics/gmail-notifications,EVENTBUS_BACKEND=pubsub,SESSION_BACKEND=firestore,MEMORY_BACKEND=memory,ARTIFACT_BACKEND=gcs,ARTIFACT_GCS_BUCKET=antigravity-bank-ynab-artifacts,MODEL_RATE_LIMIT_QPM=1500,CONTEXT_CACHE_MIN_TOKENS=2048,CONTEXT_CACHE_TTL_SECONDS=1800,CONTEXT_CACHE_INTERVALS=10 \
  --set-secrets=GOOGLE_API_KEY=google-api-key:latest,YNAB_ACCESS_TOKEN=ynab-access-token:latest,API_KEY_SECRET=api-key-secret:latest,TODOIST_API_TOKEN=todoist-api-token:latest,AIRTABLE_PERSONAL_ACCESS_TOKEN=airtable-personal-access-token:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,/secrets/credentials/credentials.json=gmail-credentials:latest,/secrets/token/token.json=gmail-token:latest \
  --cpu-boost
```

### Critical Flag Breakdown & Rationale

| Flag                | Purpose / DevOps Rationale                                                                                                                                                                                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--set-secrets=...` | **Injected Config & Mounts**. Secures scalar API keys in memory as environment variables. Also creates **Immutable Volume Mounts** by mapping Secret Manager Payloads to the container filesystem via absolute paths (`/secrets/...`). Essential for Gmail's `credentials.json` and `token.json`, as the Google SDK strictly requires physical files. |
| `--min-instances=0` | **Pure Edge**. Enforces scale-to-zero, strictly aligning cost with compute value (events processed).                                                                                                                                                                                                                                                  |
| `--max-instances=5` | **Blast Radius Limitation**. Caps concurrent container scaling to prevent accidental billing attacks or overwhelming downstream APIs (like YNAB's rate limits).                                                                                                                                                                                       |
| `--concurrency=80`  | **ASGI Optimized**. Maximizes FastAPI's asynchronous event loop efficiency. Unlike synchronous WSGI servers, a single container can comfortably juggle 80 concurrent I/O-bound requests.                                                                                                                                                              |
| `--cpu-boost`       | **Cold Start Mitigation**. Dynamically allocates maximum base CPU resources during container initialization, dramatically reducing edge routing latency for the first request.                                                                                                                                                                        |

> [!IMPORTANT]
> **Dockerfile Standards**: The Dockerfile uses an **explicit COPY allowlist** (`app.py`, `autopilot/`, `workflows/`) instead of `COPY . .`. Secrets are excluded by omission. Uvicorn runs with **1 worker** (no `--workers` flag) because the app is I/O-bound and `--cpu=1` — multiple workers waste memory with zero CPU gain. Override via `WEB_CONCURRENCY` env var if needed. See `ARCHITECTURE.md §10.4`.

## 4. Post-Deployment Verification (Smoke Tests)

Deployments are not considered finished until production health is actively verified.

### 1. Check Platform Health

Ensure the platform, models, and registered workflows are active and untainted:

```bash
curl -s https://bank-to-ynab-1005597011634.us-central1.run.app/health
```

### 2. Verify Watch Registration Status

Ensure the PubSub connector has successfully registered its Gmail Watch upon cold start. Look for `"active": true`.

```bash
curl -s https://bank-to-ynab-1005597011634.us-central1.run.app/gmail/watch/status
```

### 3. Force Renew Keep-Alive (If Watch Expired)

If you missed a deployment window and the watch expired during a scale-to-zero phase, immediately trigger the renew endpoint. Thanks to the auto-recovery (watch stealing) logic, production will automatically intercept lock errors and forcefully take back control from local developer environments.

```bash
curl -s -X POST https://bank-to-ynab-1005597011634.us-central1.run.app/gmail/watch/renew
```

### 4. Break-Glass: Manual Watch Stop

If for any reason you need to deliberately sever the connection between the Gmail topic and the current environment (for instance, releasing the lock back to a local testing environment), you can manually kill the watch hook:

```bash
curl -s -X POST https://bank-to-ynab-1005597011634.us-central1.run.app/gmail/watch/stop
```
