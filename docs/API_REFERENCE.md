# Autopilot API Reference

> **Version**: v1 &nbsp;|&nbsp; **Protocol**: HTTP/JSON + SSE &nbsp;|&nbsp; **Auth**: `X-API-Key` header

Autopilot is a **headless AI workflow platform**. This document is the canonical reference for building integrations, dashboards, and tooling on top of the API.

> [!TIP]
> Interactive API docs are also available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the server is running.

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Base URL & CORS](#2-base-url--cors)
3. [Error Handling](#3-error-handling)
4. [Endpoint Reference](#4-endpoint-reference)
   - [Workflows](#41-workflows)
   - [Runs](#42-runs)
   - [Events](#43-events--sse)
   - [HITL](#44-human-in-the-loop-hitl)
   - [Copilot](#45-copilot)
   - [System](#46-system)
5. [Webhook Integration](#5-webhook-integration)
6. [SSE Streaming Guide](#6-sse-streaming-guide)
7. [A2A Protocol](#7-a2a-protocol)
8. [Rate Limiting](#8-rate-limiting)
9. [API Versioning](#9-api-versioning)
10. [Error Code Reference](#10-error-code-reference)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Authentication

All `/api/v1/*` endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: $API_KEY" https://your-instance.run.app/api/v1/workflows
```

The server validates the key against the `API_KEY_SECRET` environment variable using timing-safe HMAC comparison.

| Scenario                 | Response                                               |
| ------------------------ | ------------------------------------------------------ |
| Missing header           | `401 — Invalid API Key`                                |
| Wrong key                | `401 — Invalid API Key`                                |
| `API_KEY_SECRET` not set | `401 — API_KEY_SECRET is not configured on the server` |

**Webhooks** (`/gmail/webhook`, `/telegram/webhook`) do **not** require `X-API-Key` — they use their own verification mechanisms (Pub/Sub signatures, Telegram secret tokens).

---

## 2. Base URL & CORS

| Environment | Base URL                           |
| ----------- | ---------------------------------- |
| Local dev   | `http://localhost:8080`            |
| Cloud Run   | `https://<service>-<hash>.run.app` |

CORS is **disabled by default** (headless API — most clients are server-to-server). Opt-in by setting:

```bash
API_CORS_ORIGINS=https://admin.example.com,https://dashboard.example.com
```

---

## 3. Error Handling

All errors return a consistent JSON envelope:

```json
{
  "error": {
    "error_code": "WORKFLOW_NOT_FOUND",
    "message": "Workflow 'xyz' not found",
    "detail": "Available: ['bank_to_ynab', 'conversational_assistant']",
    "retryable": false,
    "http_status": 404
  }
}
```

| Field         | Type             | Description                                        |
| ------------- | ---------------- | -------------------------------------------------- |
| `error_code`  | `string`         | Machine-readable code (e.g., `WORKFLOW_NOT_FOUND`) |
| `message`     | `string`         | Human-readable error message                       |
| `detail`      | `string \| null` | Additional context                                 |
| `retryable`   | `boolean`        | If `true`, the caller should consider retrying     |
| `http_status` | `integer`        | HTTP status code                                   |

> [!IMPORTANT]
> Use `retryable` for automated retry logic — don't infer retryability from HTTP status codes alone. For example, `429` from the LLM provider is `retryable: true`, but `422` from a guardrail block is `retryable: false`.

---

## 4. Endpoint Reference

### 4.1 Workflows

#### `GET /api/v1/workflows` — List all workflows

Returns all registered workflows with enriched metadata including run stats.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/workflows
```

**Response** `200`:

```json
{
  "workflows": [
    {
      "id": "bank_to_ynab",
      "display_name": "Bank to YNAB",
      "description": "Parse bank emails and create YNAB transactions",
      "version": "1.0.0",
      "icon": "🏦",
      "color": "#4CAF50",
      "enabled": true,
      "triggers": [{ "type": "GMAIL_PUSH", "path": null }],
      "tags": ["finance", "automation"],
      "strategy": "DAG",
      "step_count": 9,
      "agent_count": 1,
      "total_runs": 42,
      "success_rate": 95.2,
      "last_run": null
    }
  ],
  "total": 1
}
```

---

#### `GET /api/v1/workflows/{workflow_id}` — Full workflow detail

Returns manifest, pipeline graph, agent cards, and stats.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/workflows/bank_to_ynab
```

**Response** `200`:

```json
{
  "manifest": {
    "name": "bank_to_ynab",
    "display_name": "Bank to YNAB",
    "...": "..."
  },
  "pipeline": {
    "strategy": "DAG",
    "nodes": [
      {
        "name": "email_parser",
        "type": "agent",
        "layer": 1,
        "dependencies": ["format_parser_prompt"]
      }
    ],
    "edges": [{ "source": "format_parser_prompt", "target": "email_parser" }],
    "layers": [
      ["format_parser_prompt"],
      ["email_parser"],
      ["match_account", "format_researcher_input"]
    ]
  },
  "agents": [
    {
      "name": "email_parser",
      "display_name": "Email Parser",
      "type": "LLM",
      "model": "gemini-3-flash-preview",
      "tools": ["ynab.get_accounts"],
      "guardrails_before": ["input_length_guard"],
      "guardrails_after": ["uuid_format_guard"]
    }
  ],
  "stats": { "total": 42, "successful": 40, "failed": 2 }
}
```

**Errors**: `404` — Workflow not found.

---

#### `GET /api/v1/workflows/{workflow_id}/pipeline` — Pipeline graph topology

Returns nodes, edges, and topological layers for DAG visualization.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/workflows/bank_to_ynab/pipeline
```

---

#### `GET /api/v1/workflows/{workflow_id}/agents` — Agent cards

Returns all `.agent.yaml` agent card definitions for a workflow.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/workflows/bank_to_ynab/agents
```

---

#### `POST /api/v1/workflows/{workflow_id}/trigger` — Manually trigger

Dispatches a workflow execution via EventBus (async — fires and returns).

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"body": "email content..."}}' \
  http://localhost:8080/api/v1/workflows/bank_to_ynab/trigger
```

**Request body** (optional):

```json
{ "payload": { "body": "...", "auto_create": true } }
```

**Response** `200`:

```json
{
  "workflow_id": "bank_to_ynab",
  "status": "dispatched",
  "trigger_type": "MANUAL"
}
```

**Errors**: `404` — Workflow not found. `500` — Workflow is disabled.

---

#### `PATCH /api/v1/workflows/{workflow_id}` — Toggle enable/disable

Updates the workflow's enabled state at runtime (in-memory only, does not modify `manifest.yaml`).

```bash
curl -X PATCH -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}' \
  http://localhost:8080/api/v1/workflows/bank_to_ynab
```

**Request body**:

```json
{ "enabled": false }
```

**Response** `200`:

```json
{ "workflow_id": "bank_to_ynab", "enabled": false, "status": "updated" }
```

**Errors**: `404` — Workflow not found.

> [!NOTE]
> This toggles the in-memory state only. The change persists until the next container restart.

---

### 4.2 Runs

#### `GET /api/v1/workflows/{workflow_id}/runs` — List recent runs

Supports cursor-based pagination. Returns newest-first.

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8080/api/v1/workflows/bank_to_ynab/runs?limit=10"
```

**Query params**:

| Param         | Type     | Default | Description                                      |
| ------------- | -------- | ------- | ------------------------------------------------ |
| `limit`       | `int`    | `50`    | Max runs to return                               |
| `start_after` | `string` | `null`  | Cursor for pagination                            |
| `status`      | `string` | `null`  | Filter by status (`failed`, `success`, `paused`) |
| `since`       | `string` | `null`  | ISO 8601 datetime — runs started after this time |

**Response** `200`:

```json
{
  "workflow_id": "bank_to_ynab",
  "runs": [
    {
      "id": "run_abc123",
      "workflow_id": "bank_to_ynab",
      "status": "SUCCESS",
      "trigger_type": "GMAIL_PUSH",
      "started_at": "2026-02-26T10:00:00Z",
      "completed_at": "2026-02-26T10:00:05Z",
      "duration_ms": 5200.5
    }
  ],
  "meta": { "next_cursor": "run_abc122" },
  "stats": { "total": 42, "successful": 40, "failed": 2 }
}
```

---

#### `GET /api/v1/workflows/{workflow_id}/runs/{run_id}` — Full run trace

Returns run metadata + per-step artifact data from GCS.

```bash
curl -H "X-API-Key: $API_KEY" \
  http://localhost:8080/api/v1/workflows/bank_to_ynab/runs/run_abc123
```

**Response** `200`:

```json
{
  "run": { "id": "run_abc123", "status": "SUCCESS", "...": "..." },
  "steps": [
    {
      "name": "email_parser",
      "artifact_key": "email_parser.json",
      "output": { "parsed_email": { "date": "2026-02-21" } },
      "duration_ms": 5668.35,
      "has_llm_response": true,
      "llm_response": {
        "agent": "email_parser",
        "final_text": "{...}",
        "parsed_json": { "date": "2026-02-21" }
      }
    }
  ]
}
```

---

#### `POST /api/v1/workflows/{workflow_id}/runs/{run_id}/cancel` — Cancel a run

Cancels a `RUNNING` or `PENDING` run. Updates status to `CANCELLED` and publishes an `api.run_cancelled` event.

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  http://localhost:8080/api/v1/workflows/bank_to_ynab/runs/run_abc123/cancel
```

**Response** `200`:

```json
{ "run_id": "run_abc123", "workflow_id": "bank_to_ynab", "status": "cancelled" }
```

**Errors**: `404` — Workflow or run not found. `409` — Run is not in `RUNNING` or `PENDING` state.

---

#### `DELETE /api/v1/workflows/{workflow_id}/runs/{run_id}` — Delete a run

Permanently deletes a run from history. Adjusts aggregate stats.

```bash
curl -X DELETE -H "X-API-Key: $API_KEY" \
  http://localhost:8080/api/v1/workflows/bank_to_ynab/runs/run_abc123
```

**Response** `200`:

```json
{ "run_id": "run_abc123", "workflow_id": "bank_to_ynab", "deleted": true }
```

**Errors**: `404` — Workflow or run not found.

> [!CAUTION]
> This is a destructive operation. Deleted runs cannot be recovered.

---

### 4.3 Events & SSE

#### `GET /api/v1/events` — EventBus history

```bash
# All events (default)
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/events

# Filter by topic
curl -H "X-API-Key: $API_KEY" "http://localhost:8080/api/v1/events?topic=pipeline.completed&limit=10"
```

**Query params**:

| Param   | Type     | Default | Description                     |
| ------- | -------- | ------- | ------------------------------- |
| `topic` | `string` | `*`     | Topic filter (`*` = all topics) |
| `limit` | `int`    | `50`    | Max events to return            |

---

#### `GET /api/v1/events/stream` — SSE live stream

See [§6 SSE Streaming Guide](#6-sse-streaming-guide) for details.

---

### 4.4 Human-in-the-Loop (HITL)

#### `GET /api/v1/runs/pending-action` — List paused runs

Returns all globally `PAUSED` runs awaiting human intervention.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/runs/pending-action
```

**Response** `200`:

```json
{
  "pending": [
    {
      "run_id": "run_xyz",
      "workflow_id": "bank_to_ynab",
      "status": "PAUSED",
      "trigger_type": "GMAIL_PUSH",
      "started_at": "2026-02-26T10:00:00Z"
    }
  ],
  "total": 1
}
```

---

#### `POST /api/v1/workflows/{workflow_id}/runs/{run_id}/resume` — Resume a paused run

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"approved": true, "notes": "Looks good"}}' \
  http://localhost:8080/api/v1/workflows/bank_to_ynab/runs/run_xyz/resume
```

**Request body** (optional):

```json
{ "payload": { "approved": true, "notes": "Manual override" } }
```

**Response** `200`:

```json
{ "run_id": "run_xyz", "workflow_id": "bank_to_ynab", "status": "dispatched" }
```

**Errors**: `404` — Run not found. `409` — Run is not in `PAUSED` state.

> [!NOTE]
> The `status: "dispatched"` indicates the resume event was published to the EventBus. The actual resume happens asynchronously — the workflow's subscriber picks it up.

---

### 4.5 Copilot

#### `POST /api/v1/copilot/ask` — Ask the platform copilot

Natural language query about workflows, failures, run history, and events.

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which workflows failed in the last 24 hours?"}' \
  http://localhost:8080/api/v1/copilot/ask
```

**Request body**:

```json
{ "query": "What were the most common errors today?" }
```

**Response** `200`:

```json
{
  "answer": "There were 3 failures in bank_to_ynab today, all caused by CONNECTOR_AUTH errors from the YNAB API...",
  "tools_used": [
    {
      "tool": "get_recent_errors",
      "args": { "hours": 24 },
      "result_summary": "3 errors found"
    }
  ],
  "iterations": 2
}
```

---

### 4.6 System

#### `GET /health` — Platform health (system-level)

```bash
curl http://localhost:8080/health
```

**Response** `200`:

```json
{
  "status": "healthy",
  "version": "5.0.0",
  "platform": "AutoPilot",
  "model": "gemini-3-flash-preview",
  "workflows_registered": 2,
  "workflows_enabled": 2,
  "workflow_details": { "bank_to_ynab": { "status": "registered" } }
}
```

#### `GET /api/v1/health` — API-level health

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/health
```

Returns workflow count and EventBus stats.

#### `GET /metrics` — Prometheus metrics

```bash
curl http://localhost:8080/metrics
```

Returns Prometheus-format metrics (counters, histograms).

#### `GET /api/v1/stats` — Global platform statistics

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/stats
```

**Response** `200`:

```json
{
  "total_workflows": 3,
  "enabled_workflows": 2,
  "total_runs": 128,
  "total_successful": 120,
  "total_failed": 8,
  "global_success_rate": 93.8,
  "top_workflow": "bank_to_ynab",
  "workflows": [
    {
      "workflow_id": "bank_to_ynab",
      "display_name": "Bank to YNAB",
      "total_runs": 90,
      "successful": 86,
      "failed": 4,
      "success_rate": 95.6,
      "enabled": true
    }
  ],
  "bus_stats": { "published": 256, "delivered": 256, "errors": 0 }
}
```

#### `GET /api/v1/openapi.json` — V1-only OpenAPI spec

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8080/api/v1/openapi.json
```

Returns the OpenAPI specification filtered to only `/api/v1` endpoints. Useful for code generation and SDK building.

---

## 5. Webhook Integration

Webhooks are **thin event adapters** that publish typed events to the `EventBus`. Workflows subscribe and react independently.

### Gmail Pub/Sub

**Endpoint**: `POST /gmail/webhook`

Google Cloud Pub/Sub pushes Gmail notifications here. The adapter decodes the message, fetches new emails, and publishes `email.received` events.

```
Pub/Sub Push → POST /gmail/webhook
                    │
                    ▼
              bus.publish("email.received", {...})
                    │
                    ▼
              subscribed workflows react
```

### Telegram Bot

**Endpoint**: `POST /telegram/webhook`

Receives Telegram Bot API updates. Validates the secret token, extracts message text, and publishes `telegram.message_received` events.

| Env Var                   | Purpose                               |
| ------------------------- | ------------------------------------- |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token for webhook verification |
| `TELEGRAM_CHAT_ID`        | Restrict to a single authorized chat  |

### Generic Webhooks

**Endpoint**: `POST /api/webhook/{path}`

Routes to the workflow that handles the given path (configured in `manifest.yaml` triggers).

---

## 6. SSE Streaming Guide

The `/api/v1/events/stream` endpoint provides real-time event streaming via [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

### Connecting

```javascript
const source = new EventSource(
  "https://your-instance.run.app/api/v1/events/stream",
  { headers: { "X-API-Key": API_KEY } }, // Note: native EventSource doesn't support headers
);

// For custom headers, use a polyfill like eventsource-polyfill or fetch-based SSE
```

### Event Format

```
event: pipeline.completed
data: {"topic":"pipeline.completed","sender":"bank_to_ynab","payload":{...},"timestamp":"..."}
id: 1708900000.123456
```

### Edge-Safe Design

| Feature              | Detail                                                               |
| -------------------- | -------------------------------------------------------------------- |
| **Keepalive**        | Heartbeat every 30 seconds (`event: keepalive`)                      |
| **Auto-disconnect**  | Intentional close after 5 minutes (Cloud Run LB safety)              |
| **Reconnect signal** | `event: reconnect` sent before disconnecting                         |
| **Replay**           | `Last-Event-ID` header → replays missed events from in-memory buffer |

### Python Client Example

```python
import httpx

async def stream_events(base_url: str, api_key: str):
    """Consume SSE events with automatic reconnection."""
    last_event_id = None

    while True:
        headers = {"X-API-Key": api_key}
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET", f"{base_url}/api/v1/events/stream", headers=headers
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("id: "):
                        last_event_id = line[4:]
                    elif line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        print(f"Event: {event_data['topic']}")
                    elif line.startswith("event: reconnect"):
                        break  # Server asked us to reconnect
```

---

## 7. A2A Protocol

Autopilot implements the [Agent-to-Agent (A2A) Protocol](https://google.github.io/A2A/) for multi-agent ecosystem interoperability.

### Agent Discovery

```bash
# Public — no auth required (per A2A spec)
curl https://your-instance.run.app/.well-known/agent-card.json
```

Returns an A2A `AgentCard` with skills dynamically built from the `WorkflowRegistry`.

### Sending Tasks

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "message": {
        "parts": [{"text": "{\"workflow\": \"bank_to_ynab\", \"body\": \"email...\"}"}]
      }
    }
  }' \
  https://your-instance.run.app/a2a
```

### Task Lifecycle

| Platform `RunStatus` | A2A `TaskState` |
| -------------------- | --------------- |
| `PENDING`            | `submitted`     |
| `RUNNING`            | `working`       |
| `SUCCESS`            | `completed`     |
| `FAILED`             | `failed`        |

---

## 8. Rate Limiting

The API includes a sliding-window rate limiter per client IP. Configuration:

| Variable             | Default | Description                               |
| -------------------- | ------- | ----------------------------------------- |
| `API_RATE_LIMIT_RPM` | `60`    | Max requests per minute per IP. `0` = off |

When exceeded, the API returns:

```json
{
  "error": {
    "error_code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded (60 requests/min)",
    "detail": "Retry after 45 seconds",
    "retryable": true,
    "http_status": 429
  }
}
```

The response includes a `Retry-After` header (seconds). Health, metrics, and webhook endpoints are exempt.

---

## 9. API Versioning

All `/api/v1/*` responses include versioning headers:

| Header        | Value   | Description                    |
| ------------- | ------- | ------------------------------ |
| `API-Version` | `v1`    | Current API version            |
| `X-API-Docs`  | `/docs` | Link to interactive Swagger UI |

Use these headers for client-side version negotiation and documentation discovery.

---

## 10. Error Code Reference

All errors extend `AutoPilotError` with `error_code`, `retryable`, and `http_status`.

| Layer         | Error Code                | HTTP | Retryable | Description                        |
| ------------- | ------------------------- | ---- | --------- | ---------------------------------- |
| **API**       | `WORKFLOW_NOT_FOUND`      | 404  | ❌        | Workflow not in registry           |
| **API**       | `RUN_NOT_FOUND`           | 404  | ❌        | Run ID not found                   |
| **API**       | `RUN_NOT_PAUSED`          | 409  | ❌        | Attempted resume of non-paused run |
| **API**       | `RUN_NOT_CANCELLABLE`     | 409  | ❌        | Run not in RUNNING/PENDING state   |
| **API**       | `RATE_LIMIT_EXCEEDED`     | 429  | ✅        | API rate limit exceeded            |
| **Pipeline**  | `PIPELINE_TIMEOUT`        | 504  | ✅        | Execution exceeded time limit      |
| **Pipeline**  | `PIPELINE_EMPTY_RESPONSE` | 502  | ✅        | No usable output                   |
| **Pipeline**  | `MAX_RETRIES_EXCEEDED`    | 422  | ❌        | Loop exhausted iterations          |
| **Pipeline**  | `DAG_CYCLE`               | 422  | ❌        | Circular dependency in DAG         |
| **Agent**     | `LLM_RATE_LIMIT`          | 429  | ✅        | Gemini API quota exceeded          |
| **Agent**     | `LLM_CONTENT_FILTER`      | 422  | ❌        | Content safety block               |
| **Agent**     | `AGENT_OUTPUT_PARSE`      | 502  | ✅        | Output format mismatch             |
| **Connector** | `CONNECTOR_AUTH`          | 401  | ❌        | External service auth failure      |
| **Connector** | `CONNECTOR_UNAVAILABLE`   | 503  | ✅        | External service unreachable       |
| **Connector** | `CONNECTOR_RATE_LIMIT`    | 429  | ✅        | External service rate limit        |
| **Guardrail** | `GUARDRAIL_BLOCKED`       | 422  | ❌        | Request blocked by guard           |
| **Tools**     | `TOOL_REGISTRY_ERROR`     | 422  | ❌        | Tool not found or duplicate        |
| **Tools**     | `MCP_BRIDGE_ERROR`        | 502  | ✅        | MCP server connection failure      |
| **Tools**     | `TOOL_AUTH_ERROR`         | 401  | ❌        | Tool credential missing            |
| **Bus**       | `BUS_TIMEOUT`             | 504  | ✅        | EventBus operation timed out       |
| **DSL**       | `DSL_VALIDATION`          | 422  | ❌        | Invalid YAML workflow schema       |

---

## 11. Configuration Reference

All configuration follows the [12-Factor App](https://12factor.net/config) methodology — environment variables only, no config files in production.

### API & Security

| Variable           | Default      | Description                       |
| ------------------ | ------------ | --------------------------------- |
| `API_KEY_SECRET`   | _(required)_ | Secret for `X-API-Key` validation |
| `API_CORS_ORIGINS` | _(empty)_    | Comma-separated allowed origins   |

### Backend Selection

| Variable           | Default  | Options                           | Description         |
| ------------------ | -------- | --------------------------------- | ------------------- |
| `SESSION_BACKEND`  | `memory` | `memory`, `firestore`             | Session persistence |
| `MEMORY_BACKEND`   | `memory` | `memory`, `firestore`, `vertexai` | Long-term memory    |
| `ARTIFACT_BACKEND` | `memory` | `memory`, `gcs`                   | Artifact storage    |
| `EVENTBUS_BACKEND` | `memory` | `memory`, `pubsub`                | Event bus backend   |

### LLM & Performance

| Variable                             | Default        | Description                        |
| ------------------------------------ | -------------- | ---------------------------------- |
| `MODEL_RATE_LIMIT_QPM`               | `0` (disabled) | Proactive rate limit (queries/min) |
| `CONTEXT_CACHE_MIN_TOKENS`           | `2048`         | Min tokens for context caching     |
| `CONTEXT_CACHE_TTL_SECONDS`          | `1800`         | Cache TTL (30 min)                 |
| `CONTEXT_COMPRESSION_TRIGGER_TOKENS` | `100000`       | Start context compression          |
| `CONTEXT_COMPRESSION_TARGET_TOKENS`  | `80000`        | Compress down to                   |

### Memory & Embeddings

| Variable                          | Default                | Description               |
| --------------------------------- | ---------------------- | ------------------------- |
| `MEMORY_EMBEDDING_MODEL`          | `gemini-embedding-001` | Embedding model           |
| `MEMORY_EMBEDDING_DIMENSIONALITY` | `768`                  | Vector dimensions         |
| `MEMORY_SEARCH_LIMIT`             | `20`                   | Max memory search results |

### Webhooks & External Services

| Variable                  | Default   | Description                     |
| ------------------------- | --------- | ------------------------------- |
| `TELEGRAM_WEBHOOK_SECRET` | _(empty)_ | Telegram webhook verification   |
| `TELEGRAM_CHAT_ID`        | _(empty)_ | Authorized Telegram chat        |
| `HASS_URL`                | _(empty)_ | Home Assistant MCP URL          |
| `HASS_TOKEN`              | _(empty)_ | Home Assistant long-lived token |
| `API_RATE_LIMIT_RPM`      | `60`      | Rate limit (requests/min/IP)    |
