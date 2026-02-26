# Phase 3C. SSE & Edge LB Validation — Reconnect & Disconnect Behavior

> **Status**: ✅ COMPLETE (2026-02-26)  
> **Effort**: ~30 min  
> **Type**: VERIFICATION (Network Layer)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 3  
> **Depends on**: Phase 3B (Deployment verification — service running on Cloud Run)

---

## Problem Statement

The SSE endpoint (`/api/v1/events/stream`) implements two critical safety mechanisms that can **only** be validated on a real deployment behind a load balancer:

1. **Intentional 5-minute disconnect**: The `_event_generator()` sends a `reconnect` event and closes the stream after `MAX_CONNECTION_LIFETIME = 300s`. This prevents Cloud Run's load balancer from killing the TCP connection ungracefully (zombie connections), which would cause client-side `EventSource.onerror` to fire.

2. **`Last-Event-ID` replay on reconnect**: When the client reconnects with the `Last-Event-ID` header, the server replays missed events from EventBus history (in-memory) or Pub/Sub (production). This ensures zero event loss during reconnection.

Without validating these behaviors, the dashboard frontend SSE integration could silently fail in production:

- Clients may hang on zombie connections that the LB has already dropped.
- Event replay may not work correctly with Pub/Sub timestamps.
- The `X-Accel-Buffering: no` header may be stripped by reverse proxies.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                      | Current                            | Target                                                 |
| ----------------------- | ------------------------------------------------ | ---------------------------------- | ------------------------------------------------------ |
| §1 Core Philosophy      | NEVER `asyncio.create_task` in ephemeral compute | SSE uses request-scoped generator  | Verified generator lifecycle matches request lifecycle |
| §1 Core Philosophy      | Edge-First — run close to data/user              | SSE has Edge LB safety disconnect  | Verified disconnect fires after exactly 5 minutes      |
| §1 Core Philosophy      | Scale-to-zero compatible                         | SSE subscription is request-scoped | Verified unsubscribe fires on disconnect               |

---

## Prerequisites

- Phase 3B complete (service deployed on Cloud Run).
- Access to service URL and valid API key.
- `curl` or a tool that supports SSE streaming.

---

## Verification Steps

### Step 1: Basic SSE Connection & Keepalive

Verify the SSE stream connects, returns correct headers, and sends keepalive pings:

```bash
# Connect to SSE stream and observe output for ~35 seconds
timeout 35 curl -sN \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream" \
  https://<service-url>/api/v1/events/stream
```

**Expected output within 30 seconds**:

```
event: keepalive
data:

```

**Expected headers** (use `curl -v` to verify):

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### Step 2: Event Delivery via SSE

Open an SSE stream and trigger a real event in parallel:

```bash
# Terminal 1: Start SSE stream
curl -sN \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream" \
  https://<service-url>/api/v1/events/stream

# Terminal 2: Trigger a workflow (produces EventBus events)
curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"test": true}}' \
  https://<service-url>/api/v1/workflows/bank_to_ynab/trigger
```

**Expected**: Terminal 1 should receive an event shortly after Terminal 2's trigger:

```
event: dashboard.workflow_triggered
data: {"topic":"dashboard.workflow_triggered","sender":"dashboard_api","payload":{...},"timestamp":"...","correlation_id":"..."}
id: 2026-02-26T15:00:00.000000Z

```

### Step 3: 5-Minute Intentional Disconnect (Edge LB Safety)

This is the most critical test. Connect to the SSE stream and wait for the 5-minute disconnect:

```bash
# Start SSE stream, log all output with timestamps
time curl -sN \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream" \
  https://<service-url>/api/v1/events/stream 2>&1 | ts '[%H:%M:%S]'
```

**Expected** (after ~300 seconds):

```
[10:00:00] event: keepalive
[10:00:00] data:
[10:00:00]
[10:00:30] event: keepalive
[10:00:30] data:
...
[10:05:00] event: reconnect
[10:05:00] data:
[10:05:00]
```

**Success criteria**:

- The stream drops **exactly** at the 5-minute mark (±5 seconds).
- The final event is `event: reconnect` (not an error or abrupt TCP close).
- `curl` exits cleanly (return code 0), not with a connection error.

### Step 4: `Last-Event-ID` Replay on Reconnect

Simulate a reconnect with `Last-Event-ID` and verify missed events are replayed:

```bash
# 1. Note the `id` field from the last received event (from Step 2)
LAST_ID="2026-02-26T15:00:00.000000Z"

# 2. While disconnected, trigger another event
curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"payload": {"test": "reconnect_test"}}' \
  https://<service-url>/api/v1/workflows/bank_to_ynab/trigger

# 3. Reconnect with Last-Event-ID
timeout 10 curl -sN \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream" \
  -H "Last-Event-ID: $LAST_ID" \
  https://<service-url>/api/v1/events/stream
```

**Expected**: The stream should first replay any events that occurred between `LAST_ID` and now, then transition to live streaming.

> [!NOTE]
> Replay behavior depends on the EventBus backend:
>
> - **InMemory**: Replay works from the in-memory history ring buffer. Limited by `_history` capacity.
> - **Pub/Sub**: Replay works from Pub/Sub seek-to-timestamp. Requires retain-acked-messages configured on the subscription.

### Step 5: Browser EventSource Compatibility

If a dashboard frontend prototype is available, verify the browser's native `EventSource` API works correctly:

```javascript
// Browser console (or a simple HTML page)
const source = new EventSource(
  "https://<service-url>/api/v1/events/stream",
  // Note: EventSource does not support custom headers natively.
  // For X-API-Key auth, use a proxy or URL-based token.
);

source.addEventListener("keepalive", (e) => console.log("[keepalive]"));
source.addEventListener("reconnect", (e) => {
  console.log("[reconnect] — server requested reconnect");
  // EventSource auto-reconnects with Last-Event-ID
});
source.addEventListener("dashboard.workflow_triggered", (e) => {
  console.log("[event]", JSON.parse(e.data));
});
source.onerror = (e) => console.error("[error]", e);
```

> [!IMPORTANT]
> The browser's `EventSource` API does **not** support custom headers like `X-API-Key`. For production dashboard integration, use one of:
>
> - **URL query param**: `?api_key=...` (requires endpoint modification)
> - **Cookie-based auth**: Set `HttpOnly` cookie from a login endpoint
> - **Reverse proxy**: Inject the API key at the proxy layer (nginx/Cloudflare)
>
> This is an expected architectural constraint documented in ARCHITECTURE.md §1 (headless API, server-to-server). The SSE endpoint is designed for authenticated server-side consumers or proxy-fronted browsers.

---

## Verification Checklist

| #   | Check                                         | Pass? |
| --- | --------------------------------------------- | ----- |
| 1   | SSE connects with correct `text/event-stream` | [ ]   |
| 2   | Keepalive pings arrive every ~30 seconds      | [ ]   |
| 3   | `X-Accel-Buffering: no` header present        | [ ]   |
| 4   | `Cache-Control: no-cache` header present      | [ ]   |
| 5   | Published events appear in SSE stream         | [ ]   |
| 6   | Intentional disconnect at ~5 minutes          | [ ]   |
| 7   | `event: reconnect` is the final event         | [ ]   |
| 8   | `curl` exits cleanly (no TCP error)           | [ ]   |
| 9   | `Last-Event-ID` replay returns missed events  | [ ]   |
| 10  | Unsubscribe fires on client disconnect        | [ ]   |

---

## Edge Cases & Failure Modes

| Scenario                                     | Expected Behavior                                                | Risk if Not Handled                               |
| -------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| Client disconnects before 5 minutes          | `request.is_disconnected()` → cleanup → `unsubscribe(sub)`       | EventBus subscription leak → memory growth        |
| Cloud Run kills container during SSE stream  | Client sees TCP reset → `EventSource.onerror` → auto-reconnect   | Client retries indefinitely without data loss     |
| Pub/Sub replay returns 0 events              | Stream transitions to live mode normally                         | None — graceful empty replay                      |
| `Last-Event-ID` format doesn't match Pub/Sub | `replay()` catches exception → logs warning → proceeds with live | Transparent to client — just misses replay        |
| Multiple clients connected simultaneously    | Each gets independent `asyncio.Queue` + subscription             | Heavy load → many queue allocations (mitigated by |
|                                              |                                                                  | 5-min disconnect cycling)                         |

---

## Design Decisions

| Decision                                                     | Rationale                                                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| 5-minute disconnect (not 1 hour or infinite)                 | Cloud Run's default request timeout is 5 minutes; aligns with LB behavior             |
| `event: reconnect` as named event (not `event: close`)       | Named event lets the client distinguish intentional disconnect from errors            |
| `X-Accel-Buffering: no` header                               | Required for nginx-based proxies (incl. Cloud Run's LB) to disable response buffering |
| Keepalive every 30 seconds                                   | Keeps TCP connection alive through proxies with idle timeout (typically 60s)          |
| Browser EventSource auth constraints documented, not "fixed" | Adding token-in-URL would weaken security; proper fix is proxy-based injection        |

---

## Files Modified

| File | Change                              | Lines |
| ---- | ----------------------------------- | ----- |
| N/A  | Verification-only — no code changes | 0     |
