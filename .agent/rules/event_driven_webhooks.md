# Event-Driven Webhook Architecture

> [!CAUTION]
> This rule is **mandatory** for all external triggers (webhooks, Pub/Sub, etc.).
> Violations create architectural debt and inconsistency.

## Rule: Webhooks are Platform-Level Thin Adapters

**NEVER** define HTTP endpoints inside workflow code (`register_routes`).
**ALWAYS** follow the event-driven adapter pattern:

### Pattern

1. **Platform layer** (`autopilot/api/webhooks.py`): Thin HTTP adapter that:
   - Validates the request (auth, secret tokens, format)
   - Extracts the relevant payload
   - Publishes a typed event to the `EventBus`
   - Returns immediately — **zero business logic**

2. **Workflow layer** (`workflow.py` → `setup()`): Subscribes to the event via `get_subscriber_registry().register()` and reacts independently.

### Why

- **Separation of concerns**: HTTP is infrastructure, not business logic
- **Consistency**: All triggers follow the same EventBus pattern
- **Testability**: Workflows are testable without HTTP/FastAPI
- **Composability**: Multiple workflows can subscribe to the same event
- **No `from __future__ import annotations` + FastAPI conflicts**

### Reference Implementations

```
Gmail:     POST /gmail/webhook     → bus.publish("email.received")              → bank_to_ynab subscribes
Telegram:  POST /telegram/webhook  → bus.publish("telegram.message_received")   → conversational_assistant subscribes
```

### Workflow Code Must NEVER

- Import `FastAPI`, `APIRouter`, `Request`, or `HTTPException`
- Define `register_routes()`
- Handle auth/secret tokens (that's platform responsibility)
- Accept raw HTTP requests

### Workflow Code MUST

- Define `async def setup()` to register event subscribers
- Define `async def _on_<event>(self, msg)` handlers
- Call `self.run(TriggerType.WEBHOOK, payload)` from the handler
