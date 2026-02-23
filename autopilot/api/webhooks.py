"""
Platform Routes — Unified HTTP endpoints for the Autopilot platform.

Handles:
  - Generic Webhooks: POST /api/webhook/{path}
  - Gmail Push: POST /gmail/webhook (Pub/Sub) → publishes email.received to AgentBus
  - Telegram: POST /telegram/webhook → publishes telegram.message_received to AgentBus
"""

from __future__ import annotations

import os
import structlog
from fastapi import APIRouter, HTTPException, Request

from autopilot.router import get_router
from autopilot.connectors import get_connector_registry
from autopilot.core.bus import get_event_bus

logger = structlog.get_logger(__name__)

router = APIRouter()

# The secret token used to verify Telegram webhook requests
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
# Authorized Telegram chat ID — if set, only this chat can interact
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ── Generic Webhook ──────────────────────────────────────────────────


@router.post("/api/webhook/{webhook_path:path}")
async def generic_webhook(webhook_path: str, request: Request):
    """
    Route a webhook request to the workflow that handles the given path.
    ID is resolved via TriggerConfig.path matches.
    """
    # Parse body (JSON or Form)
    try:
        data = await request.json()
    except Exception:
        data = {}

    # Add headers/query for completeness if needed
    data.update(
        {
            "_headers": dict(request.headers),
            "_query": dict(request.query_params),
        }
    )

    router_svc = get_router()
    try:
        # Prepend slash if missing, or handle strict matching
        path_key = (
            f"/{webhook_path}" if not webhook_path.startswith("/") else webhook_path
        )

        run = await router_svc.route_webhook(path_key, data)
        return {
            "status": "success",
            "run_id": run.id,
            "workflow_id": run.workflow_id,
        }
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"No workflow found for webhook path: {path_key}"
        )
    except Exception as e:
        logger.error("webhook_failed", path=webhook_path, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Gmail Pub/Sub Webhook (Event-Driven Adapter) ────────────────────


@router.post("/gmail/webhook")
async def gmail_push_webhook(request: Request):
    """
    Handle Google Cloud Pub/Sub push notifications for Gmail.

    This is a **thin event adapter** — it decodes the Pub/Sub message,
    fetches new emails, and publishes each one as an ``email.received``
    event on the AgentBus. Subscribed workflows react independently.

    The bus awaits all subscribers via ``asyncio.gather`` before this
    endpoint responds, ensuring Pub/Sub only ACKs after full processing.

    Flow:
      1. Receive Pub/Sub message
      2. PubSubConnector decodes & fetches new emails
      3. Publish each email as ``email.received`` on the AgentBus
      4. Subscribed workflows self-match and execute
    """
    try:
        pubsub_message = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. Decode & Fetch
    registry = get_connector_registry()
    try:
        pubsub = registry.get("pubsub")
    except KeyError:
        raise HTTPException(status_code=503, detail="PubSub connector not initialized")

    try:
        # Returns list of dicts: {'id', 'snippet', 'body', 'from', 'subject', ...}
        emails = await pubsub.handle_notification(pubsub_message)
    except Exception as e:
        logger.error("pubsub_handler_failed", error=str(e))
        # Return 200 to acknowledge Pub/Sub (otherwise it retries endlessly)
        return {"status": "error", "message": "Handler failed, acking to stop retry"}

    if not emails:
        return {"status": "ignored", "reason": "no_new_emails"}

    # 2. Publish each email as an event (bus awaits all subscribers)
    bus = get_event_bus()
    for email in emails:
        event_payload = {
            "email_id": email.get("id", ""),
            "sender": email.get("from", ""),
            "subject": email.get("subject", ""),
            "body": email.get("body", ""),
            "label_ids": email.get("labelIds", []),
            "labelNames": email.get("labelNames", []),
            "source": "pubsub",
            # Preserve full email for workflows that need it
            "email": email,
        }

        logger.info(
            "email_event_publishing",
            email_id=event_payload["email_id"],
            sender=event_payload["sender"],
            subject=event_payload["subject"][:80],
        )

        await bus.publish("email.received", event_payload, sender="gmail_webhook")

    return {"status": "ok", "emails_published": len(emails)}


# ── Telegram Webhook (Event-Driven Adapter) ─────────────────────────


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Handle incoming Telegram Bot API webhook updates.

    This is a **thin event adapter** — it validates the request,
    extracts the message text and chat ID, and publishes a
    ``telegram.message_received`` event on the AgentBus.
    Subscribed workflows react independently.

    Flow:
      1. Verify secret token (optional)
      2. Verify authorized chat ID (optional)
      3. Extract message text and chat_id
      4. Publish ``telegram.message_received`` on the AgentBus
    """
    # Verify secret token if configured
    if TELEGRAM_WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if token != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret token")

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract message — ignore non-text updates silently
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"status": "ignored", "reason": "no_text_message"}

    text = message.get("text", "")
    if not text:
        return {"status": "ignored", "reason": "no_text_message"}

    chat_id = str(message.get("chat", {}).get("id", ""))
    if not chat_id:
        return {"status": "ignored", "reason": "no_chat_id"}

    # Verify authorized user if configured
    if TELEGRAM_CHAT_ID and chat_id != TELEGRAM_CHAT_ID:
        logger.warning(
            "telegram_unauthorized_chat",
            chat_id=chat_id,
            authorized=TELEGRAM_CHAT_ID,
        )
        return {"status": "ignored", "reason": "unauthorized_chat"}

    logger.info(
        "telegram_message_received",
        chat_id=chat_id,
        text_length=len(text),
        text_preview=text[:80],
    )

    # Publish to EventBus — subscribed workflows react independently
    bus = get_event_bus()
    await bus.publish(
        "telegram.message_received",
        {
            "message": text,
            "telegram_chat_id": chat_id,
            "update": update,
            "source": "telegram_webhook",
        },
        sender="telegram_webhook",
    )

    return {"status": "ok", "event": "telegram.message_received"}
