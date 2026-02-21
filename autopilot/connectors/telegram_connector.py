"""
TelegramConnector — Async-only Telegram Bot API integration block.

Wraps the Telegram Bot API as a platform connector. Provides message sending,
photo/document delivery, and webhook management — all async via native httpx
(no third-party Telegram libraries).

Usage:
    from autopilot.connectors import get_connector_registry
    telegram = get_connector_registry().get("telegram")
    me = await telegram.client.get_me()
    await telegram.client.send_message(chat_id="123", text="Hello!")
"""

from __future__ import annotations

import os
import time
import structlog
import httpx
from typing import Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)

# ── Exceptions ───────────────────────────────────────────────────────

from autopilot.errors import (
    ConnectorError as TelegramError,
    ConnectorRateLimitError as TelegramRateLimitError,
)


# ── Async Telegram Client ────────────────────────────────────────────


class AsyncTelegramClient:
    """
    Production-grade async Telegram Bot API client.

    Features:
    - httpx.AsyncClient with HTTP/2 and connection pooling
    - Automatic retry with exponential backoff on HTTP 429
    - Structured logging for every API call
    - Full webhook management (set, delete, info)
    - Message sending (text, photo, document)
    """

    def __init__(self, bot_token: str):
        self._token = bot_token
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            http2=True,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Execute an async API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise TelegramError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise TelegramError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            # Telegram rate limit — extract retry_after if available
            retry_after = None
            try:
                body = resp.json()
                retry_after = body.get("parameters", {}).get("retry_after")
            except Exception:
                pass
            logger.warning(
                "telegram_rate_limited",
                path=path,
                latency_ms=round(latency_ms),
                retry_after=retry_after,
            )
            raise TelegramRateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise TelegramError(
                "Authentication failed — check your bot token", detail="401"
            )

        data = resp.json()

        # Telegram API always wraps responses in {"ok": bool, "result": ...}
        if not data.get("ok", False):
            error_code = data.get("error_code", resp.status_code)
            description = data.get("description", "Unknown error")
            raise TelegramError(
                f"API error {error_code}: {description}",
                detail=str(error_code),
            )

        logger.debug(
            "telegram_request",
            method=method,
            path=path,
            status=resp.status_code,
            latency_ms=round(latency_ms),
            http_version=resp.http_version,
        )

        return data.get("result")

    # ── Bot Info ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def get_me(self) -> dict:
        """
        Get basic info about the bot (id, name, username).
        Useful for verifying the token is valid.
        """
        result = await self._request("GET", "/getMe")
        logger.info(
            "telegram_get_me",
            bot_id=result.get("id"),
            bot_username=result.get("username"),
        )
        return result

    # ── Sending Messages ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> dict:
        """
        Send a text message to a chat.

        Args:
            chat_id: Target chat ID or @channel_username.
            text: Message text (up to 4096 characters).
            parse_mode: Optional "Markdown", "MarkdownV2", or "HTML".
            disable_notification: Send silently if True.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if disable_notification:
            payload["disable_notification"] = True

        logger.info("telegram_sending_message", chat_id=chat_id, text_length=len(text))
        result = await self._request("POST", "/sendMessage", json=payload)
        logger.info("telegram_message_sent", message_id=result.get("message_id"))
        return result

    async def send_message_string(self, chat_id: str, text: str) -> str:
        """
        AI-friendly wrapper — sends a message and returns a human-readable
        confirmation string.
        """
        result = await self.send_message(chat_id=chat_id, text=text)
        msg_id = result.get("message_id", "unknown")
        return f"Message sent successfully (message_id: {msg_id})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def send_photo(
        self,
        chat_id: str,
        photo: str,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> dict:
        """
        Send a photo to a chat.

        Args:
            chat_id: Target chat ID or @channel_username.
            photo: Photo file_id (reuse), HTTP URL, or "attach://" for uploads.
            caption: Optional photo caption (up to 1024 characters).
            parse_mode: Optional "Markdown", "MarkdownV2", or "HTML" for caption.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "photo": photo,
        }
        if caption:
            payload["caption"] = caption
        if parse_mode:
            payload["parse_mode"] = parse_mode

        logger.info("telegram_sending_photo", chat_id=chat_id)
        result = await self._request("POST", "/sendPhoto", json=payload)
        logger.info("telegram_photo_sent", message_id=result.get("message_id"))
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def send_document(
        self,
        chat_id: str,
        document: str,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> dict:
        """
        Send a document to a chat.

        Args:
            chat_id: Target chat ID or @channel_username.
            document: Document file_id, HTTP URL, or "attach://" for uploads.
            caption: Optional document caption (up to 1024 characters).
            parse_mode: Optional "Markdown", "MarkdownV2", or "HTML" for caption.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "document": document,
        }
        if caption:
            payload["caption"] = caption
        if parse_mode:
            payload["parse_mode"] = parse_mode

        logger.info("telegram_sending_document", chat_id=chat_id)
        result = await self._request("POST", "/sendDocument", json=payload)
        logger.info("telegram_document_sent", message_id=result.get("message_id"))
        return result

    # ── Webhook Management ───────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def set_webhook(
        self,
        url: str,
        secret_token: Optional[str] = None,
        allowed_updates: Optional[list[str]] = None,
        max_connections: int = 40,
        drop_pending_updates: bool = False,
    ) -> bool:
        """
        Set a webhook URL for receiving updates.

        Args:
            url: HTTPS URL to receive updates. Empty string removes the webhook.
            secret_token: Secret token sent in X-Telegram-Bot-Api-Secret-Token header.
            allowed_updates: List of update types to receive (e.g. ["message", "callback_query"]).
            max_connections: Max simultaneous connections (1–100, default 40).
            drop_pending_updates: Drop all pending updates when setting webhook.
        """
        payload: dict[str, Any] = {"url": url}
        if secret_token:
            payload["secret_token"] = secret_token
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        if max_connections != 40:
            payload["max_connections"] = max_connections
        if drop_pending_updates:
            payload["drop_pending_updates"] = True

        logger.info("telegram_setting_webhook", url=url)
        result = await self._request("POST", "/setWebhook", json=payload)
        logger.info("telegram_webhook_set", success=result)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        """
        Remove the current webhook integration.

        Args:
            drop_pending_updates: Drop all pending updates.
        """
        payload: dict[str, Any] = {}
        if drop_pending_updates:
            payload["drop_pending_updates"] = True

        logger.info("telegram_deleting_webhook")
        result = await self._request("POST", "/deleteWebhook", json=payload)
        logger.info("telegram_webhook_deleted", success=result)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def get_webhook_info(self) -> dict:
        """
        Get current webhook status (URL, pending update count, errors, etc.).
        """
        result = await self._request("GET", "/getWebhookInfo")
        logger.info(
            "telegram_webhook_info",
            url=result.get("url", ""),
            pending_update_count=result.get("pending_update_count", 0),
        )
        return result

    # ── Polling (dev/testing) ────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TelegramRateLimitError),
    )
    async def get_updates(
        self,
        offset: Optional[int] = None,
        limit: int = 100,
        timeout: int = 0,
    ) -> list[dict]:
        """
        Receive incoming updates via long polling (for dev/testing).
        Not recommended for production — use webhooks instead.

        Args:
            offset: Identifier of the first update to return.
            limit: Max number of updates (1–100, default 100).
            timeout: Long polling timeout in seconds (0 = short poll).
        """
        params: dict[str, Any] = {"limit": limit, "timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        result = await self._request("GET", "/getUpdates", params=params)
        logger.debug("telegram_updates_received", count=len(result))
        return result

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self):
        """Close the underlying HTTP/2 connection pool."""
        await self._client.aclose()


# ── Connector ────────────────────────────────────────────────────────


class TelegramConnector(BaseConnector):
    """
    Telegram Bot API integration block — send messages, photos, documents,
    and manage webhooks for receiving updates.

    Provides an async-only client via `self.client`.
    """

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def icon(self) -> str:
        return "📱"

    @property
    def description(self) -> str:
        return "Send messages and manage Telegram bot webhooks"

    def __init__(self):
        self._client: AsyncTelegramClient | None = None

    @property
    def client(self) -> AsyncTelegramClient:
        """Get the async Telegram client. Lazy-initializes on first access."""
        if self._client is None:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            if not token:
                raise TelegramError(
                    "TELEGRAM_BOT_TOKEN environment variable is not set"
                )
            self._client = AsyncTelegramClient(bot_token=token)
        return self._client

    async def setup(self) -> None:
        """Pre-initialize the client."""
        _ = self.client

    async def teardown(self) -> None:
        """Close the HTTP/2 connection pool."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """
        Verify bot token validity via getMe.
        """
        try:
            await self.client.get_me()
            return True
        except Exception:
            return False
