"""
PubSubConnector — Gmail Push Notifications via Google Cloud Pub/Sub.

Wraps the Gmail push notification lifecycle as a reusable connector.
Any workflow that needs real-time email notifications can use this
connector with its own GmailConnector instance.

Usage:
    from autopilot.connectors import get_connector_registry
    pubsub = get_connector_registry().get("pubsub")
    await pubsub.start(gmail_connector)
    messages = await pubsub.handle_notification(pubsub_data)
"""

import base64
import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _PubSubConfig:
    """Platform-level PubSub configuration, read from environment."""

    gcp_pubsub_topic: str = ""
    gmail_push_label_names: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "_PubSubConfig":
        label_raw = os.environ.get("GMAIL_PUSH_LABEL_NAMES", "")
        labels = (
            [lbl.strip() for lbl in label_raw.split(",") if lbl.strip()]
            if label_raw
            else []
        )
        return cls(
            gcp_pubsub_topic=os.environ.get("GCP_PUBSUB_TOPIC", ""),
            gmail_push_label_names=labels,
        )


class PubSubConnector(BaseConnector):
    """
    Gmail Push Notification block using Cloud Pub/Sub.

    Manages the lifecycle of Gmail watch() registrations,
    processes incoming Pub/Sub webhook notifications, and
    fetches changed messages via the Gmail history API.

    Architecture:
      Gmail → Pub/Sub Topic → POST /gmail/webhook → history.list() → process_email()

    Watch Renewal:
      Gmail watch() expires after ~7 days. Cloud Scheduler
      (ping-bank-to-ynab) calls POST /gmail/watch/renew daily
      to keep it alive across scale-to-zero cycles.
    """

    @property
    def name(self) -> str:
        return "pubsub"

    @property
    def icon(self) -> str:
        return "📡"

    @property
    def description(self) -> str:
        return "Real-time Gmail push notifications via Google Cloud Pub/Sub"

    def __init__(self):
        self._gmail = None  # GmailConnector instance (injected)
        self._settings = None
        self._history_id: str | None = None
        self._watch_expiration: int = 0
        self._resolved_label_ids: list[str] = []
        self._label_id_to_name: dict[str, str] = {}  # Reverse map for email enrichment
        self._gmail_lock = (
            asyncio.Lock()
        )  # Serialize Gmail API calls (httplib2 not thread-safe)
        self._processed_ids: set[str] = (
            set()
        )  # Dedup: message IDs already processed this instance

    @property
    def history_id(self) -> str | None:
        """Current history checkpoint — messages after this ID are new."""
        return self._history_id

    @property
    def watch_expiration(self) -> datetime | None:
        """When the current watch() will expire."""
        if self._watch_expiration:
            return datetime.fromtimestamp(
                self._watch_expiration / 1000, tz=timezone.utc
            )
        return None

    @property
    def is_watch_active(self) -> bool:
        """Whether the watch is currently active (not expired)."""
        if not self._watch_expiration:
            return False
        now_millis = int(time.time() * 1000)
        return now_millis < self._watch_expiration

    @property
    def watch_status(self) -> dict:
        """Diagnostic snapshot of the current watch state."""
        return {
            "active": self.is_watch_active,
            "history_id": self._history_id,
            "expiration": str(self.watch_expiration) if self.watch_expiration else None,
            "expiration_epoch_ms": self._watch_expiration,
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Platform setup hook. Starts watching for Gmail push notifications.

        Idempotent: Gmail watch() is safe to call repeatedly.
        On every cold start, this re-registers the watch to ensure
        it hasn't expired while the instance was scaled to zero.
        """
        from autopilot.connectors import get_connector_registry

        try:
            gmail = get_connector_registry().get("gmail")
            logger.info(
                "pubsub_setup_cold_start", reason="re-registering watch on startup"
            )
            await self.start_watching(gmail)
        except Exception as e:
            logger.error("pubsub_setup_failed", error=str(e))

    async def start_watching(self, gmail_connector, settings=None) -> None:
        """
        Start the push notification service.

        Args:
            gmail_connector: A GmailConnector instance for Gmail API access.
            settings: Optional _PubSubConfig (auto-loaded from env if not provided).
        """
        self._gmail = gmail_connector
        self._settings = settings or _PubSubConfig.from_env()

        topic = self._settings.gcp_pubsub_topic
        if not topic:
            raise ValueError("GCP_PUBSUB_TOPIC is required for push ingestion mode")

        # Dynamically aggregate label names from ALL registered workflows
        from autopilot.registry import get_registry
        from autopilot.models import TriggerType

        registry = get_registry()
        required_label_names = set()

        # 1. Add labels from settings (global overrides)
        if self._settings.gmail_push_label_names:
            required_label_names.update(self._settings.gmail_push_label_names)

        # 2. Add labels from all enabled workflows
        for wf in registry.get_all_workflows():
            if not wf.manifest.enabled:
                continue
            for trigger in wf.manifest.triggers:
                # If trigger has specific label_ids (names in manifest), add them
                # Note: Manifest triggers usually define 'label_ids' as a list of strings (names or IDs).
                # We assume they are NAMES here because we resolve them below.
                # If they are IDs (like "INBOX"), resolve_label_ids handles them (it queries by name).
                # Wait, "INBOX" is an ID. "IMPORTANT" is an ID. User labels are names.
                # resolve_label_ids maps Name -> ID.
                # If a workflow uses "INBOX", we should probably just use it directly.
                # unique standard IDs: INBOX, SPAM, TRASH, UNREAD, STARRED, IMPORTANT, SENT, DRAFT.
                # anything else is a user label name.
                if trigger.type == TriggerType.GMAIL_PUSH and trigger.label_ids:
                    required_label_names.update(trigger.label_ids)

        if required_label_names:
            self._resolved_label_ids = await asyncio.to_thread(
                self._gmail.resolve_label_ids,
                list(required_label_names),
            )
            logger.info(
                "pubsub_labels_resolved",
                sources="workflows+settings",
                names=list(required_label_names),
                ids=self._resolved_label_ids,
            )

        # Build full label ID → Name reverse map for email enrichment.
        # This lets _fetch_message_detail include human-readable label names
        # so workflows can match on names (e.g. "Bancolombia") rather than IDs.
        try:
            all_labels = await asyncio.to_thread(
                lambda: self._gmail.service.users().labels().list(userId="me").execute()
            )
            self._label_id_to_name = {
                label["id"]: label["name"] for label in all_labels.get("labels", [])
            }
            logger.info(
                "pubsub_label_map_built",
                total_labels=len(self._label_id_to_name),
            )
        except Exception as e:
            logger.warning("pubsub_label_map_failed", error=str(e))

        logger.info(
            "pubsub_starting", topic=topic, label_filter=self._resolved_label_ids
        )

        # Register initial watch (Cloud Scheduler handles renewals)
        await self._register_watch()

        logger.info(
            "pubsub_started",
            history_id=self._history_id,
            watch_expires=str(self.watch_expiration),
        )

    async def stop_watching(self) -> None:
        """Stop receiving push notifications."""
        logger.info("pubsub_stopping")

        try:
            await asyncio.to_thread(self._stop_watch)
        except Exception as e:
            logger.warning("pubsub_stop_watch_failed", error=str(e))

        logger.info("pubsub_stopped")

    async def teardown(self) -> None:
        """Platform shutdown hook."""
        # Intentionally do not call stop_watching() here.
        # The watch must outlive the ephemeral Cloud Run instances
        # (which scale to zero) so that Gmail continues sending
        # notifications for new emails.
        pass

    async def health_check(self) -> bool:
        """Check if the watch is active."""
        return self.is_watch_active

    async def force_rewatch(self) -> dict:
        """Force re-register the Gmail watch.

        Designed to be called from an HTTP endpoint (e.g. by Cloud Scheduler)
        to guarantee the watch stays alive even when the renewal background
        task has been killed by Cloud Run scaling to zero.

        Returns the updated watch_status dict.
        """
        logger.info(
            "pubsub_force_rewatch",
            was_active=self.is_watch_active,
            previous_expiration=str(self.watch_expiration),
        )
        await self._register_watch()
        logger.info(
            "pubsub_force_rewatch_complete",
            is_active=self.is_watch_active,
            new_expiration=str(self.watch_expiration),
        )
        return self.watch_status

    # ── Watch Management ─────────────────────────────────────────────

    async def _register_watch(self) -> None:
        """Register a watch() with Gmail for push notifications."""
        topic = self._settings.gcp_pubsub_topic

        request_body = {
            "topicName": topic,
            "labelFilterBehavior": "INCLUDE",
        }

        if self._resolved_label_ids:
            request_body["labelIds"] = self._resolved_label_ids

        try:
            response = await asyncio.to_thread(
                lambda: (
                    self._gmail.service.users()
                    .watch(userId="me", body=request_body)
                    .execute()
                )
            )
        except Exception as e:
            error_msg = str(e)
            if "Only one user push notification client allowed" in error_msg:
                logger.warning(
                    "pubsub_stolen_watch_detected",
                    topic=topic,
                    recovering=True,
                )
                await asyncio.to_thread(self._stop_watch)
                try:
                    response = await asyncio.to_thread(
                        lambda: (
                            self._gmail.service.users()
                            .watch(userId="me", body=request_body)
                            .execute()
                        )
                    )
                except Exception as retry_e:
                    logger.error(
                        "pubsub_watch_retry_failed",
                        error=str(retry_e),
                        topic=topic,
                    )
                    raise
            else:
                logger.error(
                    "pubsub_watch_registration_failed",
                    error=error_msg,
                    topic=topic,
                )
                raise

        self._history_id = str(response.get("historyId", ""))
        self._watch_expiration = int(response.get("expiration", 0))

        logger.info(
            "pubsub_watch_registered",
            history_id=self._history_id,
            expiration=str(self.watch_expiration),
            topic=topic,
        )

    def _stop_watch(self) -> None:
        """Stop receiving push notifications for this user."""
        try:
            self._gmail.service.users().stop(userId="me").execute()
            logger.info("pubsub_watch_stopped")
        except Exception as e:
            logger.warning("pubsub_watch_stop_failed", error=str(e))

    # ── Notification Handling ────────────────────────────────────────

    async def handle_notification(self, pubsub_message: dict) -> list[dict]:
        """
        Process an incoming Pub/Sub notification from Gmail.

        Decodes the notification, fetches new messages via history.list(),
        and returns them in standard format.

        Cold-start race fix (v2): On cold start, ``_register_watch()`` sets
        ``_history_id`` to the CURRENT Gmail historyId, which is the SAME
        or AHEAD of the email that triggered this notification.  Gmail's
        history.list(startHistoryId=X) returns changes AFTER X, not AT X.
        So we subtract 1 from the effective historyId to look back far enough.
        If history still returns 0, we fall back to fetching recent unread.
        """
        try:
            encoded_data = pubsub_message.get("message", {}).get("data", "")
            if not encoded_data:
                logger.warning("pubsub_empty_notification")
                return []

            decoded = base64.urlsafe_b64decode(encoded_data).decode("utf-8")
            notification = json.loads(decoded)
            new_history_id = str(notification.get("historyId", ""))

            logger.info(
                "pubsub_notification_received",
                new_history_id=new_history_id,
                previous_history_id=self._history_id,
            )
        except Exception as e:
            logger.error("pubsub_decode_failed", error=str(e))
            return []

        if not self._history_id:
            logger.warning("pubsub_no_history_id")
            return []

        # ── Cold-start race condition fix (v2) ────────────────────────
        # On cold start, _register_watch() obtains the CURRENT historyId
        # which is the SAME as (or ahead of) the triggering email's.
        # Gmail history.list(startHistoryId=X) returns changes AFTER X.
        # So we use min(notification, watch) - 1 to look back far enough.
        effective_history_id = self._history_id
        is_cold_start_race = False
        if new_history_id and new_history_id.isdigit() and self._history_id.isdigit():
            min_id = min(int(new_history_id), int(self._history_id))
            # Subtract 1 to ensure we see the email AT this historyId
            effective_history_id = str(max(1, min_id - 1))
            if int(new_history_id) <= int(self._history_id):
                is_cold_start_race = True
                logger.warning(
                    "pubsub_cold_start_race_detected",
                    notification_history_id=new_history_id,
                    watch_history_id=self._history_id,
                    effective_history_id=effective_history_id,
                    action="using min(notification, watch) - 1",
                )

        async with self._gmail_lock:
            try:
                messages = await self._get_new_messages(effective_history_id)
                if new_history_id:
                    self._history_id = new_history_id

                # If history returned nothing on a cold-start race, use fallback
                if not messages and is_cold_start_race:
                    logger.warning(
                        "pubsub_cold_start_zero_messages",
                        effective_history_id=effective_history_id,
                        action="falling back to recent unread fetch",
                    )
                    messages = await self._fallback_fetch_recent()

                # ── Deduplication ─────────────────────────────────────
                # When multiple Pub/Sub notifications arrive during the
                # same cold-start burst, history.list() can return the
                # same emails for both requests. Filter out already-seen IDs.
                if messages:
                    before_count = len(messages)
                    messages = [
                        m for m in messages if m.get("id") not in self._processed_ids
                    ]
                    new_ids = {m.get("id") for m in messages if m.get("id")}
                    self._processed_ids.update(new_ids)
                    # Cap the set to prevent unbounded growth
                    if len(self._processed_ids) > 500:
                        self._processed_ids = set(list(self._processed_ids)[-200:])
                    if before_count != len(messages):
                        logger.info(
                            "pubsub_dedup_filtered",
                            before=before_count,
                            after=len(messages),
                            duplicates_skipped=before_count - len(messages),
                        )

                logger.info(
                    "pubsub_messages_fetched",
                    count=len(messages),
                    new_checkpoint=self._history_id,
                    cold_start_race=is_cold_start_race,
                )
                return messages
            except Exception as e:
                logger.error("pubsub_history_fetch_failed", error=str(e))
                if new_history_id:
                    self._history_id = new_history_id
                # On cold-start errors (like SSL issues), try fallback
                if is_cold_start_race:
                    logger.warning(
                        "pubsub_cold_start_error_fallback",
                        error=str(e),
                        action="falling back to recent unread fetch",
                    )
                    try:
                        return await self._fallback_fetch_recent()
                    except Exception as fb_err:
                        logger.error("pubsub_fallback_also_failed", error=str(fb_err))
                return []

    async def _get_new_messages(self, start_history_id: str) -> list[dict]:
        """Fetch ALL new messages since the given history ID.

        The connector is a platform-level pipe — it does NOT filter by
        sender, labels, or read status.  Each workflow's
        ``_matches_gmail_trigger()`` handles its own filtering.
        """
        try:
            history_records = await asyncio.to_thread(
                self._fetch_history_records, start_history_id
            )
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "notFound" in error_str:
                logger.warning(
                    "pubsub_history_id_expired",
                    history_id=start_history_id,
                    fallback="direct_message_search",
                )
                return await self._fallback_fetch_recent()
            raise

        if not history_records:
            return []

        # Collect ALL messagesAdded — no label or read-status filtering
        message_ids: set[str] = set()
        for record in history_records:
            for added in record.get("messagesAdded", []):
                msg = added.get("message", {})
                msg_id = msg.get("id")
                if msg_id:
                    message_ids.add(msg_id)

        if not message_ids:
            return []

        logger.info("pubsub_candidate_messages", count=len(message_ids))

        emails = []
        for msg_id in message_ids:
            try:
                email = await asyncio.to_thread(self._fetch_message_detail, msg_id)
                if email:
                    emails.append(email)
            except Exception as e:
                logger.error(
                    "pubsub_message_fetch_failed", message_id=msg_id, error=str(e)
                )

        return emails

    def _fetch_history_records(self, start_history_id: str) -> list[dict]:
        """Synchronous call to Gmail history.list() API.

        Does NOT filter by label — the connector is a platform-level pipe
        that fetches ALL changes. Each workflow's ``_matches_gmail_trigger()``
        handles its own label filtering. This allows multiple workflows
        with different Gmail labels to coexist correctly.

        The Gmail ``watch()`` already limits Pub/Sub notifications to the
        registered labels (INCLUDE behavior = OR logic), so this only
        processes relevant changes.
        """
        all_records = []
        page_token = None

        while True:
            params = {
                "userId": "me",
                "startHistoryId": start_history_id,
                "historyTypes": ["messageAdded"],
            }
            if page_token:
                params["pageToken"] = page_token

            response = self._gmail.service.users().history().list(**params).execute()

            records = response.get("history", [])
            all_records.extend(records)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return all_records

    def _fetch_message_detail(self, message_id: str) -> dict | None:
        """Fetch full message details and extract body + labelIds.

        Includes ``labelIds`` so that downstream workflows can filter by
        Gmail labels via ``_matches_gmail_trigger()``.
        """
        msg = (
            self._gmail.service.users()
            .messages()
            .get(userId="me", id=message_id)
            .execute()
        )

        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = next(
            (h["value"] for h in headers if h["name"].lower() == "subject"),
            "No Subject",
        )
        from_addr = next(
            (h["value"] for h in headers if h["name"].lower() == "from"),
            "",
        )

        body = self._gmail.decode_body(payload)

        return {
            "id": message_id,
            "subject": subject,
            "body": body,
            "from": from_addr,
            "snippet": msg.get("snippet", ""),
            "labelIds": msg.get("labelIds", []),
            "labelNames": [
                self._label_id_to_name.get(lid, lid) for lid in msg.get("labelIds", [])
            ],
        }

    async def _fallback_fetch_recent(self) -> list[dict]:
        """Fallback: fetch the latest messages when history.list() fails.

        Used when the historyId is too old (expired) and the Gmail API
        returns a 404, or when the cold-start race returns 0 messages.

        Does NOT filter by label — same principle as ``_fetch_history_records``.
        Each workflow's ``_matches_gmail_trigger()`` handles filtering.
        We use a broad ``newer_than:1h`` query to limit scope.
        """
        try:
            results = await asyncio.to_thread(
                lambda: (
                    self._gmail.service.users()
                    .messages()
                    .list(userId="me", q="newer_than:1h", maxResults=10)
                    .execute()
                )
            )
            message_ids = [m["id"] for m in results.get("messages", [])]
        except Exception as e:
            logger.error("pubsub_fallback_search_failed", error=str(e))
            return []

        emails = []
        for msg_id in message_ids:
            try:
                email = await asyncio.to_thread(self._fetch_message_detail, msg_id)
                if email:
                    emails.append(email)
            except Exception as e:
                logger.error(
                    "pubsub_fallback_fetch_failed", message_id=msg_id, error=str(e)
                )

        logger.info("pubsub_fallback_fetched", count=len(emails))
        return emails
