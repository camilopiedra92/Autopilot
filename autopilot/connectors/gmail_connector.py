"""
GmailConnector — Reusable Gmail integration block.

Wraps the Gmail API client as a platform connector. Any workflow
that needs to read emails, mark-as-read, or resolve labels can
use this connector without duplicating Gmail auth/client logic.

Usage:
    from autopilot.connectors import get_connector_registry
    gmail = get_connector_registry().get("gmail")
    emails = gmail.get_unread_emails()
"""

from __future__ import annotations

import os
import base64
import re
import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailConnector(BaseConnector):
    """
    Gmail integration block — read, filter, and manage emails.

    Capabilities:
      - OAuth2 authentication with token refresh
      - Fetch unread emails with sender filtering
      - Mark messages as read
      - Resolve human-readable label names to IDs
      - Decode email bodies (plain text + HTML stripping)
    """

    @property
    def name(self) -> str:
        return "gmail"

    @property
    def icon(self) -> str:
        return "📧"

    @property
    def description(self) -> str:
        return "Read, filter, and manage Gmail messages via the Gmail API"

    def __init__(self):
        self._service = None

    # ── Auth ──────────────────────────────────────────────────────────

    # Paths to search for OAuth files (Cloud Run mounts → local fallback)
    _CREDENTIALS_PATHS = [
        "/secrets/credentials/credentials.json",  # Cloud Run secret mount
        "credentials.json",  # Local development
    ]
    _TOKEN_PATHS = [
        "/secrets/token/token.json",  # Cloud Run secret mount
        "token.json",  # Local development
    ]

    def _find_file(self, candidates: list[str]) -> str | None:
        """Return the first path that exists, or None."""
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _authenticate(self):
        """Authenticate with Gmail API using OAuth2."""
        creds = None

        token_path = self._find_file(self._TOKEN_PATHS)
        credentials_path = self._find_file(self._CREDENTIALS_PATHS)

        if token_path:
            logger.info("gmail_loading_token", path=token_path)
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("gmail_refreshing_token")
                creds.refresh(Request())
                # Save refreshed token (only if writable path exists)
                writable_token = "token.json"
                try:
                    with open(writable_token, "w") as f:
                        f.write(creds.to_json())
                except OSError:
                    logger.warning(
                        "gmail_token_save_skipped", reason="read-only filesystem"
                    )
            else:
                if not credentials_path:
                    raise FileNotFoundError(
                        "credentials.json not found. Searched: "
                        + ", ".join(self._CREDENTIALS_PATHS)
                    )
                logger.info("gmail_starting_oauth_flow", path=credentials_path)
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

                with open("token.json", "w") as token:
                    token.write(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        logger.info("gmail_authenticated")

    @property
    def service(self):
        """Lazy-init Gmail service."""
        if self._service is None:
            self._authenticate()
        return self._service

    async def setup(self) -> None:
        """Pre-authenticate on startup."""
        try:
            _ = self.service
        except Exception as e:
            logger.warning("gmail_setup_failed", error=str(e))

    async def health_check(self) -> bool:
        """Check if Gmail API is reachable."""
        try:
            self.service.users().getProfile(userId="me").execute()
            return True
        except Exception:
            return False

    # ── Email Operations ─────────────────────────────────────────────

    def resolve_label_ids(self, label_names: list[str]) -> list[str]:
        """Resolve human-readable Gmail label names to internal IDs."""
        results = self.service.users().labels().list(userId="me").execute()
        all_labels = results.get("labels", [])
        name_to_id = {label["name"]: label["id"] for label in all_labels}

        resolved = []
        for name in label_names:
            if lid := name_to_id.get(name):
                resolved.append(lid)
                logger.info("gmail_label_resolved", label_name=name, label_id=lid)
            else:
                logger.warning(
                    "gmail_label_not_found",
                    label_name=name,
                    available=[
                        lbl["name"] for lbl in all_labels if lbl.get("type") == "user"
                    ],
                )
        return resolved

    def _strip_html(self, html: str) -> str:
        """Basic HTML → plain text conversion."""
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def decode_body(self, payload: dict) -> str:
        """Extract and decode email body from Gmail payload."""
        data = ""

        if "parts" in payload:
            for mime in ("text/plain", "text/html"):
                for part in payload["parts"]:
                    if part["mimeType"] == mime and "data" in part.get("body", {}):
                        data = part["body"]["data"]
                        break
                if data:
                    break
        elif "body" in payload and "data" in payload["body"]:
            data = payload["body"]["data"]

        if not data:
            return "[No body found]"

        try:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8")
            if "<html" in decoded.lower() or "<div" in decoded.lower():
                decoded = self._strip_html(decoded)
            return decoded
        except Exception as e:
            logger.warning("gmail_decode_error", error=str(e))
            return "[Error decoding body]"

    def get_unread_emails(self, sender_filter: str) -> list[dict]:
        """Fetch unread emails from a specific sender."""
        query = f"from:{sender_filter} is:unread"

        results = self.service.users().messages().list(userId="me", q=query).execute()
        messages = results.get("messages", [])

        logger.info("gmail_unread_found", count=len(messages), sender=sender_filter)

        emails = []
        for message in messages:
            msg = (
                self.service.users()
                .messages()
                .get(userId="me", id=message["id"])
                .execute()
            )
            payload = msg["payload"]
            headers = payload.get("headers", [])
            subject = next(
                (h["value"] for h in headers if h["name"].lower() == "subject"),
                "No Subject",
            )

            body = self.decode_body(payload)

            emails.append(
                {
                    "id": message["id"],
                    "subject": subject,
                    "body": body,
                    "snippet": msg.get("snippet", ""),
                }
            )

        return emails

    def mark_as_read(self, message_id: str) -> None:
        """Mark a message as read."""
        self.service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        logger.info("gmail_marked_read", message_id=message_id)
