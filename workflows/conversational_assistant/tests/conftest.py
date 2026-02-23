"""
Shared pytest fixtures for the Conversational Assistant test suite.
"""

import pytest
from unittest.mock import AsyncMock

from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set default environment variables only when real keys are not present."""
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    if not os.environ.get("GOOGLE_GENAI_API_KEY"):
        monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key")
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    if not os.environ.get("TODOIST_API_TOKEN"):
        monkeypatch.setenv("TODOIST_API_TOKEN", "test-todoist-token")
    if not os.environ.get("YNAB_ACCESS_TOKEN"):
        monkeypatch.setenv("YNAB_ACCESS_TOKEN", "test-ynab-token")


# ── Sample Telegram Updates ──────────────────────────────────────────


@pytest.fixture
def sample_text_update():
    """A realistic Telegram Update with a text message."""
    return {
        "update_id": 123456789,
        "message": {
            "message_id": 42,
            "from": {
                "id": 1093871758,
                "is_bot": False,
                "first_name": "Camilo",
                "language_code": "es",
            },
            "chat": {
                "id": 1093871758,
                "first_name": "Camilo",
                "type": "private",
            },
            "date": 1708700000,
            "text": "Recuérdame comprar leche mañana",
        },
    }


@pytest.fixture
def sample_photo_update():
    """A Telegram Update with a photo (no text) — should be ignored."""
    return {
        "update_id": 123456790,
        "message": {
            "message_id": 43,
            "from": {"id": 1093871758, "is_bot": False, "first_name": "Camilo"},
            "chat": {"id": 1093871758, "type": "private"},
            "date": 1708700001,
            "photo": [{"file_id": "abc123", "width": 320, "height": 240}],
        },
    }


@pytest.fixture
def sample_callback_query_update():
    """A Telegram Update with no message at all — should be ignored."""
    return {
        "update_id": 123456791,
        "callback_query": {
            "id": "query123",
            "from": {"id": 1093871758, "is_bot": False},
            "data": "some_callback_data",
        },
    }


# ── Mock Telegram Client ─────────────────────────────────────────────


@pytest.fixture
def mock_telegram_client():
    """Mocked AsyncTelegramClient for tests."""
    client = AsyncMock()
    client.send_message = AsyncMock(return_value={"message_id": 42})
    client.send_message_string = AsyncMock(
        return_value="Message sent successfully (message_id: 42)"
    )
    client.close = AsyncMock()
    return client
