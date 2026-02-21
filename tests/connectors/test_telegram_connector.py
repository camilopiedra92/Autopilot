import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from autopilot.errors import ConnectorError
from autopilot.connectors.telegram_connector import (
    AsyncTelegramClient,
    TelegramConnector,
)


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mocks the httpx.AsyncClient to prevent actual network calls."""
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)

    # Default: a successful Telegram API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.http_version = "HTTP/2"
    mock_response.json.return_value = {"ok": True, "result": {}}
    mock_client_instance.request.return_value = mock_response
    mock_client_instance.aclose = AsyncMock()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client_instance)
    return mock_client_instance


def _make_response(
    status_code=200, result=None, ok=True, error_code=None, description=None
):
    """Helper to create a mock Telegram API response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.http_version = "HTTP/2"
    body = {"ok": ok, "result": result}
    if error_code:
        body["error_code"] = error_code
        body["description"] = description or "Error"
    resp.json.return_value = body
    return resp


@pytest.mark.asyncio
async def test_telegram_client_initialization(mock_httpx_client):
    client = AsyncTelegramClient("123456:ABC-DEF")
    assert client._token == "123456:ABC-DEF"
    assert "123456:ABC-DEF" in client._base_url


@pytest.mark.asyncio
async def test_telegram_get_me(mock_httpx_client):
    bot_info = {
        "id": 123456,
        "is_bot": True,
        "first_name": "TestBot",
        "username": "test_bot",
    }
    mock_httpx_client.request.return_value = _make_response(result=bot_info)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.get_me()

    assert result["id"] == 123456
    assert result["username"] == "test_bot"
    mock_httpx_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_send_message(mock_httpx_client):
    sent_msg = {"message_id": 42, "chat": {"id": 789}, "text": "Hello!"}
    mock_httpx_client.request.return_value = _make_response(result=sent_msg)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.send_message(chat_id="789", text="Hello!")

    assert result["message_id"] == 42
    # Verify the JSON payload
    call_args = mock_httpx_client.request.call_args
    assert call_args[0] == ("POST", "/sendMessage")
    payload = call_args[1]["json"]
    assert payload["chat_id"] == "789"
    assert payload["text"] == "Hello!"


@pytest.mark.asyncio
async def test_telegram_send_message_with_parse_mode(mock_httpx_client):
    sent_msg = {"message_id": 43, "chat": {"id": 789}}
    mock_httpx_client.request.return_value = _make_response(result=sent_msg)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    await client.send_message(chat_id="789", text="*bold*", parse_mode="MarkdownV2")

    payload = mock_httpx_client.request.call_args[1]["json"]
    assert payload["parse_mode"] == "MarkdownV2"


@pytest.mark.asyncio
async def test_telegram_send_message_string(mock_httpx_client):
    sent_msg = {"message_id": 99, "chat": {"id": 789}, "text": "Hi"}
    mock_httpx_client.request.return_value = _make_response(result=sent_msg)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.send_message_string(chat_id="789", text="Hi")

    assert "99" in result
    assert "successfully" in result.lower()


@pytest.mark.asyncio
async def test_telegram_send_photo(mock_httpx_client):
    sent_msg = {"message_id": 50, "chat": {"id": 789}, "photo": []}
    mock_httpx_client.request.return_value = _make_response(result=sent_msg)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.send_photo(
        chat_id="789", photo="https://example.com/photo.jpg", caption="Nice!"
    )

    assert result["message_id"] == 50
    payload = mock_httpx_client.request.call_args[1]["json"]
    assert payload["photo"] == "https://example.com/photo.jpg"
    assert payload["caption"] == "Nice!"


@pytest.mark.asyncio
async def test_telegram_set_webhook(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(result=True)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.set_webhook(
        url="https://myserver.com/webhook",
        secret_token="my_secret",
        allowed_updates=["message"],
    )

    assert result is True
    payload = mock_httpx_client.request.call_args[1]["json"]
    assert payload["url"] == "https://myserver.com/webhook"
    assert payload["secret_token"] == "my_secret"
    assert payload["allowed_updates"] == ["message"]


@pytest.mark.asyncio
async def test_telegram_delete_webhook(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(result=True)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.delete_webhook(drop_pending_updates=True)

    assert result is True
    payload = mock_httpx_client.request.call_args[1]["json"]
    assert payload["drop_pending_updates"] is True


@pytest.mark.asyncio
async def test_telegram_get_updates(mock_httpx_client):
    updates = [
        {"update_id": 1, "message": {"text": "hi"}},
        {"update_id": 2, "message": {"text": "hello"}},
    ]
    mock_httpx_client.request.return_value = _make_response(result=updates)

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.get_updates(offset=1, limit=10)

    assert len(result) == 2
    assert result[0]["update_id"] == 1


@pytest.mark.asyncio
async def test_telegram_rate_limit_retry(mock_httpx_client):
    """Verifies that a 429 triggers retry and succeeds on the second attempt."""
    rate_limit_resp = MagicMock()
    rate_limit_resp.status_code = 429
    rate_limit_resp.http_version = "HTTP/2"
    rate_limit_resp.json.return_value = {"ok": False, "parameters": {"retry_after": 1}}

    success_resp = _make_response(result={"id": 1, "username": "bot"})

    mock_httpx_client.request.side_effect = [rate_limit_resp, success_resp]

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    result = await client.get_me()

    assert result["username"] == "bot"
    assert mock_httpx_client.request.call_count == 2


@pytest.mark.asyncio
async def test_telegram_api_error(mock_httpx_client):
    """Verifies that a non-ok Telegram response raises ConnectorError."""
    error_resp = _make_response(
        status_code=400,
        ok=False,
        result=None,
        error_code=400,
        description="Bad Request: chat not found",
    )
    mock_httpx_client.request.return_value = error_resp

    client = AsyncTelegramClient("dummy_token")
    client._client = mock_httpx_client

    with pytest.raises(ConnectorError, match="chat not found"):
        await client.send_message(chat_id="invalid", text="test")


@pytest.mark.asyncio
async def test_telegram_health_check(mock_httpx_client):
    bot_info = {"id": 1, "username": "bot"}
    mock_httpx_client.request.return_value = _make_response(result=bot_info)

    connector = TelegramConnector()
    connector._client = AsyncTelegramClient("dummy_token")
    connector._client._client = mock_httpx_client

    assert await connector.health_check() is True


@pytest.mark.asyncio
async def test_telegram_health_check_failure(mock_httpx_client):
    mock_httpx_client.request.side_effect = Exception("Network error")

    connector = TelegramConnector()
    connector._client = AsyncTelegramClient("dummy_token")
    connector._client._client = mock_httpx_client

    assert await connector.health_check() is False
