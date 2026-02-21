import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

# Re-exporting from our actual codebase
from autopilot.connectors.airtable_connector import AsyncAirtableClient, AirtableConnector

@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mocks the httpx.AsyncClient to prevent actual network calls."""
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    # By default, a 200 OK response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_client_instance.request.return_value = mock_response
    
    # Needs aclose method for teardown
    mock_client_instance.aclose = AsyncMock()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client_instance)
    return mock_client_instance

@pytest.mark.asyncio
async def test_airtable_client_initialization(mock_httpx_client):
    client = AsyncAirtableClient("dummy_token")
    assert client._token == "dummy_token"

@pytest.mark.asyncio
async def test_airtable_get_records(mock_httpx_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Simulate paginated response
    mock_response.json.side_effect = [
        {"records": [{"id": "rec1"}], "offset": "token123"},
        {"records": [{"id": "rec2"}]} # No offset, pagination stops
    ]
    mock_httpx_client.request.return_value = mock_response

    client = AsyncAirtableClient("dummy_token")
    # Provide the mock instance we configured
    client._client = mock_httpx_client
    
    records = await client.get_records("base123", "table123")
    
    # It should have collected both pages
    assert len(records) == 2
    assert records[0]["id"] == "rec1"
    assert records[1]["id"] == "rec2"
    
    # It should have called request twice
    assert mock_httpx_client.request.call_count == 2
    
@pytest.mark.asyncio
async def test_airtable_create_records_chunking(mock_httpx_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"records": [{"id": "new_rec"}]}
    mock_httpx_client.request.return_value = mock_response
    
    client = AsyncAirtableClient("dummy_token")
    client._client = mock_httpx_client

    # 15 records should be split into 2 chunks (10 and 5)
    dummy_records = [{"fields": {"Name": f"Item {i}"}} for i in range(15)]
    
    res = await client.create_records("base123", "table123", dummy_records)
    
    # We expect 2 return records (one for each batch response mock)
    assert len(res) == 2
    assert mock_httpx_client.request.call_count == 2
    
@pytest.mark.asyncio
async def test_airtable_client_rate_limit(mock_httpx_client, monkeypatch):
    # Shorten Tenacity retries for test speed
    monkeypatch.setattr("autopilot.connectors.airtable_connector.wait_exponential", lambda **kw: wait_fixed(0.01))

    # Fast wait helper for the decorator
    from tenacity import wait_none, wait_fixed
    monkeypatch.setattr("tenacity.wait_exponential", lambda **kw: wait_none())
    
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {"records": [{"id": "rec1"}]}
    
    # Fail once, then succeed
    mock_httpx_client.request.side_effect = [mock_response_429, mock_response_200]
    
    client = AsyncAirtableClient("dummy_token")
    client._client = mock_httpx_client
    
    # Replace the method decorator so it uses wait_none for immediate retry
    # Actually, we can just rely on the side_effect since Tenacity is already applied to the method.
    records = await client.get_records("b", "t")
    
    assert len(records) == 1
    assert mock_httpx_client.request.call_count == 2

@pytest.mark.asyncio
async def test_airtable_health_check(mock_httpx_client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_httpx_client.request.return_value = mock_response

    connector = AirtableConnector()
    connector._client = AsyncAirtableClient("dummy_token")
    connector._client._client = mock_httpx_client
    
    is_healthy = await connector.health_check()
    assert is_healthy is True

@pytest.mark.asyncio
async def test_airtable_health_check_failure(mock_httpx_client):
    # Mocks a failed request
    mock_httpx_client.request.side_effect = Exception("Network error")

    connector = AirtableConnector()
    connector._client = AsyncAirtableClient("dummy_token")
    connector._client._client = mock_httpx_client
    
    is_healthy = await connector.health_check()
    assert is_healthy is False
