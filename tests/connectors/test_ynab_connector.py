"""
Tests for the YNAB connector — AsyncYNABClient and YNABConnector.

Follows the same pattern as test_telegram_connector.py: mock httpx.AsyncClient,
inject _client, verify endpoint paths /payloads / error handling.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from autopilot.errors import ConnectorError
from autopilot.connectors.ynab_connector import (
    AsyncYNABClient,
    YNABConnector,
)


# ── Fixtures ─────────────────────────────────────────────────────────

BUDGET_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
ACCOUNT_ID = "f1e2d3c4-b5a6-7890-abcd-ef0987654321"
TX_ID = "tx-0001-0002-0003-0004"
SCHEDULED_TX_ID = "stx-0001-0002-0003-0004"
PAYEE_ID = "payee-0001-0002-0003"
CATEGORY_ID = "cat-0001-0002-0003"


def _make_response(status_code=200, json_data=None):
    """Helper to create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.http_version = "HTTP/2"
    resp.text = ""
    resp.json.return_value = json_data or {}
    return resp


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mocks httpx.AsyncClient to prevent actual network calls."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    default_resp = _make_response(json_data={"data": {}})
    mock_client.request.return_value = default_resp
    mock_client.aclose = AsyncMock()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)
    return mock_client


def _make_client(mock_httpx) -> AsyncYNABClient:
    """Create an AsyncYNABClient with injected mock transport."""
    client = AsyncYNABClient("test-token")
    client._client = mock_httpx
    return client


# ── Budgets ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_all_budgets(mock_httpx_client):
    budgets = [{"id": BUDGET_ID, "name": "My Budget"}]
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"budgets": budgets}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_all_budgets()

    assert result == budgets
    call = mock_httpx_client.request.call_args
    assert call[0] == ("GET", "/budgets")


# ── Accounts ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_accounts(mock_httpx_client):
    accounts = [
        {"id": ACCOUNT_ID, "name": "Checking", "closed": False, "deleted": False}
    ]
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"accounts": accounts}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_accounts(BUDGET_ID)

    assert len(result) == 1
    assert result[0]["id"] == ACCOUNT_ID


@pytest.mark.asyncio
async def test_get_accounts_filters_closed(mock_httpx_client):
    accounts = [
        {"id": "a1", "name": "Open", "closed": False, "deleted": False},
        {"id": "a2", "name": "Closed", "closed": True, "deleted": False},
        {"id": "a3", "name": "Deleted", "closed": False, "deleted": True},
    ]
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"accounts": accounts}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_accounts(BUDGET_ID)

    assert len(result) == 1
    assert result[0]["name"] == "Open"


@pytest.mark.asyncio
async def test_create_account(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"account": {"id": "new-acc-id", "name": "Savings"}}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.create_account(
        BUDGET_ID, {"name": "Savings", "type": "savings", "balance": 0}
    )

    assert result["data"]["account"]["id"] == "new-acc-id"
    call = mock_httpx_client.request.call_args
    assert call[0] == ("POST", f"/budgets/{BUDGET_ID}/accounts")
    assert call[1]["json"]["account"]["name"] == "Savings"


# ── Categories ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_category_by_id(mock_httpx_client):
    cat = {"id": CATEGORY_ID, "name": "Dining", "balance": 180000}
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"category": cat}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_category_by_id(BUDGET_ID, CATEGORY_ID)

    assert result["name"] == "Dining"


@pytest.mark.asyncio
async def test_update_category(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"category": {"id": CATEGORY_ID, "goal_target": 500000}}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.update_category(
        BUDGET_ID, CATEGORY_ID, {"goal_target": 500000}
    )

    assert result["data"]["category"]["goal_target"] == 500000
    call = mock_httpx_client.request.call_args
    assert call[0] == ("PATCH", f"/budgets/{BUDGET_ID}/categories/{CATEGORY_ID}")


@pytest.mark.asyncio
async def test_update_month_category(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"category": {"id": CATEGORY_ID, "budgeted": 300000}}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.update_month_category(
        BUDGET_ID, "2026-02-01", CATEGORY_ID, {"budgeted": 300000}
    )

    call = mock_httpx_client.request.call_args
    assert call[0] == (
        "PATCH",
        f"/budgets/{BUDGET_ID}/months/2026-02-01/categories/{CATEGORY_ID}",
    )
    assert result["data"]["category"]["budgeted"] == 300000


# ── Transactions ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transaction": {"id": TX_ID}}}
    )
    client = _make_client(mock_httpx_client)

    payload = {"payee_name": "Test", "amount": -50000, "date": "2026-02-23"}
    result = await client.create_transaction(BUDGET_ID, payload)

    assert result["data"]["transaction"]["id"] == TX_ID
    call = mock_httpx_client.request.call_args
    assert call[0] == ("POST", f"/budgets/{BUDGET_ID}/transactions")
    assert call[1]["json"]["transaction"] == payload


@pytest.mark.asyncio
async def test_bulk_create_transactions(mock_httpx_client):
    txs = [
        {"payee_name": "A", "amount": -10000, "date": "2026-02-23"},
        {"payee_name": "B", "amount": -20000, "date": "2026-02-23"},
    ]
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {
                "transactions": [{"id": "tx-1"}, {"id": "tx-2"}],
                "transaction_ids": ["tx-1", "tx-2"],
            }
        }
    )
    client = _make_client(mock_httpx_client)

    result = await client.bulk_create_transactions(BUDGET_ID, txs)

    assert len(result["data"]["transactions"]) == 2
    call = mock_httpx_client.request.call_args
    assert call[1]["json"]["transactions"] == txs


@pytest.mark.asyncio
async def test_get_transactions(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {
                "transactions": [{"id": TX_ID}],
                "server_knowledge": 100,
            }
        }
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_transactions(BUDGET_ID, since_date="2026-01-01")

    assert result["data"]["transactions"][0]["id"] == TX_ID
    call = mock_httpx_client.request.call_args
    assert call[0] == ("GET", f"/budgets/{BUDGET_ID}/transactions")
    assert call[1]["params"]["since_date"] == "2026-01-01"


@pytest.mark.asyncio
async def test_get_transactions_with_delta(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transactions": [], "server_knowledge": 101}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_transactions(BUDGET_ID, last_knowledge_of_server=100)

    call = mock_httpx_client.request.call_args
    assert call[1]["params"]["last_knowledge_of_server"] == 100
    assert result["data"]["server_knowledge"] == 101


@pytest.mark.asyncio
async def test_get_transaction(mock_httpx_client):
    tx = {"id": TX_ID, "payee_name": "Store", "amount": -30000}
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transaction": tx}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_transaction(BUDGET_ID, TX_ID)

    assert result["id"] == TX_ID
    assert result["payee_name"] == "Store"


@pytest.mark.asyncio
async def test_update_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transaction": {"id": TX_ID, "memo": "Updated"}}}
    )
    client = _make_client(mock_httpx_client)

    await client.update_transaction(BUDGET_ID, TX_ID, {"memo": "Updated"})

    call = mock_httpx_client.request.call_args
    assert call[0] == ("PATCH", f"/budgets/{BUDGET_ID}/transactions/{TX_ID}")
    assert call[1]["json"]["transaction"]["memo"] == "Updated"


@pytest.mark.asyncio
async def test_bulk_update_transactions(mock_httpx_client):
    txs = [{"id": TX_ID, "memo": "Bulk updated"}]
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transactions": txs}}
    )
    client = _make_client(mock_httpx_client)

    await client.bulk_update_transactions(BUDGET_ID, txs)

    call = mock_httpx_client.request.call_args
    assert call[0] == ("PATCH", f"/budgets/{BUDGET_ID}/transactions")
    assert call[1]["json"]["transactions"] == txs


@pytest.mark.asyncio
async def test_delete_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transaction": {"id": TX_ID, "deleted": True}}}
    )
    client = _make_client(mock_httpx_client)

    await client.delete_transaction(BUDGET_ID, TX_ID)

    call = mock_httpx_client.request.call_args
    assert call[0] == ("DELETE", f"/budgets/{BUDGET_ID}/transactions/{TX_ID}")


@pytest.mark.asyncio
async def test_get_recent_transactions(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"transactions": [{"id": TX_ID}]}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_recent_transactions(BUDGET_ID, ACCOUNT_ID, "2026-02-01")

    assert result[0]["id"] == TX_ID
    call = mock_httpx_client.request.call_args
    assert f"/accounts/{ACCOUNT_ID}/transactions" in call[0][1]


# ── Scheduled Transactions ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_scheduled_transactions(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {
                "scheduled_transactions": [{"id": SCHEDULED_TX_ID}],
                "server_knowledge": 50,
            }
        }
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_scheduled_transactions(BUDGET_ID)

    assert result["data"]["scheduled_transactions"][0]["id"] == SCHEDULED_TX_ID


@pytest.mark.asyncio
async def test_get_scheduled_transaction(mock_httpx_client):
    stx = {"id": SCHEDULED_TX_ID, "payee_name": "Netflix", "frequency": "monthly"}
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"scheduled_transaction": stx}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_scheduled_transaction(BUDGET_ID, SCHEDULED_TX_ID)

    assert result["payee_name"] == "Netflix"


@pytest.mark.asyncio
async def test_create_scheduled_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"scheduled_transaction": {"id": SCHEDULED_TX_ID}}}
    )
    client = _make_client(mock_httpx_client)

    payload = {"payee_name": "Spotify", "amount": -15000, "frequency": "monthly"}
    await client.create_scheduled_transaction(BUDGET_ID, payload)

    call = mock_httpx_client.request.call_args
    assert call[0] == ("POST", f"/budgets/{BUDGET_ID}/scheduled_transactions")
    assert call[1]["json"]["scheduled_transaction"] == payload


@pytest.mark.asyncio
async def test_update_scheduled_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {"scheduled_transaction": {"id": SCHEDULED_TX_ID, "amount": -20000}}
        }
    )
    client = _make_client(mock_httpx_client)

    await client.update_scheduled_transaction(
        BUDGET_ID, SCHEDULED_TX_ID, {"amount": -20000}
    )

    call = mock_httpx_client.request.call_args
    assert call[0] == (
        "PUT",
        f"/budgets/{BUDGET_ID}/scheduled_transactions/{SCHEDULED_TX_ID}",
    )


@pytest.mark.asyncio
async def test_delete_scheduled_transaction(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"scheduled_transaction": {"id": SCHEDULED_TX_ID}}}
    )
    client = _make_client(mock_httpx_client)

    await client.delete_scheduled_transaction(BUDGET_ID, SCHEDULED_TX_ID)

    call = mock_httpx_client.request.call_args
    assert call[0] == (
        "DELETE",
        f"/budgets/{BUDGET_ID}/scheduled_transactions/{SCHEDULED_TX_ID}",
    )


# ── Payees ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_payees(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {
                "payees": [{"id": PAYEE_ID, "name": "Amazon"}],
                "server_knowledge": 75,
            }
        }
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_payees(BUDGET_ID)

    assert result["data"]["payees"][0]["name"] == "Amazon"


@pytest.mark.asyncio
async def test_get_payee(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"payee": {"id": PAYEE_ID, "name": "Amazon"}}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_payee(BUDGET_ID, PAYEE_ID)

    assert result["name"] == "Amazon"


@pytest.mark.asyncio
async def test_update_payee(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"payee": {"id": PAYEE_ID, "name": "Amazon.com"}}}
    )
    client = _make_client(mock_httpx_client)

    await client.update_payee(BUDGET_ID, PAYEE_ID, {"name": "Amazon.com"})

    call = mock_httpx_client.request.call_args
    assert call[0] == ("PUT", f"/budgets/{BUDGET_ID}/payees/{PAYEE_ID}")
    assert call[1]["json"]["payee"]["name"] == "Amazon.com"


# ── Budget Months ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_months(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={
            "data": {
                "months": [{"month": "2026-02-01"}, {"month": "2026-01-01"}],
                "server_knowledge": 88,
            }
        }
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_months(BUDGET_ID)

    assert len(result["data"]["months"]) == 2


@pytest.mark.asyncio
async def test_get_month(mock_httpx_client):
    month_data = {
        "month": "2026-02-01",
        "income": 5000000,
        "budgeted": 4500000,
        "activity": -3200000,
        "categories": [{"id": CATEGORY_ID, "budgeted": 500000}],
    }
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"month": month_data}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_month(BUDGET_ID, "2026-02-01")

    assert result["month"] == "2026-02-01"
    assert len(result["categories"]) == 1


# ── User ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_user(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"user": {"id": "user-123"}}}
    )
    client = _make_client(mock_httpx_client)

    result = await client.get_user()

    assert result["id"] == "user-123"
    call = mock_httpx_client.request.call_args
    assert call[0] == ("GET", "/user")


# ── Error Handling ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_retry(mock_httpx_client):
    """429 triggers retry and succeeds on second attempt."""
    rate_limited = _make_response(status_code=429)
    success = _make_response(json_data={"data": {"budgets": [{"id": BUDGET_ID}]}})
    mock_httpx_client.request.side_effect = [rate_limited, success]

    client = _make_client(mock_httpx_client)
    result = await client.get_all_budgets()

    assert result[0]["id"] == BUDGET_ID
    assert mock_httpx_client.request.call_count == 2


@pytest.mark.asyncio
async def test_auth_error(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(status_code=401)

    client = _make_client(mock_httpx_client)

    with pytest.raises(ConnectorError, match="Authentication failed"):
        await client.get_all_budgets()


@pytest.mark.asyncio
async def test_generic_api_error(mock_httpx_client):
    resp = _make_response(status_code=500)
    resp.text = "Internal Server Error"
    mock_httpx_client.request.return_value = resp

    client = _make_client(mock_httpx_client)

    with pytest.raises(ConnectorError, match="API error: 500"):
        await client.get_user()


@pytest.mark.asyncio
async def test_connection_error(mock_httpx_client):
    mock_httpx_client.request.side_effect = httpx.ConnectError("DNS failure")

    client = _make_client(mock_httpx_client)

    with pytest.raises(ConnectorError, match="Connection failed"):
        await client.get_user()


@pytest.mark.asyncio
async def test_timeout_error(mock_httpx_client):
    mock_httpx_client.request.side_effect = httpx.ReadTimeout("Read timed out")

    client = _make_client(mock_httpx_client)

    with pytest.raises(ConnectorError, match="Request timed out"):
        await client.get_user()


# ── Connector Lifecycle ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connector_health_check(mock_httpx_client):
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"budgets": [{"id": BUDGET_ID}]}}
    )

    connector = YNABConnector()
    connector._client = _make_client(mock_httpx_client)

    assert await connector.health_check() is True


@pytest.mark.asyncio
async def test_connector_health_check_failure(mock_httpx_client):
    mock_httpx_client.request.side_effect = Exception("Network error")

    connector = YNABConnector()
    connector._client = _make_client(mock_httpx_client)

    assert await connector.health_check() is False


@pytest.mark.asyncio
async def test_connector_teardown(mock_httpx_client):
    connector = YNABConnector()
    connector._client = _make_client(mock_httpx_client)

    await connector.teardown()

    assert connector._client is None
    mock_httpx_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_connector_missing_token(monkeypatch):
    monkeypatch.delenv("YNAB_ACCESS_TOKEN", raising=False)

    connector = YNABConnector()
    connector._client = None

    with pytest.raises(ConnectorError, match="YNAB_ACCESS_TOKEN"):
        _ = connector.client


# ── Cache Invalidation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_account_invalidates_cache(mock_httpx_client):
    """Creating an account should invalidate the accounts cache."""
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"account": {"id": "new-acc"}}}
    )
    client = _make_client(mock_httpx_client)
    # Seed cache
    client._accounts_cache.set("accounts:test", [{"id": "old"}])

    await client.create_account(
        BUDGET_ID, {"name": "New", "type": "checking", "balance": 0}
    )

    assert client._accounts_cache.get("accounts:test") is None


@pytest.mark.asyncio
async def test_update_category_invalidates_cache(mock_httpx_client):
    """Updating a category should invalidate the categories cache."""
    mock_httpx_client.request.return_value = _make_response(
        json_data={"data": {"category": {"id": CATEGORY_ID}}}
    )
    client = _make_client(mock_httpx_client)
    # Seed cache
    client._categories_cache.set("categories:test", [{"id": "old"}])

    await client.update_category(BUDGET_ID, CATEGORY_ID, {"goal_target": 100000})

    assert client._categories_cache.get("categories:test") is None
