"""
Tests for PolymarketConnector — Async Polymarket API client tests.

Follows the same pattern as test_ynab_connector.py:
- Mock httpx.AsyncClient for Gamma and CLOB API calls
- Test all read methods (events, markets, prices, order book)
- Test error handling (timeouts, rate limits, auth errors)
- Test connector lifecycle (setup, teardown, health check)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx

from autopilot.connectors.polymarket_connector import (
    AsyncPolymarketClient,
    PolymarketConnector,
)
from autopilot.errors import ConnectorError, ConnectorRateLimitError


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_response(
    json_data: dict | list | None = None,
    status_code: int = 200,
    text: str = "",
) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    resp.http_version = "HTTP/2"
    return resp


def _make_client() -> AsyncPolymarketClient:
    """Create a test client with mocked HTTP clients."""
    client = AsyncPolymarketClient()

    # Mock Gamma HTTP client
    client._gamma = AsyncMock(spec=httpx.AsyncClient)
    client._gamma.request = AsyncMock(return_value=_make_response(json_data=[]))
    client._gamma.aclose = AsyncMock()

    # Mock CLOB HTTP client
    client._clob_http = AsyncMock(spec=httpx.AsyncClient)
    client._clob_http.request = AsyncMock(return_value=_make_response(json_data={}))
    client._clob_http.aclose = AsyncMock()

    return client


# ── Gamma API Tests ──────────────────────────────────────────────────


class TestGammaAPI:
    """Tests for Gamma API methods (market data)."""

    @pytest.mark.asyncio
    async def test_get_events(self):
        client = _make_client()
        events = [
            {"id": "1", "title": "BTC 5-min UP", "closed": False},
            {"id": "2", "title": "ETH price", "closed": False},
        ]
        client._gamma.request.return_value = _make_response(json_data=events)

        result = await client.get_events(limit=10)

        assert len(result) == 2
        assert result[0]["title"] == "BTC 5-min UP"

    @pytest.mark.asyncio
    async def test_get_events_with_filters(self):
        client = _make_client()
        client._gamma.request.return_value = _make_response(json_data=[])

        await client.get_events(limit=5, offset=10, closed=True, tag_id=42)

        call_args = client._gamma.request.call_args
        params = call_args.kwargs.get("params", {})
        assert params["limit"] == 5
        assert params["offset"] == 10
        assert params["closed"] == "true"
        assert params["tag_id"] == 42

    @pytest.mark.asyncio
    async def test_get_markets(self):
        client = _make_client()
        markets = [
            {"condition_id": "abc", "question": "Will BTC go up?"},
        ]
        client._gamma.request.return_value = _make_response(json_data=markets)

        result = await client.get_markets(limit=5)

        assert len(result) == 1
        assert result[0]["question"] == "Will BTC go up?"

    @pytest.mark.asyncio
    async def test_get_market_by_id(self):
        client = _make_client()
        market = {"condition_id": "abc123", "question": "BTC up?", "tokens": []}
        client._gamma.request.return_value = _make_response(json_data=market)

        result = await client.get_market("abc123")

        assert result["condition_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_search_markets(self):
        client = _make_client()
        markets = [
            {"condition_id": "btc1", "question": "Bitcoin 5-minute"},
            {"condition_id": "btc2", "question": "Bitcoin hourly"},
        ]
        client._gamma.request.return_value = _make_response(json_data=markets)

        result = await client.search_markets("Bitcoin", limit=10)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_search_markets_cached(self):
        client = _make_client()
        markets = [{"condition_id": "btc1", "question": "Bitcoin"}]
        client._gamma.request.return_value = _make_response(json_data=markets)

        # First call — hits API
        result1 = await client.search_markets("Bitcoin", limit=10)
        # Second call — should use cache
        result2 = await client.search_markets("Bitcoin", limit=10)

        assert result1 == result2
        assert client._gamma.request.call_count == 1  # Only 1 API call

    @pytest.mark.asyncio
    async def test_find_active_btc_markets(self):
        client = _make_client()
        # The new code queries /events?slug=btc-updown-5m-{ts}
        # Returns an event with a "markets" list inside
        event = [
            {
                "title": "Bitcoin Up or Down - 5 Minutes",
                "markets": [
                    {
                        "conditionId": "btc1",
                        "question": "Bitcoin Up or Down - Feb 23, 10:00PM-10:05PM ET",
                        "outcomes": '["Up", "Down"]',
                    },
                ],
            }
        ]
        client._gamma.request.return_value = _make_response(json_data=event)

        result = await client.find_active_btc_markets(duration="5m")

        assert len(result) == 1
        assert result[0]["conditionId"] == "btc1"


# ── CLOB API Tests ───────────────────────────────────────────────────


class TestCLOBAPI:
    """Tests for CLOB API methods (pricing & order book)."""

    @pytest.mark.asyncio
    async def test_get_midpoint(self):
        client = _make_client()
        client._clob_http.request.return_value = _make_response(
            json_data={"mid": "0.55"}
        )

        result = await client.get_midpoint("token123")

        assert result["mid"] == "0.55"
        call_args = client._clob_http.request.call_args
        assert call_args.kwargs["params"]["token_id"] == "token123"

    @pytest.mark.asyncio
    async def test_get_price(self):
        client = _make_client()
        client._clob_http.request.return_value = _make_response(
            json_data={"price": "0.60"}
        )

        result = await client.get_price("token123", "BUY")

        assert result["price"] == "0.60"
        call_args = client._clob_http.request.call_args
        assert call_args.kwargs["params"]["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_get_order_book(self):
        client = _make_client()
        book = {
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "200"}],
        }
        client._clob_http.request.return_value = _make_response(json_data=book)

        result = await client.get_order_book("token123")

        assert len(result["bids"]) == 1
        assert len(result["asks"]) == 1

    @pytest.mark.asyncio
    async def test_get_spread(self):
        client = _make_client()
        client._clob_http.request.return_value = _make_response(
            json_data={"spread": "0.10"}
        )

        result = await client.get_spread("token123")

        assert result["spread"] == "0.10"

    @pytest.mark.asyncio
    async def test_get_fee_rate(self):
        client = _make_client()
        client._clob_http.request.return_value = _make_response(
            json_data={"base_fee": 30}
        )

        result = await client.get_fee_rate("token123")

        assert result["base_fee"] == 30

    @pytest.mark.asyncio
    async def test_get_order_book_imbalance(self):
        client = _make_client()
        book = {
            "bids": [
                {"price": "0.55", "size": "300"},
                {"price": "0.54", "size": "200"},
            ],
            "asks": [
                {"price": "0.56", "size": "100"},
                {"price": "0.57", "size": "100"},
            ],
        }
        client._clob_http.request.return_value = _make_response(json_data=book)

        result = await client.get_order_book_imbalance("token123")

        assert result["total_bid_size"] == 500.0
        assert result["total_ask_size"] == 200.0
        assert result["imbalance_ratio"] > 0  # Bid-heavy
        assert result["spread"] == pytest.approx(0.01, abs=0.001)
        assert result["bid_levels"] == 2
        assert result["ask_levels"] == 2


# ── Error Handling Tests ─────────────────────────────────────────────


class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_gamma_rate_limit(self):
        client = _make_client()
        client._gamma.request.return_value = _make_response(
            status_code=429, text="Rate limited"
        )

        with pytest.raises(ConnectorRateLimitError):
            await client.get_events.__wrapped__(client, limit=10)

    @pytest.mark.asyncio
    async def test_gamma_server_error(self):
        client = _make_client()
        client._gamma.request.return_value = _make_response(
            status_code=500, text="Internal Server Error"
        )

        with pytest.raises(ConnectorError, match="Gamma API error: 500"):
            await client.get_events.__wrapped__(client, limit=10)

    @pytest.mark.asyncio
    async def test_clob_auth_error(self):
        client = _make_client()
        client._clob_http.request.return_value = _make_response(
            status_code=401, text="Unauthorized"
        )

        with pytest.raises(ConnectorError, match="Authentication failed"):
            await client.get_midpoint.__wrapped__(client, "token123")

    @pytest.mark.asyncio
    async def test_gamma_connection_error(self):
        client = _make_client()
        client._gamma.request.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ConnectorError, match="Connection failed"):
            await client.get_events.__wrapped__(client, limit=10)

    @pytest.mark.asyncio
    async def test_gamma_timeout_error(self):
        client = _make_client()
        client._gamma.request.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(ConnectorError, match="timed out"):
            await client.get_events.__wrapped__(client, limit=10)

    @pytest.mark.asyncio
    async def test_clob_connection_error(self):
        client = _make_client()
        client._clob_http.request.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ConnectorError, match="Connection failed"):
            await client.get_midpoint.__wrapped__(client, "token123")


# ── Trading Tests ────────────────────────────────────────────────────


class TestTrading:
    """Tests for trading methods."""

    @pytest.mark.asyncio
    async def test_place_limit_order_requires_private_key(self):
        client = _make_client()

        with pytest.raises(ConnectorError, match="POLYMARKET_PRIVATE_KEY"):
            await client.place_limit_order("token", "BUY", 0.5, 10)

    @pytest.mark.asyncio
    async def test_place_market_order_requires_private_key(self):
        client = _make_client()

        with pytest.raises(ConnectorError, match="POLYMARKET_PRIVATE_KEY"):
            await client.place_market_order("token", "BUY", 10)

    @pytest.mark.asyncio
    async def test_cancel_order_requires_private_key(self):
        client = _make_client()

        with pytest.raises(ConnectorError, match="POLYMARKET_PRIVATE_KEY"):
            await client.cancel_order("order123")


# ── Connector Lifecycle Tests ────────────────────────────────────────


class TestConnectorLifecycle:
    """Tests for the PolymarketConnector wrapper."""

    @pytest.mark.asyncio
    async def test_connector_name(self):
        connector = PolymarketConnector()
        assert connector.name == "polymarket"
        assert connector.icon == "🔮"

    @pytest.mark.asyncio
    async def test_connector_setup(self):
        connector = PolymarketConnector()
        await connector.setup()
        assert connector._client is not None

    @pytest.mark.asyncio
    async def test_connector_teardown(self):
        connector = PolymarketConnector()
        await connector.setup()
        # Mock the client close method
        connector._client._gamma = AsyncMock(spec=httpx.AsyncClient)
        connector._client._gamma.aclose = AsyncMock()
        connector._client._clob_http = AsyncMock(spec=httpx.AsyncClient)
        connector._client._clob_http.aclose = AsyncMock()

        await connector.teardown()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_health_check_success(self):
        connector = PolymarketConnector()
        await connector.setup()
        # Mock the events call
        connector._client._gamma = AsyncMock(spec=httpx.AsyncClient)
        connector._client._gamma.request = AsyncMock(
            return_value=_make_response(json_data=[{"id": "1"}])
        )

        result = await connector.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_connector_health_check_failure(self):
        connector = PolymarketConnector()
        await connector.setup()
        connector._client._gamma = AsyncMock(spec=httpx.AsyncClient)
        connector._client._gamma.request.side_effect = httpx.ConnectError("Failed")

        result = await connector.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_connector_client_close(self):
        client = _make_client()
        await client.close()
        client._gamma.aclose.assert_awaited_once()
        client._clob_http.aclose.assert_awaited_once()
