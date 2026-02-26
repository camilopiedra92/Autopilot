"""
PolymarketConnector — Async Polymarket integration block.

Wraps the Polymarket Gamma API (market data) and CLOB API (trading) as a
platform connector. Provides market discovery, pricing, order book, and
order placement — all async.

The Gamma API (https://gamma-api.polymarket.com) is public, no auth needed.
The CLOB API (https://clob.polymarket.com) requires L1/L2 auth for trading
but is public for read-only market data.

Usage:
    from autopilot.connectors import get_connector_registry
    pm = get_connector_registry().get("polymarket")
    events = await pm.client.get_events(limit=10)
"""

import os
import time
import structlog
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from autopilot.connectors.base_connector import BaseConnector
from autopilot.errors import (
    ConnectorError as PolymarketError,
    ConnectorRateLimitError as PolymarketRateLimitError,
)

logger = structlog.get_logger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"


# ── TTL Cache ────────────────────────────────────────────────────────


class TTLCache:
    """Simple thread-safe TTL cache for market data."""

    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, any]] = {}

    def get(self, key: str):
        if key in self._store:
            ts, data = self._store[key]
            if time.monotonic() - ts < self._ttl:
                return data
            del self._store[key]
        return None

    def set(self, key: str, value):
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str | None = None):
        if key:
            self._store.pop(key, None)
        else:
            self._store.clear()


# ── Async Polymarket Client ──────────────────────────────────────────


class AsyncPolymarketClient:
    """
    Production-grade async Polymarket API client.

    Features:
    - httpx.AsyncClient with HTTP/2 and connection pooling
    - Dual API support: Gamma (market data) + CLOB (trading)
    - TTL-based caching for market discovery
    - Automatic retry with exponential backoff on rate limits
    - Structured logging for every API call

    CLOB trading requires:
    - L1 auth: Private key → derive API creds
    - L2 auth: HMAC-SHA256 with API key/secret/passphrase
    The py-clob-client library handles signing automatically.
    """

    def __init__(
        self,
        *,
        private_key: str | None = None,
        proxy_wallet: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        passphrase: str | None = None,
    ):
        self._private_key = private_key
        self._proxy_wallet = proxy_wallet
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._clob_client = None  # Lazy-initialized py-clob-client

        # Gamma API — public, no auth needed
        self._gamma = httpx.AsyncClient(
            base_url=GAMMA_API_URL,
            http2=True,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )

        # CLOB API — public for reads, auth for trades
        self._clob_http = httpx.AsyncClient(
            base_url=CLOB_API_URL,
            http2=True,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(15.0, connect=5.0, read=10.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )

        self._markets_cache = TTLCache(ttl_seconds=30)

    async def _gamma_request(self, method: str, path: str, **kwargs) -> dict | list:
        """Execute a Gamma API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._gamma.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise PolymarketError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise PolymarketError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            logger.warning("polymarket_rate_limited", path=path)
            raise PolymarketRateLimitError("Rate limit exceeded")
        if resp.status_code >= 400:
            raise PolymarketError(
                f"Gamma API error: {resp.status_code} — {resp.text}",
                detail=str(resp.status_code),
            )

        logger.debug(
            "polymarket_gamma_request",
            method=method,
            path=path,
            status=resp.status_code,
            latency_ms=round(latency_ms),
        )
        return resp.json()

    async def _clob_request(self, method: str, path: str, **kwargs) -> dict | list:
        """Execute a CLOB API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._clob_http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise PolymarketError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise PolymarketError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            logger.warning("polymarket_clob_rate_limited", path=path)
            raise PolymarketRateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise PolymarketError(
                "Authentication failed — check private key / API creds",
                detail="401",
            )
        if resp.status_code >= 400:
            raise PolymarketError(
                f"CLOB API error: {resp.status_code} — {resp.text}",
                detail=str(resp.status_code),
            )

        logger.debug(
            "polymarket_clob_request",
            method=method,
            path=path,
            status=resp.status_code,
            latency_ms=round(latency_ms),
        )
        return resp.json()

    # ── Gamma API — Market Discovery ────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        closed: bool = False,
        tag_id: int | None = None,
        order: str = "id",
        ascending: bool = False,
    ) -> list[dict]:
        """Fetch prediction events with optional filtering.

        Args:
            limit: Results per page (max 100).
            offset: Pagination offset.
            closed: Include closed events.
            tag_id: Filter by tag ID.
            order: Sort field.
            ascending: Sort direction.
        """
        params: dict = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag_id is not None:
            params["tag_id"] = tag_id

        data = await self._gamma_request("GET", "/events", params=params)
        return data if isinstance(data, list) else data.get("events", [data])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_markets(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        closed: bool = False,
        tag_id: int | None = None,
    ) -> list[dict]:
        """Fetch markets with optional filtering."""
        params: dict = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        if tag_id is not None:
            params["tag_id"] = tag_id

        data = await self._gamma_request("GET", "/markets", params=params)
        return data if isinstance(data, list) else data.get("markets", [data])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_market(self, condition_id: str) -> dict:
        """Fetch a single market by condition ID."""
        data = await self._gamma_request("GET", f"/markets/{condition_id}")
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def search_markets(self, query: str, *, limit: int = 20) -> list[dict]:
        """Search markets by keyword (full-text on Gamma API)."""
        cached = self._markets_cache.get(f"search:{query}:{limit}")
        if cached is not None:
            return cached

        params = {"search": query, "limit": limit, "closed": "false"}
        data = await self._gamma_request("GET", "/markets", params=params)
        results = data if isinstance(data, list) else data.get("markets", [data])
        self._markets_cache.set(f"search:{query}:{limit}", results)
        return results

    async def find_active_btc_markets(self, *, duration: str = "5m") -> list[dict]:
        """Find currently active BTC price prediction markets.

        Polymarket creates a new market every interval with a slug like:
        - `btc-updown-5m-{end_timestamp}` (every 300 seconds)
        - `btc-updown-15m-{end_timestamp}` (every 900 seconds)
        - `btc-updown-1h-{end_timestamp}` (every 3600 seconds)

        We calculate the current interval's end timestamp and query by slug.

        Args:
            duration: Market duration to search for ('5m', '15m', '1h').

        Returns:
            List of active BTC market dicts from the Gamma API event.
        """

        now = int(time.time())

        # Map duration to slug prefix and interval seconds
        slug_config = {
            "5m": ("btc-updown-5m", 300),
            "15m": ("btc-updown-15m", 900),
            "1h": ("btc-updown-1h", 3600),
        }
        prefix, interval = slug_config.get(duration, slug_config["5m"])

        # Calculate the CURRENT interval's start timestamp (floor = active window)
        current_start = (now // interval) * interval
        window_end = current_start + interval
        slug = f"{prefix}-{current_start}"

        logger.info(
            "polymarket_slug_lookup",
            slug=slug,
            duration=duration,
            seconds_elapsed=now - current_start,
            seconds_remaining=window_end - now,
        )

        # Query the Gamma API by exact slug
        try:
            events = await self._gamma_request(
                "GET", "/events", params={"slug": slug, "limit": 1}
            )
        except Exception as e:
            logger.warning("polymarket_slug_lookup_failed", slug=slug, error=str(e))
            return []

        if not events or not isinstance(events, list) or len(events) == 0:
            logger.info("polymarket_no_event_for_slug", slug=slug)
            return []

        event = events[0]
        markets = event.get("markets", [])

        # Inject window metadata into each market for timing decisions
        for m in markets:
            m["_window_start"] = current_start
            m["_window_end"] = window_end
            m["_window_interval"] = interval

        logger.info(
            "polymarket_btc_markets_found",
            duration=duration,
            slug=slug,
            title=event.get("title", ""),
            count=len(markets),
            seconds_elapsed=now - current_start,
            seconds_remaining=window_end - now,
        )
        return markets

    # ── CLOB API — Pricing & Order Book ─────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_midpoint(self, token_id: str) -> dict:
        """Get the midpoint price for a token (0–1 probability)."""
        data = await self._clob_request(
            "GET", "/midpoint", params={"token_id": token_id}
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_price(self, token_id: str, side: str = "BUY") -> dict:
        """Get the best bid/ask price for a token.

        Args:
            token_id: The token asset ID.
            side: 'BUY' or 'SELL'.
        """
        data = await self._clob_request(
            "GET", "/price", params={"token_id": token_id, "side": side}
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_order_book(self, token_id: str) -> dict:
        """Get the full order book for a token."""
        data = await self._clob_request("GET", "/book", params={"token_id": token_id})
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_spread(self, token_id: str) -> dict:
        """Get the bid-ask spread for a token."""
        data = await self._clob_request("GET", "/spread", params={"token_id": token_id})
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_fee_rate(self, token_id: str) -> dict:
        """Get the fee rate in basis points for a token."""
        data = await self._clob_request(
            "GET", "/fee-rate", params={"token_id": token_id}
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_simplified_markets(self, *, next_cursor: str = "") -> dict:
        """Get paginated simplified market list with token IDs."""
        params: dict = {}
        if next_cursor:
            params["next_cursor"] = next_cursor
        data = await self._clob_request("GET", "/simplified-markets", params=params)
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_prices_history(
        self,
        token_id: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str = "1m",
        fidelity: int = 1,
    ) -> list[dict]:
        """Get historical price data for a token.

        Args:
            token_id: The token asset ID.
            start_ts: Start Unix timestamp.
            end_ts: End Unix timestamp.
            interval: Data interval ('1m', '1h', '1d', '1w', 'max').
            fidelity: Accuracy in minutes (lower = more precise).

        Returns:
            List of price points: [{"t": unix_ts, "p": price_str}, ...]
        """
        params: dict = {
            "token_id": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        data = await self._clob_request("GET", "/prices-history", params=params)
        return data.get("history", data) if isinstance(data, dict) else data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(PolymarketRateLimitError),
    )
    async def get_last_trade_price(self, token_id: str) -> dict:
        """Get the last trade price and side for a token.

        Returns:
            {"price": "0.55", "side": "BUY", "token_id": "..."}
        """
        data = await self._clob_request(
            "GET", "/last-trade-price", params={"token_id": token_id}
        )
        return data

    # ── CLOB API — Trading (requires auth) ──────────────────────────

    def _ensure_clob_client(self):
        """Lazy-initialize the py-clob-client for trading operations.

        Supports two auth modes:
        1. Pre-derived L2 creds (api_key + api_secret + passphrase) — skip derivation
        2. Private key only — auto-derive L2 creds on first call
        """
        if self._clob_client is not None:
            return

        if not self._private_key:
            raise PolymarketError(
                "POLYMARKET_PRIVATE_KEY is required for trading operations"
            )

        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            raise PolymarketError(
                "py-clob-client package is required for trading. "
                "Install it with: pip install py-clob-client"
            )

        # Mode 1: Pre-derived L2 creds (faster, no on-chain call)
        if self._api_key and self._api_secret and self._passphrase:
            from py_clob_client.clob_types import ApiCreds

            api_creds = ApiCreds(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._passphrase,
            )
            logger.info("polymarket_l2_creds_loaded_from_env")
        else:
            # Mode 2: Derive L2 creds from private key
            client = ClobClient(
                host=CLOB_API_URL,
                key=self._private_key,
                chain_id=137,  # Polygon mainnet
            )
            api_creds = client.create_or_derive_api_creds()
            logger.info("polymarket_l2_creds_derived")

        # Initialize with full auth
        self._clob_client = ClobClient(
            host=CLOB_API_URL,
            key=self._private_key,
            chain_id=137,
            creds=api_creds,
            signature_type=2,  # POLY_GNOSIS_SAFE
            funder=self._proxy_wallet or "",
        )
        logger.info("polymarket_clob_client_initialized")

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict:
        """Place a limit order (maker — 0% fees).

        Args:
            token_id: The token asset ID.
            side: 'BUY' or 'SELL'.
            price: Limit price (0.01–0.99).
            size: Number of shares.

        Returns:
            Order response with order_id and status.
        """
        self._ensure_clob_client()

        from py_clob_client.order_builder.constants import BUY, SELL

        order_side = BUY if side.upper() == "BUY" else SELL

        logger.info(
            "polymarket_placing_limit_order",
            token_id=token_id[:20] + "...",
            side=side,
            price=price,
            size=size,
        )

        try:
            order = self._clob_client.create_and_post_order(
                order_args={
                    "token_id": token_id,
                    "price": price,
                    "size": size,
                    "side": order_side,
                },
            )
        except Exception as e:
            raise PolymarketError(f"Order placement failed: {e}") from e

        logger.info(
            "polymarket_order_placed",
            order_id=order.get("orderID", "unknown"),
            status=order.get("status", "unknown"),
        )
        return order

    async def place_market_order(
        self,
        token_id: str,
        side: str,
        size: float,
    ) -> dict:
        """Place a market order (taker — pays fees).

        Args:
            token_id: The token asset ID.
            side: 'BUY' or 'SELL'.
            size: Amount in USDC.

        Returns:
            Order response.
        """
        self._ensure_clob_client()

        from py_clob_client.order_builder.constants import BUY, SELL

        order_side = BUY if side.upper() == "BUY" else SELL

        logger.info(
            "polymarket_placing_market_order",
            token_id=token_id[:20] + "...",
            side=side,
            size=size,
        )

        try:
            order = self._clob_client.create_and_post_market_order(
                order_args={
                    "token_id": token_id,
                    "amount": size,
                    "side": order_side,
                },
            )
        except Exception as e:
            raise PolymarketError(f"Market order failed: {e}") from e

        logger.info(
            "polymarket_market_order_placed",
            order_id=order.get("orderID", "unknown"),
        )
        return order

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        self._ensure_clob_client()

        logger.info("polymarket_canceling_order", order_id=order_id)
        try:
            result = self._clob_client.cancel(order_id)
        except Exception as e:
            raise PolymarketError(f"Order cancellation failed: {e}") from e

        logger.info("polymarket_order_canceled", order_id=order_id)
        return result

    async def cancel_all_orders(self) -> dict:
        """Cancel all open orders."""
        self._ensure_clob_client()

        logger.info("polymarket_canceling_all_orders")
        try:
            result = self._clob_client.cancel_all()
        except Exception as e:
            raise PolymarketError(f"Cancel all failed: {e}") from e

        logger.info("polymarket_all_orders_canceled")
        return result

    async def get_open_orders(self) -> list[dict]:
        """List all open orders."""
        self._ensure_clob_client()

        try:
            orders = self._clob_client.get_orders()
        except Exception as e:
            raise PolymarketError(f"Failed to fetch orders: {e}") from e

        return orders if isinstance(orders, list) else orders.get("orders", [])

    async def get_trades(
        self, *, limit: int = 50, after: str | None = None
    ) -> list[dict]:
        """Fetch trade history."""
        self._ensure_clob_client()

        try:
            trades = self._clob_client.get_trades(limit=limit)
        except Exception as e:
            raise PolymarketError(f"Failed to fetch trades: {e}") from e

        return trades if isinstance(trades, list) else trades.get("trades", [])

    # ── Order Book Analysis ─────────────────────────────────────────

    async def get_order_book_imbalance(self, token_id: str) -> dict:
        """Analyze order book for bid/ask imbalance.

        The CLOB API returns bids sorted ascending (lowest first) and
        asks sorted descending (highest first). Best bid = max(bids),
        best ask = min(asks). We also call /spread for the canonical
        spread value as a safeguard.

        Returns:
            Dict with imbalance_ratio (-1 to 1, positive = bid-heavy),
            total_bid_size, total_ask_size, spread, and midpoint.
        """
        book = await self.get_order_book(token_id)

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        total_bid = sum(float(b.get("size", 0)) for b in bids)
        total_ask = sum(float(a.get("size", 0)) for a in asks)
        total = total_bid + total_ask

        imbalance = (total_bid - total_ask) / total if total > 0 else 0.0

        # Best bid = highest bid price, best ask = lowest ask price.
        # CLOB sorts bids ascending and asks descending, so use
        # max/min to be safe regardless of sort order.
        best_bid = max(float(b["price"]) for b in bids) if bids else 0.0
        best_ask = min(float(a["price"]) for a in asks) if asks else 1.0
        spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2

        # Sanity check: if computed spread looks wrong (> 0.5), fall back
        # to the /spread endpoint which Polymarket computes server-side.
        if spread > 0.5:
            try:
                spread_data = await self.get_spread(token_id)
                api_spread = float(spread_data.get("spread", spread))
                logger.debug(
                    "spread_fallback_used",
                    computed_spread=round(spread, 4),
                    api_spread=round(api_spread, 4),
                )
                spread = api_spread
                # Recompute midpoint from the outcome prices if spread
                # was corrected — midpoint of best_bid/best_ask is still
                # valid as a directional indicator.
            except Exception:
                pass  # Keep the computed spread if /spread fails

        return {
            "imbalance_ratio": round(imbalance, 4),
            "total_bid_size": round(total_bid, 2),
            "total_ask_size": round(total_ask, 2),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 4),
            "midpoint": round(midpoint, 4),
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            # Depth clustering — wall detection
            **self._compute_depth_clustering(bids, asks, total_bid, total_ask),
        }

    @staticmethod
    def _compute_depth_clustering(
        bids: list[dict], asks: list[dict], total_bid: float, total_ask: float
    ) -> dict:
        """Detect order book walls and volume concentration.

        A 'wall' = single price level with > 30% of total side volume.
        Concentration = how much volume sits in the top-3 levels (thin book indicator).
        """
        result: dict = {}

        # Bid side
        if bids and total_bid > 0:
            sorted_bids = sorted(
                bids, key=lambda b: float(b.get("size", 0)), reverse=True
            )
            largest = float(sorted_bids[0].get("size", 0))
            result["bid_wall_pct"] = round(largest / total_bid, 4)
            result["bid_wall_price"] = float(sorted_bids[0].get("price", 0))
            result["bid_top3_concentration"] = round(
                sum(float(b.get("size", 0)) for b in sorted_bids[:3]) / total_bid, 4
            )
        else:
            result["bid_wall_pct"] = 0.0
            result["bid_wall_price"] = 0.0
            result["bid_top3_concentration"] = 0.0

        # Ask side
        if asks and total_ask > 0:
            sorted_asks = sorted(
                asks, key=lambda a: float(a.get("size", 0)), reverse=True
            )
            largest = float(sorted_asks[0].get("size", 0))
            result["ask_wall_pct"] = round(largest / total_ask, 4)
            result["ask_wall_price"] = float(sorted_asks[0].get("price", 0))
            result["ask_top3_concentration"] = round(
                sum(float(a.get("size", 0)) for a in sorted_asks[:3]) / total_ask, 4
            )
        else:
            result["ask_wall_pct"] = 0.0
            result["ask_wall_price"] = 0.0
            result["ask_top3_concentration"] = 0.0

        return result

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self):
        """Close all HTTP connection pools."""
        await self._gamma.aclose()
        await self._clob_http.aclose()


# ── Connector ────────────────────────────────────────────────────────


class PolymarketConnector(BaseConnector):
    """
    Polymarket integration block — prediction markets data & trading.

    Provides an async client via `self.client`. Supports:
    - Market discovery (Gamma API — public, no auth)
    - Pricing & order book (CLOB API — public reads)
    - Trading (CLOB API — requires POLYMARKET_PRIVATE_KEY)
    """

    @property
    def name(self) -> str:
        return "polymarket"

    @property
    def icon(self) -> str:
        return "🔮"

    @property
    def description(self) -> str:
        return "Polymarket prediction markets — data & trading"

    def __init__(self):
        self._client: AsyncPolymarketClient | None = None

    @property
    def client(self) -> AsyncPolymarketClient:
        """Get the async Polymarket client. Lazy-initializes on first access."""
        if self._client is None:
            private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            proxy_wallet = os.environ.get("POLYMARKET_PROXY_WALLET", "")
            api_key = os.environ.get("POLYMARKET_API_KEY", "")
            api_secret = os.environ.get("POLYMARKET_API_SECRET", "")
            passphrase = os.environ.get("POLYMARKET_PASSPHRASE", "")
            self._client = AsyncPolymarketClient(
                private_key=private_key or None,
                proxy_wallet=proxy_wallet or None,
                api_key=api_key or None,
                api_secret=api_secret or None,
                passphrase=passphrase or None,
            )
        return self._client

    async def setup(self) -> None:
        """Pre-initialize the client."""
        _ = self.client

    async def teardown(self) -> None:
        """Close the HTTP connection pools."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """Check Polymarket API connectivity via Gamma /events."""
        try:
            events = await self.client.get_events(limit=1)
            return len(events) > 0
        except Exception:
            return False
