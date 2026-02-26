"""
ADK-native tools for the Polymarket BTC trading workflow.

Each tool is registered via ``@tool`` and referenced by string name
in the agent's ``tools=[...]`` list. The platform auto-resolves them
at agent creation time (§4 Tool Ecosystem, ARCHITECTURE.md).

Tools use signal functions from ``steps.py`` as backends.
"""

import json
import math
import time

import httpx
import structlog

from autopilot.core.tools import tool
from autopilot.connectors.binance_feed import BinancePriceFeed, Candle

from workflows.polymarket_btc.steps import (
    compute_window_timing,
    compute_momentum_signal,
    compute_ta_signal,
    compute_orderbook_signal,
    compute_market_price_signal,
)

logger = structlog.get_logger(__name__)


# ── Shared state (process-scoped, recreated on cold start) ──────────


def _get_risk_manager():
    """Lazy singleton for the risk manager (delegates to risk.py).

    Reads settings from env vars (12-Factor), falling back to
    manifest.yaml defaults: max_trade_size=20, daily_loss_cap=100,
    min_conviction=65.
    """
    import os

    from workflows.polymarket_btc.risk import get_risk_manager

    return get_risk_manager(
        max_trade_size_usd=float(os.getenv("MAX_TRADE_SIZE_USD", "20.0")),
        daily_loss_cap_usd=float(os.getenv("DAILY_LOSS_CAP_USD", "100.0")),
        min_conviction_score=int(os.getenv("MIN_CONVICTION_SCORE", "65")),
    )


# ── Shared: Binance TA Snapshot ─────────────────────────────────────
# Cached per-process to avoid duplicate API calls when agent calls
# both get_btc_price and get_ta_indicators in the same cycle.

_cached_ta: dict | None = None
_cached_ta_ts: float = 0.0
_TA_CACHE_TTL = 5.0  # seconds — near real-time, 1m candles update every second


async def _fetch_ta_snapshot() -> tuple[float, object]:
    """Fetch BTC price + TA snapshot from Binance (cached 30s)."""
    global _cached_ta, _cached_ta_ts

    now = time.time()
    if _cached_ta and (now - _cached_ta_ts) < _TA_CACHE_TTL:
        return _cached_ta["price"], _cached_ta["snapshot"]

    async with httpx.AsyncClient() as http:
        # Current price
        resp = await http.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=10.0,
        )
        resp.raise_for_status()
        price = float(resp.json()["price"])

        # 1-minute candles for TA
        resp = await http.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": "50"},
            timeout=10.0,
        )
        resp.raise_for_status()
        klines = resp.json()

    feed = BinancePriceFeed()
    for k in klines:
        candle = Candle(
            timestamp=k[0] / 1000.0,
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            is_closed=True,
        )
        feed._candles.append(candle)

    ta = feed.get_snapshot()

    _cached_ta = {"price": price, "snapshot": ta}
    _cached_ta_ts = now

    return price, ta


# ── Tool 1: BTC Price + Momentum ───────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_btc_price() -> dict:
    """Fetch current BTC/USDT price and short-term momentum from Binance.

    Momentum measures price movement direction and magnitude over the
    last 1 and 3 minutes — the PRIMARY directional edge for 5-min markets.

    Returns:
        price: Current BTC/USDT price.
        direction_1m/3m: Price direction over 1 and 3 minutes.
        magnitude_1m/3m: Size of the move in USD.
        momentum_signal: Composite momentum score (0-100) and direction.
    """
    price, ta = await _fetch_ta_snapshot()
    momentum = compute_momentum_signal(ta)

    return {
        "price": price,
        "direction_1m": ta.direction_1m,
        "magnitude_1m": round(ta.magnitude_1m, 2),
        "direction_3m": ta.direction_3m,
        "magnitude_3m": round(ta.magnitude_3m, 2),
        "momentum_signal": {
            "score": momentum["score"],
            "direction": momentum["direction"],
        },
    }


# ── Tool 2: Technical Analysis Indicators ──────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_ta_indicators() -> dict:
    """Fetch technical analysis indicators for BTC from Binance.

    Each indicator provides an independent directional signal:
    - RSI: Overbought (>70) → bearish, Oversold (<30) → bullish
    - MACD histogram: Positive → bullish momentum, Negative → bearish
    - Bollinger Band position: >1 → near upper band (overbought), <0 → near lower (oversold)
    - VWAP deviation: Positive → price above VWAP (bullish), Negative → below (bearish)

    Returns:
        rsi, macd_histogram, bb_position, vwap, vwap_deviation: Raw values.
        ta_signal: Composite TA confluence score (0-100) and direction.
    """
    price, ta = await _fetch_ta_snapshot()
    ta_signal = compute_ta_signal(ta)

    return {
        "rsi": round(ta.rsi, 1),
        "macd_histogram": round(ta.macd_histogram, 2),
        "bb_position": round(ta.bb_position, 2),
        "vwap": round(ta.vwap, 2),
        "vwap_deviation": round(ta.vwap_deviation, 3),
        "ta_signal": {
            "score": ta_signal["score"],
            "direction": ta_signal["direction"],
        },
    }


# ── Tool 2: Polymarket Market State ────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_market_state(duration: str = "5m") -> dict:
    """Find the active Polymarket BTC market for the given duration and analyze timing + outcome prices.

    Args:
        duration: Market window duration to search for ('5m', '15m', '1h').

    Returns market question, window timing (elapsed/remaining/should_trade),
    outcome prices (Up/Down), and the market price signal (delta from 0.50).
    """
    from autopilot.connectors.polymarket_connector import AsyncPolymarketClient

    client = AsyncPolymarketClient()
    try:
        markets = await client.find_active_btc_markets(duration=duration)

        if not markets:
            return {
                "found": False,
                "reason": "No active BTC 5-min markets on Polymarket",
            }

        market = markets[0]

        # Window timing
        window_start = market.get("_window_start", 0)
        window_interval = market.get("_window_interval", 300)
        timing = compute_window_timing(window_start, window_interval)

        # Outcome prices → Price to Beat signal
        outcome_prices = json.loads(market.get("outcomePrices", '["0.5", "0.5"]'))
        outcomes = json.loads(market.get("outcomes", '["Up", "Down"]'))
        mkt_signal = compute_market_price_signal(outcome_prices, outcomes)

        # Extract token IDs
        tokens = market.get("tokens", [])
        if not tokens:
            clob_ids_raw = market.get("clobTokenIds", "")
            if isinstance(clob_ids_raw, str) and clob_ids_raw.startswith("["):
                token_ids = json.loads(clob_ids_raw)
                tokens = [{"token_id": tid} for tid in token_ids]

        up_token = tokens[0].get("token_id", "") if tokens else ""
        down_token = tokens[1].get("token_id", "") if len(tokens) > 1 else ""

        return {
            "found": True,
            "question": market.get("question", ""),
            "condition_id": market.get("conditionId", ""),
            "up_token": up_token,
            "down_token": down_token,
            "timing": {
                "should_trade": timing["should_trade"],
                "reason": timing["reason"],
                "elapsed": timing["elapsed"],
                "remaining": timing["remaining"],
                "timing_score": timing["timing_score"],
            },
            "market_price_signal": {
                "score": mkt_signal["score"],
                "direction": mkt_signal["direction"],
                "delta": mkt_signal["delta"],
                "up_price": mkt_signal["up_price"],
                "down_price": mkt_signal["down_price"],
            },
        }
    finally:
        await client.close()


# ── Tool 3: Order Book Analysis ────────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_order_book(up_token: str) -> dict:
    """Analyze order book imbalance for the active market's Up token.

    Args:
        up_token: The token ID for the "Up" outcome.

    Returns bid/ask imbalance, spread, and directional bias.
    """
    from autopilot.connectors.polymarket_connector import AsyncPolymarketClient

    client = AsyncPolymarketClient()
    try:
        result = await compute_orderbook_signal(client, up_token)
        return result
    finally:
        await client.close()


# ── Tool 4: Risk State ─────────────────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_risk_state() -> dict:
    """Check current risk limits — whether trading is allowed.

    Returns daily P&L, trade count, consecutive losses, and whether
    the next trade is allowed based on current risk parameters.
    State is loaded from platform ArtifactService for cross-run durability.
    """
    risk = _get_risk_manager()

    # Load persisted state from ArtifactService
    await risk.load()

    # Reset daily if new day
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    risk.reset_daily(today)

    return {
        **risk.get_summary(),
        "min_conviction_required": risk.min_conviction,
        "max_trade_size_usd": risk.max_trade_size,
    }


# ── Tool 7: Derivatives Sentiment ──────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_derivatives_sentiment() -> dict:
    """Fetch BTC perpetual futures sentiment from Binance Futures.

    Reveals institutional positioning via:
    - Funding rate: Positive = longs paying shorts (overleveraged longs → bearish pressure)
    - Open interest: Rising OI = new money entering, falling OI = liquidations
    - OI trend: Compares current OI to 1-hour average for momentum

    Key patterns:
    - High funding (>0.01%) + rising OI = longs piling in → crash risk
    - Negative funding + dropping OI = short squeeze potential
    - OI dropping + price dropping = long liquidation cascade (very bearish)
    - OI dropping + price rising = short squeeze (very bullish)
    """
    async with httpx.AsyncClient() as http:
        # Current funding rate
        resp = await http.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": "3"},
            timeout=10.0,
        )
        resp.raise_for_status()
        funding_data = resp.json()

        # Current open interest
        resp = await http.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=10.0,
        )
        resp.raise_for_status()
        oi_data = resp.json()

        # Historical OI (5-min intervals, last hour)
        resp = await http.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": "BTCUSDT", "period": "5m", "limit": "12"},
            timeout=10.0,
        )
        resp.raise_for_status()
        oi_hist = resp.json()

    # Current funding rate
    current_funding = float(funding_data[-1]["fundingRate"]) if funding_data else 0.0
    prev_funding = (
        float(funding_data[-2]["fundingRate"])
        if len(funding_data) > 1
        else current_funding
    )

    # Open interest analysis
    current_oi = float(oi_data.get("openInterest", 0))

    # OI trend (compare to 1-hour average)
    oi_values = [float(h.get("sumOpenInterest", 0)) for h in oi_hist] if oi_hist else []
    avg_oi = sum(oi_values) / len(oi_values) if oi_values else current_oi
    oi_change_pct = ((current_oi - avg_oi) / avg_oi * 100) if avg_oi > 0 else 0

    # Classify sentiment
    if current_funding > 0.0005:
        funding_bias = "very_bearish"
    elif current_funding > 0.0001:
        funding_bias = "bearish"
    elif current_funding < -0.0005:
        funding_bias = "very_bullish"
    elif current_funding < -0.0001:
        funding_bias = "bullish"
    else:
        funding_bias = "neutral"

    if oi_change_pct > 3:
        oi_trend = "rising_fast"
    elif oi_change_pct > 0.5:
        oi_trend = "rising"
    elif oi_change_pct < -3:
        oi_trend = "falling_fast"
    elif oi_change_pct < -0.5:
        oi_trend = "falling"
    else:
        oi_trend = "stable"

    return {
        "funding_rate": round(current_funding * 100, 4),  # as percentage
        "funding_rate_prev": round(prev_funding * 100, 4),
        "funding_bias": funding_bias,
        "open_interest_btc": round(current_oi, 2),
        "oi_1h_avg_btc": round(avg_oi, 2),
        "oi_change_pct": round(oi_change_pct, 2),
        "oi_trend": oi_trend,
        "interpretation": (
            f"Funding {funding_bias} ({current_funding * 100:.4f}%), "
            f"OI {oi_trend} ({oi_change_pct:+.2f}% vs 1h avg)"
        ),
    }


# ── Tool 7B: Liquidation Cascade Detection ─────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_liquidation_data() -> dict:
    """Detect BTC liquidation cascades from Binance Futures using Open Interest & Price.

    Since the raw forceOrders API is restricted, we infer liquidations by
    looking for simultaneous drops in Open Interest (OI) and rapid price movement.

    Key signals:
    - OI drops + Price drops rapidly → Longs liquidated (cascading sell pressure)
    - OI drops + Price rises rapidly → Shorts liquidated (short squeeze)

    Returns:
        long_liq_usd: Estimated long liquidation volume (USDT)
        short_liq_usd: Estimated short liquidation volume (USDT)
        net_direction: "up" (short squeeze), "down" (long cascade), "neutral"
        cascade_level: "none", "minor", "major"
    """
    try:
        async with httpx.AsyncClient() as http:
            # Get 5m historical OI
            resp_oi = await http.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": "BTCUSDT", "period": "5m", "limit": "4"},
                timeout=5.0,
            )
            resp_oi.raise_for_status()
            oi_hist = resp_oi.json()

            # Get 5m klines
            resp_kl = await http.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "5m", "limit": "4"},
                timeout=5.0,
            )
            resp_kl.raise_for_status()
            klines = resp_kl.json()
    except Exception:
        return {
            "long_liq_usd": 0.0,
            "short_liq_usd": 0.0,
            "net_direction": "neutral",
            "cascade_level": "none",
            "count": 0,
        }

    if len(oi_hist) < 2 or len(klines) < 2:
        return {
            "long_liq_usd": 0.0,
            "short_liq_usd": 0.0,
            "net_direction": "neutral",
            "cascade_level": "none",
            "count": 0,
        }

    oi_start = float(oi_hist[0].get("sumOpenInterestValue", 0))
    oi_end = float(oi_hist[-1].get("sumOpenInterestValue", 0))
    price_start = float(klines[0][1])  # Open of first candle
    price_end = float(klines[-1][4])  # Close of last candle

    oi_drop = oi_start - oi_end
    price_change_pct = ((price_end - price_start) / price_start) * 100

    long_liq = 0.0
    short_liq = 0.0
    count = 0

    # If OI dropped significantly, it implies liquidations or mass closures
    if oi_drop > 1_000_000:  # $1M drop minimum to be considered meaningful
        count = int(oi_drop / 50_000)  # Estimate ~50k per order on average
        if price_change_pct < -0.2:
            # OI down, Price down = Longs capitulating/liquidated
            long_liq = oi_drop * 0.8  # Assume 80% of OI drop was longs
            short_liq = oi_drop * 0.2
        elif price_change_pct > 0.2:
            # OI down, Price up = Shorts squeezed/liquidated
            short_liq = oi_drop * 0.8
            long_liq = oi_drop * 0.2
        else:
            # Mixed/neutral closure
            long_liq = oi_drop * 0.5
            short_liq = oi_drop * 0.5

    total_liq = long_liq + short_liq
    if total_liq > 20_000_000:
        cascade_level = "major"
    elif total_liq > 5_000_000:
        cascade_level = "minor"
    else:
        cascade_level = "none"

    if long_liq > short_liq * 2 and long_liq > 1_000_000:
        net_direction = "down"
    elif short_liq > long_liq * 2 and short_liq > 1_000_000:
        net_direction = "up"
    else:
        net_direction = "neutral"

    return {
        "long_liq_usd": round(long_liq, 2),
        "short_liq_usd": round(short_liq, 2),
        "net_direction": net_direction,
        "cascade_level": cascade_level,
        "count": count,
    }


# ── Tool 8: Volatility Regime ─────────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_volatility_regime() -> dict:
    """Classify current BTC volatility regime from 1-minute candles.

    Different strategies work in different regimes:
    - LOW volatility (choppy): Favor mean-reversion, smaller positions
    - NORMAL volatility: Standard strategy, moderate positions
    - HIGH volatility (trending): Favor trend-following, larger positions
    - EXTREME volatility: SKIP — too unpredictable

    Computes ATR (Average True Range) and Bollinger Band width to determine regime.
    """
    _, ta = await _fetch_ta_snapshot()

    # We need the raw candles for ATR calculation
    global _cached_ta
    if not _cached_ta:
        return {"regime": "unknown", "reason": "No candle data available"}

    # Reconstruct candles from the BinancePriceFeed
    # Use a fresh fetch for the raw candle data
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": "50"},
            timeout=10.0,
        )
        resp.raise_for_status()
        klines = resp.json()

    # Compute ATR (14-period)
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    true_ranges = []
    for i in range(1, len(klines)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < 14:
        return {"regime": "unknown", "reason": "Not enough candles for ATR"}

    # Current ATR (14-period EMA)
    atr_14 = sum(true_ranges[:14]) / 14
    for tr in true_ranges[14:]:
        atr_14 = (atr_14 * 13 + tr) / 14

    # ATR as percentage of price
    current_price = closes[-1]
    atr_pct = (atr_14 / current_price) * 100

    # Historical ATR for percentile ranking
    atr_history = []
    window_trs = true_ranges[:14]
    atr_val = sum(window_trs) / 14
    atr_history.append(atr_val)
    for tr in true_ranges[14:]:
        atr_val = (atr_val * 13 + tr) / 14
        atr_history.append(atr_val)

    sorted_atrs = sorted(atr_history)
    percentile = sorted_atrs.index(min(sorted_atrs, key=lambda x: abs(x - atr_14)))
    atr_percentile = (percentile / len(sorted_atrs)) * 100

    # Bollinger Band width (measure of compression/expansion)
    if len(closes) >= 20:
        bb_window = closes[-20:]
        bb_mean = sum(bb_window) / 20
        bb_std = math.sqrt(sum((x - bb_mean) ** 2 for x in bb_window) / 20)
        bb_width = (4 * bb_std / bb_mean) * 100  # Width as % of price
    else:
        bb_width = 0

    # Classify regime — calibrated for BTC's natural 1m volatility.
    # BTC typically has ATR of 0.10-0.25% on 1m candles; the old
    # 0.15% threshold classified ~80% of the time as "extreme".
    # New thresholds reserve "extreme" for true black-swan events.
    if atr_pct > 0.35 or atr_percentile > 95:
        regime = "extreme"
        strategy = "SKIP — volatility too high, unpredictable"
    elif atr_pct > 0.20 or atr_percentile > 80:
        regime = "high"
        strategy = "Trend-following preferred, full position sizing"
    elif atr_pct > 0.10 or atr_percentile > 40:
        regime = "normal"
        strategy = "Standard strategy, moderate positions"
    else:
        regime = "low"
        strategy = "Mean-reversion preferred, smaller positions"

    return {
        "regime": regime,
        "strategy_recommendation": strategy,
        "atr_14": round(atr_14, 2),
        "atr_pct": round(atr_pct, 4),
        "atr_percentile": round(atr_percentile, 1),
        "bb_width_pct": round(bb_width, 4),
        "current_price": round(current_price, 2),
    }


# ── Tool 9: Multi-Timeframe Momentum ──────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_multi_timeframe() -> dict:
    """Analyze BTC momentum across multiple timeframes (1m, 5m, 15m).

    Key insight: A 1m downtrend inside a 15m uptrend usually reverses.
    Alignment across all timeframes = highest conviction trades.

    Patterns:
    - All DOWN: Strong bearish (trend continuation)
    - All UP: Strong bullish (trend continuation)
    - 1m DOWN, 5m/15m UP: Counter-trend pullback → likely bounce UP
    - 1m UP, 5m/15m DOWN: Counter-trend rally → likely reversal DOWN
    """
    async with httpx.AsyncClient() as http:
        # 5-minute candles
        resp_5m = await http.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "5m", "limit": "20"},
            timeout=10.0,
        )
        resp_5m.raise_for_status()
        klines_5m = resp_5m.json()

        # 15-minute candles
        resp_15m = await http.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": "10"},
            timeout=10.0,
        )
        resp_15m.raise_for_status()
        klines_15m = resp_15m.json()

    def analyze_tf(klines, label):
        closes = [float(k[4]) for k in klines]
        if len(closes) < 4:
            return {"direction": "neutral", "magnitude": 0, "label": label}

        # Direction: last 3 candles
        changes = [closes[i] - closes[i - 1] for i in range(-3, 0)]
        if all(c > 0 for c in changes):
            direction = "up"
        elif all(c < 0 for c in changes):
            direction = "down"
        else:
            direction = "neutral"

        magnitude = closes[-1] - closes[-4]

        # Simple RSI for this timeframe
        if len(closes) >= 15:
            deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            gains = [max(d, 0) for d in deltas[-14:]]
            losses = [abs(min(d, 0)) for d in deltas[-14:]]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = round(100 - (100 / (1 + rs)), 1)
        else:
            rsi = 50.0

        return {
            "direction": direction,
            "magnitude": round(magnitude, 2),
            "rsi": rsi,
            "label": label,
        }

    # Get 1m direction from cached snapshot
    _, ta = await _fetch_ta_snapshot()
    tf_1m = {
        "direction": ta.direction_3m,  # Use 3m window for 1m TF
        "magnitude": round(ta.magnitude_3m, 2),
        "label": "1m",
    }

    tf_5m = analyze_tf(klines_5m, "5m")
    tf_15m = analyze_tf(klines_15m, "15m")

    directions = [tf_1m["direction"], tf_5m["direction"], tf_15m["direction"]]

    # Alignment classification
    if all(d == "up" for d in directions):
        alignment = "all_bullish"
        interpretation = "Strong bullish — all timeframes aligned UP"
    elif all(d == "down" for d in directions):
        alignment = "all_bearish"
        interpretation = "Strong bearish — all timeframes aligned DOWN"
    elif tf_1m["direction"] == "down" and tf_15m["direction"] == "up":
        alignment = "bullish_pullback"
        interpretation = (
            "1m pullback in 15m uptrend → likely bounce UP (contrarian buy)"
        )
    elif tf_1m["direction"] == "up" and tf_15m["direction"] == "down":
        alignment = "bearish_rally"
        interpretation = (
            "1m rally in 15m downtrend → likely reversal DOWN (contrarian sell)"
        )
    elif tf_5m["direction"] == tf_15m["direction"] and tf_5m["direction"] != "neutral":
        alignment = f"higher_tf_{tf_5m['direction']}"
        interpretation = f"5m+15m agree {tf_5m['direction'].upper()} — 1m is noise"
    else:
        alignment = "mixed"
        interpretation = "No clear alignment — low conviction"

    return {
        "timeframes": {
            "1m": tf_1m,
            "5m": tf_5m,
            "15m": tf_15m,
        },
        "alignment": alignment,
        "interpretation": interpretation,
    }


# ── Tool 11: Order Flow Toxicity (VPIN) ────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_vpin_signal() -> dict:
    """Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

    Approximates VPIN using 1-minute Binance candles over a 30-minute window.
    VPIN measures order flow toxicity — when informed traders (smart money)
    aggressively fill the order book in one direction.

    High VPIN (> 70) = high toxicity = strong directional continuation likely.
    Returns VPIN score, buy/sell volume split, and directional bias.
    """
    async with httpx.AsyncClient() as http:
        # Fetch last 30 1-minute candles
        resp = await http.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": "30"},
            timeout=10.0,
        )
        resp.raise_for_status()
        klines = resp.json()

    # VPIN Calculation (Time-bucket approximation)
    # k[5] is total volume, k[9] is taker buy base asset volume
    # sell volume = total - buy volume
    total_vol = 0.0
    total_imbalance = 0.0
    total_buy = 0.0
    total_sell = 0.0

    for k in klines:
        vol = float(k[5])
        buy_vol = float(k[9])
        sell_vol = max(0.0, vol - buy_vol)

        imbalance = abs(buy_vol - sell_vol)

        total_vol += vol
        total_imbalance += imbalance
        total_buy += buy_vol
        total_sell += sell_vol

    # VPIN formula = sum(|buy - sell|) / sum(volume)
    vpin_raw = (total_imbalance / total_vol) if total_vol > 0 else 0.0

    # Scale VPIN to 0-100 for our risk engine
    # Typically VPIN fluctuates between 0.1 and 0.5. Let's normalize it roughly.
    # A VPIN of 0.4+ is extremely toxic/directional.
    vpin_score = min(100.0, (vpin_raw / 0.4) * 100)

    if total_buy > total_sell * 1.2:
        direction = "bullish"
        if vpin_score > 60:
            interp = f"High toxicity ({vpin_score:.1f}) — aggressive BUY flow"
        else:
            interp = f"Moderate BUY imbalance ({vpin_score:.1f})"
    elif total_sell > total_buy * 1.2:
        direction = "bearish"
        if vpin_score > 60:
            interp = f"High toxicity ({vpin_score:.1f}) — aggressive SELL flow"
        else:
            interp = f"Moderate SELL imbalance ({vpin_score:.1f})"
    else:
        direction = "neutral"
        interp = f"Balanced flow ({vpin_score:.1f})"

    return {
        "vpin_score": round(vpin_score, 1),
        "buy_volume": round(total_buy, 2),
        "sell_volume": round(total_sell, 2),
        "direction": direction,
        "interpretation": interp,
    }


# ── Tool 10: Trade History ─────────────────────────────────────────


@tool(tags=["polymarket", "btc"])
async def get_trade_history() -> dict:
    """Query your past trade performance for Kelly Criterion calibration.

    Returns aggregate stats: win rate, total P&L, resolved/pending counts.
    Use win_rate_pct as 'p' in Kelly formula.
    If resolved < 20, use conservative p=0.55 estimate instead.

    Stats stored as compact counters (~200 bytes). Never grows.
    """
    from workflows.polymarket_btc.trade_history import get_trade_history as _get_history

    history = _get_history()
    return await history.get_stats()
