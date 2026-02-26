"""
Pipeline step functions and signal helpers for the Polymarket BTC workflow.

Pipeline steps (referenced by pipeline.yaml):
    resolve_outcomes_step — Resolve pending trade outcomes via Polymarket API
    gather_market_data    — Multi-market scan + parallel data gathering from 14 sources
    score_trade           — 3-layer Signal Confluence Engine (Bayesian sizing)
    risk_gate             — Deterministic 10-rule risk enforcement
    execute_trade_step    — Polymarket CLOB order with price drift guard
    log_performance       — Structured performance metrics for monitoring

Signal helpers (used by tools.py and gather_market_data):
    compute_window_timing, compute_momentum_signal, compute_ta_signal,
    compute_orderbook_signal, compute_market_price_signal

Confluence Engine internals (used by score_trade):
    _score_binance_composite  — Layer 1A: aggregate Binance TA
    _score_polymarket_flow    — Layer 1B: aggregate Polymarket flow + smart money

Utilities:
    execute_trade         — Paper/live order placement via Polymarket CLOB
"""

from typing import TYPE_CHECKING

import os
import structlog

from workflows.polymarket_btc.signal_dna import (
    compute_fingerprint,
    get_dna_tracker,
)
from workflows.polymarket_btc.models import (
    BinanceComposite,
    ConfluenceResult,
    MarketAnalysis,
    PolymarketFlow,
    TradeProposal,
)

if TYPE_CHECKING:
    from autopilot.core.context import AgentContext

logger = structlog.get_logger(__name__)


# ── Timing Constants ────────────────────────────────────────────────

# Multi-duration timing constants (scaled proportionally by window size).
# Each window needs: MIN_ELAPSED before trading, IDEAL sweet spot, and
# MAX_ELAPSED after which too little time for order to fill.
_TIMING_BY_DURATION = {
    "5m": {"min": 30, "ideal": 75, "max": 240},  # 300s window — 210s tradeable (70%)
    "15m": {"min": 90, "ideal": 300, "max": 600},  # 900s window
    "1h": {"min": 300, "ideal": 900, "max": 2400},  # 3600s window
}

# Backwards-compatible defaults for unrecognized durations
MIN_WINDOW_ELAPSED = 30
IDEAL_WINDOW_ELAPSED = 75
MAX_WINDOW_ELAPSED = 240


# ── Signal: Window Timing Analysis ──────────────────────────────────


def compute_window_timing(
    window_start: int,
    window_interval: int,
    current_time: int | None = None,
    duration: str = "5m",
) -> dict:
    """Decide if this is the right moment to trade within the window.

    The bot runs every minute. This function determines:
    - How far into the window we are
    - Whether we have enough intra-window data
    - Whether there's enough time left for the trade to resolve
    - A timing quality score (0-100)

    Args:
        window_start: Start time of the window (epoch seconds).
        window_interval: Total window duration in seconds.
        current_time: Override for testing (epoch seconds).
        duration: Market duration string (5m, 15m, 1h) for adaptive constants.

    Returns:
        Dict with should_trade, reason, elapsed, remaining, and timing_score.
    """
    import time as _time

    now = current_time or int(_time.time())
    elapsed = now - window_start
    remaining = window_interval - elapsed

    # Select timing constants for this duration
    tc = _TIMING_BY_DURATION.get(duration, _TIMING_BY_DURATION["5m"])
    min_elapsed = tc["min"]
    ideal_elapsed = tc["ideal"]
    max_elapsed = tc["max"]

    # Too early — not enough data formed inside this window
    if elapsed < min_elapsed:
        return {
            "should_trade": False,
            "reason": f"Too early — {elapsed}s elapsed, need {min_elapsed}s for intra-window confirmation",
            "elapsed": elapsed,
            "remaining": remaining,
            "timing_score": 0,
        }

    # Too late — not enough time for order to fill
    if elapsed > max_elapsed:
        return {
            "should_trade": False,
            "reason": f"Too late — only {remaining}s remaining, need ≥{window_interval - max_elapsed}s",
            "elapsed": elapsed,
            "remaining": remaining,
            "timing_score": 0,
        }

    # In the sweet spot — compute timing quality
    # Peak quality at ideal, tapering off either side
    distance_from_ideal = abs(elapsed - ideal_elapsed)
    max_distance = max(
        ideal_elapsed - min_elapsed,
        max_elapsed - ideal_elapsed,
    )
    timing_score = max(0, 100 - (distance_from_ideal / max_distance * 40))

    # ── Decaying Edge: informational advantage decays linearly ──
    # 1.0 at min_elapsed → 0.0 at max_elapsed
    # This front-loads position sizing to when your Binance TA edge
    # hasn't been priced into Polymarket yet.
    tradeable_range = max_elapsed - min_elapsed
    edge_decay = (
        max(0.0, 1.0 - (elapsed - min_elapsed) / tradeable_range)
        if tradeable_range > 0
        else 0.0
    )

    # Timing zone labels for logging/strategy
    if elapsed < ideal_elapsed:
        timing_zone = "alpha"  # Early — max informational edge
    elif elapsed < (ideal_elapsed + max_elapsed) / 2:
        timing_zone = "confirm"  # Mid — signal confirmed, edge fading
    else:
        timing_zone = "late"  # Late — edge nearly dead

    return {
        "should_trade": True,
        "reason": f"Good timing — {elapsed}s elapsed, {remaining}s remaining ({timing_zone} zone)",
        "elapsed": elapsed,
        "remaining": remaining,
        "timing_score": round(timing_score, 1),
        "edge_decay": round(edge_decay, 3),
        "timing_zone": timing_zone,
    }


# ── Signal: Momentum ────────────────────────────────────────────────


def compute_momentum_signal(ta_snapshot) -> dict:
    """Compute momentum signal from Binance TA snapshot.

    The primary edge: if BTC moves consistently in one direction
    for 3 minutes of a 5-minute candle, there's a ~78-96%
    probability of continuation.

    Returns:
        Dict with score (0-100), direction ('up'/'down'/'neutral'),
        and confidence metrics.
    """
    score = 50.0  # Neutral baseline
    direction = "neutral"

    # Strong 3-minute consistency = primary signal
    # Graduated by magnitude: bigger moves = higher continuation probability
    mag = abs(ta_snapshot.magnitude_3m)
    if ta_snapshot.direction_3m == "up":
        direction = "up"
        if mag >= 200:
            score += 40  # Monster move — ~96% continuation
        elif mag >= 100:
            score += 30  # Strong move — ~85% continuation
        elif mag >= 50:
            score += 20  # Moderate move — ~70% continuation
        else:
            score += 10  # Weak move — might be noise
    elif ta_snapshot.direction_3m == "down":
        direction = "down"
        if mag >= 200:
            score += 40
        elif mag >= 100:
            score += 30
        elif mag >= 50:
            score += 20
        else:
            score += 10

    # 1-minute confirmation
    if ta_snapshot.direction_1m == ta_snapshot.direction_3m:
        score += 5

    # Clamp to 0-100
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "direction": direction,
        "magnitude_3m": ta_snapshot.magnitude_3m,
        "direction_3m": ta_snapshot.direction_3m,
        "direction_1m": ta_snapshot.direction_1m,
    }


# ── Signal: Technical Analysis Confluence ───────────────────────────


def compute_ta_signal(ta_snapshot) -> dict:
    """Compute technical analysis confluence signal.

    Combines RSI, MACD, Bollinger Bands, and VWAP to determine
    overall directional bias and confidence.

    Returns:
        Dict with score (0-100), direction, and individual indicator values.
    """
    score = 50.0
    bullish_count = 0
    bearish_count = 0

    # RSI
    if ta_snapshot.rsi < 30:
        bullish_count += 1  # Oversold → expect bounce up
        score += 10
    elif ta_snapshot.rsi > 70:
        bearish_count += 1  # Overbought → expect pullback
        score += 10

    # MACD Histogram
    if ta_snapshot.macd_histogram > 0:
        bullish_count += 1
        score += 5
    elif ta_snapshot.macd_histogram < 0:
        bearish_count += 1
        score += 5

    # Bollinger Band Position
    if ta_snapshot.bb_position < 0.2:
        bullish_count += 1  # Near lower band → expect bounce
        score += 8
    elif ta_snapshot.bb_position > 0.8:
        bearish_count += 1  # Near upper band → expect drop
        score += 8

    # VWAP Deviation
    if ta_snapshot.vwap_deviation > 0.1:
        bullish_count += 1  # Above VWAP = bullish
        score += 3
    elif ta_snapshot.vwap_deviation < -0.1:
        bearish_count += 1  # Below VWAP = bearish
        score += 3

    # Confluence bonus: all indicators agree
    if bullish_count >= 3 or bearish_count >= 3:
        score += 15  # Strong confluence

    direction = (
        "up"
        if bullish_count > bearish_count
        else "down"
        if bearish_count > bullish_count
        else "neutral"
    )

    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "direction": direction,
        "rsi": ta_snapshot.rsi,
        "macd_histogram": ta_snapshot.macd_histogram,
        "bb_position": round(ta_snapshot.bb_position, 3),
        "vwap_deviation": round(ta_snapshot.vwap_deviation, 4),
        "bullish_indicators": bullish_count,
        "bearish_indicators": bearish_count,
    }


# ── Signal: Order Book Imbalance ────────────────────────────────────


async def compute_orderbook_signal(
    polymarket_client,
    token_id: str,
) -> dict:
    """Compute order book imbalance signal.

    Bid-heavy imbalance → market expects YES (up).
    Ask-heavy imbalance → market expects NO (down).

    Returns:
        Dict with score (0-100), direction, and imbalance metrics.
    """
    try:
        imbalance = await polymarket_client.get_order_book_imbalance(token_id)
    except Exception as e:
        logger.warning("orderbook_signal_failed", error=str(e))
        return {"score": 50.0, "direction": "neutral", "error": str(e)}

    score = 50.0
    ratio = imbalance["imbalance_ratio"]

    # Strong imbalance = strong signal
    if abs(ratio) > 0.3:
        score += 25
    elif abs(ratio) > 0.15:
        score += 12
    elif abs(ratio) > 0.05:
        score += 5

    # Spread bonus: tight spread = high confidence
    if imbalance["spread"] < 0.05:
        score += 10
    elif imbalance["spread"] < 0.10:
        score += 5

    direction = "up" if ratio > 0.05 else "down" if ratio < -0.05 else "neutral"

    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "direction": direction,
        "imbalance_ratio": ratio,
        "spread": imbalance["spread"],
        "midpoint": imbalance["midpoint"],
        "total_bid": imbalance["total_bid_size"],
        "total_ask": imbalance["total_ask_size"],
    }


# ── Signal: Market Price (Price to Beat) ────────────────────────────


def compute_market_price_signal(
    outcome_prices: list[str],
    outcomes: list[str],
) -> dict:
    """Compute signal from Polymarket outcome prices (Price to Beat proxy).

    The outcome prices encode the market's view of where BTC is relative
    to the Price to Beat (set by Chainlink oracle at window start):
    - Up=0.82, Down=0.18 → BTC is well ABOVE target
    - Up=0.50, Down=0.50 → BTC is right AT the target
    - Up=0.20, Down=0.80 → BTC is well BELOW target

    The delta from 0.50 is the key signal:
    - Large positive delta → Up is strongly favored
    - Large negative delta → Down is strongly favored
    - Near zero → coin flip, no edge

    Args:
        outcome_prices: List of price strings, e.g. ["0.545", "0.455"]
        outcomes: List of outcome names, e.g. ["Up", "Down"]

    Returns:
        Dict with score (0-100), direction, delta, and market implied probability.
    """
    try:
        # Parse prices — outcomes[0] = Up, outcomes[1] = Down
        up_price = float(outcome_prices[0])
        down_price = float(outcome_prices[1])
    except (IndexError, ValueError, TypeError):
        return {
            "score": 50.0,
            "direction": "neutral",
            "delta": 0.0,
            "up_price": 0.5,
            "down_price": 0.5,
        }

    # Delta from fair value (0.50)
    delta = up_price - 0.50  # Positive = Up favored, Negative = Down favored

    score = 50.0
    direction = "neutral"

    # The further from 0.50, the stronger the signal
    abs_delta = abs(delta)

    if abs_delta >= 0.30:
        # Very strong: Up=0.80+ or Down=0.80+ → one side is heavily favored
        score += 35
    elif abs_delta >= 0.15:
        # Strong: Up=0.65+ or Down=0.65+
        score += 25
    elif abs_delta >= 0.08:
        # Moderate: slight edge
        score += 15
    elif abs_delta >= 0.03:
        # Weak: barely off center
        score += 5
    # else: near 0.50 = no edge → stays at 50

    # Direction follows the favored outcome
    if delta > 0.03:
        direction = "up"
    elif delta < -0.03:
        direction = "down"

    score = max(0, min(100, score))

    logger.info(
        "market_price_signal",
        up_price=up_price,
        down_price=down_price,
        delta=round(delta, 4),
        direction=direction,
        score=round(score, 1),
    )

    return {
        "score": round(score, 1),
        "direction": direction,
        "delta": round(delta, 4),
        "up_price": up_price,
        "down_price": down_price,
    }


# ── Edge signal helpers ─────────────────────────────────────────────


def _compute_intra_window_trend(price_history: list) -> dict:
    """Compute price trend WITHIN the current window from /prices-history.

    Analyzes the last N price points to detect momentum:
    - 3+ consecutive higher prices = strong up trend
    - 3+ consecutive lower prices = strong down trend
    - Mixed = flat

    Returns:
        {"direction": "up"|"down"|"flat", "strength": 0.0-1.0, "points": N}
    """
    if not price_history or len(price_history) < 2:
        return {"direction": "flat", "strength": 0.0, "points": 0}

    # Extract prices (handle both "p" and "price" keys)
    prices = []
    for point in price_history:
        if isinstance(point, dict):
            p = point.get("p", point.get("price", None))
            if p is not None:
                prices.append(float(p))

    if len(prices) < 2:
        return {"direction": "flat", "strength": 0.0, "points": len(prices)}

    # Use last 5 points max (most recent momentum matters most)
    recent = prices[-5:]
    n = len(recent)

    # Count consecutive directional moves
    ups = sum(1 for i in range(1, n) if recent[i] > recent[i - 1])
    downs = sum(1 for i in range(1, n) if recent[i] < recent[i - 1])
    moves = n - 1

    if moves == 0:
        return {"direction": "flat", "strength": 0.0, "points": n}

    # Direction = which side dominates
    up_ratio = ups / moves
    down_ratio = downs / moves

    if up_ratio >= 0.7:
        direction = "up"
        strength = up_ratio
    elif down_ratio >= 0.7:
        direction = "down"
        strength = down_ratio
    else:
        direction = "flat"
        strength = 0.0

    # Amplify strength by magnitude of total move
    total_move = abs(recent[-1] - recent[0])
    if total_move > 0.10:
        strength = min(1.0, strength * 1.3)  # Big move = high conviction
    elif total_move > 0.05:
        strength = min(1.0, strength * 1.1)

    return {
        "direction": direction,
        "strength": round(strength, 4),
        "points": n,
        "start_price": round(recent[0], 4),
        "end_price": round(recent[-1], 4),
        "total_move": round(recent[-1] - recent[0], 4),
    }


def _compute_volume_quality(market_volume: dict) -> dict:
    """Classify volume/liquidity as a signal quality multiplier.

    High volume = smart money is active, our signals are more reliable.
    Low volume = noise dominates, need to be more cautious.

    Returns:
        {"quality": "high"|"normal"|"low", "multiplier": 0.5-1.3}
    """
    if not market_volume:
        return {"quality": "unknown", "multiplier": 1.0}

    vol = market_volume.get("volume", 0)
    liq = market_volume.get("liquidity", 0)

    # If no volume data at all, treat as unknown (don't penalize)
    if vol == 0 and liq == 0:
        return {"quality": "unknown", "multiplier": 1.0}

    # Volume thresholds (USD) — calibrated for BTC 5m markets
    if vol > 5000 or liq > 10000:
        return {"quality": "high", "multiplier": 1.2}
    elif vol > 1000 or liq > 3000:
        return {"quality": "normal", "multiplier": 1.0}
    elif vol > 200:
        return {"quality": "low", "multiplier": 0.7}
    else:
        return {"quality": "very_low", "multiplier": 0.5}


def _compute_last_trade_signal(last_trade: dict) -> dict:
    """Derive a directional signal from the last trade(s).

    If the last trade was a BUY on UP token → someone just went long UP.
    If the last trade was a SELL on UP token → someone just dumped UP.

    Returns:
        {"direction": "up"|"down"|"neutral", "confidence": 0.0-1.0}
    """
    if not last_trade:
        return {"direction": "neutral", "confidence": 0.0}

    up_price = last_trade.get("price", 0.5)
    up_side = last_trade.get("side", "unknown").upper()

    # A BUY at high price = strong conviction in UP
    # A SELL at low price = weak conviction (or exit)
    if up_side == "BUY" and up_price > 0.55:
        return {"direction": "up", "confidence": min(1.0, (up_price - 0.50) * 2)}
    elif up_side == "SELL" and up_price < 0.45:
        return {"direction": "down", "confidence": min(1.0, (0.50 - up_price) * 2)}
    elif up_side == "BUY":
        return {"direction": "up", "confidence": 0.3}
    elif up_side == "SELL":
        return {"direction": "down", "confidence": 0.3}

    return {"direction": "neutral", "confidence": 0.0}


def _compute_depth_signal(depth_clustering: dict) -> dict:
    """Derive directional signal from order book depth/walls.

    A bid wall (large buy order) at a price level = support.
    An ask wall (large sell order) at a price level = resistance.

    If bid wall > ask wall → buyers are defending → bullish.
    If ask wall > bid wall → sellers are defending → bearish.

    Returns:
        {"direction": "up"|"down"|"neutral", "confidence": 0.0-1.0,
         "has_bid_wall": bool, "has_ask_wall": bool}
    """
    bid_wall = depth_clustering.get("bid_wall_pct", 0)
    ask_wall = depth_clustering.get("ask_wall_pct", 0)
    bid_conc = depth_clustering.get("bid_top3_concentration", 0)
    ask_conc = depth_clustering.get("ask_top3_concentration", 0)

    WALL_THRESHOLD = 0.30  # 30% of volume in one level = wall

    has_bid_wall = bid_wall >= WALL_THRESHOLD
    has_ask_wall = ask_wall >= WALL_THRESHOLD

    direction = "neutral"
    confidence = 0.0

    if has_bid_wall and not has_ask_wall:
        # Buyers defending → bullish
        direction = "up"
        confidence = min(1.0, bid_wall * 1.5)
    elif has_ask_wall and not has_bid_wall:
        # Sellers defending → bearish
        direction = "down"
        confidence = min(1.0, ask_wall * 1.5)
    elif has_bid_wall and has_ask_wall:
        # Both walls → use concentration difference
        if bid_conc > ask_conc + 0.1:
            direction = "up"
            confidence = 0.4
        elif ask_conc > bid_conc + 0.1:
            direction = "down"
            confidence = 0.4

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "has_bid_wall": has_bid_wall,
        "has_ask_wall": has_ask_wall,
    }


# ── Utility: Execute Trade ──────────────────────────────────────────


async def execute_trade(
    polymarket_client,
    token_id: str,
    direction: str,
    size_usd: float,
    *,
    dry_run: bool = True,
) -> dict:
    """Execute a trade on Polymarket.

    Places a limit order (maker = 0% fees) on the predicted side.

    Args:
        polymarket_client: AsyncPolymarketClient instance.
        token_id: Token ID for YES or NO outcome.
        direction: 'up' (buy YES) or 'down' (buy NO).
        size_usd: Position size in USDC.
        dry_run: If True, simulate without placing real order.

    Returns:
        Trade result dict.
    """
    side = "BUY"  # Always buying outcome shares

    # Get current best price to place limit order slightly better
    try:
        price_data = await polymarket_client.get_price(token_id, side)
        price = float(price_data.get("price", 0.50))
    except Exception:
        price = 0.50  # Default to 50% probability

    # Calculate shares: size_usd / price
    if price <= 0 or price >= 1:
        price = 0.50
    shares = size_usd / price

    trade_info = {
        "token_id": token_id,
        "direction": direction,
        "side": side,
        "price": round(price, 4),
        "size_usd": size_usd,
        "shares": round(shares, 2),
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info("dry_run_trade", **trade_info)
        trade_info["status"] = "simulated"
        trade_info["order_id"] = "DRY_RUN"
        return trade_info

    try:
        result = await polymarket_client.place_limit_order(
            token_id=token_id,
            side=side,
            price=price,
            size=shares,
        )
        trade_info["order_id"] = result.get("orderID", "unknown")
        trade_info["status"] = result.get("status", "unknown")
        logger.info("trade_executed", **trade_info)
    except Exception as e:
        trade_info["status"] = "failed"
        trade_info["error"] = str(e)
        logger.error("trade_failed", error=str(e), **trade_info)

    return trade_info


# ── Pipeline Steps ──────────────────────────────────────────────────
# Deterministic code steps — only the Strategist uses an LLM.


async def resolve_outcomes_step() -> dict:
    """Resolve pending trade outcomes by checking Polymarket API.

    First pipeline step — runs before gather so the Strategist sees
    accurate win rates in trade history. Checks each pending trade's
    condition_id against the Polymarket API to determine won/lost.

    Returns:
        ``{"outcomes_resolved": {...}}`` with resolution summary.
    """
    from workflows.polymarket_btc.trade_history import get_trade_history

    history = get_trade_history()

    try:
        resolved_count = await history.resolve_pending_outcomes()
    except Exception as e:
        logger.warning("outcome_resolution_failed", error=str(e))
        resolved_count = 0

    # Also resolve paper trading ledger (monitor_trades.json)
    try:
        from workflows.polymarket_btc.monitor.pnl_display import PnLDisplay

        await PnLDisplay().resolve_pending()
    except Exception as e:
        logger.warning("pnl_display_resolve_failed", error=str(e))

    stats = await history.get_stats()

    logger.info(
        "resolve_outcomes_step_complete",
        resolved=resolved_count,
        total_trades=stats.get("total_trades", 0),
        win_rate=stats.get("win_rate_pct", 0),
    )

    return {
        "outcomes_resolved": {
            "resolved_count": resolved_count,
            "total_trades": stats.get("total_trades", 0),
            "pending": stats.get("pending", 0),
            "win_rate_pct": stats.get("win_rate_pct", 0),
        }
    }


async def gather_market_data(ctx: "AgentContext") -> dict:
    """Gather all market data via parallel tool calls + recall past trades.

    Multi-market scan: evaluates all configured durations (5m, 15m, 1h),
    checks each for liquidity via order book spread, and picks the one
    with the tightest spread. If no liquid market found, short-circuits
    with market_viable=false — zero wasted API calls.

    Uses ``ctx: AgentContext`` (auto-injected by FunctionalAgent) for
    ``ctx.recall()`` — semantic search over past trade reasoning.

    Returns:
        ``{"market_analysis": {...}, "past_trade_context": "..."}``
    """
    import asyncio
    import json
    from datetime import datetime, timezone

    from autopilot.connectors.polymarket_connector import AsyncPolymarketClient

    MAX_SPREAD = 0.15  # Maximum acceptable bid-ask spread
    DURATIONS = os.environ.get("BTC_DURATIONS", "5m,15m,1h").split(",")  # configurable

    # ── Phase 1: Multi-market liquidity scan ──────────────────────────
    client = AsyncPolymarketClient()
    try:
        best_market = None
        best_spread = float("inf")
        best_ob = None
        best_down_ob = {}
        best_up_token = ""
        best_down_token = ""
        best_duration = "5m"

        for dur in DURATIONS:
            try:
                markets = await client.find_active_btc_markets(duration=dur)
            except Exception as e:
                logger.warning("market_scan_failed", duration=dur, error=str(e))
                continue

            if not markets:
                continue

            market = markets[0]

            # Extract token IDs
            tokens = market.get("tokens", [])
            if not tokens:
                clob_ids_raw = market.get("clobTokenIds", "")
                if isinstance(clob_ids_raw, str) and clob_ids_raw.startswith("["):
                    token_ids = json.loads(clob_ids_raw)
                    tokens = [{"token_id": tid} for tid in token_ids]

            up_token = tokens[0].get("token_id", "") if tokens else ""
            down_token = tokens[1].get("token_id", "") if len(tokens) > 1 else ""

            if not up_token:
                continue

            # Fetch order books for BOTH tokens (real-time CLOB prices)
            up_ob = await compute_orderbook_signal(client, up_token)
            down_ob = (
                await compute_orderbook_signal(client, down_token) if down_token else {}
            )
            spread = up_ob.get("spread", 999)

            logger.info(
                "market_liquidity_check",
                duration=dur,
                spread=round(spread, 4),
                up_mid=round(up_ob.get("midpoint", 0.5), 4),
                down_mid=round(down_ob.get("midpoint", 0.5), 4),
                question=market.get("question", ""),
            )

            if spread < best_spread:
                best_spread = spread
                best_market = market
                best_ob = up_ob
                best_down_ob = down_ob
                best_up_token = up_token
                best_down_token = down_token
                best_duration = dur

    finally:
        await client.close()

    # ── Short-circuit if no liquid market ──────────────────────────────
    if best_market is None or best_spread > MAX_SPREAD:
        logger.warning(
            "no_liquid_market",
            best_spread=round(best_spread, 4) if best_spread < float("inf") else None,
            max_spread=MAX_SPREAD,
            durations_checked=DURATIONS,
        )
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "past_trade_context": "No liquid market found — skipping.",
            "market_analysis": {
                "market_viable": False,
                "selected_duration": "5m",
                "price": 0.0,
                "momentum": {
                    "direction_1m": "neutral",
                    "magnitude_1m": 0.0,
                    "direction_3m": "neutral",
                    "magnitude_3m": 0.0,
                },
                "ta_indicators": {
                    "rsi": 50.0,
                    "macd_histogram": 0.0,
                    "bb_position": 0.5,
                    "vwap_deviation": 0.0,
                },
                "derivatives": {
                    "funding_rate": 0.0,
                    "funding_bias": "neutral",
                    "oi_trend": "stable",
                    "interpretation": "",
                },
                "volatility": {
                    "regime": "normal",
                    "strategy_recommendation": "",
                    "atr_pct": 0.0,
                },
                "multi_timeframe": {
                    "alignment": "mixed",
                    "interpretation": "",
                },
                "market": {
                    "should_trade": False,
                    "elapsed": 0,
                    "remaining": 0,
                    "up_price": 0.5,
                    "down_price": 0.5,
                },
                "order_book": {
                    "direction": "neutral",
                    "imbalance_ratio": 0.0,
                    "spread": best_spread if best_spread < float("inf") else 0.0,
                },
                "risk_state": {
                    "daily_pnl": 0.0,
                    "allowed": True,
                    "max_trade_size": 20.0,
                },
                "trade_history_win_rate": 0.0,
                "trade_history_total": 0,
                "trade_history_recent_outcomes": [],
                "analysis_completed_utc": now_utc,
            },
        }

    # ── Phase 2: Full data gathering (liquid market found) ────────────
    from workflows.polymarket_btc.tools import (
        get_btc_price,
        get_ta_indicators,
        get_derivatives_sentiment,
        get_volatility_regime,
        get_multi_timeframe,
        get_trade_history,
        get_risk_state,
        get_liquidation_data,
        get_vpin_signal,
    )

    # 9 tools in parallel (order book already fetched)
    (
        price_data,
        ta_data,
        deriv_data,
        vol_data,
        mtf_data,
        history_data,
        risk_data,
        liq_data,
        vpin_data,
    ) = await asyncio.gather(
        get_btc_price(),
        get_ta_indicators(),
        get_derivatives_sentiment(),
        get_volatility_regime(),
        get_multi_timeframe(),
        get_trade_history(),
        get_risk_state(),
        get_liquidation_data(),
        get_vpin_signal(),
    )

    # ── Phase 3: Edge signal collection (parallel) ────────────────
    async def _empty_list():
        return []

    async def _empty_dict():
        return {}

    edge_client = AsyncPolymarketClient()
    try:
        window_start_ts = best_market.get("_window_start", 0)
        edge_results = await asyncio.gather(
            # Signal A: Intra-window price trend
            edge_client.get_prices_history(
                best_up_token,
                start_ts=window_start_ts,
                interval="1m",
                fidelity=1,
            )
            if best_up_token
            else _empty_list(),
            # Signal C: Last trade price + side (UP token)
            edge_client.get_last_trade_price(best_up_token)
            if best_up_token
            else _empty_dict(),
            # Signal C: Last trade price + side (DOWN token)
            edge_client.get_last_trade_price(best_down_token)
            if best_down_token
            else _empty_dict(),
            return_exceptions=True,
        )
    finally:
        await edge_client.close()

    # Parse edge results (graceful degradation on failures)
    raw_price_history = (
        edge_results[0] if not isinstance(edge_results[0], Exception) else []
    )
    up_last_trade = (
        edge_results[1] if not isinstance(edge_results[1], Exception) else {}
    )
    down_last_trade = (
        edge_results[2] if not isinstance(edge_results[2], Exception) else {}
    )

    # Signal A: Compute intra-window trend from price history
    intra_window_trend = _compute_intra_window_trend(raw_price_history)

    # Signal B: Extract volume/liquidity from Gamma market data (FREE — already returned)
    market_volume = {
        "volume": float(best_market.get("volume", 0) or 0),
        "volume_24hr": float(
            best_market.get("volume_24hr", best_market.get("volume24hr", 0)) or 0
        ),
        "liquidity": float(best_market.get("liquidity", 0) or 0),
    }

    # Signal C: Last trade (prefer UP token data, fallback to DOWN)
    last_trade = {}
    if up_last_trade and isinstance(up_last_trade, dict):
        last_trade = {
            "price": float(up_last_trade.get("price", 0.5)),
            "side": up_last_trade.get("side", "unknown"),
            "token": "up",
        }
    if down_last_trade and isinstance(down_last_trade, dict):
        last_trade["down_price"] = float(down_last_trade.get("price", 0.5))
        last_trade["down_side"] = down_last_trade.get("side", "unknown")

    # Signal D: Depth clustering (already computed in order book imbalance)
    depth_clustering = {
        "bid_wall_pct": float(best_ob.get("bid_wall_pct", 0)),
        "bid_wall_price": float(best_ob.get("bid_wall_price", 0)),
        "ask_wall_pct": float(best_ob.get("ask_wall_pct", 0)),
        "ask_wall_price": float(best_ob.get("ask_wall_price", 0)),
        "bid_top3_concentration": float(best_ob.get("bid_top3_concentration", 0)),
        "ask_top3_concentration": float(best_ob.get("ask_top3_concentration", 0)),
    }

    # ── Extract timing info ───────────────────────────────────────
    window_start = best_market.get("_window_start", 0)
    window_interval = best_market.get("_window_interval", 300)
    timing = compute_window_timing(window_start, window_interval)

    outcome_prices = json.loads(best_market.get("outcomePrices", '["0.5", "0.5"]'))
    outcomes = json.loads(best_market.get("outcomes", '["Up", "Down"]'))
    mkt_signal = compute_market_price_signal(outcome_prices, outcomes)

    # ── Extract trade history stats ───────────────────────────────
    overall_stats = history_data.get("overall", {})
    win_rate = overall_stats.get("win_rate_pct", 0)
    total_trades = overall_stats.get("total_trades", 0)
    recent_outcomes = overall_stats.get("recent_outcomes", [])
    by_regime = overall_stats.get("by_regime", {})

    # ── Pack into MarketAnalysis shape ────────────────────────────
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── Recall past trades from long-term memory ──────────────────
    past_trade_context = "No past trade memories available yet."
    try:
        vol_regime = vol_data.get("regime", "normal")
        momentum_dir = price_data.get("direction_3m", "neutral")
        query = (
            f"BTC trade outcome with {vol_regime} volatility "
            f"and {momentum_dir} momentum"
        )
        recall_result = await ctx.recall(query)
        if (
            recall_result
            and hasattr(recall_result, "memories")
            and recall_result.memories
        ):
            entries = []
            for mem in recall_result.memories[:5]:  # Cap at 5 most relevant
                if hasattr(mem, "events"):
                    for ev in mem.events:
                        if hasattr(ev, "content") and ev.content and ev.content.parts:
                            entries.append(ev.content.parts[0].text)
            if entries:
                past_trade_context = "\n---\n".join(entries)
                logger.info(
                    "past_trades_recalled",
                    count=len(entries),
                    query=query,
                )
    except Exception as e:
        logger.debug("memory_recall_skipped", error=str(e))

    logger.info(
        "market_selected",
        duration=best_duration,
        spread=round(best_spread, 4),
        question=best_market.get("question", ""),
    )

    return {
        "past_trade_context": past_trade_context,
        "market_analysis": {
            "market_viable": True,
            "selected_duration": best_duration,
            "price": float(price_data.get("price", 0)),
            "momentum": {
                "direction_1m": price_data.get("direction_1m", "neutral"),
                "magnitude_1m": float(price_data.get("magnitude_1m", 0)),
                "direction_3m": price_data.get("direction_3m", "neutral"),
                "magnitude_3m": float(price_data.get("magnitude_3m", 0)),
            },
            "ta_indicators": {
                "rsi": float(ta_data.get("rsi", 50)),
                "macd_histogram": float(ta_data.get("macd_histogram", 0)),
                "bb_position": float(ta_data.get("bb_position", 0.5)),
                "vwap_deviation": float(ta_data.get("vwap_deviation", 0)),
            },
            "derivatives": {
                "funding_rate": float(deriv_data.get("funding_rate", 0)),
                "funding_bias": deriv_data.get("funding_bias", "neutral"),
                "oi_trend": deriv_data.get("oi_trend", "stable"),
                "interpretation": deriv_data.get("interpretation", ""),
            },
            "volatility": {
                "regime": vol_data.get("regime", "normal"),
                "strategy_recommendation": vol_data.get("strategy_recommendation", ""),
                "atr_pct": float(vol_data.get("atr_pct", 0)),
            },
            "multi_timeframe": {
                "alignment": mtf_data.get("alignment", "mixed"),
                "interpretation": mtf_data.get("interpretation", ""),
            },
            "market": {
                "should_trade": timing.get("should_trade", False),
                "elapsed": timing.get("elapsed", 0),
                "remaining": timing.get("remaining", 0),
                "edge_decay": timing.get("edge_decay", 1.0),
                "timing_zone": timing.get("timing_zone", "alpha"),
                # Real-time CLOB midpoints for each token (independent order books)
                "up_price": float(
                    best_ob.get("midpoint", mkt_signal.get("up_price", 0.5))
                ),
                "down_price": float(
                    best_down_ob.get("midpoint", mkt_signal.get("down_price", 0.5))
                ),
            },
            "order_book": {
                "direction": best_ob.get("direction", "neutral"),
                "imbalance_ratio": float(best_ob.get("imbalance_ratio", 0)),
                "spread": float(best_ob.get("spread", 0)),
                "midpoint": float(best_ob.get("midpoint", 0.5)),
                "best_bid": float(best_ob.get("best_bid", 0.0)),
                "best_ask": float(best_ob.get("best_ask", 1.0)),
                "down_midpoint": float(best_down_ob.get("midpoint", 0.5)),
                "down_best_bid": float(best_down_ob.get("best_bid", 0.0)),
                "down_best_ask": float(best_down_ob.get("best_ask", 1.0)),
            },
            "risk_state": {
                "daily_pnl": float(risk_data.get("daily_pnl", 0)),
                "allowed": risk_data.get("is_allowed", risk_data.get("allowed", True)),
                "max_trade_size": float(
                    risk_data.get(
                        "max_trade_size",
                        risk_data.get("max_trade_size_usd", 20),
                    )
                ),
            },
            "trade_history_win_rate": float(win_rate),
            "trade_history_total": int(total_trades),
            "trade_history_recent_outcomes": recent_outcomes,
            "trade_history_by_regime": by_regime,
            "analysis_completed_utc": now_utc,
            # Token IDs for CLOB execution — explicit Pydantic fields
            "condition_id": best_market.get("conditionId", ""),
            "up_token_id": best_up_token,
            "down_token_id": best_down_token,
            # Edge signals
            "intra_window_trend": intra_window_trend,
            "market_volume": market_volume,
            "last_trade": last_trade,
            "depth_clustering": depth_clustering,
            # Tier 2: BTC-native edge signals
            "liquidation": liq_data if isinstance(liq_data, dict) else {},
            "vpin": vpin_data if isinstance(vpin_data, dict) else {},
            "pre_window_carry": _compute_pre_window_carry(price_data),
            "window_streak": _compute_window_streak(recent_outcomes),
        },
    }


# ── Tier 2: BTC-Native Edge Signal Helpers ──────────────────────────


def _compute_pre_window_carry(price_data: dict) -> dict:
    """Compute momentum carry from price action context.

    Uses momentum direction and magnitude from the BTC price data
    to infer whether pre-window momentum is bullish or bearish.
    Momentum data captures the last 1-3 minutes of price action.
    """
    if not isinstance(price_data, dict):
        return {
            "direction": "neutral",
            "strength": 0.0,
            "consecutive_bullish": 0,
            "consecutive_bearish": 0,
        }

    # Use the 3m momentum (most predictive) as carry signal
    direction_3m = price_data.get("direction_3m", "neutral")
    magnitude_3m = abs(float(price_data.get("magnitude_3m", 0)))
    direction_1m = price_data.get("direction_1m", "neutral")

    # Determine carry direction
    if direction_3m in ("up", "down"):
        carry_dir = direction_3m
    elif direction_1m in ("up", "down"):
        carry_dir = direction_1m
    else:
        carry_dir = "neutral"

    # Strength: normalize magnitude (100 is "monster" move)
    strength = min(1.0, magnitude_3m / 150.0) if magnitude_3m > 0 else 0.0

    # Consecutive candle count estimation from alignment
    bullish = 1 if direction_3m == "up" and direction_1m == "up" else 0
    bearish = 1 if direction_3m == "down" and direction_1m == "down" else 0
    if bullish:
        bullish += 1 if magnitude_3m > 50 else 0
    if bearish:
        bearish += 1 if magnitude_3m > 50 else 0

    return {
        "direction": carry_dir,
        "strength": round(strength, 3),
        "consecutive_bullish": bullish,
        "consecutive_bearish": bearish,
    }


def _compute_window_streak(recent_outcomes: list[str]) -> dict:
    """Compute window streak context from recent trade outcomes.

    Args:
        recent_outcomes: List of 'W' or 'L' from last 5 trades.

    Returns:
        streak_length, streak_direction, pattern classification.
    """
    if not recent_outcomes:
        return {
            "last_outcomes": [],
            "streak_length": 0,
            "streak_direction": "neutral",
            "pattern": "mixed",
        }

    # Compute current streak
    streak = 0
    last = recent_outcomes[-1]
    for outcome in reversed(recent_outcomes):
        if outcome == last:
            streak += 1
        else:
            break

    streak_direction = "winning" if last == "W" else "losing"
    if streak < 2:
        streak_direction = "neutral"

    # Pattern classification
    wins = sum(1 for o in recent_outcomes if o == "W")
    losses = sum(1 for o in recent_outcomes if o == "L")
    total = len(recent_outcomes)

    if total >= 3:
        # Check for alternating pattern (choppy)
        alternations = sum(
            1 for i in range(1, total) if recent_outcomes[i] != recent_outcomes[i - 1]
        )
        if alternations >= total - 1:
            pattern = "choppy"
        elif streak >= 3:
            pattern = "mean_reversion" if total >= 4 else "trend"
        elif wins >= total * 0.7:
            pattern = "trend"
        elif losses >= total * 0.7:
            pattern = "trend"
        else:
            pattern = "mixed"
    else:
        pattern = "mixed"

    return {
        "last_outcomes": recent_outcomes,
        "streak_length": streak if streak_direction != "neutral" else 0,
        "streak_direction": streak_direction,
        "pattern": pattern,
    }


# ── Signal Confluence Engine ────────────────────────────────────────
# Layer 1: Aggregate correlated signals into independent source groups
# Layer 2: Cross-validate Binance vs Polymarket for confluence
# Layer 3: Bayesian probability from market price + confluence edge

_MOMENTUM_MONSTER = 100

_MIN_CONFLUENCE_BY_REGIME = {
    "low": {"full": 0.25, "partial": 0.35, "single": 0.50},
    "normal": {"full": 0.20, "partial": 0.30, "single": 0.45},
    "high": {"full": 0.15, "partial": 0.25, "single": 0.40},
    "extreme": {"full": 0.30, "partial": 0.45, "single": 0.60},
}


def _score_binance_composite(market_analysis: MarketAnalysis) -> BinanceComposite:
    """Layer 1A: Aggregate all Binance TA into a single composite."""
    mom = market_analysis.momentum
    ta = market_analysis.ta_indicators
    deriv = market_analysis.derivatives
    mtf = market_analysis.multi_timeframe

    up_v = 0.0
    dn_v = 0.0
    details = []

    # Momentum 1m (w=1.0 + monster bonus)
    if mom.direction_1m == "up":
        b = 0.5 if mom.magnitude_1m > _MOMENTUM_MONSTER else 0
        up_v += 1.0 + b
        details.append("Mom1m UP" + (" +monster" if b else ""))
    elif mom.direction_1m == "down":
        b = 0.5 if abs(mom.magnitude_1m) > _MOMENTUM_MONSTER else 0
        dn_v += 1.0 + b
        details.append("Mom1m DN" + (" +monster" if b else ""))

    # Momentum 3m (w=1.5 — most predictive TA)
    if mom.direction_3m == "up":
        b = 0.5 if mom.magnitude_3m > _MOMENTUM_MONSTER else 0
        up_v += 1.5 + b
        details.append("Mom3m UP" + (" +monster" if b else ""))
    elif mom.direction_3m == "down":
        b = 0.5 if abs(mom.magnitude_3m) > _MOMENTUM_MONSTER else 0
        dn_v += 1.5 + b
        details.append("Mom3m DN" + (" +monster" if b else ""))

    # RSI, MACD, BB, VWAP — correlated cluster (0.5 each)
    if ta.rsi > 60:
        up_v += 0.5
        details.append("RSI UP")
    elif ta.rsi < 40:
        dn_v += 0.5
        details.append("RSI DN")
    if ta.macd_histogram > 0:
        up_v += 0.5
        details.append("MACD UP")
    elif ta.macd_histogram < 0:
        dn_v += 0.5
        details.append("MACD DN")
    if ta.bb_position > 0.5:
        up_v += 0.5
        details.append("BB UP")
    elif ta.bb_position < 0.5:
        dn_v += 0.5
        details.append("BB DN")
    if ta.vwap_deviation > 0.10:
        up_v += 0.5
        details.append("VWAP UP")
    elif ta.vwap_deviation < -0.10:
        dn_v += 0.5
        details.append("VWAP DN")

    # Derivatives (0.3 — 8h cycle, macro only)
    if deriv.funding_bias in ("bullish", "very_bullish"):
        up_v += 0.3
        details.append("Fund UP")
    elif deriv.funding_bias in ("bearish", "very_bearish"):
        dn_v += 0.3
        details.append("Fund DN")

    # Order Flow Toxicity (VPIN) (0.6 max)
    vpin = market_analysis.vpin
    if vpin and vpin.direction != "neutral":
        if vpin.direction == "bullish":
            up_v += 0.6 * (vpin.vpin_score / 100)
            if vpin.vpin_score > 60:
                details.append(f"VPIN Toxic UP ({vpin.vpin_score:.0f})")
            else:
                details.append(f"VPIN UP ({vpin.vpin_score:.0f})")
        elif vpin.direction == "bearish":
            dn_v += 0.6 * (vpin.vpin_score / 100)
            if vpin.vpin_score > 60:
                details.append(f"VPIN Toxic DN ({vpin.vpin_score:.0f})")
            else:
                details.append(f"VPIN DN ({vpin.vpin_score:.0f})")

    # MTF alignment (0.5)
    if mtf.alignment in ("all_bullish", "higher_tf_up", "bullish_pullback"):
        up_v += 0.5
        details.append("MTF UP")
    elif mtf.alignment in ("all_bearish", "higher_tf_down", "bearish_rally"):
        dn_v += 0.5
        details.append("MTF DN")

    total = up_v + dn_v
    if total == 0:
        return BinanceComposite(
            direction="neutral", strength=0.0, details="All TA neutral"
        )
    if up_v > dn_v:
        return BinanceComposite(
            direction="up", strength=round(up_v / total, 4), details="; ".join(details)
        )
    elif dn_v > up_v:
        return BinanceComposite(
            direction="down",
            strength=round(dn_v / total, 4),
            details="; ".join(details),
        )
    return BinanceComposite(
        direction="neutral", strength=0.0, details="; ".join(details)
    )


def _score_polymarket_flow(market_analysis: MarketAnalysis) -> PolymarketFlow:
    """Layer 1B: Aggregate Polymarket-native flow signals + smart money."""
    up_s = 0.0
    dn_s = 0.0
    details = []

    # Market Price — the ANCHOR (scales with delta magnitude)
    up_price = market_analysis.market.up_price
    mkt_delta = up_price - 0.50
    if mkt_delta > 0.05:
        w = 1.5 + min(mkt_delta, 0.30) * 5.0
        up_s += w
        details.append(f"Mkt UP={up_price:.2f}(w={w:.1f})")
    elif mkt_delta < -0.05:
        w = 1.5 + min(abs(mkt_delta), 0.30) * 5.0
        dn_s += w
        details.append(f"Mkt DN(UP={up_price:.2f},w={w:.1f})")

    # Intra-window trend
    iwt = market_analysis.intra_window_trend
    iwt_dir = iwt.get("direction", "flat")
    iwt_str = iwt.get("strength", 0)
    if iwt_dir == "up" and iwt_str > 0:
        w = 2.0 * iwt_str
        up_s += w
        details.append(f"IWT UP(s={iwt_str:.2f})")
    elif iwt_dir == "down" and iwt_str > 0:
        w = 2.0 * iwt_str
        dn_s += w
        details.append(f"IWT DN(s={iwt_str:.2f})")

    # Last trade
    lt = _compute_last_trade_signal(market_analysis.last_trade)
    lt_dir, lt_c = lt.get("direction", "neutral"), lt.get("confidence", 0)
    if lt_dir == "up" and lt_c > 0:
        up_s += 1.0 * lt_c
        details.append(f"LT BUY(c={lt_c:.2f})")
    elif lt_dir == "down" and lt_c > 0:
        dn_s += 1.0 * lt_c
        details.append(f"LT SELL(c={lt_c:.2f})")

    # Depth walls
    ds = _compute_depth_signal(market_analysis.depth_clustering)
    ds_dir, ds_c = ds.get("direction", "neutral"), ds.get("confidence", 0)
    if ds_dir == "up" and ds_c > 0:
        up_s += 1.0 * ds_c
        details.append(f"Wall UP(c={ds_c:.2f})")
    elif ds_dir == "down" and ds_c > 0:
        dn_s += 1.0 * ds_c
        details.append(f"Wall DN(c={ds_c:.2f})")

    total = up_s + dn_s
    if total == 0:
        return PolymarketFlow(
            direction="neutral",
            strength=0.0,
            is_smart_money=False,
            details="Flow neutral",
        )

    if up_s > dn_s:
        direction = "up"
        strength = up_s / total
    elif dn_s > up_s:
        direction = "down"
        strength = dn_s / total
    else:
        return PolymarketFlow(
            direction="neutral",
            strength=0.0,
            is_smart_money=False,
            details="; ".join(details),
        )

    # Smart money detection
    vq = _compute_volume_quality(market_analysis.market_volume)
    is_smart = (
        vq.get("quality") == "high"
        and (ds.get("has_bid_wall") or ds.get("has_ask_wall"))
        and iwt_dir == direction
        and strength > 0.6
    )
    if is_smart:
        details.append("⚡SMART_MONEY")

    return PolymarketFlow(
        direction=direction,
        strength=round(strength, 4),
        is_smart_money=is_smart,
        volume_quality=vq.get("quality", "unknown"),
        details="; ".join(details),
    )


async def score_trade(market_analysis: MarketAnalysis) -> dict:
    """World-class 3-layer Signal Confluence Engine.

    Layer 1: Aggregate correlated signals into 2 independent groups
    Layer 2: Cross-validate Binance vs Polymarket for confluence
    Layer 3: Bayesian probability from market price + confluence edge
    """
    if not market_analysis.market_viable:
        return _skip_proposal(0, 0, [], "No liquid market found across all durations.")

    # ═══ LAYER 1: Signal Groups ═══════════════════════════════════
    binance = _score_binance_composite(market_analysis)
    polymarket = _score_polymarket_flow(market_analysis)

    notes = [
        f"Binance: {binance.direction}(str={binance.strength:.2f})",
        f"Polymarket: {polymarket.direction}(str={polymarket.strength:.2f})",
    ]

    # ═══ LAYER 2: Confluence ══════════════════════════════════════
    b_dir = binance.direction
    b_str = binance.strength
    p_dir = polymarket.direction
    p_str = polymarket.strength
    is_smart = polymarket.is_smart_money

    if b_dir == p_dir and b_dir != "neutral":
        conf_type = "full"
        conf_strength = 0.40 * b_str + 0.60 * p_str
        if is_smart:
            conf_strength = min(1.0, conf_strength * 1.25)
        direction_raw = b_dir
        label = "FULL CONFLUENCE" + (" + SMART MONEY" if is_smart else "")
    elif b_dir == "neutral" and p_dir != "neutral":
        conf_type = "single_polymarket"
        conf_strength = p_str * 0.70
        direction_raw = p_dir
        label = "Polymarket-only"
    elif p_dir == "neutral" and b_dir != "neutral":
        conf_type = "single_binance"
        conf_strength = b_str * 0.50
        direction_raw = b_dir
        label = "Binance-only (low conviction)"
    elif b_dir != "neutral" and p_dir != "neutral" and b_dir != p_dir:
        conf_type = "conflict"
        conf_strength = 0.0
        direction_raw = "neutral"
        label = f"CONFLICT: Binance={b_dir} vs Polymarket={p_dir}"
    else:
        conf_type = "none"
        conf_strength = 0.0
        direction_raw = "neutral"
        label = "No directional signal"

    conf_strength = round(conf_strength, 4)
    confluence = ConfluenceResult(
        type=conf_type,
        direction=direction_raw,
        strength=conf_strength,
        label=label,
    )
    notes.append(f"Confluence: {confluence.label} (str={confluence.strength:.2f})")

    if confluence.direction == "neutral":
        return _skip_proposal(0, 0, notes, f"No edge: {confluence.label}")

    # ═══ Signal DNA ══════════════════════════════════════════════
    fingerprint, _fp_signals = compute_fingerprint(market_analysis)
    dna_tracker = get_dna_tracker()
    await dna_tracker.load()
    dna_weight = dna_tracker.get_weight(fingerprint)
    dna_win_rate = dna_tracker.get_win_rate(fingerprint)
    dna_trades = dna_tracker.get_total_trades(fingerprint)
    notes.append(
        f"DNA: {fingerprint[:40]}... w={dna_weight:.2f} wr={dna_win_rate or 'n/a'}"
    )

    direction = "BUY_UP" if confluence.direction == "up" else "BUY_DOWN"
    entry_price = (
        market_analysis.market.up_price
        if direction == "BUY_UP"
        else market_analysis.market.down_price
    )

    # Note expensive entries for the scorecard — actual hard block is in risk_gate Rule 3
    if entry_price > 0.70:
        notes.append(
            f"High-entry: {entry_price:.2f} > 0.70 — payout tight, risk_gate Rule 3 will veto"
        )

    # Regime-based minimum confluence
    vol_regime = market_analysis.volatility.regime
    regime_mins = _MIN_CONFLUENCE_BY_REGIME.get(
        vol_regime, _MIN_CONFLUENCE_BY_REGIME["normal"]
    )
    regime_adj = 0.0
    by_regime = market_analysis.trade_history_by_regime
    regime_stats = by_regime.get(vol_regime, {})
    if regime_stats.get("total", 0) >= 10:
        wr = regime_stats.get("win_rate_pct", 50)
        if wr > 60:
            regime_adj = -0.05
            notes.append(f"Regime boost: WR={wr:.0f}%")
        elif wr < 45:
            regime_adj = 0.10
            notes.append(f"Regime penalty: WR={wr:.0f}%")

    if entry_price > 0.55:
        zone = "single"
    elif entry_price > 0.45:
        zone = "partial"
    else:
        zone = "full"
    min_str = regime_mins.get(zone, 0.30) + regime_adj

    if confluence.strength < min_str:
        return _skip_proposal(
            0,
            0,
            notes,
            f"Confluence {confluence.strength:.2f}/{min_str:.2f} too weak "
            f"for entry {entry_price:.2f} in {vol_regime}.",
        )

    # ═══ LAYER 3: Bayesian Posterior + A+ Setup Quality ════════════
    if direction == "BUY_UP":
        market_prob = market_analysis.market.up_price
    else:
        market_prob = market_analysis.market.down_price

    # Confluence edge adjustment (heuristic alpha)
    if confluence.type == "full":
        edge = 0.03 + confluence.strength * 0.04
    elif confluence.type.startswith("single"):
        edge = 0.01 + confluence.strength * 0.02
    else:
        edge = 0.0

    # ── Bayesian posterior: DNA likelihood × market prior ──────────
    # When DNA has enough data, its win rate IS P(WIN|signals) — the
    # empirical posterior, mathematically optimal.
    if dna_win_rate is not None and dna_trades >= 10:
        # DNA-based posterior: blend DNA likelihood with market prior
        # DNA carries more weight as sample size grows (max 60%)
        dna_blend = min(dna_trades / 50, 0.60)
        p = (1 - dna_blend) * (market_prob + edge) + dna_blend * dna_win_rate
        notes.append(
            f"Bayes: DNA posterior p={p:.3f} (dna_wr={dna_win_rate:.3f}, blend={dna_blend:.2f})"
        )
    else:
        # Fallback: heuristic blend with historical performance
        total_trades = market_analysis.trade_history_total
        win_rate_pct = market_analysis.trade_history_win_rate
        if total_trades >= 20:
            hist_p = win_rate_pct / 100.0
            blend_w = min(total_trades / 100, 0.50)
            p = (1 - blend_w) * (market_prob + edge) + blend_w * hist_p
        else:
            p = market_prob + edge
    p = max(0.40, min(0.75, p))

    # ── A+ Setup Quality Score ────────────────────────────────────
    # Score 0-100 measuring overall setup quality.
    # Cold-start achievable: ~55 (confluence + timing + price).
    # As DNA accumulates, quality naturally rises above 70.
    setup_q = 0

    # Signal agreement (all same direction?) → +30 max
    if confluence.type == "full":
        setup_q += 30
    elif confluence.type == "single_polymarket":
        setup_q += 15
    elif confluence.type == "single_binance":
        setup_q += 10

    # DNA fingerprint quality → +20 max
    if dna_win_rate is not None and dna_win_rate > 0.65:
        setup_q += 20
    elif dna_win_rate is not None and dna_win_rate > 0.55:
        setup_q += 10

    # Favorable entry price → +15 max
    if entry_price < 0.48:
        setup_q += 15
    elif entry_price < 0.53:
        setup_q += 10
    elif entry_price < 0.58:
        setup_q += 5

    # Edge timing: early entry → +15 max
    edge_decay = market_analysis.market.edge_decay
    if edge_decay > 0.7:
        setup_q += 15  # Alpha zone
    elif edge_decay > 0.4:
        setup_q += 10  # Confirm zone

    # Liquidation confirmation → +20 max
    liq = market_analysis.liquidation
    if liq.cascade_level in ("major", "minor"):
        if (liq.net_direction == "down" and direction_raw == "down") or (
            liq.net_direction == "up" and direction_raw == "up"
        ):
            setup_q += 20 if liq.cascade_level == "major" else 10

    # Smart money active → +15 max
    if polymarket.is_smart_money:
        setup_q += 15

    notes.append(f"SetupQ: {setup_q}/100")

    if setup_q < 40:
        return _skip_proposal(
            0,
            0,
            notes,
            f"Not A+ setup: quality={setup_q}/100 < 40. Skipping coin-flip.",
        )

    # ═══ LAYER 4: Anti-Fragile Intelligence ═══════════════════════
    # 4A. Anomaly Rule Engine — deterministic pattern guards
    anomaly_flags = []

    # Rule 1: Liquidation trap — funding extreme + rising OI = cascade imminent
    deriv = market_analysis.derivatives
    if deriv.funding_bias in ("very_bearish", "very_bullish"):
        if deriv.oi_trend in ("rising", "rising_fast"):
            anomaly_flags.append("liquidation_trap")
            notes.append("⚠ ANOMALY: liquidation trap (extreme funding + rising OI)")

    # Rule 2: DNA poison — fingerprint has proven losing track record
    if dna_win_rate is not None and dna_win_rate < 0.38 and dna_trades >= 12:
        anomaly_flags.append("dna_poison")
        notes.append(f"⚠ ANOMALY: DNA poison (wr={dna_win_rate:.2f}, n={dna_trades})")

    # Rule 3: Expensive late entry — high price + fading edge = bad risk/reward
    if entry_price > 0.58 and edge_decay < 0.4:
        anomaly_flags.append("expensive_late")
        notes.append(
            f"⚠ ANOMALY: expensive late (entry={entry_price:.2f}, decay={edge_decay:.2f})"
        )

    # Rule 4: OI divergence — price direction vs OI direction conflict
    if deriv.oi_trend in ("falling", "falling_fast") and direction_raw == "up":
        anomaly_flags.append("oi_divergence")
        notes.append("⚠ ANOMALY: OI falling but bullish bias → smart money exiting")
    elif deriv.oi_trend in ("rising_fast",) and direction_raw == "down":
        anomaly_flags.append("oi_divergence")
        notes.append("⚠ ANOMALY: OI rising fast but bearish bias → new longs entering")

    # Rule 5: Carry contradiction with force
    carry = market_analysis.pre_window_carry
    if (
        carry.direction != "neutral"
        and carry.direction != direction_raw
        and carry.strength > 0.7
    ):
        anomaly_flags.append("strong_carry_contra")
        notes.append(
            f"⚠ ANOMALY: strong carry contradicts ({carry.direction} vs {direction_raw})"
        )

    # Rule 6: Losing streak + choppy market = regime change
    streak = market_analysis.window_streak
    if (
        streak.streak_direction == "losing"
        and streak.streak_length >= 3
        and streak.pattern == "choppy"
    ):
        anomaly_flags.append("regime_change")
        notes.append("⚠ ANOMALY: losing streak + choppy = possible regime change")

    # Anomaly veto: 2+ anomalies = SKIP, 1 = reduce size
    if len(anomaly_flags) >= 2:
        return _skip_proposal(
            0,
            0,
            notes,
            f"Anomaly veto: {len(anomaly_flags)} flags ({', '.join(anomaly_flags)}). Too dangerous.",
        )

    anomaly_penalty = 0.5 if len(anomaly_flags) == 1 else 1.0

    # ── Anti-Fragile Dynamic Sizing via Bayesian Kelly ────────────────────
    # Delegates all position sizing math to RiskManager.compute_kelly_size(),
    # which applies Fractional (Quarter) Kelly × dynamic multipliers.
    from workflows.polymarket_btc.risk import get_risk_manager

    ev_per_unit = p * (1 - entry_price) - (1 - p) * entry_price
    edge_decay = market_analysis.market.edge_decay

    # Streak factor: adapts size to recent performance
    if streak.streak_direction == "winning" and streak.streak_length >= 3:
        streak_factor = 1.3  # Hot hand — lean in
    elif streak.streak_direction == "losing" and streak.streak_length >= 3:
        streak_factor = 0.5  # Regime shift — pull back
    elif streak.pattern == "choppy":
        streak_factor = 0.7  # Choppy = uncertain
    else:
        streak_factor = 1.0

    # Quality² amplifies A+ setups, crushes mediocre
    # 60/100 → 0.36, 80/100 → 0.64, 100/100 → 1.0
    quality_sq = (setup_q / 100.0) ** 2
    smart_mult = 1.2 if polymarket.is_smart_money else 1.0

    # Load RiskManager singleton and sync to live capital.
    # Capital is dynamic runtime state — NOT an env var. It changes with every
    # trade, so we read it directly from PnLDisplay (source of truth: monitor_trades.json)
    # and update the singleton's bankroll in-place before computing Kelly size.
    _kelly_cap = float(os.environ.get("KELLY_FRACTION_CAP", 0.25))
    _rm = get_risk_manager(
        bankroll_usdc=100.0,  # Bootstrap default — immediately overwritten below
        kelly_fraction_cap=_kelly_cap,
        max_trade_size_usd=market_analysis.risk_state.max_trade_size,
    )
    try:
        from workflows.polymarket_btc.monitor.pnl_display import PnLDisplay as _PD

        _live_capital = _PD().capital
        _rm.update_bankroll(_live_capital)  # Always reflects actual equity
    except Exception as _e:
        logger.warning("bankroll_sync_failed", error=str(_e))

    # Quick pre-check: compute raw Kelly to detect zero-edge before delegating
    _b = (1.0 - entry_price) / entry_price if entry_price > 0 else 0
    kelly_f = (p * _b - (1.0 - p)) / _b if _b > 0 else 0.0

    if kelly_f <= 0:
        return _skip_proposal(
            0,
            0,
            notes,
            f"No math edge: Kelly={kelly_f:.4f}, p={p:.3f}, entry={entry_price:.2f}",
        )

    # Delegate final USD sizing to RiskManager (Fractional Kelly × multipliers)
    size_usd = _rm.compute_kelly_size(
        p=p,
        entry_price=entry_price,
        quality_sq=quality_sq,
        edge_decay=edge_decay,
        streak_factor=streak_factor,
        smart_money_mult=smart_mult,
        anomaly_penalty=anomaly_penalty,
    )

    if size_usd <= 0:
        return _skip_proposal(
            0,
            0,
            notes,
            f"Kelly sizing returned zero: p={p:.3f}, entry={entry_price:.2f}",
        )

    # Strategy selection
    if vol_regime in ("high", "extreme"):
        strategy = "trend-following"
    elif vol_regime == "low":
        strategy = "mean-reversion"
    else:
        strategy = "standard"

    notes.append(
        f"Kelly: f*={kelly_f:.4f} → capped={min(kelly_f, _kelly_cap):.4f} → "
        f"${_rm.bankroll_usdc:.0f}×{min(kelly_f, _kelly_cap):.3f}={size_usd:.2f} "
        f"(q²={quality_sq:.2f} decay={edge_decay:.2f} streak={streak_factor:.1f} "
        f"smart={smart_mult:.1f} anom={anomaly_penalty:.1f})"
    )

    # Confidence scoring
    confidence = 50
    if confluence.type == "full":
        confidence += 20
    elif confluence.type == "single_polymarket":
        confidence += 12
    elif confluence.type == "single_binance":
        confidence += 5
    confidence += int(confluence.strength * 15)
    if polymarket.is_smart_money:
        confidence += 8
    # Edge-aware confidence: front-load confidence to alpha zone
    edge_decay = market_analysis.market.edge_decay
    if edge_decay > 0.7 and entry_price < 0.55:
        confidence += 10  # Alpha zone + cheap entry = aggressive
    elif edge_decay > 0.4 and entry_price < 0.58:
        confidence += 5  # Confirm zone, acceptable
    elif edge_decay < 0.2:
        confidence -= 15  # Dead zone penalty
    recent = market_analysis.trade_history_recent_outcomes
    if recent:
        net = sum(1 for o in recent if o == "W") - sum(1 for o in recent if o == "L")
        confidence += net * 2

    # DNA weight: boost confidence for proven patterns, dampen losers
    if dna_trades >= 8:
        # Scale confidence by DNA weight (0.5x → 1.5x)
        confidence = int(confidence * dna_weight)

    # ═══ Tier 2: BTC-Native Edge Signals ═════════════════════════
    # Liquidation cascade: strongest short-term signal
    liq = market_analysis.liquidation
    liq_dir = liq.net_direction
    if liq.cascade_level == "major":
        if (liq_dir == "down" and direction_raw == "down") or (
            liq_dir == "up" and direction_raw == "up"
        ):
            confidence += 12  # Liquidation confirms our direction
            notes.append(f"LIQ: {liq.cascade_level} cascade confirms {liq_dir}")
        elif liq_dir != "neutral" and liq_dir != direction_raw:
            confidence -= 15  # Liquidation contradicts — danger
            notes.append(
                f"LIQ: {liq.cascade_level} cascade CONTRADICTS ({liq_dir} vs {direction_raw})"
            )
    elif liq.cascade_level == "minor":
        if (liq_dir == "down" and direction_raw == "down") or (
            liq_dir == "up" and direction_raw == "up"
        ):
            confidence += 5
            notes.append(f"LIQ: minor cascade confirms {liq_dir}")

    # Pre-window carry: momentum continuation
    carry = market_analysis.pre_window_carry
    if carry.direction == direction_raw and carry.strength > 0.3:
        confidence += int(5 * carry.strength)  # Up to +5
        notes.append(f"CARRY: {carry.direction} (str={carry.strength:.2f})")
    elif (
        carry.direction != "neutral"
        and carry.direction != direction_raw
        and carry.strength > 0.5
    ):
        confidence -= 5  # Carry contradicts
        notes.append(f"CARRY: contradicts ({carry.direction} vs {direction_raw})")

    # Window streak: regime awareness
    streak = market_analysis.window_streak
    if streak.pattern == "choppy":
        confidence -= 8  # Choppy market = coin flip territory
        notes.append("STREAK: choppy pattern — reduce confidence")
    elif streak.streak_direction == "losing" and streak.streak_length >= 3:
        confidence -= 10  # 3+ losses in a row = regime might have changed
        notes.append(f"STREAK: {streak.streak_length} losses — pull back")
    elif streak.streak_direction == "winning" and streak.streak_length >= 3:
        confidence += 5  # Hot hand — lean in
        notes.append(f"STREAK: {streak.streak_length} wins — lean in")

    confidence = max(0, min(100, confidence))

    # Backward-compatible scorecard
    if direction == "BUY_UP":
        sig_up = round(confluence.strength * 10, 2)
        sig_dn = round((1 - confluence.strength) * 3, 2)
    else:
        sig_dn = round(confluence.strength * 10, 2)
        sig_up = round((1 - confluence.strength) * 3, 2)

    payout_ratio = round(_b, 4)

    logger.info(
        "score_trade_result",
        direction=direction,
        confluence=confluence.type,
        conf_strength=round(confluence.strength, 3),
        binance=binance.direction,
        polymarket=polymarket.direction,
        smart_money=polymarket.is_smart_money,
        p=round(p, 4),
        entry=entry_price,
        kelly=round(kelly_f, 4),
        ev=round(ev_per_unit, 4),
        size=size_usd,
        confidence=confidence,
        strategy=strategy,
        regime=vol_regime,
    )

    return {
        "trade_proposal": {
            "scorecard": {
                "signals_up": sig_up,
                "signals_down": sig_dn,
                "scorecard_notes": "; ".join(notes),
            },
            "recommended_direction": direction,
            "entry_price": entry_price,
            "payout_ratio": payout_ratio,
            "kelly_fraction": round(kelly_f, 4),
            "position_size_usd": size_usd,
            "confidence": confidence,
            "strategy": strategy,
            "reasoning": (
                f"{label}. {direction} @ {entry_price:.2f}, "
                f"p={p:.3f} (mkt {market_prob:.2f} + {edge:+.3f} edge). "
                f"Kelly={kelly_f:.4f}, EV={ev_per_unit:.4f}, "
                f"${size_usd:.2f}. {strategy} ({vol_regime})."
            ),
            "signal_fingerprint": fingerprint,
            "setup_quality": setup_q,
        }
    }


def _skip_proposal(signals_up, signals_down, notes, reasoning):
    """Build a SKIP trade proposal."""
    return {
        "trade_proposal": {
            "scorecard": {
                "signals_up": round(signals_up, 2),
                "signals_down": round(signals_down, 2),
                "scorecard_notes": "; ".join(notes) if notes else "",
            },
            "recommended_direction": "SKIP",
            "entry_price": 0.0,
            "payout_ratio": 0.0,
            "kelly_fraction": 0.0,
            "position_size_usd": 0.0,
            "confidence": 0,
            "strategy": "skip",
            "reasoning": reasoning,
        }
    }


async def risk_gate(
    market_analysis: MarketAnalysis,
    trade_proposal: TradeProposal,
    dry_run: bool = True,
) -> dict:
    """Evaluate hard risk rules against a trade proposal.

    Pure if/else logic. Uses Pydantic type hints for auto-hydration
    from pipeline state.

    Args:
        market_analysis: From gather_market_data() step (auto-hydrated).
        trade_proposal: From score_trade() step (auto-hydrated).

    Returns:
        ``{"trade_decision": {...}}`` — dict matching TradeDecision schema.
    """
    from datetime import datetime, timezone

    # ── Short-circuit if no liquid market ──────────────────────────
    if not market_analysis.market_viable:
        return {
            "trade_decision": {
                "action": "SKIP",
                "confidence": 0,
                "reasoning": "No liquid market found across all durations",
                "size_usd": 0.0,
                "market_delta": 0.0,
            }
        }

    violations = []

    # Extract data via Pydantic model accessors
    risk = market_analysis.risk_state
    vol = market_analysis.volatility
    market = market_analysis.market
    mtf = market_analysis.multi_timeframe
    deriv = market_analysis.derivatives

    entry_price = trade_proposal.entry_price
    kelly = trade_proposal.kelly_fraction
    confidence = trade_proposal.confidence
    size_usd = trade_proposal.position_size_usd
    direction = trade_proposal.recommended_direction

    # ── 9 Hard Rules ──────────────────────────────────────────────
    # Rule 1: Risk manager
    if not risk.allowed:
        violations.append("Rule 1: Risk manager says trading not allowed")

    # Rule 2: Extreme volatility — WARNING only, not a hard veto.
    # The scorer already applies elevated thresholds for extreme regime,
    # so a duplicate hard block here created an un-bypassable double-veto.
    if vol.regime == "extreme":
        logger.warning("risk_extreme_vol_warning", regime=vol.regime)

    # Rule 3: Entry price too high
    # At entry > 0.70, payout ratio < 0.43:1. Even with full confluence
    # you'd need a sustained 70%+ win rate — very unlikely in efficient
    # 5-minute BTC markets. Trades at 0.50-0.70 have acceptable payout.
    if entry_price > 0.70:
        violations.append(f"Rule 3: Entry price {entry_price} > 0.70 (payout too poor)")

    # Rule 4: Edge Check (Kelly)
    # Binary options pricing implies a lot of noise. A strict Kelly > 0
    # often filters out highly probable momentum trades just because the
    # market maker spread eats the theoretical edge. We tolerate slight
    # theoretical negative Kelly (down to -0.05) to capture momentum sweeps.
    if kelly < -0.05:
        violations.append(f"Rule 4: Kelly fraction {kelly} < -0.05 (too negative)")

    # Rule 5: Insufficient confidence (minimum conviction threshold)
    # In binary options, taking trades with marginal conviction (<75)
    # leads to long-term EV bleed due to fees/spreads. We only want
    # A-grade setups.
    if confidence < 75:
        violations.append(
            f"Rule 5: Confidence {confidence} < 75 (need higher conviction)"
        )

    # Rule 6: Mixed alignment + neutral derivatives
    # Only applies for 15m+ markets. For 5m markets, derivatives
    # funding rate updates every 8 hours and is meaningless, and
    # MTF is "mixed" ~60% of the time due to 1m noise.  Blocking
    # on this combination in 5m windows filtered out nearly all
    # tradeable setups.  For longer windows (15m/1h), it still
    # acts as a safety net.
    selected_dur = market_analysis.selected_duration
    if selected_dur not in ("5m",):
        if mtf.alignment == "mixed" and deriv.funding_bias == "neutral":
            signal_count = (
                trade_proposal.scorecard.signals_up
                if direction == "BUY_UP"
                else trade_proposal.scorecard.signals_down
            )
            if confidence < 70 or signal_count < 5:
                violations.append(
                    f"Rule 6: Multi-TF mixed + derivatives neutral "
                    f"(confidence={confidence}, signals={signal_count})"
                )

    # Rule 7: Timing
    if not market.should_trade:
        violations.append(
            f"Rule 7: should_trade=false "
            f"(elapsed={market.elapsed}s, "
            f"remaining={market.remaining}s)"
        )

    # Rule 8: Stale data
    analysis_ts = market_analysis.analysis_completed_utc
    if analysis_ts:
        try:
            ts = datetime.fromisoformat(analysis_ts.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > 90:
                violations.append(f"Rule 8: Data is {age:.0f}s old (> 90s)")
        except (ValueError, TypeError):
            pass  # Can't parse — skip this check

    # Rule 9: Order book spread too wide (illiquid market)
    # A wide spread means limit orders won't fill at modeled prices,
    # and market orders will suffer severe slippage.
    ob_spread = market_analysis.order_book.spread
    if ob_spread > 0.10:
        violations.append(
            f"Rule 9: Order book spread {ob_spread:.2f} > 0.10 (illiquid)"
        )

    # Rule 10: Per-window exposure cap — multiple trades per window
    # are valid (each binary token purchase is independent), but total
    # exposure to a single condition_id must be bounded.
    _MAX_EXPOSURE_PER_WINDOW = float(os.environ.get("MAX_EXPOSURE_PER_WINDOW", "50.0"))
    condition_id = market_analysis.condition_id
    if condition_id:
        try:
            from workflows.polymarket_btc.trade_history import (
                get_trade_history as _get_th,
            )

            _th = _get_th()
            _stats = await _th._load()
            window_exposure = sum(
                p["size"]
                for p in _stats.get("pending", [])
                if p.get("cid") == condition_id
            )
            if window_exposure + size_usd > _MAX_EXPOSURE_PER_WINDOW:
                violations.append(
                    f"Rule 10: Window exposure ${window_exposure:.2f} + "
                    f"${size_usd:.2f} exceeds "
                    f"${_MAX_EXPOSURE_PER_WINDOW:.2f} cap"
                )
        except Exception as e:
            logger.warning("rule10_check_failed", error=str(e))

    # Rule 11: Market Conflict Guard — PREVIOUSLY a hard veto, now an informational log.
    # In 5-minute binary markets, taking contrarian trades (mean reversion) when TA
    # strongly disagrees with the current Polymarket price can offer massive EV+
    # (e.g., buying at 0.20 when TA models 0.40 probability).
    up_price = market_analysis.market.up_price
    mkt_delta_rg = up_price - 0.50
    if mkt_delta_rg > 0.05:
        market_dir = "up"
    elif mkt_delta_rg < -0.05:
        market_dir = "down"
    else:
        market_dir = "neutral"  # Near 0.50 — no opinion

    ta_dir = "up" if direction == "BUY_UP" else "down"
    if market_dir != "neutral" and market_dir != ta_dir:
        logger.info(
            "contrarian_trade_setup",
            ta_dir=ta_dir,
            market_dir=market_dir,
            up_price=up_price,
            note="TA fights the market — entering potentially high-EV contrarian trade.",
        )

    # Rule 12: Intra-window cooldown guard
    # Allows scale-in (pilot → confirm → add) but enforces 45s between
    # trades to prevent rapid-fire at 1s cadence. Rule 10 (exposure cap)
    # is the hard ceiling on total window exposure.
    import time as _time

    try:
        from workflows.polymarket_btc.risk import get_risk_manager as _get_rm

        _rm = _get_rm()
        await _rm.load()
        since_last = _time.time() - _rm.state.last_trade_time
        if _rm.state.last_trade_time > 0 and since_last < _rm.INTRA_WINDOW_COOLDOWN:
            remaining = int(_rm.INTRA_WINDOW_COOLDOWN - since_last)
            violations.append(
                f"Rule 12: Intra-window cooldown — {remaining}s remaining "
                f"(min {_rm.INTRA_WINDOW_COOLDOWN}s between trades)"
            )
    except Exception as e:
        logger.warning("rule12_check_failed", error=str(e))

    # Rule 13: Window gate — direction lock + per-window trade cap (persisted)
    try:
        from workflows.polymarket_btc.risk import get_risk_manager as _get_rm_wg

        _rm_wg = _get_rm_wg()
        await _rm_wg.load()
        _gate_ok, _gate_reason = _rm_wg.check_window_gate(
            direction, market_analysis.condition_id
        )
        if not _gate_ok:
            violations.append(f"Rule 13 (Window Gate): {_gate_reason}")
    except Exception as e:
        logger.warning("rule13_window_gate_failed", error=str(e))

    # ── Build decision ────────────────────────────────────────────
    market_delta = round(market.up_price - market.down_price, 4)

    if violations:
        return {
            "trade_decision": {
                "action": "SKIP",
                "confidence": 0,
                "reasoning": f"VETO: {'; '.join(violations)}",
                "size_usd": 0.0,
                "market_delta": market_delta,
            }
        }

    # ── APPROVED — cap size if needed ─────────────────────────────
    max_size = risk.max_trade_size
    final_size = min(size_usd, max_size)

    # Record window trade to lock direction and increment cap counter
    try:
        from workflows.polymarket_btc.risk import get_risk_manager as _get_rm_rec

        _rm_rec = _get_rm_rec()
        await _rm_rec.load()
        _rm_rec.record_window_trade(direction, market_analysis.condition_id)
        await _rm_rec.save()
    except Exception as e:
        logger.warning("risk_window_record_failed", error=str(e))

    return {
        "trade_decision": {
            "action": direction,
            "confidence": confidence,
            "reasoning": (
                f"All 13 risk rules passed. "
                f"Direction: {direction}, "
                f"Entry: {entry_price}, "
                f"Kelly: {kelly:.4f}, "
                f"Size: ${final_size:.2f} "
                f"(capped from ${size_usd:.2f} if needed)"
            ),
            "size_usd": final_size,
            "market_delta": market_delta,
            "entry_price": entry_price,
            "dry_run": dry_run,
            "signal_fingerprint": trade_proposal.signal_fingerprint,
        }
    }


# ── Pipeline Step 4: Execute Trade ──────────────────────────────────


# Price drift threshold: max absolute drift in outcome price
# 0.05 = 5 cents, which on a ~0.50 entry represents ~10% edge erosion
_PRICE_DRIFT_THRESHOLD = 0.05


async def execute_trade_step(
    ctx: "AgentContext",
    market_analysis: MarketAnalysis,
    trade_decision: dict,
) -> dict:
    """Execute the approved trade on Polymarket CLOB.

    Deterministic pipeline step — runs after risk_gate. If the trade
    was vetoed (action=SKIP), publishes skip event and returns a no-op.
    Otherwise, places the order with a price drift guard, then handles
    all post-trade lifecycle:

    1. Publish trade event to EventBus via ``ctx.publish()``
    2. Record trade to ArtifactService-backed ledger
    3. Persist risk state

    Uses ``ctx: AgentContext`` for EventBus access (auto-injected by
    the platform's FunctionalAgent).

    Args:
        ctx: Platform context — auto-injected, provides EventBus access.
        market_analysis: Market data from gather step (auto-hydrated).
        trade_decision: Risk gate output with action, size, confidence.

    Returns:
        ``{"execution_result": {...}}`` with order status and details.
    """
    action = trade_decision.get("action", "SKIP")

    # ── SKIP → publish event + return no-op ──────────────────────
    if action in ("SKIP", "skip", "no_trade"):
        logger.info("execute_trade_step_skip", reason="trade_decision is SKIP")

        result = {
            "status": "skipped",
            "action": action,
            "reason": trade_decision.get("reasoning", ""),
        }

        # Publish skip event via ctx
        await ctx.publish(
            "trade.skipped",
            payload={
                "workflow": "polymarket_btc",
                "status": "skipped",
                "action": action,
                "confidence": trade_decision.get("confidence", 0),
                "size_usd": 0,
            },
        )

        return {"execution_result": result}

    direction = "up" if action == "BUY_UP" else "down"
    size_usd = trade_decision.get("size_usd", 0)
    entry_price = trade_decision.get("entry_price", 0)

    if size_usd <= 0:
        return {
            "execution_result": {
                "status": "skipped",
                "action": action,
                "reason": "size_usd is 0",
            }
        }

    # ── Resolve token ID from market_analysis (no re-lookup!) ─────
    # Token IDs were already resolved during gather_market_data and
    # stored in MarketAnalysis. Avoids redundant API call + race
    # condition of looking up a different market than we analyzed.
    up_token = market_analysis.up_token_id
    down_token = market_analysis.down_token_id
    token_id = up_token if direction == "up" else down_token

    if not token_id:
        return {
            "execution_result": {
                "status": "failed",
                "error": "Could not resolve token ID from market_analysis",
            }
        }

    from autopilot.connectors.polymarket_connector import AsyncPolymarketClient

    client = AsyncPolymarketClient()
    try:
        # ── PRICE DRIFT GUARD ────────────────────────────────────
        side = "BUY"
        try:
            price_data = await client.get_price(token_id, side)
            clob_price = float(price_data.get("price", 0.50))
        except Exception:
            clob_price = 0.50

        if entry_price > 0:
            price_drift = abs(clob_price - entry_price)
            logger.info(
                "price_drift_check",
                direction=direction,
                analyzed_price=entry_price,
                clob_price=clob_price,
                drift=round(price_drift, 4),
                threshold=_PRICE_DRIFT_THRESHOLD,
                passed=price_drift <= _PRICE_DRIFT_THRESHOLD,
            )

            if price_drift > _PRICE_DRIFT_THRESHOLD:
                result = {
                    "status": "aborted",
                    "error": "price_drift_exceeded",
                    "analyzed_entry_price": entry_price,
                    "clob_price": clob_price,
                    "drift": round(price_drift, 4),
                    "threshold": _PRICE_DRIFT_THRESHOLD,
                    "reason": (
                        f"CLOB price drifted {price_drift:.3f} "
                        f"(analyzed {entry_price} vs CLOB {clob_price}) "
                        f"since analysis — exceeds {_PRICE_DRIFT_THRESHOLD} threshold."
                    ),
                }
                await ctx.publish(
                    "trade.failed",
                    payload={
                        "workflow": "polymarket_btc",
                        "status": "aborted",
                        "action": action,
                        "reason": "price_drift_exceeded",
                    },
                )
                return {"execution_result": result}

        # ── Execute ──────────────────────────────────────────────
        is_dry_run = trade_decision.get("dry_run", True)
        result = await execute_trade(
            client,
            token_id=token_id,
            direction=direction,
            size_usd=size_usd,
            dry_run=is_dry_run,
        )

        # Add condition_id for outcome resolution
        result["condition_id"] = market_analysis.condition_id

        logger.info(
            "execute_trade_step_complete",
            action=action,
            status=result.get("status", "unknown"),
            size_usd=size_usd,
            selected_duration=market_analysis.selected_duration,
        )

        # ── Post-trade lifecycle (via ctx) ───────────────────────
        status = result.get("status", "unknown")
        topic = (
            "trade.completed" if status in ("simulated", "filled") else "trade.failed"
        )

        # Remember this trade for future learning
        try:
            memory_text = (
                f"TRADE {action} | "
                f"Status: {status} | "
                f"Duration: {market_analysis.selected_duration} | "
                f"Market: {result.get('condition_id', 'unknown')} | "
                f"Entry: {result.get('price', 0):.4f} | "
                f"Size: ${result.get('size_usd', 0):.2f} | "
                f"Reasoning: {trade_decision.get('reasoning', 'N/A')}"
            )
            await ctx.remember(memory_text)
            logger.info("trade_remembered", action=action, status=status)
        except Exception as e:
            logger.debug("memory_remember_skipped", error=str(e))

        await ctx.publish(
            topic,
            payload={
                "workflow": "polymarket_btc",
                "status": status,
                "action": action,
                "condition_id": result.get("condition_id", ""),
                "confidence": trade_decision.get("confidence", 0),
                "size_usd": result.get("size_usd", 0),
                "order_id": result.get("order_id", ""),
                "selected_duration": market_analysis.selected_duration,
            },
        )

        # Record trade to ArtifactService-backed ledger
        try:
            from workflows.polymarket_btc.trade_history import get_trade_history

            history = get_trade_history()
            await history.record_trade(
                action=action,
                entry_price=result.get("price", 0.50),
                size_usd=result.get("size_usd", 0),
                condition_id=result.get("condition_id", ""),
                vol_regime=market_analysis.volatility.regime,
                alignment=market_analysis.multi_timeframe.alignment,
                signal_fingerprint=trade_decision.get("signal_fingerprint", ""),
            )
        except Exception as e:
            logger.warning("trade_record_failed", error=str(e))

        # Persist risk state (with window tracking)
        try:
            from workflows.polymarket_btc.risk import get_risk_manager
            import time as _time

            risk = get_risk_manager(
                bankroll_usdc=float(os.environ.get("BANKROLL_USDC", 200.0)),
                kelly_fraction_cap=float(os.environ.get("KELLY_FRACTION_CAP", 0.25)),
                max_trade_size_usd=float(os.environ.get("MAX_TRADE_SIZE_USD", 20.0)),
            )
            risk.record_trade(pnl=0.0)  # P&L resolved later via Polymarket API
            # Record window to prevent double-trading
            window_start = int(_time.time()) - market_analysis.market.elapsed
            risk.state.last_traded_window_start = window_start
            await risk.save()
        except Exception as e:
            logger.warning("risk_state_persist_failed", error=str(e))

        # Paper trading ledger (dev/paper — writes to monitor_trades.json)
        try:
            from workflows.polymarket_btc.monitor.pnl_display import PnLDisplay

            _display = PnLDisplay()
            _display.record_trade(
                action=action,
                entry_price=result.get("price", 0.50),
                size_usd=size_usd,
                confidence=trade_decision.get("confidence", 0),
                btc_price=market_analysis.price,
                condition_id=result.get("condition_id", ""),
                window_remaining=market_analysis.market.remaining,
                spread=market_analysis.order_book.spread,
                window_start=int(_time.time()) - market_analysis.market.elapsed,
                duration=market_analysis.selected_duration,
            )
        except Exception as e:
            logger.warning("pnl_display_record_failed", error=str(e))

        return {"execution_result": result}

    finally:
        await client.close()


# ── Pipeline Step 5: Performance Dashboard ──────────────────────────


async def log_performance(
    market_analysis: MarketAnalysis,
    trade_decision: dict,
) -> dict:
    """Emit structured performance metrics for monitoring.

    Non-blocking — does not affect trade execution. Logs a complete
    snapshot of the pipeline run for dashboards and alerting.

    Args:
        market_analysis: Market data from gather step.
        trade_decision: Risk gate output.

    Returns:
        ``{"performance_log": {...}}`` with run metrics.
    """
    from workflows.polymarket_btc.trade_history import get_trade_history

    try:
        history = get_trade_history()
        stats = await history.get_stats()
    except Exception:
        stats = {}

    action = trade_decision.get("action", "SKIP")
    was_trade = action not in ("SKIP", "skip", "no_trade")

    metrics = {
        "run_action": action,
        "was_trade": was_trade,
        "selected_duration": market_analysis.selected_duration,
        "market_viable": market_analysis.market_viable,
        "vol_regime": market_analysis.volatility.regime,
        "entry_price": trade_decision.get("entry_price", 0),
        "confidence": trade_decision.get("confidence", 0),
        "size_usd": trade_decision.get("size_usd", 0),
        # Cumulative stats
        "total_trades": stats.get("total_trades", 0),
        "win_rate_pct": stats.get("win_rate_pct", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "pending_trades": stats.get("pending", 0),
        # Risk state
        "daily_pnl": market_analysis.risk_state.daily_pnl,
        "risk_allowed": market_analysis.risk_state.allowed,
    }

    logger.info("performance_dashboard", **metrics)
    return {"performance_log": metrics}
