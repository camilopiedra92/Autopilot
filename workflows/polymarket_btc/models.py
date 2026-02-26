"""
Pydantic models for the Polymarket BTC trading workflow.

Defines structured output schemas used by each pipeline step:

    gather_market_data()  → MarketAnalysis   (code step)
    score_trade()         → TradeProposal    (code step)
    risk_gate()           → TradeDecision    (code step)

All inter-step communication is type-safe via FunctionalAgent
auto-hydration.
"""

from pydantic import BaseModel, Field


# ── Agent 1: Market Analyst Output ──────────────────────────────────


class Momentum(BaseModel):
    """BTC price momentum over 1m and 3m windows."""

    direction_1m: str = Field(
        description="Price direction over last 1 minute: up, down, or neutral"
    )
    magnitude_1m: float = Field(description="Price change in USD over last 1 minute")
    direction_3m: str = Field(
        description="Price direction over last 3 minutes: up, down, or neutral"
    )
    magnitude_3m: float = Field(description="Price change in USD over last 3 minutes")


class TAIndicators(BaseModel):
    """Technical analysis indicators from Binance 1m candles."""

    rsi: float = Field(
        description="Relative Strength Index (0-100). <30=oversold, >70=overbought"
    )
    macd_histogram: float = Field(
        description="MACD histogram value. Positive=bullish, Negative=bearish"
    )
    bb_position: float = Field(
        description="Bollinger Band position (0-1). <0.2=near lower, >0.8=near upper"
    )
    vwap_deviation: float = Field(
        description="VWAP deviation as percentage. Positive=above VWAP (bullish)"
    )


class Derivatives(BaseModel):
    """Perpetual futures sentiment from Binance Futures."""

    funding_rate: float = Field(description="Current funding rate as percentage")
    funding_bias: str = Field(
        description="Classified bias: very_bullish, bullish, neutral, bearish, very_bearish"
    )
    oi_trend: str = Field(
        description="Open interest trend: rising_fast, rising, stable, falling, falling_fast"
    )
    interpretation: str = Field(
        description="Human-readable summary of derivatives sentiment"
    )


class VolatilityRegime(BaseModel):
    """ATR-based volatility classification."""

    regime: str = Field(description="Volatility regime: low, normal, high, or extreme")
    strategy_recommendation: str = Field(
        description="Recommended strategy for this regime"
    )
    atr_pct: float = Field(description="ATR as percentage of price")


class MultiTimeframe(BaseModel):
    """Multi-timeframe momentum alignment (1m/5m/15m)."""

    alignment: str = Field(
        description="Alignment classification: all_bullish, all_bearish, bullish_pullback, bearish_rally, mixed"
    )
    interpretation: str = Field(description="Human-readable alignment interpretation")


class MarketTiming(BaseModel):
    """Window timing and outcome prices from Polymarket."""

    should_trade: bool = Field(description="Whether the timing window allows trading")
    elapsed: int = Field(description="Seconds elapsed in current 5-min window")
    remaining: int = Field(description="Seconds remaining in current 5-min window")
    up_price: float = Field(description="Current outcome price for Up (0-1)")
    down_price: float = Field(description="Current outcome price for Down (0-1)")
    edge_decay: float = Field(
        default=1.0,
        description="Informational edge decay multiplier (1.0=fresh, 0.0=dead). "
        "Front-loads position sizing to early window.",
    )
    timing_zone: str = Field(
        default="alpha",
        description="Timing zone: alpha (max edge), confirm (fading), late (dead)",
    )


class OrderBook(BaseModel):
    """Polymarket CLOB order book analysis."""

    direction: str = Field(
        description="Implied direction from order book: up, down, or neutral"
    )
    imbalance_ratio: float = Field(
        description="Bid/ask imbalance ratio. Positive=bid-heavy (bullish)"
    )
    spread: float = Field(description="Bid-ask spread")


class RiskState(BaseModel):
    """Current risk manager state."""

    daily_pnl: float = Field(description="Daily P&L in USD")
    allowed: bool = Field(
        description="Whether trading is currently allowed by risk rules"
    )
    max_trade_size: float = Field(description="Maximum allowed trade size in USD")


class LiquidationSignal(BaseModel):
    """Binance Futures liquidation cascade detection."""

    long_liq_usd: float = Field(
        default=0.0, description="Long liquidation volume (USDT) in last 3m"
    )
    short_liq_usd: float = Field(
        default=0.0, description="Short liquidation volume (USDT) in last 3m"
    )
    net_direction: str = Field(
        default="neutral",
        description="up (short squeeze), down (long cascade), neutral",
    )
    cascade_level: str = Field(default="none", description="none, minor, major")
    count: int = Field(default=0, description="Number of liquidation orders in window")


class VPINSignal(BaseModel):
    """Volume-Synchronized Probability of Informed Trading (VPIN) based on 1m candles."""

    vpin_score: float = Field(
        description="VPIN score (0-100). Higher = more toxic order flow."
    )
    buy_volume: float = Field(description="Total buy volume in the window")
    sell_volume: float = Field(description="Total sell volume in the window")
    direction: str = Field(
        description="Direction of the toxicity: bullish, bearish, or neutral"
    )
    interpretation: str = Field(description="Human-readable summary of VPIN signal")


class PreWindowCarry(BaseModel):
    """Pre-window momentum carry — last 3 candles before window opened."""

    direction: str = Field(
        default="neutral", description="up, down, or neutral carry from pre-window"
    )
    strength: float = Field(default=0.0, description="0-1 strength of the carry signal")
    consecutive_bullish: int = Field(
        default=0, description="Number of consecutive bullish candles pre-window"
    )
    consecutive_bearish: int = Field(
        default=0, description="Number of consecutive bearish candles pre-window"
    )


class WindowStreak(BaseModel):
    """Multi-window outcome streak context."""

    last_outcomes: list[str] = Field(
        default_factory=list, description="Last 5 outcomes: ['W', 'L', ...]"
    )
    streak_length: int = Field(
        default=0, description="Current streak length (positive=wins, negative=losses)"
    )
    streak_direction: str = Field(
        default="neutral", description="winning, losing, or neutral"
    )
    pattern: str = Field(
        default="mixed", description="trend, mean_reversion, choppy, or mixed"
    )


class MarketAnalysis(BaseModel):
    """Structured output from the gather_market_data step.

    Contains ALL market data gathered from 9 tools, organized into
    typed sections. Downstream steps (score_trade, risk_gate) read
    this from pipeline state via Pydantic auto-hydration.
    """

    market_viable: bool = Field(
        default=True,
        description="Whether a liquid market was found. If false, downstream steps skip.",
    )
    selected_duration: str = Field(
        default="5m",
        description="Selected market duration (5m, 15m, or 1h)",
    )
    price: float = Field(description="Current BTC/USDT price")
    momentum: Momentum = Field(description="1m/3m price momentum")
    ta_indicators: TAIndicators = Field(description="Technical analysis indicators")
    derivatives: Derivatives = Field(description="Perpetual futures sentiment")
    volatility: VolatilityRegime = Field(
        description="ATR-based volatility regime classification"
    )
    multi_timeframe: MultiTimeframe = Field(
        description="Multi-timeframe momentum alignment"
    )
    market: MarketTiming = Field(
        description="Polymarket window timing and outcome prices"
    )
    order_book: OrderBook = Field(description="Order book imbalance analysis")
    risk_state: RiskState = Field(description="Current risk manager state")
    # Tier 2: BTC-native edge signals
    liquidation: LiquidationSignal = Field(
        default_factory=LiquidationSignal,
        description="Liquidation cascade detection from Binance Futures",
    )
    vpin: VPINSignal | None = Field(
        default=None,
        description="Order flow toxicity (VPIN) based on recent trade volumes",
    )
    pre_window_carry: PreWindowCarry = Field(
        default_factory=PreWindowCarry,
        description="Momentum carry from pre-window price action",
    )
    window_streak: WindowStreak = Field(
        default_factory=WindowStreak,
        description="Multi-window outcome streak context",
    )
    trade_history_win_rate: float = Field(
        description="Overall win rate from trade history (0-100). Use 0 if no trades yet."
    )
    trade_history_total: int = Field(description="Total number of historical trades")
    trade_history_recent_outcomes: list[str] = Field(
        default_factory=list,
        description="Last 5 trade outcomes ('W' or 'L') for streak tracking",
    )
    trade_history_by_regime: dict = Field(
        default_factory=dict,
        description="Per-regime win rate stats: {'low': {'wins': 3, 'total': 5, 'win_rate_pct': 60}, ...}",
    )
    analysis_completed_utc: str = Field(
        description="ISO 8601 UTC timestamp of when this analysis was completed"
    )
    # Token IDs for CLOB execution — set by gather_market_data
    condition_id: str = Field(default="", description="Polymarket condition ID")
    up_token_id: str = Field(default="", description="UP outcome token ID for CLOB")
    down_token_id: str = Field(default="", description="DOWN outcome token ID for CLOB")

    # ── Edge signals (populated by gather_market_data) ──────────
    intra_window_trend: dict = Field(
        default_factory=dict,
        description="Intra-window price trend from /prices-history: {direction, strength, points}",
    )
    market_volume: dict = Field(
        default_factory=dict,
        description="Market volume/liquidity from Gamma: {volume, liquidity, volume_24hr}",
    )
    last_trade: dict = Field(
        default_factory=dict,
        description="Last trade on this token: {price, side, token_id}",
    )
    depth_clustering: dict = Field(
        default_factory=dict,
        description="Order book wall/depth: {bid_wall_pct, ask_wall_pct, bid_top3, ask_top3}",
    )


# ── Signal Confluence Engine Models ─────────────────────────────────


class BinanceComposite(BaseModel):
    """Layer 1A: Aggregated Binance TA composite direction."""

    direction: str = Field(
        description="Composite direction from all Binance TA: up, down, or neutral"
    )
    strength: float = Field(
        ge=0, le=1, description="Proportion of signals agreeing with winning side (0-1)"
    )
    details: str = Field(default="", description="Semicolon-separated signal details")


class PolymarketFlow(BaseModel):
    """Layer 1B: Aggregated Polymarket-native flow signal."""

    direction: str = Field(
        description="Flow direction from Polymarket signals: up, down, or neutral"
    )
    strength: float = Field(
        ge=0, le=1, description="Proportion of flow signals agreeing (0-1)"
    )
    is_smart_money: bool = Field(
        default=False,
        description="True when volume high + depth wall + trend aligned = institutional positioning",
    )
    volume_quality: str = Field(
        default="unknown", description="Volume quality: high, medium, low, unknown"
    )
    details: str = Field(default="", description="Semicolon-separated signal details")


class ConfluenceResult(BaseModel):
    """Layer 2: Cross-validation result between Binance and Polymarket."""

    type: str = Field(
        description="Confluence type: full, single_polymarket, single_binance, conflict, none"
    )
    direction: str = Field(description="Agreed direction: up, down, or neutral")
    strength: float = Field(
        ge=0, le=1, description="Combined confluence strength (0-1)"
    )
    label: str = Field(description="Human-readable confluence description for logging")


# ── Agent 2: Strategist Output ──────────────────────────────────────


class SignalScorecard(BaseModel):
    """Dual-direction signal evaluation."""

    signals_up: float = Field(
        ge=0, description="Weighted signal score supporting UP direction"
    )
    signals_down: float = Field(
        ge=0, description="Weighted signal score supporting DOWN direction"
    )
    scorecard_notes: str = Field(
        description="Brief notes on which signals support each direction"
    )


class TradeProposal(BaseModel):
    """Structured output from the Strategist agent.

    Contains the complete trade analysis: scorecard, entry/payout,
    Kelly sizing, and final recommendation. Read by the Gatekeeper
    from session state via ``{trade_proposal}``.
    """

    scorecard: SignalScorecard = Field(
        description="Dual-direction signal scorecard (UP vs DOWN)"
    )
    recommended_direction: str = Field(
        description="BUY_UP, BUY_DOWN, or SKIP based on best mathematical edge"
    )
    entry_price: float = Field(
        description="Entry price for the recommended direction (e.g. 0.505 for BUY_UP)"
    )
    payout_ratio: float = Field(
        description="Payout ratio for the recommended direction (e.g. 0.98:1)"
    )
    kelly_fraction: float = Field(
        description="Kelly criterion fraction f*. Negative means no edge."
    )
    position_size_usd: float = Field(
        ge=0, description="Half-Kelly position size in USD (0 if SKIP)"
    )
    confidence: int = Field(ge=0, le=100, description="Trade confidence (0-100)")
    strategy: str = Field(
        description="Selected strategy: trend-following, mean-reversion, or skip"
    )
    reasoning: str = Field(
        description="Detailed reasoning including decisive factor for the recommendation"
    )
    signal_fingerprint: str = Field(
        default="",
        description="Signal DNA fingerprint: sorted pipe-separated signal key string for pattern tracking",
    )
    setup_quality: int = Field(
        default=0,
        description="A+ Setup Quality score (0-100). Only trades with ≥65 are executed.",
    )


# ── Agent 3: Risk Gatekeeper Output ─────────────────────────────────


class TradeDecision(BaseModel):
    """Structured output from the Risk Gatekeeper agent.

    The final, authoritative trade decision after risk rule enforcement.
    This is the terminal output of the multi-agent pipeline.
    """

    action: str = Field(description="Trade action: BUY_UP, BUY_DOWN, or SKIP")
    confidence: int = Field(
        ge=0, le=100, description="Confidence in the decision (0-100)"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for the trade decision including which rules passed/failed"
    )
    size_usd: float = Field(ge=0, description="Position size in USD (0 if SKIP)")
    market_delta: float = Field(
        description="Outcome price delta from 0.50 (positive = Up favored)"
    )
