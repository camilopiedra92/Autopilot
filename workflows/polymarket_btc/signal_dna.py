"""
Signal DNA — Self-learning pattern fingerprinting & weight evolution.

Every trade gets a "DNA fingerprint" — a frozen snapshot of which signals
fired and in what direction. After resolution, we update win/loss counts
per fingerprint. Signals with 70%+ win rate get boosted; <45% get dampened.

Persistence: JSON file via WorkflowStateService (same as trade_history).

Example fingerprints:
    "mom3m:up|macd:up|funding:up|ob:up"  → 78% win → weight 1.4x
    "rsi:dn|funding:dn"                  → 42% win → weight 0.6x
"""

import math
import time

import structlog

logger = structlog.get_logger(__name__)

_WORKFLOW_NAME = "polymarket_btc"
_STATE_KEY = "signal_dna"

# Minimum trades required before a fingerprint influences decisions.
_MIN_SAMPLES = 8


def compute_fingerprint(market_analysis) -> tuple[str, dict]:
    """Build a signal fingerprint from the current MarketAnalysis.

    Returns:
        Tuple of (fingerprint_string, component_signals_dict).
        The fingerprint is a sorted, pipe-separated key string:
            "bb:up|funding:up|macd:up|mom1m:up|mom3m:up|mkt:up|rsi:up|vwap:up"
    """
    signals = {}

    # ── Binance TA signals ────────────────────────────────────────
    mom = market_analysis.momentum
    ta = market_analysis.ta_indicators
    deriv = market_analysis.derivatives
    mtf = market_analysis.multi_timeframe

    # Momentum (the most predictive signals)
    if mom.direction_1m in ("up", "down"):
        signals["mom1m"] = mom.direction_1m
    if mom.direction_3m in ("up", "down"):
        signals["mom3m"] = mom.direction_3m

    # RSI (oversold/overbought)
    if ta.rsi > 60:
        signals["rsi"] = "up"
    elif ta.rsi < 40:
        signals["rsi"] = "dn"

    # MACD
    if ta.macd_histogram > 0:
        signals["macd"] = "up"
    elif ta.macd_histogram < 0:
        signals["macd"] = "dn"

    # Bollinger Band position
    if ta.bb_position > 0.5:
        signals["bb"] = "up"
    elif ta.bb_position < 0.5:
        signals["bb"] = "dn"

    # VWAP
    if ta.vwap_deviation > 0.10:
        signals["vwap"] = "up"
    elif ta.vwap_deviation < -0.10:
        signals["vwap"] = "dn"

    # Funding rate
    if deriv.funding_bias in ("bullish", "very_bullish"):
        signals["funding"] = "up"
    elif deriv.funding_bias in ("bearish", "very_bearish"):
        signals["funding"] = "dn"

    # Multi-timeframe alignment
    if mtf.alignment in ("all_bullish", "higher_tf_up", "bullish_pullback"):
        signals["mtf"] = "up"
    elif mtf.alignment in ("all_bearish", "higher_tf_down", "bearish_rally"):
        signals["mtf"] = "dn"

    # ── Polymarket-native signals ────────────────────────────────
    up_price = market_analysis.market.up_price
    mkt_delta = up_price - 0.50
    if mkt_delta > 0.05:
        signals["mkt"] = "up"
    elif mkt_delta < -0.05:
        signals["mkt"] = "dn"

    iwt = market_analysis.intra_window_trend
    if iwt.get("direction") == "up" and iwt.get("strength", 0) > 0.3:
        signals["iwt"] = "up"
    elif iwt.get("direction") == "down" and iwt.get("strength", 0) > 0.3:
        signals["iwt"] = "dn"

    ob_dir = market_analysis.order_book.direction
    if ob_dir in ("up", "down"):
        signals["ob"] = ob_dir

    # ── Build sorted fingerprint string ───────────────────────────
    if not signals:
        return "neutral", {}

    fp = "|".join(f"{k}:{v}" for k, v in sorted(signals.items()))
    return fp, signals


class DNATracker:
    """Tracks signal DNA fingerprints with win/loss stats.

    Persistence via WorkflowStateService (same mechanism as TradeHistory).
    """

    def __init__(self):
        self._loaded = False
        self._patterns: dict[str, dict] = {}  # fp → {wins, losses, last_seen}

    async def load(self) -> None:
        """Load DNA stats from persistence."""
        if self._loaded:
            return
        try:
            from autopilot.core.state import get_workflow_state_service

            svc = get_workflow_state_service()
            raw = await svc.load_state(_WORKFLOW_NAME, _STATE_KEY)
            self._patterns = raw if raw else {}
        except Exception as e:
            logger.warning("dna_load_failed", error=str(e))
            self._patterns = {}
        self._loaded = True

    async def _save(self) -> None:
        """Persist DNA stats."""
        try:
            from autopilot.core.state import get_workflow_state_service

            svc = get_workflow_state_service()
            await svc.save_state(_WORKFLOW_NAME, _STATE_KEY, self._patterns)
        except Exception as e:
            logger.warning("dna_save_failed", error=str(e))

    def get_weight(self, fingerprint: str) -> float:
        """Get the weight multiplier for a fingerprint.

        Returns 1.0 if not enough data (<MIN_SAMPLES trades).
        Uses sigmoid scaling: 45% → 0.6x, 55% → 1.2x, 70% → 1.5x.
        """
        entry = self._patterns.get(fingerprint)
        if not entry:
            return 1.0

        wins = entry.get("wins", 0)
        losses = entry.get("losses", 0)
        total = wins + losses

        if total < _MIN_SAMPLES:
            return 1.0  # Not enough data — neutral

        win_rate = wins / total
        # Sigmoid: centers at 0.50, steep transition in 0.40-0.60 range
        return round(0.5 + 1.0 / (1 + math.exp(-10 * (win_rate - 0.50))), 3)

    def get_win_rate(self, fingerprint: str) -> float | None:
        """Get raw win rate for a fingerprint. None if not enough data."""
        entry = self._patterns.get(fingerprint)
        if not entry:
            return None
        total = entry.get("wins", 0) + entry.get("losses", 0)
        if total < _MIN_SAMPLES:
            return None
        return round(entry["wins"] / total, 4)

    def get_total_trades(self, fingerprint: str) -> int:
        """Get total trades for a fingerprint."""
        entry = self._patterns.get(fingerprint)
        if not entry:
            return 0
        return entry.get("wins", 0) + entry.get("losses", 0)

    async def record_outcome(self, fingerprint: str, won: bool) -> None:
        """Record a win or loss for a fingerprint."""
        if not fingerprint or fingerprint == "neutral":
            return

        if fingerprint not in self._patterns:
            self._patterns[fingerprint] = {"wins": 0, "losses": 0, "last_seen": 0}

        if won:
            self._patterns[fingerprint]["wins"] += 1
        else:
            self._patterns[fingerprint]["losses"] += 1
        self._patterns[fingerprint]["last_seen"] = time.time()

        await self._save()

        entry = self._patterns[fingerprint]
        total = entry["wins"] + entry["losses"]
        wr = entry["wins"] / total if total > 0 else 0
        logger.info(
            "dna_outcome_recorded",
            fingerprint=fingerprint[:60],
            won=won,
            wins=entry["wins"],
            losses=entry["losses"],
            win_rate=round(wr, 3),
            weight=self.get_weight(fingerprint),
        )

    def get_stats_summary(self) -> dict:
        """Return a summary of all tracked patterns."""
        total_patterns = len(self._patterns)
        patterns_with_data = sum(
            1
            for p in self._patterns.values()
            if (p.get("wins", 0) + p.get("losses", 0)) >= _MIN_SAMPLES
        )
        if not self._patterns:
            return {"total_patterns": 0, "mature_patterns": 0}

        # Top 5 best patterns
        ranked = sorted(
            (
                (fp, d["wins"] / (d["wins"] + d["losses"]))
                for fp, d in self._patterns.items()
                if (d.get("wins", 0) + d.get("losses", 0)) >= _MIN_SAMPLES
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "total_patterns": total_patterns,
            "mature_patterns": patterns_with_data,
            "top_patterns": [
                {"fp": fp[:50], "win_rate": round(wr, 3)} for fp, wr in ranked[:5]
            ],
        }


# ── Singleton ──────────────────────────────────────────────────────

_instance: DNATracker | None = None


def get_dna_tracker() -> DNATracker:
    """Get the singleton DNA tracker instance."""
    global _instance
    if _instance is None:
        _instance = DNATracker()
    return _instance
