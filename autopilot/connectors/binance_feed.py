"""
BinancePriceFeed — Real-time BTC/USDT price via Binance WebSocket.

Provides live 1-minute candle data and technical analysis indicators
for the Polymarket BTC trading workflow. No API key needed — uses
Binance's public WebSocket API.

Usage:
    feed = BinancePriceFeed()
    await feed.start()
    snapshot = feed.get_snapshot()
    await feed.stop()
"""

import asyncio
import json
import time
import math
from collections import deque
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"


@dataclass
class Candle:
    """A single 1-minute OHLCV candle."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False


@dataclass
class TASnapshot:
    """Technical analysis snapshot computed from recent candles."""

    price: float = 0.0
    timestamp: float = 0.0

    # Momentum
    direction_1m: str = "neutral"  # "up", "down", "neutral"
    direction_3m: str = "neutral"
    magnitude_1m: float = 0.0  # Absolute price change in last 1 min
    magnitude_3m: float = 0.0

    # RSI (14-period on 1-min candles)
    rsi: float = 50.0

    # MACD (12, 26, 9)
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Bollinger Bands (20-period, 2 std)
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5  # 0=at lower, 1=at upper

    # VWAP
    vwap: float = 0.0
    vwap_deviation: float = 0.0  # % above/below VWAP

    # Volume
    volume_ratio: float = 1.0  # Current vs average volume

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BinancePriceFeed:
    """
    Real-time BTC/USDT price feed via Binance WebSocket.

    Maintains a rolling window of 1-minute candles and computes
    technical analysis indicators for the trading strategy.
    """

    def __init__(self, max_candles: int = 100):
        self._candles: deque[Candle] = deque(maxlen=max_candles)
        self._current_candle: Candle | None = None
        self._ws = None
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def current_price(self) -> float:
        """Get the latest BTC/USDT price."""
        if self._current_candle:
            return self._current_candle.close
        if self._candles:
            return self._candles[-1].close
        return 0.0

    async def start(self) -> None:
        """Start the WebSocket connection and price feed."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._ws_loop())
        logger.info("binance_feed_started")

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("binance_feed_stopped")

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "binance_feed_missing_dep",
                msg="websockets package required. pip install websockets",
            )
            return

        while self._running:
            try:
                async with websockets.connect(BINANCE_WS_URL) as ws:
                    self._ws = ws
                    logger.info("binance_ws_connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_message(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("binance_ws_error", error=str(e))
                if self._running:
                    await asyncio.sleep(5)  # Reconnect delay

    def _process_message(self, data: dict) -> None:
        """Process a Binance kline WebSocket message."""
        kline = data.get("k", {})
        if not kline:
            return

        candle = Candle(
            timestamp=kline["t"] / 1000.0,
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            is_closed=kline.get("x", False),
        )

        if candle.is_closed:
            self._candles.append(candle)
            self._current_candle = None
        else:
            self._current_candle = candle

    def get_snapshot(self) -> TASnapshot:
        """Compute and return a full TA snapshot from current candle data."""
        candles = list(self._candles)
        if self._current_candle:
            candles.append(self._current_candle)

        if not candles:
            return TASnapshot()

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        current = closes[-1]

        snap = TASnapshot(
            price=current,
            timestamp=time.time(),
        )

        # ── Momentum ────────────────────────────────────────────────
        if len(closes) >= 2:
            snap.magnitude_1m = current - closes[-2]
            snap.direction_1m = (
                "up"
                if snap.magnitude_1m > 0
                else "down"
                if snap.magnitude_1m < 0
                else "neutral"
            )

        if len(closes) >= 4:
            snap.magnitude_3m = current - closes[-4]
            # Check direction consistency over last 3 candles
            changes = [closes[i] - closes[i - 1] for i in range(-3, 0)]
            if all(c > 0 for c in changes):
                snap.direction_3m = "up"
            elif all(c < 0 for c in changes):
                snap.direction_3m = "down"
            else:
                snap.direction_3m = "neutral"

        # ── RSI (14-period) ─────────────────────────────────────────
        snap.rsi = self._compute_rsi(closes, period=14)

        # ── MACD (12, 26, 9) ────────────────────────────────────────
        macd = self._compute_macd(closes)
        snap.macd_line = macd["macd"]
        snap.macd_signal = macd["signal"]
        snap.macd_histogram = macd["histogram"]

        # ── Bollinger Bands (20-period, 2 std) ──────────────────────
        bb = self._compute_bollinger(closes, period=20, num_std=2)
        snap.bb_upper = bb["upper"]
        snap.bb_middle = bb["middle"]
        snap.bb_lower = bb["lower"]
        if bb["upper"] != bb["lower"]:
            snap.bb_position = (current - bb["lower"]) / (bb["upper"] - bb["lower"])
        else:
            snap.bb_position = 0.5

        # ── VWAP ────────────────────────────────────────────────────
        snap.vwap = self._compute_vwap(candles)
        if snap.vwap > 0:
            snap.vwap_deviation = ((current - snap.vwap) / snap.vwap) * 100

        # ── Volume Ratio ────────────────────────────────────────────
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0:
                snap.volume_ratio = volumes[-1] / avg_vol

        return snap

    @staticmethod
    def _compute_rsi(closes: list[float], period: int = 14) -> float:
        """Compute RSI using exponential moving average method."""
        if len(closes) < period + 1:
            return 50.0

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]

        # Initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # EMA-style smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def _compute_macd(
        closes: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        """Compute MACD line, signal line, and histogram."""
        if len(closes) < slow + signal:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

        def ema(data: list[float], period: int) -> list[float]:
            multiplier = 2 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append(data[i] * multiplier + result[-1] * (1 - multiplier))
            return result

        fast_ema = ema(closes, fast)
        slow_ema = ema(closes, slow)
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = ema(macd_line, signal)

        return {
            "macd": round(macd_line[-1], 4),
            "signal": round(signal_line[-1], 4),
            "histogram": round(macd_line[-1] - signal_line[-1], 4),
        }

    @staticmethod
    def _compute_bollinger(
        closes: list[float], period: int = 20, num_std: float = 2
    ) -> dict:
        """Compute Bollinger Bands."""
        if len(closes) < period:
            return {
                "upper": closes[-1] if closes else 0,
                "middle": closes[-1] if closes else 0,
                "lower": closes[-1] if closes else 0,
            }

        window = closes[-period:]
        middle = sum(window) / period
        variance = sum((x - middle) ** 2 for x in window) / period
        std = math.sqrt(variance)

        return {
            "upper": round(middle + num_std * std, 2),
            "middle": round(middle, 2),
            "lower": round(middle - num_std * std, 2),
        }

    @staticmethod
    def _compute_vwap(candles: list[Candle]) -> float:
        """Compute Volume-Weighted Average Price."""
        if not candles:
            return 0.0

        total_pv = sum(((c.high + c.low + c.close) / 3) * c.volume for c in candles)
        total_vol = sum(c.volume for c in candles)

        return round(total_pv / total_vol, 2) if total_vol > 0 else 0.0
