"""
PnL display tracker for the live monitor.

Backed by ``data/monitor_trades.json`` — dev/paper trading persistence.
Contains fee/slippage/equity-history data needed only by the dashboard.
All resolution logic delegates to the canonical implementation.

This module is intentionally the SINGLE writer to monitor_trades.json.
It is called from workflow steps (execute_trade_step, resolve_outcomes_step,
gather_market_data) so the monitor itself has zero domain logic.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

TAKER_FEE_PCT = 0.01
DEFAULT_INITIAL_CAPITAL = 200.0

TRADES_FILE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "data",
    "monitor_trades.json",
)


class PnLDisplay:
    """Paper-trading ledger + rolling capital/drawdown display state.

    Persistence: ``data/monitor_trades.json`` (dev/paper trading only).
    Called from workflow steps — never from the monitor loop directly.
    """

    def __init__(self, initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        self.initial_capital: float = initial_capital
        self.trades: list[dict] = []
        self.total_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.total_slippage_cost: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        # Display-only rolling state
        self.equity_history: list[float] = []
        self.peak_capital: float = initial_capital
        self.max_drawdown: float = 0.0
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load trades + portfolio state from disk."""
        try:
            if not os.path.exists(TRADES_FILE):
                return
            with open(TRADES_FILE) as f:
                raw = json.load(f)

            # Support both legacy bare-list and new envelope format
            if isinstance(raw, list):
                self.trades = raw
                portfolio: dict = {}
            else:
                self.trades = raw.get("trades", [])
                portfolio = raw.get("portfolio", {})

            # Recalculate PnL totals from trade records
            for t in self.trades:
                self.total_fees += t.get("fee", 0)
                self.total_slippage_cost += t.get("slippage", 0) * t.get("size_usd", 0)
                if t.get("resolved"):
                    self.total_pnl += t.get("pnl", 0)
                    if t.get("outcome") == "WON":
                        self.wins += 1
                    elif t.get("outcome") == "LOST":
                        self.losses += 1

            # Restore persisted portfolio state
            if "initial_capital" in portfolio:
                self.initial_capital = portfolio["initial_capital"]
            cap = self.initial_capital + self.total_pnl
            self.peak_capital = portfolio.get(
                "peak_capital", max(self.initial_capital, cap)
            )
            self.max_drawdown = portfolio.get("max_drawdown", 0.0)
            self.equity_history = portfolio.get("equity_history", [cap])
            self.equity_history = self.equity_history[-500:]

            logger.info(
                "pnl_display_loaded",
                trades=len(self.trades),
                wins=self.wins,
                losses=self.losses,
                total_pnl=round(self.total_pnl, 2),
                capital=round(cap, 2),
                peak=round(self.peak_capital, 2),
            )
        except Exception as e:
            logger.warning("pnl_display_load_failed", error=str(e))

    def _save(self) -> None:
        """Persist trades + portfolio state to disk."""
        try:
            os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
            payload = {
                "portfolio": {
                    "initial_capital": self.initial_capital,
                    "peak_capital": round(self.peak_capital, 4),
                    "max_drawdown": round(self.max_drawdown, 4),
                    "equity_history": [round(v, 4) for v in self.equity_history[-500:]],
                },
                "trades": self.trades,
            }
            with open(TRADES_FILE, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("pnl_display_save_failed", error=str(e))

    # ── Trade Recording ──────────────────────────────────────────────

    def record_trade(
        self,
        action: str,
        entry_price: float,
        size_usd: float,
        confidence: float,
        btc_price: float,
        condition_id: str = "",
        window_remaining: float = 300,
        spread: float = 0.01,
        window_start: int = 0,
        duration: str = "5m",
    ) -> None:
        """Record a paper trade with fee/slippage simulation."""
        slippage = spread / 2
        effective_entry = round(entry_price + slippage, 4)
        fee = round(size_usd * TAKER_FEE_PCT, 4)
        self.total_fees += fee
        self.total_slippage_cost += slippage * size_usd

        # Build deterministic slug for resolution lookup
        dur_prefix = {
            "5m": "btc-updown-5m",
            "15m": "btc-updown-15m",
            "1h": "btc-updown-1h",
        }.get(duration, "btc-updown-5m")
        slug = f"{dur_prefix}-{window_start}" if window_start else ""

        self.trades.append(
            {
                "action": action,
                "entry_price": entry_price,
                "effective_entry": effective_entry,
                "size_usd": size_usd,
                "fee": fee,
                "slippage": slippage,
                "confidence": confidence,
                "btc_price_at_entry": btc_price,
                "condition_id": condition_id,
                "slug": slug,
                "time": datetime.now().strftime("%H:%M:%S"),
                "window_end_ts": time.time() + window_remaining,
                "resolved": False,
                "pnl": 0.0,
                "outcome": None,
            }
        )
        self._save()
        logger.info(
            "pnl_display_trade_recorded",
            action=action,
            size_usd=size_usd,
            condition_id=condition_id,
        )

    # ── Resolution (3-source canonical chain) ────────────────────────

    async def resolve_pending(self) -> None:
        """Resolve trades via Polymarket CLOB API — 3-source priority chain.

        Source priority:
          1. CLOB /markets/{cid} → tokens[0].price  (0 or 1 after resolution)
          2. Gamma API → outcomePrices[0]            (fallback)
          3. CLOB outcomePrices field                (last resort)
        """
        RESOLUTION_BUFFER = 90
        now = time.time()
        ready = [
            t
            for t in self.trades
            if not t["resolved"]
            and t.get("condition_id")
            and now > t["window_end_ts"] + RESOLUTION_BUFFER
        ]
        if not ready:
            return

        import json as _json
        import httpx

        cids_to_resolve: dict[str, list[dict]] = {}
        for t in ready:
            cids_to_resolve.setdefault(t["condition_id"], []).append(t)

        async with httpx.AsyncClient(timeout=10.0) as http:
            for cid, trades in cids_to_resolve.items():
                try:
                    # ── Source 1: CLOB tokens[].price ─────────────────
                    resp = await http.get(f"https://clob.polymarket.com/markets/{cid}")
                    if resp.status_code != 200:
                        continue

                    market = resp.json()
                    is_closed = market.get("closed", False)

                    tokens = market.get("tokens", [])
                    up_price: float | None = None
                    for tok in tokens:
                        if str(tok.get("outcome", "")).lower() in ("up", "yes"):
                            p = tok.get("price")
                            if p is not None:
                                up_price = float(p)
                            break

                    # ── Source 2: Gamma API fallback ───────────────────
                    if up_price is None or (0.04 < up_price < 0.96):
                        try:
                            g_resp = await http.get(
                                f"https://gamma-api.polymarket.com/markets?condition_ids={cid}"
                            )
                            if g_resp.status_code == 200:
                                gm = g_resp.json()
                                if gm:
                                    raw = gm[0].get("outcomePrices", "")
                                    g_prices = (
                                        _json.loads(raw)
                                        if isinstance(raw, str)
                                        else raw
                                    )
                                    if g_prices:
                                        g_up = float(g_prices[0])
                                        if g_up > 0.95 or g_up < 0.05:
                                            up_price = g_up
                        except Exception:
                            pass

                    # ── Source 3: CLOB outcomePrices (last resort) ─────
                    if up_price is None or (0.04 < up_price < 0.96):
                        raw_op = market.get("outcomePrices")
                        if raw_op:
                            try:
                                op = (
                                    _json.loads(raw_op)
                                    if isinstance(raw_op, str)
                                    else raw_op
                                )
                                if op:
                                    up_price = float(op[0])
                            except Exception:
                                pass

                    # ── Confirm decisive resolution ────────────────────
                    if up_price is None:
                        continue
                    is_resolved = is_closed and (up_price > 0.95 or up_price < 0.05)
                    if not is_resolved:
                        continue

                    winning = "BUY_UP" if up_price > 0.5 else "BUY_DOWN"

                    for trade in trades:
                        won = trade["action"] == winning
                        eff = trade["effective_entry"]
                        size = trade["size_usd"]
                        fee = trade["fee"]
                        tokens_held = size / eff if eff > 0 else 0

                        # Canonical PnL: tokens_held × 1.0 − size − fee
                        if won:
                            pnl = round(tokens_held * 1.0 - size - fee, 2)
                            self.wins += 1
                        else:
                            pnl = round(-size - fee, 2)
                            self.losses += 1

                        trade["pnl"] = pnl
                        trade["resolved"] = True
                        trade["resolved_price"] = round(up_price, 4)
                        trade["outcome"] = "WON" if won else "LOST"
                        self.total_pnl += pnl
                        self._update_capital()

                        logger.info(
                            "pnl_display_trade_resolved",
                            action=trade["action"],
                            outcome=trade["outcome"],
                            pnl=pnl,
                            close_price=up_price,
                        )
                except Exception as e:
                    logger.warning(
                        "pnl_display_resolution_failed", cid=cid, error=str(e)
                    )

        self._save()

    # ── Capital tracking ─────────────────────────────────────────────

    def _update_capital(self) -> None:
        """Recalculate capital, peak, and drawdown after a PnL change."""
        cap = self.capital
        self.equity_history.append(cap)
        if cap > self.peak_capital:
            self.peak_capital = cap
        dd = self.peak_capital - cap
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def snapshot_capital(self) -> None:
        """Record current capital; persist to disk every 60 snapshots."""
        cap = self.capital
        self.equity_history.append(cap)
        if cap > self.peak_capital:
            self.peak_capital = cap
        if len(self.equity_history) % 60 == 0:
            self._save()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self.initial_capital + self.total_pnl

    @property
    def capital_return_pct(self) -> float:
        return (
            (self.total_pnl / self.initial_capital * 100)
            if self.initial_capital > 0
            else 0.0
        )

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.capital) / self.peak_capital * 100

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self.trades if not t["resolved"])

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def gross_pnl(self) -> float:
        return self.total_pnl + self.total_fees + self.total_slippage_cost


# ── Module-level helpers (used by workflow steps) ────────────────────


def get_pnl_display(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> PnLDisplay:
    """Create a fresh PnLDisplay instance (reads from disk each time)."""
    return PnLDisplay(initial_capital=initial_capital)
