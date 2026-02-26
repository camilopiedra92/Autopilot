"""
Trade history — single-file persistence with bounded growth.

One workflow state key: ``trade_stats``

    {
      "total_trades": 42,
      "wins": 20, "losses": 18, "resolved": 38,
      "total_pnl": 12.50, "win_rate_pct": 52.6,
      "pending": [
        {"cid":"abc","action":"BUY_UP","price":0.52,"size":10,"ts":170800}
      ],
      "ledger": [
        {"action":"BUY_UP","entry":0.52,"close":0.95,"pnl":4.80,
         "outcome":"won","size":10,"cid":"abc","ts_entry":170800,"ts_resolved":171100}
      ]
    }

- **Counters**: never grow, updated in-place
- **Pending**: minimal entries (~100 bytes each), pruned on resolution
- **Ledger**: last 200 resolved trades with full details (FIFO-capped)
- **Signal snapshots**: stored by platform in per-execution artifacts
"""

import json
import time

import structlog

logger = structlog.get_logger(__name__)

_WORKFLOW_NAME = "polymarket_btc"
_STATE_KEY = "trade_stats"


_MAX_LEDGER_SIZE = 200


def _empty_stats() -> dict:
    """Default stats structure."""
    return {
        "total_trades": 0,
        "resolved": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "win_rate_pct": 0.0,
        "pending": [],
        "ledger": [],
    }


class TradeHistory:
    """Single-key trade stats with inline pending trades.

    Counters stay small forever. Pending entries are pruned
    when Polymarket markets resolve (~5 min).

    Uses ``WorkflowStateService`` for cross-run persistence.
    """

    async def _load(self) -> dict:
        """Load trade stats from WorkflowStateService."""
        from autopilot.core.workflow_state import WorkflowStateService

        state = WorkflowStateService(_WORKFLOW_NAME)
        data = await state.get(_STATE_KEY)
        return data if data else _empty_stats()

    async def _save(self, stats: dict) -> None:
        """Save trade stats to WorkflowStateService."""
        from autopilot.core.workflow_state import WorkflowStateService

        state = WorkflowStateService(_WORKFLOW_NAME)
        await state.put(_STATE_KEY, stats)

    # ── Record a new trade ────────────────────────────────────────

    async def record_trade(
        self,
        *,
        action: str,
        entry_price: float,
        size_usd: float,
        condition_id: str,
        vol_regime: str = "unknown",
        alignment: str = "unknown",
        signal_fingerprint: str = "",
    ) -> None:
        """Record a trade: increment counter + add to pending."""
        stats = await self._load()

        stats["total_trades"] += 1
        stats["pending"].append(
            {
                "cid": condition_id,
                "action": action,
                "price": round(entry_price, 4),
                "size": round(size_usd, 2),
                "ts": time.time(),
                "vol_regime": vol_regime,
                "alignment": alignment,
                "signal_fingerprint": signal_fingerprint,
            }
        )

        stats["total_trades"] = stats["resolved"] + len(stats["pending"])
        await self._save(stats)
        logger.info(
            "trade_recorded",
            action=action,
            pending=len(stats["pending"]),
            fingerprint=signal_fingerprint[:50],
        )

    # ── Resolve pending outcomes ──────────────────────────────────

    async def resolve_pending_outcomes(self) -> int:
        """Check Polymarket API for resolved markets.

        Uses a 3-source priority chain (canonical — matches monitor logic):
          1. CLOB /markets/{cid} → tokens[].price  (authoritative post-resolution)
          2. Gamma API → outcomePrices[0]           (fallback)
          3. CLOB outcomePrices field               (last resort)

        Only resolves trades 90 seconds after the window close (on-chain
        finality buffer). PnL formula: tokens_held × 1.0 − size − fee
        (1% taker fee included).

        Returns:
            Number of trades resolved.
        """
        import httpx as _httpx

        TAKER_FEE_PCT = 0.01
        RESOLUTION_BUFFER = 90  # seconds post-close for on-chain finality

        stats = await self._load()
        pending = stats.get("pending", [])
        now = time.time()

        # Only check trades whose window has closed + buffer elapsed
        ready = [
            p
            for p in pending
            if (now - p.get("ts", 0)) > (300 + RESOLUTION_BUFFER) and p.get("cid")
        ]
        if not ready:
            return 0

        resolved_count = 0
        resolved_cids: set[str] = set()

        async with _httpx.AsyncClient(timeout=10.0) as http:
            for entry in ready:
                cid = entry["cid"]
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
                                        json.loads(raw) if isinstance(raw, str) else raw
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
                                    json.loads(raw_op)
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

                    winning_direction = "BUY_UP" if up_price > 0.5 else "BUY_DOWN"
                    outcome = "won" if entry["action"] == winning_direction else "lost"

                    # PnL: tokens_held × 1.0 − size − fee  (canonical formula)
                    entry_price = entry["price"]
                    size = entry["size"]
                    fee = round(size * TAKER_FEE_PCT, 4)
                    tokens_held = size / entry_price if entry_price > 0 else 0

                    if outcome == "won":
                        pnl = round(tokens_held * 1.0 - size - fee, 2)
                        stats["wins"] += 1
                    else:
                        pnl = round(-size - fee, 2)
                        stats["losses"] += 1

                    stats["resolved"] += 1
                    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
                    resolved_cids.add(cid)
                    resolved_count += 1

                    # Record in ledger
                    ledger = stats.setdefault("ledger", [])
                    ledger.append(
                        {
                            "action": entry["action"],
                            "entry": entry_price,
                            "close": round(up_price, 4),
                            "pnl": pnl,
                            "fee": fee,
                            "outcome": outcome,
                            "size": size,
                            "cid": cid,
                            "ts_entry": entry.get("ts", 0),
                            "ts_resolved": now,
                            "vol_regime": entry.get("vol_regime", "unknown"),
                            "alignment": entry.get("alignment", "unknown"),
                        }
                    )

                    # Update RiskManager with actual PnL
                    try:
                        from workflows.polymarket_btc.risk import get_risk_manager

                        risk_mgr = get_risk_manager()
                        risk_mgr.record_trade(pnl=pnl)
                        await risk_mgr.save()
                    except Exception as e:
                        logger.warning("risk_update_on_resolve_failed", error=str(e))

                    logger.info(
                        "trade_resolved",
                        action=entry["action"],
                        outcome=outcome,
                        pnl=pnl,
                        close_price=up_price,
                        source="3-source-chain",
                    )

                    # Update Signal DNA
                    fp = entry.get("signal_fingerprint", "")
                    if fp:
                        try:
                            from workflows.polymarket_btc.signal_dna import (
                                get_dna_tracker,
                            )

                            dna = get_dna_tracker()
                            await dna.load()
                            await dna.record_outcome(fp, won=(outcome == "won"))
                        except Exception as e:
                            logger.warning("dna_update_failed", error=str(e))

                except Exception as e:
                    logger.warning(
                        "resolution_failed", cid=cid, error=str(e), exc_info=True
                    )

        if resolved_count > 0:
            total_resolved = stats["resolved"]
            stats["win_rate_pct"] = round(
                (stats["wins"] / total_resolved * 100) if total_resolved > 0 else 0, 1
            )
            stats["pending"] = [p for p in pending if p.get("cid") not in resolved_cids]
            ledger = stats.get("ledger", [])
            if len(ledger) > _MAX_LEDGER_SIZE:
                stats["ledger"] = ledger[-_MAX_LEDGER_SIZE:]
            stats["total_trades"] = stats["resolved"] + len(stats["pending"])
            await self._save(stats)
            logger.info(
                "outcomes_resolved",
                resolved=resolved_count,
                still_pending=len(stats["pending"]),
                win_rate=stats["win_rate_pct"],
            )

        return resolved_count

    # ── Query ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Get trade stats (counters + pending count + per-regime/alignment breakdowns)."""
        stats = await self._load()
        ledger = stats.get("ledger", [])

        # Compute per-regime win rates from ledger
        by_regime: dict[str, dict] = {}
        by_alignment: dict[str, dict] = {}
        for trade in ledger:
            regime = trade.get("vol_regime", "unknown")
            align = trade.get("alignment", "unknown")
            won = trade.get("outcome") == "won"

            bucket = by_regime.setdefault(regime, {"wins": 0, "losses": 0, "total": 0})
            bucket["total"] += 1
            bucket["wins" if won else "losses"] += 1

            bucket = by_alignment.setdefault(
                align, {"wins": 0, "losses": 0, "total": 0}
            )
            bucket["total"] += 1
            bucket["wins" if won else "losses"] += 1

        # Add win_rate_pct to each bucket
        for bucket in list(by_regime.values()) + list(by_alignment.values()):
            total = bucket["total"]
            bucket["win_rate_pct"] = (
                round(bucket["wins"] / total * 100, 1) if total > 0 else 0
            )

        return {
            "total_trades": stats["total_trades"],
            "resolved": stats["resolved"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total_pnl": stats["total_pnl"],
            "win_rate_pct": stats["win_rate_pct"],
            "pending": len(stats.get("pending", [])),
            "by_regime": by_regime,
            "by_alignment": by_alignment,
            "recent_outcomes": self._get_recent_outcomes(ledger),
        }

    @staticmethod
    def _get_recent_outcomes(ledger: list[dict], n: int = 5) -> list[str]:
        """Extract last N trade outcomes as ['W', 'L', ...] for streak tracking."""
        if not ledger:
            return []
        recent = ledger[-n:]
        return ["W" if t.get("outcome") == "won" else "L" for t in recent]


# ── Singleton ──────────────────────────────────────────────────────

_instance: TradeHistory | None = None


def get_trade_history() -> TradeHistory:
    """Get the singleton trade history instance."""
    global _instance
    if _instance is None:
        _instance = TradeHistory()
    return _instance
