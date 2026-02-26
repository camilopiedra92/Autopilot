"""
LiveMonitor — pure orchestration loop for the live dashboard.

Responsibilities (ONLY):
  - Run the PolymarketBTC workflow on each 1-second tick
  - Track ephemeral display state: history[], btc_history[], start_time
  - Call snapshot_capital() on PnLDisplay (display-only equity tick)
  - Render the dashboard via renderer.render()

Explicitly NOT responsible for:
  - Trade gate logic (→ RiskManager via risk_gate step)
  - Trade recording (→ execute_trade_step → PnLDisplay.record_trade())
  - Trade resolution (→ resolve_outcomes_step → PnLDisplay.resolve_pending())
  - Bankroll sync (→ gather_market_data step)
  - Any domain logic whatsoever
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import structlog

from workflows.polymarket_btc.monitor.pnl_display import (
    DEFAULT_INITIAL_CAPITAL,
    PnLDisplay,
)
from workflows.polymarket_btc.monitor.renderer import render

logger = structlog.get_logger(__name__)

INTERVAL_SECONDS = 1


class LiveMonitor:
    """Thin orchestration loop — visualization only."""

    def __init__(self, interval_seconds: int = INTERVAL_SECONDS) -> None:
        self.interval = interval_seconds
        self.history: list[dict] = []
        self.btc_history: list[float] = []
        self.start_time = time.time()

        initial_capital = float(
            os.environ.get("INITIAL_CAPITAL", DEFAULT_INITIAL_CAPITAL)
        )
        self.tracker = PnLDisplay(initial_capital=initial_capital)

        # Suppress noisy loggers at startup so the dashboard renders cleanly
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        )

        # Print startup summary (uses print intentionally — before log suppression)
        cap = self.tracker.capital
        pnl = self.tracker.total_pnl
        ret = self.tracker.capital_return_pct
        ret_sign = "+" if ret >= 0 else ""
        pnl_sign = "+" if pnl >= 0 else ""
        print("\n\033[1m\033[96m🚀 Starting Polymarket BTC Live Monitor\033[0m")
        print("\033[2m   Real-time paper trading  •  Ctrl+C to stop\033[0m\n")
        print(
            f"  💼 \033[1mCapital loaded:\033[0m  "
            f"\033[1m\033[97m${cap:,.2f} USDC\033[0m  "
            f"(start ${self.tracker.initial_capital:.2f}  •  "
            f"PnL {pnl_sign}${pnl:.2f}  •  {ret_sign}{ret:.2f}%)"
        )
        pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"
        print(f"  📊 Kelly bankroll synced to {pnl_color}${cap:.2f}\033[0m\n")

    # ── Main Loop ────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Run the monitor loop indefinitely."""
        run_number = 0
        while True:
            run_number += 1
            try:
                t0 = time.time()
                data = await self._run_once(run_number)
                elapsed_ms = (time.time() - t0) * 1000

                self.history.append(data)

                btc_price = data.get("market", {}).get("price", 0)
                if btc_price > 0:
                    self.btc_history.append(btc_price)

                # Reload tracker from disk — workflow steps may have updated it
                self.tracker = PnLDisplay(initial_capital=self.tracker.initial_capital)
                self.tracker.snapshot_capital()

                output = render(
                    data,
                    self.history,
                    elapsed_ms,
                    self.tracker,
                    self.start_time,
                    self.btc_history,
                )
                print(output)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n  \033[91m❌ Run #{run_number} failed: {e}\033[0m")
                import traceback

                traceback.print_exc()

            try:
                await asyncio.sleep(self.interval)
            except KeyboardInterrupt:
                raise

    # ── Single Workflow Invocation ───────────────────────────────────

    async def _run_once(self, run_number: int) -> dict:
        """Invoke the PolymarketBTC workflow and return mapped result.

        Forces 5m duration (monitor-level config) and clears TA cache
        between runs to ensure fresh data at 1s cadence.
        """
        # Clear TA cache between runs (performance optimization for 1s cadence)
        try:
            import workflows.polymarket_btc.tools as tools_mod

            tools_mod._cached_ta = None
            tools_mod._cached_ta_ts = 0.0
        except Exception:
            pass

        # Force 5m only — avoids duration switching which breaks slug resolution
        os.environ["BTC_DURATIONS"] = "5m"

        from autopilot.models import TriggerType
        from workflows.polymarket_btc.workflow import PolymarketBTCWorkflow

        workflow = PolymarketBTCWorkflow()
        await workflow.setup()
        result = await workflow.run(trigger_type=TriggerType.SCHEDULED, trigger_data={})

        state = result.result or {}
        return {
            "run": run_number,
            "status": result.status.value,
            "market": state.get("market_analysis", {}),
            "proposal": state.get("trade_proposal", {}),
            "decision": state.get("trade_decision", {}),
            "performance": state.get("performance_log", {}),
            "error": result.error,
        }
