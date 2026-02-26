#!/usr/bin/env python3
"""
Polymarket BTC Live Dashboard — entry point only.
All logic lives in workflows/polymarket_btc/:
  monitor/pnl_display.py  — paper trading ledger (monitor_trades.json)
  monitor/renderer.py     — ANSI dashboard renderer
  monitor/live.py         — LiveMonitor orchestration loop
  steps.py                — trade execution, resolution, risk gate, bankroll sync
  risk.py                 — RiskManager (direction lock, window cap, cooldown)
  trade_history.py        — TradeHistory (WorkflowStateService persistence)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from workflows.polymarket_btc.monitor.live import LiveMonitor  # noqa: E402


async def main() -> None:
    await LiveMonitor().run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped.\n")
