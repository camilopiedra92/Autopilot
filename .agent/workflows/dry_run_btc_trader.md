---
description: Run a dry-run of the Polymarket BTC trader pipeline with full step-by-step output
---

// turbo-all

## Run BTC Trader Pipeline Dry Run

This workflow executes the **6-step fully deterministic pipeline** end-to-end in dry-run mode (no real trades). Zero LLM calls — all steps are pure Python code:

1. **🔄 resolve_outcomes** — Resolves any pending trades from previous runs
2. **📊 gather_market_data** — Multi-market scan (5m/15m/1h) + liquidity filter + 9 data sources
3. **🎯 score_trade** — Weighted 9-signal scoring with EV-based sizing and regime-adaptive thresholds
4. **🛡️ risk_gate** — Enforces 10 hard rules with veto power (incl. one-trade-per-window)
5. **⚡ execute_trade** — Executes the trade (or skips in dry-run mode) with price drift guard
6. **📈 log_performance** — Emits structured performance metrics for monitoring

### Prerequisites

- `.env` file must be present in the project root with `GOOGLE_API_KEY` and Polymarket API keys
- Virtual environment activated with all dependencies installed

### Steps

1. **Run the pipeline dry run**:

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && source venv/bin/activate && python scripts/run_polymarket_dry_run.py
   ```

2. **Report the results**: Show the user the complete output. Key things to report:

   **📊 Market Analysis (gather_market_data):**
   - Selected market duration (5m, 15m, or 1h) and liquidity/spread
   - BTC price, momentum (1m/3m with magnitude), and TA indicators (RSI, MACD, BB, VWAP)
   - Derivatives sentiment (funding rate, OI trend)
   - Volatility regime classification
   - Multi-timeframe alignment
   - Market state (outcome prices, timing, should_trade)
   - Order book direction and imbalance
   - Risk state (daily P&L, allowed, max trade size)
   - Trade history win rate, total trades, and recent outcomes

   **🎯 Score Trade Proposal (score_trade):**
   - Weighted signal scorecard (signals_up vs signals_down, max 9.5)
   - Momentum magnitude bonuses (if any)
   - Regime-adaptive threshold applied (with dynamic history adjustment)
   - Strategy selection (trend-following vs mean-reversion vs standard)
   - Recommended direction (BUY_UP / BUY_DOWN / SKIP)
   - EV-based sizing (entry_price, payout_ratio, kelly_fraction, signal_strength)
   - Position size, confidence (with streak bonus/penalty), and reasoning

   **🛡️ Risk Gate Decision (risk_gate):**
   - Action: APPROVED or SKIP (with rule violation details)
   - Which of the 10 hard rules triggered (if any)
   - Final confidence, size_usd, and reasoning

   **📈 Performance Log (log_performance):**
   - Cumulative stats: total trades, win rate, total P&L
   - Current run: action, duration, regime, confidence

   **Summary:**
   - Final action (BUY_UP / BUY_DOWN / SKIP)
   - Execution time and run status
   - Whether risk_gate approved or vetoed the trade

### Notes

- The dry-run uses in-memory backends — no state is persisted
- Typical execution time is <2 seconds (0 LLM calls, 9 tool calls)
- Cost is $0 per run (fully deterministic, no LLM)
- The script invokes `PolymarketBTCWorkflow.run()` and pretty-prints the pipeline state
- Pipeline state keys: `market_analysis`, `trade_proposal`, `trade_decision`, `execution_result`, `performance_log`
