"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Polymarket BTC Trader — Dry Run                                        ║
║                                                                          ║
║  Full pipeline via BaseWorkflow.run() — exactly as production.           ║
║  5-Layer Conviction: DNA · Confluence · Bayesian · Kelly · Anti-Fragile  ║
║  Fully deterministic: 0 LLM calls, $0 cost, <2s execution.              ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    source venv/bin/activate
    python scripts/run_polymarket_dry_run.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# ── ANSI Colors & Styles ────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITAL = "\033[3m"

BLK = "\033[30m"
RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
WHT = "\033[97m"

BG_RED = "\033[41m"
BG_GRN = "\033[42m"
BG_YEL = "\033[43m"
BG_BLU = "\033[44m"
BG_MAG = "\033[45m"
BG_CYN = "\033[46m"
BG_DGRAY = "\033[100m"

# Box drawing
DH = "═"
DV = "║"
DTL = "╔"
DTR = "╗"
DBL = "╚"
DBR = "╝"
H = "─"
V = "│"
VL = "├"
VR = "┤"

W = 76


def c(text, color, bold=False):
    b = BOLD if bold else ""
    return f"{b}{color}{text}{RST}"


def gauge(value, max_val, width=20, filled_color=GRN, empty_color=DIM):
    pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
    filled = int(pct * width)
    return f"{filled_color}{'█' * filled}{empty_color}{'░' * (width - filled)}{RST}"


def section(icon, title):
    print(
        f"\n  {c(VL, CYN)}{c(f' {icon} {title} ', CYN, bold=True)}{c(H * (W - 6 - len(title)) + VR, CYN)}"
    )


def kv(key, val, indent=5):
    print(f"  {V}{' ' * indent}{c(key + ':', WHT, bold=True)} {val}")


def dir_str(direction):
    if "UP" in str(direction):
        return c(f"▲ {direction}", GRN, bold=True)
    elif "DOWN" in str(direction):
        return c(f"▼ {direction}", RED, bold=True)
    return c(f"○ {direction}", YEL)


def pnl_str(val):
    if val > 0:
        return c(f"+${val:.2f}", GRN, bold=True)
    elif val < 0:
        return c(f"-${abs(val):.2f}", RED, bold=True)
    return c("$0.00", DIM)


def regime_badge(regime):
    colors = {"low": CYN, "normal": GRN, "high": YEL, "extreme": RED}
    return c(regime.upper(), colors.get(regime, WHT), bold=True)


def momentum_str(direction, magnitude):
    if direction == "up":
        arrow = "▲" if magnitude > 50 else "△"
        return c(f"{arrow}+{magnitude:.0f}", GRN, bold=magnitude > 100)
    elif direction == "down":
        arrow = "▼" if abs(magnitude) > 50 else "▽"
        return c(f"{arrow}{magnitude:.0f}", RED, bold=abs(magnitude) > 100)
    return c(f"─{magnitude:.0f}", DIM)


async def main():
    from autopilot.models import TriggerType

    # ── Header ──
    bankroll = float(os.environ.get("BANKROLL_USDC", 200.0))
    kelly_cap = float(os.environ.get("KELLY_FRACTION_CAP", 0.25))
    max_exposure = float(os.environ.get("MAX_EXPOSURE_PER_WINDOW", 50.0))

    print(f"\n  {c(DTL + DH * (W - 2) + DTR, MAG, bold=True)}")
    print(
        f"  {c(DV, MAG, bold=True)}  {c('🔮', WHT)}  {c('POLYMARKET BTC TRADER', WHT, bold=True)}"
        f"  {c('— DRY RUN', MAG)}                           {c(DV, MAG, bold=True)}"
    )
    print(
        f"  {c(DV, MAG, bold=True)}  {c('5-Layer: DNA · Confluence · Bayesian · Kelly · Anti-Fragile', DIM)}"
        f"    {c(DV, MAG, bold=True)}"
    )
    print(
        f"  {c(DV, MAG, bold=True)}  {c('0 LLM calls', CYN)} {c('·', DIM)} {c('$0 cost', CYN)}"
        f" {c('·', DIM)} {c('< 2s latency', CYN)}"
        f" {c('·', DIM)} {c(f'Bankroll ${bankroll:.0f}', YEL)}"
        f" {c('·', DIM)} {c(f'Kelly cap {kelly_cap:.0%}', YEL)}"
        f" {c('·', DIM)} {c(f'Window cap ${max_exposure:.0f}', YEL)}"
        f"      {c(DV, MAG, bold=True)}"
    )
    print(
        f"  {c(DV, MAG, bold=True)}  {c(datetime.now().strftime('%H:%M:%S'), DIM)}"
        f"                                                           {c(DV, MAG, bold=True)}"
    )
    print(f"  {c(DBL + DH * (W - 2) + DBR, MAG, bold=True)}")

    # ── Run pipeline ──
    from workflows.polymarket_btc.workflow import PolymarketBTCWorkflow

    workflow = PolymarketBTCWorkflow()
    await workflow.setup()

    import time as _time

    t0 = _time.time()
    result = await workflow.run(trigger_type=TriggerType.SCHEDULED, trigger_data={})
    elapsed = (_time.time() - t0) * 1000

    state = result.result or {}
    m = state.get("market_analysis", {})
    p = state.get("trade_proposal", {})
    d = state.get("trade_decision", {})
    perf = state.get("performance_log", {})

    print(f"\n  {c(f'Pipeline completed in {elapsed:.0f}ms', DIM)}")

    if result.error:
        print(f"\n  {c('❌ ERROR: ' + str(result.error), RED, bold=True)}")
        return

    # ═══ MARKET DATA ═══
    section("📊", "MARKET DATA")

    price = m.get("price", 0)
    regime = m.get("volatility", {}).get("regime", "?")
    atr = m.get("volatility", {}).get("atr_pct", 0)
    duration = m.get("selected_duration", "?")
    mkt = m.get("market", {})
    should_trade = mkt.get("should_trade", False)
    elapsed_w = mkt.get("elapsed", 0)
    remaining = mkt.get("remaining", 0)
    edge_decay = mkt.get("edge_decay", 0)
    up_p = mkt.get("up_price", 0.5)
    down_p = mkt.get("down_price", 0.5)
    zone = mkt.get("timing_zone", "?")
    ob = m.get("order_book", {})

    trade_badge = (
        c(" TRADEABLE ", BLK, bold=True).replace(BLK, f"\033[30m{BG_GRN}")
        if should_trade
        else c(" WAIT ", BLK, bold=True).replace(BLK, f"\033[30m{BG_YEL}")
    )

    kv(
        "BTC Price",
        f"{c(f'${price:,.2f}', WHT, bold=True)}  "
        f"Regime: {regime_badge(regime)} {c(f'ATR:{atr:.4f}%', DIM)}  "
        f"Duration: {c(duration, WHT, bold=True)}",
    )

    total_w = elapsed_w + remaining if (elapsed_w + remaining) > 0 else 300
    kv(
        "Window",
        f"{gauge(elapsed_w, total_w, 25, CYN)} {c(f'{elapsed_w}s/{total_w}s', DIM)} "
        f"Zone: {c(zone.upper(), CYN, bold=True)} "
        f"Decay: {c(f'{edge_decay:.2f}', YEL if edge_decay < 0.5 else GRN)}  "
        f"{trade_badge}",
    )

    up_bid = ob.get("best_bid", 0)
    up_ask = ob.get("best_ask", 1)
    dn_bid = ob.get("down_best_bid", 0)
    dn_ask = ob.get("down_best_ask", 1)
    ob_spread = ob.get("spread", 0)
    kv(
        "Tokens",
        f"UP {c(f'{up_p:.3f}', GRN)} "
        f"{c(f'(bid:{up_bid:.3f} ask:{up_ask:.3f})', DIM)}  "
        f"DOWN {c(f'{down_p:.3f}', RED)} "
        f"{c(f'(bid:{dn_bid:.3f} ask:{dn_ask:.3f})', DIM)}  "
        f"Spread: {c(f'{ob_spread:.3f}', YEL)}",
    )

    # ═══ SIGNAL CONFLUENCE ENGINE ═══
    section("📈", "SIGNAL CONFLUENCE ENGINE")

    mom = m.get("momentum", {})
    ta = m.get("ta_indicators", {})
    mtf = m.get("multi_timeframe", {})
    deriv = m.get("derivatives", {})

    m1 = momentum_str(mom.get("direction_1m", ""), mom.get("magnitude_1m", 0))
    m3 = momentum_str(mom.get("direction_3m", ""), mom.get("magnitude_3m", 0))
    rsi = ta.get("rsi", 50)
    rsi_c = (
        RED
        if rsi > 70
        else (GRN if rsi < 30 else (YEL if rsi > 60 or rsi < 40 else DIM))
    )
    macd = ta.get("macd_histogram", 0)
    bb = ta.get("bb_position", 0.5)

    print(
        f"  {V}  {c('L1A', CYN, bold=True)} Binance  "
        f"Mom: {m1}/{m3}  RSI {c(f'{rsi:.0f}', rsi_c)}  "
        f"MACD {c(f'{macd:+.1f}', GRN if macd > 0 else RED)}  "
        f"BB {c(f'{bb:.2f}', WHT)}  "
        f"MTF {c(mtf.get('alignment', '?'), GRN if 'bullish' in mtf.get('alignment', '') else RED if 'bearish' in mtf.get('alignment', '') else YEL)}"
    )

    fb = deriv.get("funding_bias", "neutral")
    fb_c = GRN if "bullish" in fb else (RED if "bearish" in fb else DIM)
    print(
        f"  {V}         Funding: {c(deriv.get('funding_rate', 0), fb_c)} ({c(fb, fb_c)})  "
        f"OI: {c(deriv.get('oi_trend', 'stable'), WHT)}"
    )

    # Polymarket flow
    iwt = m.get("intra_window_trend", {})
    lt = m.get("last_trade", {})

    vol = m.get("market_volume", {})

    iwt_dir = iwt.get("direction", "flat")
    iwt_str = iwt.get("strength", 0)
    lt_side = lt.get("side", "?")
    lt_price = lt.get("price", 0)
    vol_val = vol.get("volume", 0)
    liq_val = vol.get("liquidity", 0)

    iwt_c = GRN if iwt_dir == "up" else (RED if iwt_dir == "down" else DIM)
    print(
        f"  {V}  {c('L1B', MAG, bold=True)} Polymarket  "
        f"IWT:{c(f'{iwt_dir}({iwt_str:.2f})', iwt_c)}  "
        f"Last:{c(f'{lt_side}@{lt_price:.2f}', GRN if lt_side == 'BUY' else RED)}  "
        f"Vol:${vol_val:,.0f}  Liq:${liq_val:,.0f}"
    )

    # Confluence
    sc = p.get("scorecard", {})
    notes_raw = sc.get("scorecard_notes", "")
    conf_text = ""
    for n in notes_raw.split("; "):
        if "Confluence:" in n:
            conf_text = n.replace("Confluence: ", "")
            break
    conf_c = (
        GRN if "FULL" in conf_text else (YEL if "only" in conf_text.lower() else RED)
    )
    print(
        f"  {V}  {c('L2', WHT, bold=True)}  {c(conf_text or 'No confluence', conf_c, bold=True)}"
    )

    # ═══ TIER 2: BTC-NATIVE SIGNALS ═══
    section("📡", "BTC-NATIVE SIGNALS (Tier 2)")

    liq = m.get("liquidation", {})
    carry = m.get("pre_window_carry", {})
    streak = m.get("window_streak", {})

    # Liquidation cascade
    liq_level = liq.get("cascade_level", "none")
    liq_dir = liq.get("net_direction", "neutral")
    liq_long = liq.get("long_liq_usd", 0)
    liq_short = liq.get("short_liq_usd", 0)
    liq_count = liq.get("count", 0)

    liq_badge = c(f" {liq_level.upper()} ", BLK, bold=True).replace(
        BLK,
        f"\033[30m{BG_RED}"
        if liq_level == "major"
        else f"\033[30m{BG_YEL}"
        if liq_level == "minor"
        else f"\033[30m{BG_DGRAY}",
    )

    print(
        f"  {V}  🔥 Liquidation  {liq_badge}  "
        f"Long:${liq_long:,.0f} Short:${liq_short:,.0f}  "
        f"Dir:{c(liq_dir, GRN if liq_dir == 'up' else RED if liq_dir == 'down' else DIM)}  "
        f"Orders:{liq_count}"
    )

    # Pre-window carry
    carry_dir = carry.get("direction", "neutral")
    carry_str = carry.get("strength", 0)
    carry_c = GRN if carry_dir == "up" else (RED if carry_dir == "down" else DIM)
    bull_bars = carry.get("consecutive_bullish", 0)
    bear_bars = carry.get("consecutive_bearish", 0)
    print(
        f"  {V}  🌊 Carry        Dir:{c(carry_dir, carry_c, bold=True)}  "
        f"Str:{gauge(carry_str, 1.0, 10, carry_c)} {c(f'{carry_str:.3f}', carry_c)}  "
        f"Candles: {c(f'▲{bull_bars}', GRN)}/{c(f'▼{bear_bars}', RED)}"
    )

    # Window streak
    outcomes = streak.get("last_outcomes", [])
    s_len = streak.get("streak_length", 0)
    s_dir = streak.get("streak_direction", "neutral")
    s_pattern = streak.get("pattern", "mixed")
    outcomes_display = ""
    for o in outcomes:
        outcomes_display += c("●", GRN) if o == "W" else c("●", RED)
    s_c = GRN if s_dir == "winning" else (RED if s_dir == "losing" else DIM)
    pat_c = RED if s_pattern == "choppy" else (GRN if s_pattern == "trend" else YEL)
    print(
        f"  {V}  📊 Streak       {outcomes_display or c('(no data)', DIM)}  "
        f"Streak:{c(str(s_len), s_c, bold=True)} ({c(s_dir, s_c)})  "
        f"Pattern:{c(s_pattern, pat_c, bold=True)}"
    )

    # ═══ TIER 3: BAYESIAN + SETUP QUALITY ═══
    section("🧠", "BAYESIAN + SETUP QUALITY (Tier 3)")

    # Parse from notes
    bayes_note = ""
    setup_note = ""
    for n in notes_raw.split("; "):
        if "Bayes:" in n:
            bayes_note = n
        if "SetupQ:" in n:
            setup_note = n

    setup_q = p.get("setup_quality", 0)
    sq_c = (
        GRN
        if setup_q >= 70
        else (YEL if setup_q >= 50 else (RED if setup_q < 40 else DIM))
    )
    print(
        f"  {V}  Setup Quality  {gauge(setup_q, 100, 20, sq_c)} "
        f"{c(f'{setup_q}/100', sq_c, bold=True)}  "
        f"{c('A+' if setup_q >= 70 else ('OK' if setup_q >= 40 else 'SKIP'), sq_c, bold=True)}"
    )

    if bayes_note:
        print(f"  {V}  Bayesian       {c(bayes_note, CYN)}")
    if setup_note:
        print(f"  {V}  Quality Score  {c(setup_note, DIM)}")

    # ═══ TIER 4: KELLY SIZING + ANTI-FRAGILE ═══
    section("💰", "KELLY SIZING + ANTI-FRAGILE (Tier 4)")

    # Parse Kelly and anomaly notes
    anomaly_notes = []
    kelly_note = ""
    for n in notes_raw.split("; "):
        if "ANOMALY:" in n:
            anomaly_notes.append(n)
        if n.strip().startswith("Kelly:"):
            kelly_note = n.strip()

    if kelly_note:
        print(f"  {V}  {c('💰', GRN)}  {c(kelly_note, GRN)}")
    else:
        _bankroll = float(os.environ.get("BANKROLL_USDC", 200.0))
        _cap = float(os.environ.get("KELLY_FRACTION_CAP", 0.25))
        print(
            f"  {V}  {c('💰', DIM)}  {c(f'Bankroll=${_bankroll:.0f} · Kelly cap={_cap:.0%} · (no trade → no sizing)', DIM)}"
        )

    if anomaly_notes:
        for an in anomaly_notes:
            print(f"  {V}  {c('⚠', YEL)}  {c(an.replace('⚠ ', ''), YEL)}")
    else:
        print(f"  {V}  {c('✅', GRN)}  {c('No anomalies detected', GRN)}")

    # ═══ SIGNAL DNA ═══
    section("🧬", "SIGNAL DNA (Tier 1)")

    fp = p.get("signal_fingerprint", "")
    dna_note = ""
    for n in notes_raw.split("; "):
        if "DNA:" in n:
            dna_note = n

    if fp:
        signals = fp.split("|")
        display = " ".join(
            c(s.split(":")[0], GRN if ":up" in s else RED if ":down" in s else DIM)
            + c(
                ":" + s.split(":")[1] if ":" in s else "",
                GRN if ":up" in s else RED if ":down" in s else DIM,
            )
            for s in signals
        )
        print(f"  {V}  Fingerprint    {display}")
    else:
        print(f"  {V}  Fingerprint    {c('(no directional signals)', DIM)}")
    if dna_note:
        print(f"  {V}  DNA Weight     {c(dna_note, MAG)}")

    # ═══ TRADE DECISION ═══
    section("🎯", "TRADE SCORING")

    sig_up = sc.get("signals_up", 0)
    sig_down = sc.get("signals_down", 0)
    direction = p.get("recommended_direction", "SKIP")
    confidence = p.get("confidence", 0)
    kelly = p.get("kelly_fraction", 0)
    entry = p.get("entry_price", 0)
    size = p.get("position_size_usd", 0)
    strategy = p.get("strategy", "skip")
    reasoning = p.get("reasoning", "")

    print(
        f"  {V}  {dir_str(direction)}  "
        f"Signals: {c(f'▲{sig_up:.1f}', GRN)} vs {c(f'▼{sig_down:.1f}', RED)}  "
        f"Strategy: {c(strategy, MAG)}"
    )

    conf_c = GRN if confidence >= 75 else (YEL if confidence >= 60 else RED)
    print(
        f"  {V}  Confidence     {gauge(confidence, 100, 15, conf_c)} {c(f'{confidence}/100', conf_c, bold=True)}  "
        f"Kelly: {c(f'{kelly:.4f}', WHT)}  "
        f"Size: {c(f'${size:.2f}', WHT, bold=True)}"
    )

    if entry > 0:
        payout = p.get("payout_ratio", 0)
        print(
            f"  {V}  Entry          {c(f'{entry:.3f}', WHT)}  "
            f"Payout: {c(f'{payout:.2f}x', CYN)}"
        )

    if reasoning and reasoning != "N/A":
        # Wrap long reasoning
        for i in range(0, len(reasoning), 68):
            prefix = "Reasoning" if i == 0 else "         "
            print(f"  {V}  {prefix}      {c(reasoning[i : i + 68], DIM)}")

    # ═══ RISK GATE ═══
    section("🛡️ ", "RISK GATE")

    action = d.get("action", "SKIP")
    r_reasoning = d.get("reasoning", "")

    if action in ("BUY_UP", "BUY_DOWN"):
        final_size = d.get("size_usd", 0)
        conf_val = d.get("confidence", 0)
        print(
            f"  {V}  {c(' ✅ APPROVED ', BLK, bold=True).replace(BLK, chr(27) + '[30m' + BG_GRN)}  "
            f"{dir_str(action)}  "
            f"Size: {c(f'${final_size:.2f}', WHT, bold=True)}  "
            f"Conf: {c(str(conf_val), GRN, bold=True)}"
        )
    else:
        print(
            f"  {V}  {c(' ⏭ SKIP ', BLK, bold=True).replace(BLK, chr(27) + '[30m' + BG_YEL)}  "
            f"{c('No trade this run', DIM)}"
        )
        violations = (
            r_reasoning.replace("VETO: ", "").split("; ")
            if "VETO" in r_reasoning
            else [r_reasoning]
        )
        for v in violations[:5]:
            if v.strip():
                print(f"  {V}    {c('•', RED)} {c(v.strip(), DIM)}")

    # ═══ SCORECARD NOTES ═══
    section("📝", "ALL SCORECARD NOTES")

    for note in notes_raw.split("; "):
        if note.strip():
            icon = (
                "⚠"
                if "ANOMALY" in note
                else "🧬"
                if "DNA" in note
                else "💰"
                if note.strip().startswith("Kelly:")
                else "🧠"
                if "Bayes" in note
                else "📊"
                if "SetupQ" in note
                else "•"
            )
            note_c = (
                YEL
                if "ANOMALY" in note
                else MAG
                if "DNA" in note
                else GRN
                if note.strip().startswith("Kelly:")
                else CYN
                if "Bayes" in note
                else DIM
            )
            print(f"  {V}  {icon} {c(note.strip(), note_c)}")

    # ═══ EXECUTION ═══
    exec_result = state.get("execution_result", {})
    if exec_result:
        section("⚡", "EXECUTION")
        status = exec_result.get("status", "unknown")
        if status == "simulated":
            print(
                f"  {V}  {c('✅ SIMULATED', GRN, bold=True)} @ {exec_result.get('price', '?')}  "
                f"Order: {c(exec_result.get('order_id', 'N/A'), DIM)}"
            )
        elif status == "aborted":
            print(
                f"  {V}  {c('🛑 ABORTED', RED, bold=True)} — {exec_result.get('reason', '?')}"
            )
        elif status == "skipped":
            print(f"  {V}  {c('⏭ SKIPPED', DIM)}")

    # ═══ PERFORMANCE ═══
    if perf:
        section("📈", "PERFORMANCE")
        print(f"  {V}  {json.dumps(perf, indent=2, default=str)[:200]}")

    # ═══ FOOTER ═══
    print(f"\n  {c(H * W, DIM)}")
    print(
        f"  {c('Run Status:', WHT, bold=True)} {result.status.value}  "
        f"{c('Workflow:', WHT, bold=True)} {result.workflow_id}  "
        f"{c(f'Elapsed: {elapsed:.0f}ms', DIM)}"
    )
    print()


if __name__ == "__main__":
    asyncio.run(main())
