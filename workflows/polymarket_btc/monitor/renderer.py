"""
ANSI dashboard renderer for the Polymarket BTC live monitor.

All visual/display logic lives here — zero domain logic.
Called by LiveMonitor with workflow result + PnLDisplay state.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime

from workflows.polymarket_btc.monitor.pnl_display import PnLDisplay

# ── ANSI Colors & Styles ────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITAL = "\033[3m"

# Foreground
BLK = "\033[30m"
RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
WHT = "\033[97m"

# Background
BG_RED = "\033[41m"
BG_GRN = "\033[42m"
BG_YEL = "\033[43m"
BG_BLU = "\033[44m"
BG_MAG = "\033[45m"
BG_CYN = "\033[46m"
BG_WHT = "\033[47m"
BG_DGRAY = "\033[100m"

# Box-drawing
H = "─"
V = "│"
TL = "┌"
TR = "┐"
BL = "└"
BR = "┘"
TT = "┬"
TB = "┴"
VL = "├"
VR = "┤"
CR = "┼"

# Double line
DH = "═"
DV = "║"
DTL = "╔"
DTR = "╗"
DBL = "╚"
DBR = "╝"

TAKER_FEE_PCT = 0.01


# ── Formatting Helpers ──────────────────────────────────────────────


def c(text: str, color: str, bold: bool = False) -> str:
    b = BOLD if bold else ""
    return f"{b}{color}{text}{RST}"


def pad(text: str, width: int, align: str = "left") -> str:
    """Pad text to width, stripping ANSI for length calc."""
    clean = re.sub(r"\033\[[0-9;]*m", "", str(text))
    diff = width - len(clean)
    if diff <= 0:
        return str(text)
    if align == "right":
        return " " * diff + str(text)
    elif align == "center":
        left = diff // 2
        right = diff - left
        return " " * left + str(text) + " " * right
    return str(text) + " " * diff


def spark(values: list[float], width: int = 20) -> str:
    """Sparkline from a list of numbers."""
    if not values or len(values) < 2:
        return ""
    chars = " ▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    recent = values[-width:]
    return "".join(chars[min(7, int((v - mn) / rng * 7.99))] for v in recent)


def gauge(
    value: float,
    max_val: float,
    width: int = 20,
    filled_color: str = GRN,
    empty_color: str = DIM,
) -> str:
    """Colored gauge bar."""
    pct = max(0, min(1, value / max_val)) if max_val > 0 else 0
    filled = int(pct * width)
    empty = width - filled
    return f"{filled_color}{'▰' * filled}{empty_color}{'▱' * empty}{RST}"


def delta_str(val: float, fmt: str = "+,.0f") -> str:
    """Colored delta string."""
    color = GRN if val > 0 else (RED if val < 0 else DIM)
    return c(f"{val:{fmt}}", color, bold=abs(val) > 50)


def pnl_str(val: float) -> str:
    """Colored PnL string."""
    if val > 0:
        return c(f"+${val:.2f}", GRN, bold=True)
    elif val < 0:
        return c(f"-${abs(val):.2f}", RED, bold=True)
    return c("$0.00", DIM)


def dir_str(direction: str) -> str:
    """Colored direction string."""
    if "UP" in str(direction):
        return c(f"▲ {direction}", GRN, bold=True)
    elif "DOWN" in str(direction):
        return c(f"▼ {direction}", RED, bold=True)
    return c(f"○ {direction}", YEL)


def regime_badge(regime: str) -> str:
    """Simple colored regime text."""
    colors = {"low": CYN, "normal": GRN, "high": YEL, "extreme": RED}
    return c(regime, colors.get(regime, WHT), bold=True)


def momentum_str(direction: str, magnitude: float) -> str:
    if direction == "up":
        arrow = "▲" if magnitude > 50 else "△"
        return c(f"{arrow}+{magnitude:.0f}", GRN, bold=magnitude > 100)
    elif direction == "down":
        arrow = "▼" if abs(magnitude) > 50 else "▽"
        return c(f"{arrow}{magnitude:.0f}", RED, bold=abs(magnitude) > 100)
    return c(f"─{magnitude:.0f}", DIM)


def box_line(left: str, fill: str, mid: str, right: str, width: int = 74) -> str:
    return f"{left}{fill * (width - 2)}{right}"


def section_header(icon: str, title: str, width: int = 74) -> str:
    title_str = f" {icon} {title} "
    remaining = width - 2 - len(title_str)
    return f"  {c(VL, CYN)}{c(title_str, CYN, bold=True)}{c(H * remaining + VR, CYN)}"


# ── Main Renderer ────────────────────────────────────────────────────


def render(
    data: dict,
    history: list[dict],
    elapsed_ms: float,
    tracker: PnLDisplay,
    start_time: float,
    btc_history: list[float],
) -> str:
    W = 78
    now_str = datetime.now().strftime("%H:%M:%S")
    uptime = int(time.time() - start_time)
    h_u, m_u, s_u = uptime // 3600, (uptime % 3600) // 60, uptime % 60
    uptime_str = f"{h_u:02d}h{m_u:02d}m{s_u:02d}s"
    run_num = data["run"]
    m = data.get("market", {})
    p = data.get("proposal", {})
    d = data.get("decision", {})

    bankroll = float(os.environ.get("BANKROLL_USDC", 200.0))
    kelly_cap = float(os.environ.get("KELLY_FRACTION_CAP", 0.25))
    max_exp = float(os.environ.get("MAX_EXPOSURE_PER_WINDOW", 50.0))

    lines = []

    def L(s: str = "") -> None:
        lines.append(s)

    def HR() -> None:
        L(f"  {c('─' * (W - 2), DIM)}")

    def sec(icon: str, title: str) -> None:
        bar = "─" * max(0, W - 6 - len(title))
        L(f"  {c('├' + f' {icon} {title} ' + bar + '┤', CYN)}")

    action_now = d.get("action", "SKIP")
    if action_now in ("BUY_UP", "BUY_DOWN"):
        bot_badge = c(f" ▶ {action_now} ", BLK, bold=True).replace(
            BLK, f"\033[30m{BG_GRN}"
        )
    else:
        bot_badge = c(" ● MONITORING ", BLK, bold=True).replace(
            BLK, f"\033[30m{BG_DGRAY}"
        )

    # ══════════════════════════════════════════════════════
    # 1. STATUS HEADER
    # ══════════════════════════════════════════════════════
    L()
    L(f"  {c(DTL + DH * (W - 2) + DTR, CYN, bold=True)}")
    L(
        f"  {c(DV, CYN, bold=True)} 🔮  {c('POLYMARKET BTC', WHT, bold=True)}  {bot_badge}"
        f"  {c(f'Run #{run_num}', DIM)}  {c(f'{elapsed_ms:.0f}ms', DIM)}"
        f"  {c(now_str, WHT, bold=True)}  {c('↑' + uptime_str, DIM)}"
        f"  {c(DV, CYN, bold=True)}"
    )
    L(
        f"  {c(DV, CYN, bold=True)}  "
        f"{c(f'Bankroll ${bankroll:.0f}', YEL)}  "
        f"{c('│', DIM)}  "
        f"{c(f'Kelly {kelly_cap:.0%}', YEL)}  "
        f"{c('│', DIM)}  "
        f"{c(f'Window cap ${max_exp:.0f}', YEL)}  "
        f"{c('│', DIM)}  "
        f"{c('5-Layer: DNA · Confluence · Bayesian · Kelly · Anti-Fragile', DIM)}"
        f"  {c(DV, CYN, bold=True)}"
    )
    L(f"  {c(DBL + DH * (W - 2) + DBR, CYN, bold=True)}")

    if data.get("error"):
        L(f"  {c(' ❌ ERROR: ' + str(data['error']), RED, bold=True)}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════
    # 2. 💼 PORTFOLIO / CAPITAL OVERVIEW
    # ══════════════════════════════════════════════════════
    L()
    wr = tracker.win_rate
    wr_c = GRN if wr >= 60 else (YEL if wr >= 50 else RED)
    pnl_v = tracker.total_pnl
    cap = tracker.capital
    ret_pct = tracker.capital_return_pct
    dd_pct = tracker.drawdown_pct
    dd_usd = tracker.peak_capital - cap

    cap_bg = BG_GRN if pnl_v > 0 else (BG_RED if pnl_v < 0 else BG_DGRAY)
    ret_sign = "+" if ret_pct >= 0 else ""
    pnl_sign = "+" if pnl_v >= 0 else ""
    costs = tracker.total_fees + tracker.total_slippage_cost

    eq_spark = (
        spark(tracker.equity_history[-40:], 40)
        if len(tracker.equity_history) >= 2
        else ""
    )
    eq_clr = GRN if pnl_v >= 0 else RED
    dd_c = GRN if dd_pct < 2 else (YEL if dd_pct < 5 else RED)

    L(
        f"  \033[97m{BOLD}{cap_bg}  💼 Capital: ${cap:.2f}  ({ret_sign}{ret_pct:.2f}%)  {RST}  "
        f"{c(f'Start ${tracker.initial_capital:.0f}', DIM)}  "
        f"{c('→', DIM)}  "
        f"{c(f'PnL {pnl_sign}${pnl_v:.2f}', eq_clr, bold=True)}  "
        f"{c('│', DIM)}  "
        f"{c(f'W: {tracker.wins}', GRN, bold=True)} / "
        f"{c(f'L: {tracker.losses}', RED, bold=True)}  "
        f"WR: {c(f'{wr:.0f}%', wr_c, bold=True)}  "
        f"Trades: {c(str(tracker.total_trades), WHT, bold=True)}  "
        f"{c(f'⏳ {tracker.pending_count} pending', YEL)}"
    )
    L(
        f"    {c('Equity:', DIM)} {c(eq_spark, eq_clr)}  "
        f"{c('│', DIM)}  "
        f"Peak: {c(f'${tracker.peak_capital:.2f}', CYN)}  "
        f"DD: {c(f'{dd_pct:.1f}% (${dd_usd:.2f})', dd_c, bold=dd_pct > 5)}  "
        f"{c('│', DIM)}  "
        f"MaxDD: {c(f'${tracker.max_drawdown:.2f}', RED if tracker.max_drawdown > 5 else DIM)}"
    )
    L(
        f"    {c('Fees:', DIM)} {c(f'${tracker.total_fees:.2f}', DIM)}  "
        f"{c('+', DIM)} {c('Slip:', DIM)} {c(f'${tracker.total_slippage_cost:.2f}', DIM)}  "
        f"= Costs: {c(f'${costs:.2f}', RED)}  "
        f"{c('Gross:', DIM)} {pnl_str(tracker.gross_pnl)}"
    )

    # ══════════════════════════════════════════════════════
    # 3. 📊 MARKET INTELLIGENCE
    # ══════════════════════════════════════════════════════
    L()
    sec("📊", "MARKET")

    price = m.get("price", 0)
    vol_d = m.get("volatility", {})
    regime = vol_d.get("regime", "?")
    atr = vol_d.get("atr_pct", 0)
    duration = m.get("selected_duration", "?")
    mkt = m.get("market", {})
    elapsed_w = mkt.get("elapsed", 0)
    remaining = mkt.get("remaining", 0)
    total_w = (elapsed_w + remaining) if (elapsed_w + remaining) > 0 else 300
    should_t = mkt.get("should_trade", False)
    up_p = mkt.get("up_price", 0.5)
    down_p = mkt.get("down_price", 0.5)
    edge_decay = mkt.get("edge_decay", 1.0)
    zone = mkt.get("timing_zone", "alpha")
    ob = m.get("order_book", {})
    spread = ob.get("spread", 0)

    btc_delta = (btc_history[-1] - btc_history[-2]) if len(btc_history) >= 2 else 0
    sess_delta = (btc_history[-1] - btc_history[0]) if len(btc_history) >= 2 else 0

    L(
        f"  {V}  BTC {c(f'${price:,.2f}', WHT, bold=True)}  "
        f"{delta_str(btc_delta, '+,.2f')} tick  {delta_str(sess_delta)} session  "
        f"Regime: {regime_badge(regime)} {c(f'ATR {atr:.4f}%', DIM)}  "
        f"{c(duration, CYN, bold=True)}"
    )

    if len(btc_history) > 2:
        spk = spark(btc_history[-60:], 44)
        spk_clr = CYN if sess_delta >= 0 else RED
        L(f"  {V}  {c(spk, spk_clr)}")

    zone_c = GRN if zone == "alpha" else (YEL if zone == "confirm" else RED)
    decay_c = GRN if edge_decay > 0.7 else (YEL if edge_decay > 0.4 else RED)
    tbadge = (
        c(" ✓ TRADEABLE ", BLK, bold=True).replace(BLK, f"\033[30m{BG_GRN}")
        if should_t
        else c(" ✗ WAIT ", BLK, bold=True).replace(BLK, f"\033[30m{BG_YEL}")
    )
    L(
        f"  {V}  Window  {gauge(elapsed_w, total_w, 30, zone_c)}  "
        f"{c(f'{elapsed_w}s', WHT)}/{c(f'{total_w}s', DIM)}  "
        f"Zone: {c(zone.upper(), zone_c, bold=True)}  "
        f"Decay: {c(f'{edge_decay:.2f}', decay_c, bold=True)}  "
        f"{tbadge}"
    )

    up_bid = ob.get("best_bid", 0)
    up_ask = ob.get("best_ask", 1)
    dn_bid = ob.get("down_best_bid", 0)
    dn_ask = ob.get("down_best_ask", 1)
    impl_mid = ob.get("midpoint", up_p)
    delta_fv = impl_mid - 0.50
    bias_str = (
        c(f"UP FAVORED  Δ={delta_fv:+.3f}", GRN, bold=True)
        if delta_fv > 0.03
        else c(f"DOWN FAVORED Δ={delta_fv:+.3f}", RED, bold=True)
        if delta_fv < -0.03
        else c(f"NEUTRAL      Δ={delta_fv:+.3f}", DIM)
    )
    sp_c = RED if spread > 0.10 else (YEL if spread > 0.05 else GRN)
    L(
        f"  {V}  UP {c(f'{up_p:.4f}', GRN, bold=True)} {c(f'b:{up_bid:.3f} a:{up_ask:.3f}', DIM)}  "
        f"DN {c(f'{down_p:.4f}', RED, bold=True)} {c(f'b:{dn_bid:.3f} a:{dn_ask:.3f}', DIM)}  "
        f"Spd: {c(f'{spread:.3f}', sp_c)}  {bias_str}"
    )

    # ══════════════════════════════════════════════════════
    # 4. ⚡ SIGNAL CONFLUENCE ENGINE
    # ══════════════════════════════════════════════════════
    L()
    sec("⚡", "SIGNAL CONFLUENCE ENGINE")

    mom = m.get("momentum", {})
    ta = m.get("ta_indicators", {})
    mtf = m.get("multi_timeframe", {})
    deriv = m.get("derivatives", {})
    scorecard = p.get("scorecard", {})
    scnotes = scorecard.get("scorecard_notes", "")

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
    vwap = ta.get("vwap_deviation", 0)
    fb = deriv.get("funding_bias", "neutral")
    fb_c = GRN if "bullish" in fb else (RED if "bearish" in fb else DIM)
    oi = deriv.get("oi_trend", "stable")
    oi_c = GRN if "rising" in oi else (RED if "falling" in oi else DIM)
    ma = mtf.get("alignment", "?")
    ma_c = GRN if "bullish" in ma else (RED if "bearish" in ma else YEL)

    L(
        f"  {V}  {c('L1A', CYN, bold=True)} {c('Binance TA', WHT, bold=True)}  "
        f"Mom: {m1} / {m3}  "
        f"RSI {c(f'{rsi:.0f}', rsi_c, bold=True)}  "
        f"MACD {c(f'{macd:+.2f}', GRN if macd > 0 else RED)}  "
        f"BB {c(f'{bb:.2f}', WHT)}  "
        f"VWAP {c(f'{vwap:+.3f}', GRN if vwap > 0.1 else RED if vwap < -0.1 else DIM)}"
    )
    fr_val = deriv.get("funding_rate", 0)
    L(
        f"  {V}       Fund: {c(f'{fr_val:.4f}', fb_c)} "
        f"({c(fb, fb_c, bold=True)})  "
        f"OI: {c(oi, oi_c, bold=True)}  "
        f"MTF: {c(ma, ma_c, bold=True)}"
    )

    iwt = m.get("intra_window_trend", {})
    lt = m.get("last_trade", {})
    dc = m.get("depth_clustering", {})
    vol = m.get("market_volume", {})
    vpin = m.get("vpin", {})

    iwt_d = iwt.get("direction", "flat")
    iwt_s = iwt.get("strength", 0)
    iwt_c = GRN if iwt_d == "up" else (RED if iwt_d == "down" else DIM)
    lt_side = lt.get("side", "?")
    lt_px = lt.get("price", 0)
    bw = dc.get("bid_wall_pct", 0)
    aw = dc.get("ask_wall_pct", 0)
    vol_v = vol.get("volume", 0)
    liq_v = vol.get("liquidity", 0)

    vpin_val = vpin.get("vpin_score", 0)
    vpin_dir = vpin.get("direction", "neutral")
    vpin_c = GRN if vpin_dir == "bullish" else (RED if vpin_dir == "bearish" else DIM)
    vpin_str = c(f"{vpin_val:.0f}", vpin_c, bold=vpin_val > 60)

    wall_disp = ""
    if bw >= 0.30:
        wall_disp += c(f" 🔵BID:{bw:.0%}", GRN, bold=True)
    if aw >= 0.30:
        wall_disp += c(f" 🔴ASK:{aw:.0%}", RED, bold=True)
    if not wall_disp:
        wall_disp = c("No walls", DIM)

    L(
        f"  {V}  {c('L1B', MAG, bold=True)} {c('Polymarket Flow', WHT, bold=True)}  "
        f"VPIN: {vpin_str}  "
        f"IWT: {c(f'{iwt_d}({iwt_s:.2f})', iwt_c, bold=True)}  "
        f"Last: {c(f'{lt_side}@{lt_px:.3f}', GRN if lt_side == 'BUY' else RED if lt_side == 'SELL' else DIM)}  "
        f"{wall_disp}  {c(f'Vol ${vol_v:,.0f}', DIM)}  {c(f'Liq ${liq_v:,.0f}', DIM)}"
    )

    conf_text = bin_note = pm_note = ""
    for note in scnotes.split("; "):
        if "Confluence:" in note:
            conf_text = note.replace("Confluence: ", "")
        if note.startswith("Binance:"):
            bin_note = note
        if note.startswith("Polymarket:"):
            pm_note = note
    cf_c = (
        GRN
        if "FULL" in conf_text
        else YEL
        if "only" in conf_text.lower()
        else RED
        if "CONFLICT" in conf_text
        else DIM
    )
    L(
        f"  {V}  {c('L2', WHT, bold=True)}  {c(conf_text or 'No confluence', cf_c, bold=True)}"
    )
    if bin_note and pm_note:
        L(f"  {V}       {c(bin_note, DIM)}  {c('│', DIM)}  {c(pm_note, DIM)}")

    fp = p.get("signal_fingerprint", "")
    dna_note = next((n.strip() for n in scnotes.split("; ") if "DNA:" in n), "")
    if fp:
        sigs = fp.split("|")
        fp_disp = " ".join(
            c(s.split(":")[0], GRN if ":up" in s else RED if ":down" in s else DIM)
            + c(
                ":" + s.split(":")[1] if ":" in s else "",
                GRN if ":up" in s else RED if ":down" in s else DIM,
            )
            for s in sigs[:10]
        )
        if len(sigs) > 10:
            fp_disp += c(f" +{len(sigs) - 10}", DIM)
        L(f"  {V}  {c('🧬 DNA', MAG, bold=True)}  {fp_disp}")
    if dna_note:
        L(f"  {V}       {c(dna_note, MAG)}")

    # ══════════════════════════════════════════════════════
    # 5. 🎯 BTC-NATIVE SIGNALS (Tier 2)
    # ══════════════════════════════════════════════════════
    L()
    sec("🎯", "BTC-NATIVE SIGNALS (Tier 2)")

    liq_data = m.get("liquidation", {})
    carry_d = m.get("pre_window_carry", {})
    streak_d = m.get("window_streak", {})

    liq_lv = liq_data.get("cascade_level", "none")
    liq_dir = liq_data.get("net_direction", "neutral")
    liq_long = liq_data.get("long_liq_usd", 0)
    liq_short = liq_data.get("short_liq_usd", 0)
    liq_cnt = liq_data.get("count", 0)
    liq_bg = (
        BG_RED if liq_lv == "major" else (BG_YEL if liq_lv == "minor" else BG_DGRAY)
    )
    liq_badge = f"{BOLD}\033[30m{liq_bg} {liq_lv.upper()} {RST}"
    liq_dc = GRN if liq_dir == "up" else (RED if liq_dir == "down" else DIM)

    L(
        f"  {V}  🔥 Liquidations  {liq_badge}  "
        f"Longs: {c(f'${liq_long:,.0f}', RED if liq_long > 0 else DIM)}  "
        f"Shorts: {c(f'${liq_short:,.0f}', GRN if liq_short > 0 else DIM)}  "
        f"→ {c(liq_dir, liq_dc, bold=True)}  "
        f"{c(f'({liq_cnt} orders)', DIM)}"
    )

    carry_dir = carry_d.get("direction", "neutral")
    carry_s = carry_d.get("strength", 0)
    carry_c = GRN if carry_dir == "up" else (RED if carry_dir == "down" else DIM)
    bear_bars = carry_d.get("consecutive_bearish", 0)
    bull_bars = carry_d.get("consecutive_bullish", 0)
    L(
        f"  {V}  🌊 Pre-window Carry  "
        f"{c(carry_dir.upper(), carry_c, bold=True)}  "
        f"{gauge(carry_s, 1.0, 12, carry_c)} {c(f'{carry_s:.3f}', carry_c)}  "
        f"{c(f'▲{bull_bars}', GRN)}/{c(f'▼{bear_bars}', RED)} consecutive candles"
    )

    outcomes = streak_d.get("last_outcomes", [])
    s_len = streak_d.get("streak_length", 0)
    s_dir = streak_d.get("streak_direction", "neutral")
    s_pat = streak_d.get("pattern", "mixed")
    dots = "".join(c("●", GRN) if o == "W" else c("●", RED) for o in outcomes)
    s_c = GRN if s_dir == "winning" else (RED if s_dir == "losing" else DIM)
    pat_c = RED if s_pat == "choppy" else (GRN if s_pat == "trend" else YEL)
    L(
        f"  {V}  📊 Outcome Streak  "
        f"{dots or c('(no trades yet)', DIM)}  "
        f"{c(str(s_len) + ' ' + s_dir, s_c, bold=True)}  "
        f"Pattern: {c(s_pat, pat_c, bold=True)}"
    )

    # ══════════════════════════════════════════════════════
    # 6. 🧠 BAYESIAN DECISION ENGINE
    # ══════════════════════════════════════════════════════
    L()
    sec("🧠", "BAYESIAN DECISION ENGINE")

    setup_q = p.get("setup_quality", 0)
    sq_c = GRN if setup_q >= 70 else (YEL if setup_q >= 40 else RED)
    sq_label = (
        c(" A+ ", BLK, bold=True).replace(BLK, f"\033[30m{BG_GRN}")
        if setup_q >= 70
        else c(" OK ", BLK, bold=True).replace(BLK, f"\033[30m{BG_YEL}")
        if setup_q >= 40
        else c(" SKIP ", BLK, bold=True).replace(BLK, f"\033[30m{BG_RED}")
    )
    L(
        f"  {V}  Setup Quality  {gauge(setup_q, 100, 24, sq_c)} "
        f"{c(f'{setup_q:3d}/100', sq_c, bold=True)}  {sq_label}"
    )

    bayes_note = next((n.strip() for n in scnotes.split("; ") if "Bayes:" in n), "")
    kelly_note = next(
        (n.strip() for n in scnotes.split("; ") if n.strip().startswith("Kelly:")), ""
    )
    anomalies = [n.strip() for n in scnotes.split("; ") if "ANOMALY:" in n]
    setq_note = next((n.strip() for n in scnotes.split("; ") if "SetupQ:" in n), "")
    highentry_note = next(
        (n.strip() for n in scnotes.split("; ") if "High-entry:" in n), ""
    )

    if bayes_note:
        L(f"  {V}  Bayesian       {c(bayes_note, CYN)}")
    if setq_note:
        L(f"  {V}  Quality Detail {c(setq_note, DIM)}")
    if kelly_note:
        L(f"  {V}  Kelly Sizing   {c(kelly_note, GRN, bold=True)}")
    else:
        L(
            f"  {V}  Kelly Sizing   {c(f'Bankroll=${bankroll:.0f} · cap={kelly_cap:.0%} · (no trade → no sizing)', DIM)}"
        )
    if highentry_note:
        L(f"  {V}  {c('⚠ Entry too high', YEL, bold=True)}  {c(highentry_note, YEL)}")
    if anomalies:
        for an in anomalies:
            L(
                f"  {V}  {c('⚠ ANOMALY', YEL, bold=True)}  {c(an.replace('⚠ ANOMALY: ', ''), YEL)}"
            )
    else:
        L(
            f"  {V}  {c('✅ No anomalies', GRN)}  Signal integrity confirmed — all 6 anti-fragile rules passed"
        )

    # ══════════════════════════════════════════════════════
    # 7. 🎰 TRADE SCORING
    # ══════════════════════════════════════════════════════
    L()
    sec("🎰", "TRADE SCORING")

    direction = p.get("recommended_direction", "SKIP")
    confidence = p.get("confidence", 0)
    kelly_f = p.get("kelly_fraction", 0)
    entry = p.get("entry_price", 0)
    size = p.get("position_size_usd", 0)
    strategy = p.get("strategy", "skip")
    reasoning = p.get("reasoning", "")
    payout = p.get("payout_ratio", 0)
    sig_up = scorecard.get("signals_up", 0)
    sig_dn = scorecard.get("signals_down", 0)

    conf_c = GRN if confidence >= 75 else (YEL if confidence >= 60 else RED)
    L(
        f"  {V}  {dir_str(direction)}  "
        f"Conf: {gauge(confidence, 100, 14, conf_c)} {c(f'{confidence}/100', conf_c, bold=True)}  "
        f"Signals: {c(f'▲{sig_up:.1f}', GRN)} vs {c(f'▼{sig_dn:.1f}', RED)}  "
        f"Strategy: {c(strategy, MAG, bold=True)}"
    )

    if entry > 0:
        eff_entry = entry + (spread / 2)
        fee_est = size * TAKER_FEE_PCT if size > 0 else 0
        net_exposure = size + fee_est
        net_c = RED if net_exposure > 30 else (YEL if net_exposure > 10 else GRN)
        entry_warn = c(f" ⚠ >{0.70:.0%} cap", YEL) if entry > 0.70 else ""
        L(
            f"  {V}  Entry: {c(f'{entry:.4f}', WHT, bold=True)}{entry_warn} → eff: {c(f'{eff_entry:.4f}', YEL)} {c('(+slip)', DIM)}  "
            f"Payout: {c(f'{payout:.3f}x', CYN)}  "
            f"f*= {c(f'{kelly_f:.4f}', WHT)} {c(f'({kelly_f * 100:.1f}%)', DIM)}"
        )
        if size > 0:
            L(
                f"  {V}  Size:  {c(f'${size:.2f}', WHT, bold=True)}  "
                f"Fee: {c(f'${fee_est:.2f}', DIM)}  "
                f"Net exposure: {c(f'${net_exposure:.2f}', net_c)}"
            )
    if reasoning:
        max_w = W - 20
        reason_color = DIM if direction != "SKIP" else YEL
        prefix = "" if direction != "SKIP" else c("⚑ Skip reason: ", YEL, bold=True)
        L(f"  {V}  {prefix}{c(reasoning[:max_w], reason_color)}")

    # ══════════════════════════════════════════════════════
    # 8. 🛡️ RISK GATE
    # ══════════════════════════════════════════════════════
    L()
    sec("🛡️", "RISK GATE  (11 Hard Rules)")

    rg_action = d.get("action", "SKIP")
    rg_reasoning = d.get("reasoning", "")

    if rg_action in ("BUY_UP", "BUY_DOWN"):
        final_size = d.get("size_usd", 0)
        conf_val = d.get("confidence", 0)
        L(
            f"  {V}  {c(' ✅ ALL 11 RULES PASSED ', BLK, bold=True).replace(BLK, chr(27) + '[30m' + BG_GRN)}  "
            f"{dir_str(rg_action)}  "
            f"Final: {c(f'${final_size:.2f}', WHT, bold=True)}  "
            f"Conf: {c(str(conf_val), GRN, bold=True)}"
        )
        L(f"  {V}  {c(rg_reasoning[: W - 6], DIM)}")
    else:
        L(
            f"  {V}  {c(' ⏭ VETO / SKIP ', BLK, bold=True).replace(BLK, chr(27) + '[30m' + BG_YEL)}"
        )
        violations = (
            rg_reasoning.replace("VETO: ", "").split("; ")
            if "VETO" in rg_reasoning
            else [rg_reasoning]
        )
        for v in violations[:6]:
            if v.strip():
                L(f"  {V}    {c('▸', RED)} {c(v.strip(), DIM)}")
        proposal_skip = direction == "SKIP" and reasoning
        rg_conf_only = all(
            ("Confidence" in v and "< 75" in v) or not v.strip() for v in violations
        )
        if proposal_skip and rg_conf_only:
            L(f"  {V}    {c('▸ Root cause:', YEL)} {c(reasoning[: W - 24], YEL)}")

    # ══════════════════════════════════════════════════════
    # 9. 📋 TRADE LOG (last 10)
    # ══════════════════════════════════════════════════════
    L()
    sec("📋", "TRADE LOG")

    if not tracker.trades:
        L(f"  {V}  {c('Waiting for first approved trade...', DIM)}")
    else:
        L(
            f"  {V}  {c('  Time    Dir.    Size     Entry→Eff.       Slip  Fee   Result', DIM)}"
        )
        HR()
        for t in tracker.trades[-10:]:
            icon = (
                "⏳" if not t["resolved"] else ("✅" if t.get("pnl", 0) > 0 else "❌")
            )
            eff = t.get("effective_entry", t["entry_price"])
            fee = t.get("fee", 0)
            slip = t.get("slippage", 0)
            t_mid = t["entry_price"]
            t_act = t["action"]
            t_size = t["size_usd"]
            act_str = (
                c("▲ UP  ", GRN, bold=True)
                if "UP" in t_act
                else c("▼ DOWN", RED, bold=True)
                if "DOWN" in t_act
                else c("○ SKIP", DIM)
            )

            if t["resolved"]:
                outcome = t.get("outcome", "?")
                t_pnl = t["pnl"]
                res_str = f"{pnl_str(t_pnl)} {c(outcome, GRN if outcome == 'WON' else RED, bold=True)}"
            else:
                wcs = max(0, int(t["window_end_ts"] - time.time()))
                rcs = max(0, int(t["window_end_ts"] + 90 - time.time()))
                res_str = (
                    c(f"⏱  {wcs:3d}s to close", YEL)
                    if wcs > 0
                    else c(f"🔄 ~{rcs:3d}s on-chain", MAG)
                    if rcs > 0
                    else c("🔍 resolving...", MAG)
                )

            L(
                f"  {V}  {icon} {c(t['time'], DIM)} {act_str} "
                f"{c(f'${t_size:.2f}', WHT):>8}  "
                f"{c(f'{t_mid:.4f}', WHT)}→{c(f'{eff:.4f}', YEL)}  "
                f"{c(f'+{slip:.3f}', DIM):>6}  {c(f'${fee:.2f}', DIM):>5}  {res_str}"
            )

    # ══════════════════════════════════════════════════════
    # 10. 📜 SESSION HISTORY
    # ══════════════════════════════════════════════════════
    L()
    sec("📜", "SESSION HISTORY")

    last_acts = []
    for h in history[-60:]:
        a = h.get("decision", {}).get("action", "SKIP")
        last_acts.append(
            c("▲", GRN)
            if "UP" in str(a)
            else c("▼", RED)
            if "DOWN" in str(a)
            else c("·", DIM)
        )
    L(f"  {V}  Last 60:  {''.join(last_acts)}")

    if len(btc_history) >= 2:
        total_d = btc_history[-1] - btc_history[0]
        L(
            f"  {V}  BTC  {c(f'${btc_history[0]:,.0f}', DIM)} → "
            f"{c(f'${btc_history[-1]:,.0f}', WHT, bold=True)}  "
            f"({delta_str(total_d)})  "
            f"{c(f'{len(history)} runs', WHT, bold=True)}  "
            f"{c(f'{len(tracker.trades)} trades', WHT)}"
        )

    # ══════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════
    L()
    HR()
    L(
        f"  {c('  5-Layer: DNA · Confluence · Bayesian · Kelly · Anti-Fragile', CYN)}  "
        f"{c('│', DIM)}  "
        f"{c(f'Bankroll ${bankroll:.0f}', YEL)}  "
        f"{c(f'Kelly {kelly_cap:.0%}', YEL)}  "
        f"{c(f'Cap ${max_exp:.0f}', YEL)}"
    )
    L(
        f"  {c('  REAL data · signals · resolution', DIM)}  "
        f"{c('│', DIM)}  "
        f"{c('SIM slippage · fees · fill', YEL)}  "
        f"{c('│', DIM)}  "
        f"{c('Cooldown 20s · Resolve 90s', DIM)}  "
        f"{c('│', DIM)}  "
        f"{c('Ctrl+C stop', DIM)}"
    )

    return "\n".join(lines)
