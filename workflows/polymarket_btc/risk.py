"""
Risk management module for the Polymarket BTC trading workflow.

State is persisted via the platform's ``ArtifactService`` with a fixed
namespace so it survives across pipeline runs (Cloud Run deploys).

Enforces:
- Per-trade position size limits
- Daily loss cap (circuit breaker)
- Cooldown after consecutive losses
"""

import time
from dataclasses import dataclass, asdict, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_WORKFLOW_NAME = "polymarket_btc"
_STATE_KEY = "risk_state"


@dataclass
class RiskState:
    """Persistent risk management state."""

    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    last_trade_time: float = 0.0
    last_reset_day: str = ""
    paused_until: float = 0.0
    last_traded_window_start: int = 0  # Prevents double-trading same window
    # Process trade gate (persisted for cross-restart durability)
    trades_per_window: dict = field(default_factory=dict)  # condition_id → trade count
    direction_per_window: dict = field(
        default_factory=dict
    )  # condition_id → locked direction


class RiskManager:
    """
    Enforces risk limits for the BTC trading agent.

    State is persisted via WorkflowStateService for cross-run durability.
    All limits are configurable via constructor parameters,
    which map to manifest.yaml settings.
    """

    def __init__(
        self,
        *,
        max_trade_size_usd: float = 20.0,
        daily_loss_cap_usd: float = 100.0,
        min_conviction_score: int = 65,
        max_consecutive_losses: int = 3,
        cooldown_seconds: int = 900,  # 15 minutes
        bankroll_usdc: float = 200.0,
        kelly_fraction_cap: float = 0.25,  # Quarter-Kelly — proven safest for live trading
    ):
        self.max_trade_size = max_trade_size_usd
        self.daily_loss_cap = daily_loss_cap_usd
        self.min_conviction = min_conviction_score
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_seconds = cooldown_seconds
        self.bankroll_usdc = bankroll_usdc
        self.kelly_fraction_cap = kelly_fraction_cap

        self._state = RiskState()

    @property
    def state(self) -> RiskState:
        return self._state

    # ── Platform Persistence (WorkflowStateService) ───────────────

    async def load(self) -> None:
        """Load risk state from WorkflowStateService."""
        from autopilot.core.workflow_state import WorkflowStateService

        state_svc = WorkflowStateService(_WORKFLOW_NAME)

        try:
            data = await state_svc.get(_STATE_KEY)
            if data:
                self._state = RiskState(**data)
                logger.debug("risk_state_loaded", state=asdict(self._state))
        except Exception as e:
            logger.warning("risk_state_load_failed", error=str(e), exc_info=True)

    async def save(self) -> None:
        """Persist risk state to WorkflowStateService."""
        from autopilot.core.workflow_state import WorkflowStateService

        state_svc = WorkflowStateService(_WORKFLOW_NAME)

        try:
            await state_svc.put(_STATE_KEY, asdict(self._state))
            logger.debug("risk_state_saved")
        except Exception as e:
            logger.warning("risk_state_save_failed", error=str(e), exc_info=True)

    def update_bankroll(self, capital: float) -> None:
        """Update bankroll to reflect current live portfolio capital.

        Capital is dynamic runtime state — call this before every Kelly
        computation so position sizes always scale with actual equity.

        Args:
            capital: Current portfolio capital in USDC. Ignored if < $1.
        """
        if capital > 1.0:
            self.bankroll_usdc = capital
            logger.debug("bankroll_updated", bankroll=round(capital, 2))

    # ── Daily Reset ───────────────────────────────────────────────

    def reset_daily(self, today: str) -> None:
        """Reset daily counters if it's a new day."""
        if self._state.last_reset_day != today:
            logger.info(
                "risk_daily_reset",
                previous_day=self._state.last_reset_day,
                previous_pnl=self._state.daily_pnl,
                previous_trades=self._state.daily_trades,
            )
            self._state.daily_pnl = 0.0
            self._state.daily_trades = 0
            self._state.consecutive_losses = 0
            self._state.last_reset_day = today
            self._state.paused_until = 0.0

    # ── Risk Checks ───────────────────────────────────────────────

    # Intra-window cooldown: 20s between trades (consolidated from monitor).
    # Fast enough for re-evaluation in 5m windows (tradeable zone=135s → ~6 entries max).
    # Rule 10 (per-window exposure cap) is the hard ceiling on total risk.
    INTRA_WINDOW_COOLDOWN = 20
    MAX_TRADES_PER_WINDOW = 3

    def check_trade_allowed(
        self,
        conviction_score: float,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed given current risk state.

        Args:
            conviction_score: Minimum conviction threshold.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        now = time.time()

        # 0. Intra-window cooldown — prevent rapid-fire at 1s cadence
        #    while allowing scale-in (pilot → confirm → add).
        #    Rule 10 (per-window exposure cap) is the hard ceiling.
        since_last = now - self._state.last_trade_time
        if self._state.last_trade_time > 0 and since_last < self.INTRA_WINDOW_COOLDOWN:
            remaining = int(self.INTRA_WINDOW_COOLDOWN - since_last)
            reason = (
                f"Intra-window cooldown — {remaining}s remaining "
                f"(min {self.INTRA_WINDOW_COOLDOWN}s between trades)"
            )
            logger.info(
                "risk_blocked_intra_cooldown",
                since_last=round(since_last, 1),
                cooldown=self.INTRA_WINDOW_COOLDOWN,
            )
            return False, reason

        # 1. Cooldown check (consecutive losses)
        if now < self._state.paused_until:
            remaining = int(self._state.paused_until - now)
            reason = f"Cooldown active — {remaining}s remaining after {self.max_consecutive_losses} consecutive losses"
            logger.info("risk_blocked_cooldown", remaining_s=remaining)
            return False, reason

        # 2. Daily loss cap
        if (
            abs(self._state.daily_pnl) >= self.daily_loss_cap
            and self._state.daily_pnl < 0
        ):
            reason = f"Daily loss cap hit: ${abs(self._state.daily_pnl):.2f} >= ${self.daily_loss_cap:.2f}"
            logger.warning("risk_blocked_daily_cap", daily_pnl=self._state.daily_pnl)
            return False, reason

        # 3. Conviction threshold
        if conviction_score < self.min_conviction:
            reason = (
                f"Conviction too low: {conviction_score:.1f} < {self.min_conviction}"
            )
            logger.info("risk_blocked_conviction", score=conviction_score)
            return False, reason

        return True, "Trade allowed"

    # ── Window Gate (direction lock + cap) ─────────────────────────

    def check_window_gate(
        self,
        action: str,
        condition_id: str,
    ) -> tuple[bool, str]:
        """Direction lock + window trade cap.

        Args:
            action: Trade action — 'BUY_UP' or 'BUY_DOWN'.
            condition_id: Market condition ID for the current window.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        trades = self._state.trades_per_window.get(condition_id, 0)
        locked = self._state.direction_per_window.get(condition_id)

        # Cap: max 3 trades per window per market
        if trades >= self.MAX_TRADES_PER_WINDOW:
            return False, (
                f"Window cap hit: {trades}/{self.MAX_TRADES_PER_WINDOW} trades "
                f"on {condition_id[:8]}…"
            )

        # Direction lock: once committed, no flip within same window
        if locked and locked != action:
            return False, (
                f"Direction locked to {locked} — cannot flip to {action} "
                f"in same window ({condition_id[:8]}…)"
            )

        return True, "Window gate passed"

    def record_window_trade(self, action: str, condition_id: str) -> None:
        """Increment window counter and lock direction after gate passes."""
        self._state.trades_per_window[condition_id] = (
            self._state.trades_per_window.get(condition_id, 0) + 1
        )
        self._state.direction_per_window[condition_id] = action
        logger.info(
            "risk_window_trade_recorded",
            action=action,
            condition_id=condition_id,
            trades_in_window=self._state.trades_per_window[condition_id],
        )

    # ── Bayesian Kelly Position Sizing ────────────────────────────

    def compute_kelly_size(
        self,
        p: float,
        entry_price: float,
        *,
        quality_sq: float = 1.0,
        edge_decay: float = 1.0,
        streak_factor: float = 1.0,
        smart_money_mult: float = 1.0,
        anomaly_penalty: float = 1.0,
    ) -> float:
        """Compute Fractional Kelly position size in USD.

        Applies the Kelly Criterion to the Bayesian posterior probability
        ``p`` and entry price, then scales by dynamic multipliers to
        produce a risk-adjusted, bankroll-relative position size.

        Kelly Formula:
            b  = net_payout_ratio      = (1 - entry_price) / entry_price
            f* = kelly_fraction        = (p*b - (1-p)) / b
            sz = bankroll × min(f*, cap) × multipliers

        Multipliers applied (all ≤ 1.0 unless winning streak bonus):
            quality_sq      — (setup_quality / 100)²: amplifies A+ setups
            edge_decay      — front-loads size to alpha zone of window
            streak_factor   — increases in hot streak, reduces in losing streak
            smart_money_mult — bonus when institutional flow is detected
            anomaly_penalty — halves size if one anomaly flag is present

        Args:
            p:               Bayesian win probability (0.40–0.75 clamped by score_trade)
            entry_price:     Outcome token entry price (e.g. 0.52)
            quality_sq:      (setup_quality / 100)²
            edge_decay:      Window edge decay multiplier (0.0–1.0)
            streak_factor:   Performance streak multiplier
            smart_money_mult: 1.2 if smart money detected, else 1.0
            anomaly_penalty: 0.5 if one anomaly flag, else 1.0

        Returns:
            Position size in USD (≥ 1.0, bounded by max_trade_size_usd).
        """
        if entry_price <= 0 or entry_price >= 1:
            return 0.0

        # Net payout ratio: if you bet $1 at price 0.52, you win $(1-0.52)/0.52 = $0.923 net
        b = (1.0 - entry_price) / entry_price
        if b <= 0:
            return 0.0

        # Raw Kelly fraction
        kelly_f = (p * b - (1.0 - p)) / b

        if kelly_f <= 0:
            # No mathematical edge — no trade
            return 0.0

        # Cap at kelly_fraction_cap (Quarter-Kelly by default)
        # This dramatically reduces variance while preserving ~75% of CAGR
        capped_f = min(kelly_f, self.kelly_fraction_cap)

        # Bankroll-relative base size
        base_size = self.bankroll_usdc * capped_f

        # Apply all dynamic multipliers
        size = (
            base_size
            * quality_sq
            * edge_decay
            * streak_factor
            * smart_money_mult
            * anomaly_penalty
        )

        # Floor at $1 (minimum meaningful bet), ceil at max_trade_size_usd
        size = round(max(1.0, min(size, self.max_trade_size)), 2)

        logger.info(
            "kelly_size_computed",
            p=round(p, 4),
            entry_price=round(entry_price, 4),
            b=round(b, 4),
            kelly_f=round(kelly_f, 4),
            capped_f=round(capped_f, 4),
            bankroll=self.bankroll_usdc,
            base_size=round(base_size, 2),
            quality_sq=round(quality_sq, 3),
            edge_decay=round(edge_decay, 3),
            streak_factor=streak_factor,
            smart_money_mult=smart_money_mult,
            anomaly_penalty=anomaly_penalty,
            final_size=size,
        )

        return size

    # ── Trade Recording ───────────────────────────────────────────

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade and update risk state."""
        self._state.daily_pnl += pnl
        self._state.daily_trades += 1
        self._state.last_trade_time = time.time()

        if pnl < 0:
            self._state.consecutive_losses += 1
            if self._state.consecutive_losses >= self.max_consecutive_losses:
                # Exponential backoff: 15m → 30m → 60m → 120m (capped)
                escalation = min(
                    self._state.consecutive_losses - self.max_consecutive_losses + 1,
                    4,  # Cap at 4 doublings (15m → 240m max)
                )
                cooldown = self.cooldown_seconds * (2 ** (escalation - 1))
                self._state.paused_until = time.time() + cooldown
                logger.warning(
                    "risk_cooldown_triggered",
                    consecutive_losses=self._state.consecutive_losses,
                    cooldown_seconds=cooldown,
                    escalation_level=escalation,
                )
        else:
            self._state.consecutive_losses = 0

        logger.info(
            "risk_trade_recorded",
            pnl=pnl,
            daily_pnl=self._state.daily_pnl,
            daily_trades=self._state.daily_trades,
            consecutive_losses=self._state.consecutive_losses,
        )

    # ── Summary ───────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Get a summary of current risk state for logging/display."""
        return {
            "daily_pnl": round(self._state.daily_pnl, 2),
            "daily_trades": self._state.daily_trades,
            "consecutive_losses": self._state.consecutive_losses,
            "is_paused": time.time() < self._state.paused_until,
            "allowed": time.time() >= self._state.paused_until,
            "max_trade_size": self.max_trade_size,
            "daily_loss_cap": self.daily_loss_cap,
            "min_conviction": self.min_conviction,
            "bankroll_usdc": self.bankroll_usdc,
            "kelly_fraction_cap": self.kelly_fraction_cap,
        }


# ── Singleton ──────────────────────────────────────────────────────

_risk_manager: RiskManager | None = None


def get_risk_manager(**kwargs: Any) -> RiskManager:
    """Get the singleton RiskManager instance."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(**kwargs)
    return _risk_manager
