"""
Polymarket BTC Trader — workflow anchor.

Identity & Triggers: loaded from manifest.yaml
Execution Logic: handled by pipeline.yaml
  resolve_outcomes → gather → score_trade → risk_gate → execute_trade

Fully deterministic pipeline — 5 code steps, 0 LLM calls.
All trade lifecycle (event publishing, ledger recording, risk persistence)
is handled inside execute_trade_step via ctx: AgentContext.
"""

from typing import Any

import structlog

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowResult

logger = structlog.get_logger(__name__)


class PolymarketBTCWorkflow(BaseWorkflow):
    """
    Deterministic BTC prediction market trading workflow.

    manifest.yaml defines WHO (identity, triggers, settings).
    pipeline.yaml defines WHAT (5 code steps: resolve → gather → score → risk → execute).
    execute_trade_step handles post-trade lifecycle via ctx: AgentContext.
    """

    async def setup(self) -> None:
        """Import tools module to trigger @tool registration at startup."""
        import workflows.polymarket_btc.tools  # noqa: F401

        logger.info("polymarket_btc_workflow_setup_complete")

    async def execute(self, trigger_data: dict[str, Any]) -> WorkflowResult:
        """Run the pipeline — all lifecycle is handled inside execute_trade_step."""
        return await super().execute(trigger_data)
