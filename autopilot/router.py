"""
WorkflowRouter — Routes incoming triggers to the correct workflow.

The router is the central dispatch point that:
  1. Receives triggers (webhook, scheduled, manual)
  2. Looks up the target workflow(s) in the registry
  3. Dispatches execution to the matched workflow

Note: Gmail push routing is handled event-driven via the AgentBus
(``email.received`` topic). See ``autopilot/api/webhooks.py``.
"""

from __future__ import annotations

import structlog
from typing import Any

from autopilot.models import (
    TriggerType,
    WorkflowRun,
)
from autopilot.registry import WorkflowRegistry, get_registry

logger = structlog.get_logger(__name__)


class WorkflowRouter:
    """
    Routes triggers to workflows via the registry.

    Usage:
        router = WorkflowRouter(registry)

        # Route a webhook
        result = await router.route_webhook("/process-email", {"body": "..."})

        # Route a manual trigger
        result = await router.route_manual("bank_to_ynab", {"body": "..."})
    """

    def __init__(self, registry: WorkflowRegistry | None = None):
        self._registry = registry or get_registry()

    # ── Webhook Routing ───────────────────────────────────────────────

    async def route_webhook(self, path: str, data: dict[str, Any]) -> WorkflowRun:
        """
        Route a webhook request to the workflow that handles the given path.

        Raises KeyError if no workflow handles the path.
        """
        workflow = self._registry.find_by_webhook_path(path)
        if workflow is None:
            raise KeyError(f"No workflow registered for webhook path: {path}")

        logger.info(
            "routing_webhook",
            path=path,
            workflow=workflow.manifest.name,
        )

        return await workflow.run(TriggerType.WEBHOOK, data)

    # ── Manual Trigger ────────────────────────────────────────────────

    async def route_manual(self, workflow_id: str, data: dict[str, Any]) -> WorkflowRun:
        """
        Manually trigger a specific workflow.

        Raises KeyError if workflow not found.
        """
        workflow = self._registry.get_or_raise(workflow_id)

        if not workflow.manifest.enabled:
            logger.warning("manual_trigger_disabled_workflow", name=workflow_id)

        logger.info("routing_manual", workflow=workflow_id)
        return await workflow.run(TriggerType.MANUAL, data)

    # ── Scheduled ─────────────────────────────────────────────────────

    async def route_scheduled(self, workflow_id: str) -> WorkflowRun:
        """
        Execute a scheduled workflow.

        Raises KeyError if workflow not found.
        """
        workflow = self._registry.get_or_raise(workflow_id)

        logger.info("routing_scheduled", workflow=workflow_id)
        return await workflow.run(TriggerType.SCHEDULED, {})


# ── Singleton ─────────────────────────────────────────────────────────

_router: WorkflowRouter | None = None


def get_router() -> WorkflowRouter:
    """Get or create the global WorkflowRouter singleton."""
    global _router
    if _router is None:
        _router = WorkflowRouter()
    return _router
