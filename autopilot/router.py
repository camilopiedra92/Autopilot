"""
WorkflowRouter — Routes incoming triggers to the correct workflow.

The router is the central dispatch point that:
  1. Receives triggers (webhook, Gmail push, scheduled, manual)
  2. Looks up the target workflow(s) in the registry
  3. Dispatches execution to the matched workflow
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

    async def route_webhook(
        self, path: str, data: dict[str, Any]
    ) -> WorkflowRun:
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

    # ── Gmail Push Routing ────────────────────────────────────────────

    async def route_gmail_push(
        self,
        email_data: dict[str, Any],
        trigger_source: str = "pubsub",
    ) -> list[WorkflowRun]:
        """
        Route a Gmail message to all workflows that match its criteria.
        
        Matching Logic:
          1. Workflow must be enabled.
          2. Workflow must have a GMAIL_PUSH trigger.
          3. Trigger 'filter' (sender) must match email 'from' (if set).
          4. Trigger 'label_ids' must overlap with email 'labelIds' (if set).
        
        Args:
            email_data: Dict containing 'id', 'from', 'subject', 'body', 'labelIds'.
            trigger_source: Origin of the trigger (default: "pubsub").

        Returns:
            List of WorkflowRun objects (one per matching workflow).
        """
        sender = email_data.get("from", "").lower()
        email_labels = set(email_data.get("labelIds", []))
        
        # Find all matching workflows
        matches: list[tuple[Any, Any]] = []  # (workflow, trigger_config)
        
        for workflow in self._registry.get_all_workflows():
            if not workflow.manifest.enabled:
                continue
                
            for trigger in workflow.manifest.triggers:
                if trigger.type != TriggerType.GMAIL_PUSH:
                    continue
                
                # 1. Match Sender (if filter is set)
                if trigger.filter:
                    if trigger.filter.lower() not in sender:
                        continue

                # 2. Match Labels (if label_ids are set in trigger)
                if trigger.label_ids:
                    trigger_labels = set(trigger.label_ids)
                    if not trigger_labels.intersection(email_labels):
                        continue
                
                matches.append((workflow, trigger))
        
        if not matches:
            logger.info("gmail_routing_no_matches", sender=sender, labels=list(email_labels))
            return []

        results = []
        for workflow, trigger in matches:
            logger.info(
                "routing_gmail_email",
                workflow=workflow.manifest.name,
                sender=sender,
                trigger_labels=trigger.label_ids,
            )
            
            # Prepare trigger data for the workflow
            trigger_payload = {
                "source": trigger_source,
                "email": email_data,
                "body": email_data.get("body", ""),
                "subject": email_data.get("subject", ""),
                "id": email_data.get("id"),
            }
            
            # Execute — BaseWorkflow.run() handles setting defaults
            run = await workflow.run(TriggerType.GMAIL_PUSH, trigger_payload)
            results.append(run)

        return results

    # ── Manual Trigger ────────────────────────────────────────────────

    async def route_manual(
        self, workflow_id: str, data: dict[str, Any]
    ) -> WorkflowRun:
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

