"""
BankToYnabWorkflow — Parse bank notification emails into YNAB transactions.

manifest.yaml, pipeline.yaml, and agent cards are auto-loaded by BaseWorkflow.
This file (workflow.py) provides the custom execute() override to map inputs
and outputs to the declarative pipeline.
"""

import structlog

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import TriggerType

logger = structlog.get_logger(__name__)


class BankToYnabWorkflow(BaseWorkflow):
    """
    Bank→YNAB: Parse bank notification emails and create YNAB transactions.

    This class loads its identity from manifest.yaml and its execution logic
    from pipeline.yaml. It relies entirely on the platform's DSL loader.

    Event Subscriptions (registered in ``setup()``):
      - ``email.received`` → Trigger pipeline when matching Bancolombia email
      - ``transaction.created`` → Telegram notifier (LLM-powered)
    """

    async def setup(self) -> None:
        """Register event subscribers for this workflow."""
        from autopilot.core.subscribers import get_subscriber_registry
        from workflows.bank_to_ynab.subscribers.telegram_subscriber import (
            on_transaction_created,
        )

        registry = get_subscriber_registry()

        # React to email events — self-match using manifest trigger config
        registry.register(
            "email.received",
            self._on_email_received,
            name="bank_to_ynab_email_trigger",
        )

        # React to transaction events — send Telegram notification
        registry.register(
            "transaction.created",
            on_transaction_created,
            name="telegram_notifier",
        )

    async def _on_email_received(self, msg) -> None:
        """
        React to ``email.received`` events from the AgentBus.

        Self-matches using the manifest's GMAIL_PUSH trigger config
        (sender filter + label IDs). If the email doesn't match, returns
        immediately without running the pipeline.
        """
        payload = msg.payload if hasattr(msg, "payload") else msg

        if not self._matches_gmail_trigger(payload):
            logger.debug(
                "email_event_skipped",
                workflow=self.manifest.name,
                sender=payload.get("sender", ""),
            )
            return

        logger.info(
            "email_event_matched",
            workflow=self.manifest.name,
            sender=payload.get("sender", ""),
            subject=payload.get("subject", "")[:80],
        )

        # Build trigger payload compatible with pipeline expectations
        email_data = payload.get("email", payload)
        trigger_payload = {
            "source": payload.get("source", "pubsub"),
            "email": email_data,
            "body": payload.get("body", "") or email_data.get("body", ""),
            "subject": payload.get("subject", ""),
            "id": payload.get("email_id", ""),
        }

        await self.run(TriggerType.GMAIL_PUSH, trigger_payload)
