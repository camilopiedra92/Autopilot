"""
ConversationalAssistantWorkflow — Personal Telegram assistant.

Subscribes to ``telegram.message_received`` events from the platform-level
webhook adapter, runs an ADK agent with Todoist/YNAB/Telegram tools, and
responds within the same ADK session (multi-turn conversations).

manifest.yaml and agent cards are auto-loaded by BaseWorkflow.
"""

from __future__ import annotations

import structlog

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowResult, RunStatus

logger = structlog.get_logger(__name__)


class ConversationalAssistantWorkflow(BaseWorkflow):
    """
    Personal Telegram assistant powered by an ADK agent.

    Subscribes to ``telegram.message_received`` events (published by the
    platform-level Telegram webhook adapter in ``webhooks.py``).
    Runs an autonomous agent that manages Todoist tasks, queries YNAB
    budgets, and replies via Telegram.

    Event Subscriptions (registered in ``setup()``):
      - ``telegram.message_received`` → Run assistant agent
    """

    async def setup(self) -> None:
        """Register event subscribers for this workflow."""
        from autopilot.core.subscribers import get_subscriber_registry

        registry = get_subscriber_registry()

        # React to Telegram messages — same pattern as bank_to_ynab + email.received
        registry.register(
            "telegram.message_received",
            self._on_telegram_message,
            name="conversational_assistant_telegram",
        )

    async def _on_telegram_message(self, msg) -> None:
        """
        React to ``telegram.message_received`` events from the AgentBus.

        Extracts the message and chat_id, then runs the workflow.
        """
        payload = msg.payload if hasattr(msg, "payload") else msg

        message = payload.get("message", "")
        chat_id = payload.get("telegram_chat_id", "")

        if not message or not chat_id:
            logger.debug(
                "telegram_event_skipped",
                workflow=self.manifest.name,
                reason="missing_message_or_chat_id",
            )
            return

        logger.info(
            "telegram_event_matched",
            workflow=self.manifest.name,
            chat_id=chat_id,
            text_preview=message[:80],
        )

        from autopilot.models import TriggerType

        await self.run(
            TriggerType.WEBHOOK,
            {
                "message": message,
                "telegram_chat_id": chat_id,
                "update": payload.get("update", {}),
            },
        )

    async def execute(self, trigger_data: dict) -> WorkflowResult:
        """
        Execute the conversational assistant.

        Uses a single ADKAgent invocation — the agent autonomously calls
        tools (Todoist, YNAB) and replies via Telegram within one ADK session.
        """
        from autopilot.core.context import AgentContext
        from autopilot.core.agent import ADKAgent
        from workflows.conversational_assistant.agents.assistant import (
            create_assistant,
        )

        message = trigger_data.get("message", "")
        chat_id = trigger_data.get("telegram_chat_id", "")

        if not message:
            return WorkflowResult(
                workflow_id=self.manifest.name,
                status=RunStatus.FAILED,
                error="No message provided",
            )

        # Create the agent — instruction placeholders like {telegram_chat_id}
        # are resolved from session state by the platform InstructionProvider.
        adk_agent = create_assistant()
        agent = ADKAgent(adk_agent)

        ctx = AgentContext(
            pipeline_name="conversational_assistant",
            metadata={
                "source": "telegram_webhook",
                # Use chat_id as session_id for conversation continuity
                "session_id": f"telegram_{chat_id}",
                "persist_memory": self.manifest.memory,
            },
        )
        # Put chat_id in state so the InstructionProvider resolves
        # {telegram_chat_id} from ReadonlyContext.state — the ADK-native way.
        ctx.update_state({"telegram_chat_id": chat_id})

        try:
            result = await agent.invoke(
                ctx,
                {
                    "message": message,
                    "telegram_chat_id": chat_id,
                },
            )

            return WorkflowResult(
                workflow_id=self.manifest.name,
                status=RunStatus.SUCCESS,
                data=result if isinstance(result, dict) else {"output": str(result)},
            )

        except Exception as e:
            logger.error(
                "assistant_execution_failed",
                error=str(e),
                message=message[:80],
            )

            # Try to send an error message to the user
            try:
                from autopilot.connectors import get_connector_registry

                telegram = get_connector_registry().get("telegram")
                await telegram.client.send_message(
                    chat_id=chat_id,
                    text="⚠️ Hubo un error procesando tu mensaje. Intenta de nuevo.",
                )
            except Exception:
                pass  # Best-effort error notification

            return WorkflowResult(
                workflow_id=self.manifest.name,
                status=RunStatus.FAILED,
                error=str(e),
            )
