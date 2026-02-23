"""
ConversationalAssistantWorkflow — Personal Telegram assistant.

Receives Telegram webhook updates, extracts the user message,
runs a ReAct agent with Todoist/YNAB/Telegram tools, and responds.

manifest.yaml and agent cards are auto-loaded by BaseWorkflow.
This file provides the custom execute() and Telegram webhook route.
"""

import os
import structlog

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowResult, RunStatus, TriggerType

logger = structlog.get_logger(__name__)

# The secret token used to verify Telegram webhook requests
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")


def _extract_message(update: dict) -> tuple[str, str] | None:
    """
    Extract message text and chat_id from a Telegram Update object.

    Returns (message_text, chat_id) or None if the update is not a text message.
    """
    message = update.get("message") or update.get("edited_message")
    if not message:
        return None

    text = message.get("text", "")
    if not text:
        return None

    chat_id = str(message.get("chat", {}).get("id", ""))
    if not chat_id:
        return None

    return text, chat_id


class ConversationalAssistantWorkflow(BaseWorkflow):
    """
    Personal Telegram assistant powered by a ReAct agent.

    Receives Telegram webhook updates and runs an autonomous agent that
    can manage Todoist tasks, query YNAB budgets, and reply via Telegram.

    Event flow:
      POST /telegram/webhook → extract message → ReAct agent → tools → reply
    """

    def register_routes(self, app) -> None:
        """Mount the Telegram webhook endpoint on the FastAPI app."""
        from fastapi import APIRouter, HTTPException, Request

        router = APIRouter(tags=["telegram"])

        @router.post("/telegram/webhook")
        async def telegram_webhook(request: Request):
            """
            Handle incoming Telegram Bot API webhook updates.

            Validates the secret token header, extracts the message,
            and runs the workflow asynchronously.
            """
            # Verify secret token if configured
            if TELEGRAM_WEBHOOK_SECRET:
                token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
                if token != TELEGRAM_WEBHOOK_SECRET:
                    raise HTTPException(status_code=403, detail="Invalid secret token")

            try:
                update = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON")

            # Extract message — ignore non-text updates silently
            result = _extract_message(update)
            if not result:
                return {"status": "ignored", "reason": "no_text_message"}

            message_text, chat_id = result

            # Verify authorized user
            authorized_chat_id = self.manifest.settings[0].default or os.environ.get(
                "TELEGRAM_CHAT_ID", ""
            )
            if authorized_chat_id and chat_id != authorized_chat_id:
                logger.warning(
                    "telegram_unauthorized_chat",
                    chat_id=chat_id,
                    authorized=authorized_chat_id,
                )
                return {"status": "ignored", "reason": "unauthorized_chat"}

            logger.info(
                "telegram_message_received",
                chat_id=chat_id,
                text_length=len(message_text),
                text_preview=message_text[:80],
            )

            # Run the workflow
            run = await self.run(
                TriggerType.WEBHOOK,
                {
                    "message": message_text,
                    "telegram_chat_id": chat_id,
                    "update": update,
                },
            )

            return {
                "status": "ok",
                "run_id": run.id,
                "workflow_id": run.workflow_id,
            }

        app.include_router(router)

    async def execute(self, trigger_data: dict) -> WorkflowResult:
        """
        Execute the conversational assistant.

        Uses a single ADKAgent invocation — the agent autonomously calls
        tools (Todoist, YNAB) and replies via Telegram within one ADK session.
        No multi-iteration ReAct loop needed for single-turn commands.
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

        # Create the agent and format the instruction with the user's chat_id
        adk_agent = create_assistant()
        adk_agent.instruction = adk_agent.instruction.format(
            telegram_chat_id=chat_id,
        )
        agent = ADKAgent(adk_agent)

        ctx = AgentContext(
            pipeline_name="conversational_assistant",
            metadata={
                "source": "telegram_webhook",
                # Use chat_id as session_id for conversation continuity
                "session_id": f"telegram_{chat_id}",
            },
        )

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
