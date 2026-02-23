"""
Unit tests for the Conversational Assistant workflow.

Tests:
  1. Manifest loading — fields, triggers, tags
  2. Agent creation — verify tool list and agent name
  3. Event subscription — verify setup() registers subscriber
  4. Execute — edge cases and error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Manifest Tests ───────────────────────────────────────────────────


class TestManifest:
    """Verify manifest.yaml loads correctly."""

    def test_manifest_loads(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )

        wf = ConversationalAssistantWorkflow()
        m = wf.manifest

        assert m.name == "conversational_assistant"
        assert m.display_name == "Asistente Personal 💬"
        assert m.version == "1.0.0"
        assert m.icon == "💬"

    def test_manifest_triggers(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )

        wf = ConversationalAssistantWorkflow()
        triggers = wf.manifest.triggers

        assert len(triggers) == 1
        assert triggers[0].type.value == "webhook"

    def test_manifest_tags(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )

        wf = ConversationalAssistantWorkflow()
        tags = wf.manifest.tags

        assert "assistant" in tags
        assert "telegram" in tags
        assert "todoist" in tags
        assert "ynab" in tags


# ── Agent Creation Tests ─────────────────────────────────────────────


class TestAgentCreation:
    """Verify the assistant agent is created with correct tools."""

    def test_create_assistant_returns_agent(self):
        from workflows.conversational_assistant.agents.assistant import (
            create_assistant,
        )

        agent = create_assistant()
        assert agent is not None
        assert agent.name == "assistant"

    def test_create_assistant_has_tools(self):
        from workflows.conversational_assistant.agents.assistant import (
            create_assistant,
        )

        agent = create_assistant()

        # The agent should have tools configured
        # Tools are string refs, auto-resolved at runtime
        assert agent.tools is not None

    def test_assistant_instruction_contains_key_elements(self):
        from workflows.conversational_assistant.agents.assistant import (
            ASSISTANT_INSTRUCTION,
        )

        # Must mention the key capabilities
        assert "Todoist" in ASSISTANT_INSTRUCTION
        assert "YNAB" in ASSISTANT_INSTRUCTION
        assert "Telegram" in ASSISTANT_INSTRUCTION
        assert "DETENTE" in ASSISTANT_INSTRUCTION

        # Must have the chat_id placeholder
        assert "{telegram_chat_id}" in ASSISTANT_INSTRUCTION


# ── Event Subscription Tests ─────────────────────────────────────────


class TestEventSubscription:
    """Verify the workflow subscribes to telegram.message_received events."""

    @pytest.mark.asyncio
    async def test_setup_registers_subscriber(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )

        wf = ConversationalAssistantWorkflow()

        with patch(
            "autopilot.core.subscribers.get_subscriber_registry"
        ) as mock_get_reg:
            mock_registry = MagicMock()
            mock_get_reg.return_value = mock_registry

            await wf.setup()

            mock_registry.register.assert_called_once_with(
                "telegram.message_received",
                wf._on_telegram_message,
                name="conversational_assistant_telegram",
            )

    @pytest.mark.asyncio
    async def test_on_telegram_message_skips_empty(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )

        wf = ConversationalAssistantWorkflow()

        # Empty payload should be skipped (no run called)
        msg = MagicMock()
        msg.payload = {"message": "", "telegram_chat_id": ""}

        with patch.object(wf, "run", new_callable=AsyncMock) as mock_run:
            await wf._on_telegram_message(msg)
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_telegram_message_triggers_run(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )
        from autopilot.models import TriggerType

        wf = ConversationalAssistantWorkflow()

        msg = MagicMock()
        msg.payload = {
            "message": "Hola",
            "telegram_chat_id": "123",
            "update": {"update_id": 1},
        }

        with patch.object(wf, "run", new_callable=AsyncMock) as mock_run:
            await wf._on_telegram_message(msg)
            mock_run.assert_called_once_with(
                TriggerType.WEBHOOK,
                {
                    "message": "Hola",
                    "telegram_chat_id": "123",
                    "update": {"update_id": 1},
                },
            )

    def test_workflow_inherits_base(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )
        from autopilot.base_workflow import BaseWorkflow

        wf = ConversationalAssistantWorkflow()
        assert isinstance(wf, BaseWorkflow)


# ── Execute Tests ────────────────────────────────────────────────────


class TestExecute:
    """Verify the execute method handles edge cases."""

    @pytest.mark.asyncio
    async def test_execute_empty_message_fails(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )
        from autopilot.models import RunStatus

        wf = ConversationalAssistantWorkflow()

        result = await wf.execute({"message": "", "telegram_chat_id": "123"})

        assert result.status == RunStatus.FAILED
        assert "No message" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_message_fails(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )
        from autopilot.models import RunStatus

        wf = ConversationalAssistantWorkflow()

        result = await wf.execute({"telegram_chat_id": "123"})

        assert result.status == RunStatus.FAILED
        assert "No message" in result.error
