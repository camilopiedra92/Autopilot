"""
Unit tests for the Conversational Assistant workflow.

Tests:
  1. Manifest loading — fields, triggers, tags
  2. Telegram update parsing — extract text and chat_id
  3. Agent creation — verify tool list and agent name
  4. Webhook route registration — verify endpoint mounts
"""

import pytest


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


# ── Message Extraction Tests ─────────────────────────────────────────


class TestMessageExtraction:
    """Verify Telegram update parsing extracts text and chat_id correctly."""

    def test_extract_text_message(self, sample_text_update):
        from workflows.conversational_assistant.workflow import _extract_message

        result = _extract_message(sample_text_update)
        assert result is not None

        text, chat_id = result
        assert text == "Recuérdame comprar leche mañana"
        assert chat_id == "1093871758"

    def test_extract_photo_returns_none(self, sample_photo_update):
        from workflows.conversational_assistant.workflow import _extract_message

        result = _extract_message(sample_photo_update)
        assert result is None

    def test_extract_callback_query_returns_none(self, sample_callback_query_update):
        from workflows.conversational_assistant.workflow import _extract_message

        result = _extract_message(sample_callback_query_update)
        assert result is None

    def test_extract_empty_update(self):
        from workflows.conversational_assistant.workflow import _extract_message

        result = _extract_message({})
        assert result is None

    def test_extract_edited_message(self):
        from workflows.conversational_assistant.workflow import _extract_message

        update = {
            "update_id": 123,
            "edited_message": {
                "message_id": 44,
                "chat": {"id": 1093871758, "type": "private"},
                "date": 1708700002,
                "text": "Mensaje editado",
            },
        }

        result = _extract_message(update)
        assert result is not None
        text, chat_id = result
        assert text == "Mensaje editado"
        assert chat_id == "1093871758"


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


# ── Webhook Route Registration Tests ─────────────────────────────────


class TestWebhookRegistration:
    """Verify the workflow registers its Telegram webhook route."""

    def test_register_routes_adds_endpoint(self):
        from workflows.conversational_assistant.workflow import (
            ConversationalAssistantWorkflow,
        )
        from fastapi import FastAPI

        app = FastAPI()
        wf = ConversationalAssistantWorkflow()

        wf.register_routes(app)

        # Check that the /telegram/webhook route was registered
        routes = [route.path for route in app.routes]
        assert "/telegram/webhook" in routes

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
