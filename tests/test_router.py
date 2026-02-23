import pytest
from unittest.mock import AsyncMock, MagicMock
from autopilot.router import WorkflowRouter
from autopilot.models import TriggerType


@pytest.mark.asyncio
class TestWorkflowRouterLogic:
    async def test_route_webhook_finds_workflow(self):
        registry = MagicMock()
        workflow = AsyncMock()
        workflow.manifest.name = "test_wf"
        registry.find_by_webhook_path.return_value = workflow

        router = WorkflowRouter(registry)
        await router.route_webhook("/test-path", {"body": "foo"})

        registry.find_by_webhook_path.assert_called_with("/test-path")
        workflow.run.assert_awaited_with(TriggerType.WEBHOOK, {"body": "foo"})

    async def test_route_webhook_raises_if_not_found(self):
        registry = MagicMock()
        # Ensure it returns None
        registry.find_by_webhook_path.return_value = None

        router = WorkflowRouter(registry)

        with pytest.raises(KeyError):
            await router.route_webhook("/unknown", {})

    async def test_route_manual_triggers_workflow(self):
        registry = MagicMock()
        workflow = AsyncMock()
        workflow.manifest.enabled = True
        registry.get_or_raise.return_value = workflow

        router = WorkflowRouter(registry)
        await router.route_manual("wf-id", {"data": 123})

        registry.get_or_raise.assert_called_with("wf-id")
        workflow.run.assert_awaited_with(TriggerType.MANUAL, {"data": 123})

    async def test_matches_gmail_trigger_sender_match(self):
        """BaseWorkflow._matches_gmail_trigger matches by sender filter."""
        from autopilot.base_workflow import BaseWorkflow
        from autopilot.models import TriggerConfig, WorkflowManifest

        wf = BaseWorkflow.__new__(BaseWorkflow)
        wf._manifest = WorkflowManifest(
            name="test_wf",
            display_name="Test",
            triggers=[
                TriggerConfig(
                    type=TriggerType.GMAIL_PUSH,
                    filter="bank.com",
                    label_ids=[],
                ),
            ],
        )

        # Should match
        assert wf._matches_gmail_trigger(
            {"sender": "alerts@bank.com", "label_ids": ["INBOX"]}
        )
        # Should not match
        assert not wf._matches_gmail_trigger(
            {"sender": "nope@other.com", "label_ids": ["INBOX"]}
        )

    async def test_matches_gmail_trigger_disabled_workflow(self):
        """Disabled workflows never match."""
        from autopilot.base_workflow import BaseWorkflow
        from autopilot.models import TriggerConfig, WorkflowManifest

        wf = BaseWorkflow.__new__(BaseWorkflow)
        wf._manifest = WorkflowManifest(
            name="test_wf",
            display_name="Test",
            enabled=False,
            triggers=[
                TriggerConfig(type=TriggerType.GMAIL_PUSH, filter="bank.com"),
            ],
        )
        assert not wf._matches_gmail_trigger({"sender": "alerts@bank.com"})

    async def test_matches_gmail_trigger_label_match(self):
        """Label IDs must overlap when specified."""
        from autopilot.base_workflow import BaseWorkflow
        from autopilot.models import TriggerConfig, WorkflowManifest

        wf = BaseWorkflow.__new__(BaseWorkflow)
        wf._manifest = WorkflowManifest(
            name="test_wf",
            display_name="Test",
            triggers=[
                TriggerConfig(
                    type=TriggerType.GMAIL_PUSH,
                    label_ids=["INBOX", "Label_123"],
                ),
            ],
        )

        assert wf._matches_gmail_trigger({"sender": "x@y.com", "label_ids": ["INBOX"]})
        assert not wf._matches_gmail_trigger(
            {"sender": "x@y.com", "label_ids": ["SPAM"]}
        )

    async def test_email_received_event_via_bus(self):
        """email.received published on bus is delivered to subscribers."""
        from autopilot.core.bus import EventBus

        bus = EventBus()
        received = []

        async def handler(msg):
            received.append(msg.payload)

        bus.subscribe("email.received", handler)
        await bus.publish("email.received", {"sender": "a@b.com"}, sender="test")

        assert len(received) == 1
        assert received[0]["sender"] == "a@b.com"
