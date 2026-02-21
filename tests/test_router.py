
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

    async def test_route_gmail_push_broadcasts(self):
        registry = MagicMock()

        # Create realistic workflow mocks with manifest and triggers
        trigger_config = MagicMock()
        trigger_config.type = TriggerType.GMAIL_PUSH
        trigger_config.filter = "bank.com"
        trigger_config.label_ids = []

        wf1 = AsyncMock()
        wf1.manifest.name = "wf1"
        wf1.manifest.enabled = True
        wf1.manifest.triggers = [trigger_config]

        wf2 = AsyncMock()
        wf2.manifest.name = "wf2"
        wf2.manifest.enabled = True
        wf2.manifest.triggers = [trigger_config]

        registry.get_all_workflows.return_value = [wf1, wf2]

        router = WorkflowRouter(registry)

        email_data = {
            "from": "alerts@bank.com",
            "subject": "Transaction Alert",
            "body": "<html>...</html>",
            "labelIds": ["INBOX"],
        }
        results = await router.route_gmail_push(email_data)

        assert len(results) == 2
        assert wf1.run.await_count == 1
        assert wf2.run.await_count == 1


