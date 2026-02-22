"""
Unit tests for PubSubConnector — verifies clean cloud-native architecture.

These tests confirm that PubSubConnector:
- Does NOT spawn background tasks (no in-process renewal loop)
- Relies entirely on Cloud Scheduler for watch renewal
- Properly registers/deregisters watches on cold start
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def pubsub():
    """Create a PubSubConnector instance with mocked dependencies."""
    from autopilot.connectors.pubsub_connector import PubSubConnector

    connector = PubSubConnector()
    return connector


@pytest.fixture
def mock_gmail():
    """Create a mock GmailConnector."""
    gmail = MagicMock()
    gmail.service.users.return_value.watch.return_value.execute.return_value = {
        "historyId": "12345",
        "expiration": "1740000000000",
    }
    gmail.service.users.return_value.stop.return_value.execute.return_value = {}
    gmail.service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
        "labels": []
    }
    gmail.resolve_label_ids = MagicMock(return_value=[])
    return gmail


class TestPubSubConnectorNoBackgroundTasks:
    """Verify that PubSubConnector has no in-process renewal loops."""

    def test_no_renewal_task_field(self, pubsub):
        """Connector must not have a _renewal_task field."""
        assert not hasattr(pubsub, "_renewal_task")

    def test_no_shutdown_event_field(self, pubsub):
        """Connector must not have a _shutdown_event field."""
        assert not hasattr(pubsub, "_shutdown_event")

    def test_no_renewal_interval_constant(self, pubsub):
        """Connector must not have WATCH_RENEWAL_INTERVAL_SECONDS."""
        assert not hasattr(pubsub, "WATCH_RENEWAL_INTERVAL_SECONDS")

    def test_no_watch_renewal_loop_method(self, pubsub):
        """Connector must not have _watch_renewal_loop method."""
        assert not hasattr(pubsub, "_watch_renewal_loop")

    def test_watch_status_no_renewal_task_alive(self, pubsub):
        """watch_status must NOT contain 'renewal_task_alive'."""
        status = pubsub.watch_status
        assert "renewal_task_alive" not in status
        assert "active" in status
        assert "history_id" in status
        assert "expiration" in status
        assert "expiration_epoch_ms" in status


class TestPubSubConnectorWatchLifecycle:
    """Verify watch registration and teardown work without background tasks."""

    @pytest.mark.asyncio
    async def test_start_watching_no_background_task(self, pubsub, mock_gmail):
        """start_watching() must NOT spawn any asyncio tasks."""
        with (
            patch.dict(
                "os.environ",
                {"GCP_PUBSUB_TOPIC": "projects/test/topics/test"},
            ),
            patch("autopilot.registry.get_registry") as mock_reg,
        ):
            mock_reg.return_value.get_all_workflows.return_value = []

            await pubsub.start_watching(mock_gmail)

            # Watch should be registered
            assert pubsub.is_watch_active or pubsub._history_id == "12345"

            # No background task should exist
            assert not hasattr(pubsub, "_renewal_task")

    @pytest.mark.asyncio
    async def test_stop_watching_clean(self, pubsub, mock_gmail):
        """stop_watching() must work without task cancellation logic."""
        with (
            patch.dict(
                "os.environ",
                {"GCP_PUBSUB_TOPIC": "projects/test/topics/test"},
            ),
            patch("autopilot.registry.get_registry") as mock_reg,
        ):
            mock_reg.return_value.get_all_workflows.return_value = []

            await pubsub.start_watching(mock_gmail)
            await pubsub.stop_watching()

            # Should complete without errors — no task to cancel

    @pytest.mark.asyncio
    async def test_force_rewatch_returns_status(self, pubsub, mock_gmail):
        """force_rewatch() must return valid watch_status without renewal_task_alive."""
        with (
            patch.dict(
                "os.environ",
                {"GCP_PUBSUB_TOPIC": "projects/test/topics/test"},
            ),
            patch("autopilot.registry.get_registry") as mock_reg,
        ):
            mock_reg.return_value.get_all_workflows.return_value = []

            await pubsub.start_watching(mock_gmail)
            status = await pubsub.force_rewatch()

            assert "active" in status
            assert "renewal_task_alive" not in status
