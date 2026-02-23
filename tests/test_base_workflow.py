"""
Tests for BaseWorkflow — verifies self-contained setting resolution.

The workflow must apply its own manifest setting defaults before
execution, so that every entry point (router, CLI, tests, direct
invocation) gets the same behavior.
"""

import pytest

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import (
    WorkflowManifest,
    SettingConfig,
    SettingType,
    TriggerType,
    WorkflowResult,
    RunStatus,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_manifest(**overrides) -> WorkflowManifest:
    """Build a WorkflowManifest with sensible defaults."""
    defaults = dict(
        name="test_workflow",
        display_name="Test Workflow",
        description="A test workflow",
        settings=[],
    )
    defaults.update(overrides)
    return WorkflowManifest(**defaults)


class StubWorkflow(BaseWorkflow):
    """
    Minimal BaseWorkflow subclass for testing.

    Overrides the manifest and execute() to avoid filesystem
    dependencies (no manifest.yaml / pipeline.yaml needed).
    """

    def __init__(self, manifest: WorkflowManifest):
        super().__init__()
        self._manifest = manifest
        self._received_trigger_data: dict | None = None

    @property
    def manifest(self) -> WorkflowManifest:
        return self._manifest

    async def execute(self, trigger_data: dict) -> WorkflowResult:
        # Capture the trigger_data so tests can inspect what was passed
        self._received_trigger_data = trigger_data
        return WorkflowResult(
            workflow_id=self.manifest.name,
            status=RunStatus.SUCCESS,
            data=trigger_data,
        )


# ── Tests: _apply_setting_defaults ────────────────────────────────────


class TestApplySettingDefaults:
    """
    Tests for BaseWorkflow._apply_setting_defaults.

    This method is the canonical setting resolution layer.
    It must:
      1. Inject defaults from manifest settings when absent
      2. Never overwrite explicit caller values
      3. Skip settings with no default (None)
      4. Return a new dict (no input mutation)
    """

    def test_injects_defaults_when_absent(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email"})

        assert result["auto_create"] is True
        assert result["body"] == "email"

    def test_does_not_overwrite_explicit_value(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email", "auto_create": False})

        assert result["auto_create"] is False

    def test_skips_settings_with_no_default(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="ynab_budget_id", type=SettingType.STRING, default=None
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({})

        assert "ynab_budget_id" not in result

    def test_does_not_mutate_input(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        original = {"body": "email"}
        result = wf._apply_setting_defaults(original)

        # The original dict must NOT be modified
        assert "auto_create" not in original
        assert result["auto_create"] is True

    def test_multiple_settings(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
                SettingConfig(
                    key="gmail_sender_filter",
                    type=SettingType.STRING,
                    default="alertasynotificaciones",
                ),
                SettingConfig(
                    key="ynab_access_token",
                    type=SettingType.SECRET,
                    default=None,
                    required=True,
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email"})

        assert result["auto_create"] is True
        assert result["gmail_sender_filter"] == "alertasynotificaciones"
        assert "ynab_access_token" not in result  # No default → not injected

    def test_no_settings(self):
        manifest = _make_manifest(settings=[])
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email"})

        assert result == {"body": "email"}


# ── Tests: run() integration with settings ────────────────────────────


@pytest.mark.asyncio
class TestRunAppliesSettingDefaults:
    """
    Integration tests verifying that run() passes enriched trigger data
    to execute(), making the workflow self-contained.
    """

    async def test_run_injects_defaults_into_execute(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.MANUAL, {"body": "email"})

        # execute() should have received the enriched data
        assert wf._received_trigger_data["auto_create"] is True
        assert wf._received_trigger_data["body"] == "email"
        assert run.status == RunStatus.SUCCESS

    async def test_run_preserves_explicit_overrides(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.MANUAL, {"body": "email", "auto_create": False})

        assert wf._received_trigger_data["auto_create"] is False
        assert run.status == RunStatus.SUCCESS

    async def test_run_records_enriched_data_in_history(self):
        manifest = _make_manifest(
            settings=[
                SettingConfig(
                    key="auto_create", type=SettingType.BOOLEAN, default=True
                ),
            ]
        )
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.WEBHOOK, {"body": "email"})

        # The run record should contain the enriched trigger data
        assert run.trigger_data["auto_create"] is True
        assert run.trigger_data["body"] == "email"


# ── Tests: Manifest memory flag ──────────────────────────────────────


class TestManifestMemoryFlag:
    """Tests for the manifest ``memory`` flag."""

    def test_memory_defaults_to_false(self):
        manifest = _make_manifest()
        assert manifest.memory is False

    def test_memory_true_parsed(self):
        manifest = _make_manifest(memory=True)
        assert manifest.memory is True

    def test_memory_false_explicit(self):
        manifest = _make_manifest(memory=False)
        assert manifest.memory is False


# ── Tests: DSL pipeline memory transfer ──────────────────────────────


@pytest.mark.asyncio
class TestDSLPipelineMemoryTransfer:
    """Tests for memory transfer in BaseWorkflow._execute_dsl_pipeline."""

    async def test_transfers_memory_when_manifest_memory_true(self):
        """When manifest.memory=True and session exists, add_session_to_memory is called."""
        from unittest.mock import AsyncMock, patch, MagicMock

        manifest = _make_manifest(memory=True)
        wf = StubWorkflow(manifest)

        mock_session = MagicMock()
        mock_memory = AsyncMock()

        # Patch the DSL pipeline execution to set ctx.session
        async def fake_execute(ctx, initial_input=None):
            ctx.session = mock_session
            ctx.memory = mock_memory
            result = MagicMock()
            result.state = {"done": True}
            return result

        with patch("autopilot.core.dsl_loader.load_workflow") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.execute = fake_execute
            mock_load.return_value = mock_pipeline

            # Create a fake pipeline.yaml path
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(b"name: test\nsteps: []")
                pipeline_path = f.name

            try:
                from pathlib import Path

                result = await wf._execute_dsl_pipeline({}, Path(pipeline_path))
            finally:
                os.unlink(pipeline_path)

        assert result.status == RunStatus.SUCCESS
        mock_memory.add_session_to_memory.assert_called_once_with(mock_session)

    async def test_skips_memory_when_manifest_memory_false(self):
        """When manifest.memory=False, add_session_to_memory is NOT called."""
        from unittest.mock import AsyncMock, patch, MagicMock

        manifest = _make_manifest(memory=False)
        wf = StubWorkflow(manifest)

        mock_session = MagicMock()
        mock_memory = AsyncMock()

        async def fake_execute(ctx, initial_input=None):
            ctx.session = mock_session
            ctx.memory = mock_memory
            result = MagicMock()
            result.state = {"done": True}
            return result

        with patch("autopilot.core.dsl_loader.load_workflow") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.execute = fake_execute
            mock_load.return_value = mock_pipeline

            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                f.write(b"name: test\nsteps: []")
                pipeline_path = f.name

            try:
                from pathlib import Path

                result = await wf._execute_dsl_pipeline({}, Path(pipeline_path))
            finally:
                os.unlink(pipeline_path)

        assert result.status == RunStatus.SUCCESS
        mock_memory.add_session_to_memory.assert_not_called()
