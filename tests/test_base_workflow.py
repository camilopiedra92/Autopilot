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
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email"})

        assert result["auto_create"] is True
        assert result["body"] == "email"

    def test_does_not_overwrite_explicit_value(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({"body": "email", "auto_create": False})

        assert result["auto_create"] is False

    def test_skips_settings_with_no_default(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="ynab_budget_id", type=SettingType.STRING, default=None),
        ])
        wf = StubWorkflow(manifest)

        result = wf._apply_setting_defaults({})

        assert "ynab_budget_id" not in result

    def test_does_not_mutate_input(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        original = {"body": "email"}
        result = wf._apply_setting_defaults(original)

        # The original dict must NOT be modified
        assert "auto_create" not in original
        assert result["auto_create"] is True

    def test_multiple_settings(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
            SettingConfig(key="gmail_sender_filter", type=SettingType.STRING, default="alertasynotificaciones"),
            SettingConfig(key="ynab_access_token", type=SettingType.SECRET, default=None, required=True),
        ])
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
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.MANUAL, {"body": "email"})

        # execute() should have received the enriched data
        assert wf._received_trigger_data["auto_create"] is True
        assert wf._received_trigger_data["body"] == "email"
        assert run.status == RunStatus.SUCCESS

    async def test_run_preserves_explicit_overrides(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.MANUAL, {"body": "email", "auto_create": False})

        assert wf._received_trigger_data["auto_create"] is False
        assert run.status == RunStatus.SUCCESS

    async def test_run_records_enriched_data_in_history(self):
        manifest = _make_manifest(settings=[
            SettingConfig(key="auto_create", type=SettingType.BOOLEAN, default=True),
        ])
        wf = StubWorkflow(manifest)

        run = await wf.run(TriggerType.WEBHOOK, {"body": "email"})

        # The run record should contain the enriched trigger data
        assert run.trigger_data["auto_create"] is True
        assert run.trigger_data["body"] == "email"
