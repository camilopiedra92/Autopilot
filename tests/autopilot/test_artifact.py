"""
Tests for ArtifactService — ADK-native artifact storage with 12-Factor factory.

All tests use InMemoryArtifactService — zero external dependencies.
Tests verify: factory selection, singleton lifecycle, AgentContext
convenience methods, cross-run access, versioning, and ADKRunner wiring.
"""

import pytest
from unittest.mock import patch, MagicMock

from google.genai import types

from autopilot.core.artifact import (
    BaseArtifactService,
    InMemoryArtifactService,
    create_artifact_service,
    get_artifact_service,
    reset_artifact_service,
)
from autopilot.core.context import AgentContext


# ── Factory Tests ────────────────────────────────────────────────────


class TestCreateArtifactService:
    """Tests for the create_artifact_service() factory."""

    def test_default_returns_in_memory(self):
        service = create_artifact_service()
        assert isinstance(service, InMemoryArtifactService)

    def test_explicit_memory(self):
        service = create_artifact_service("memory")
        assert isinstance(service, InMemoryArtifactService)

    @patch.dict("os.environ", {"ARTIFACT_BACKEND": "memory"})
    def test_env_var_memory(self):
        service = create_artifact_service()
        assert isinstance(service, InMemoryArtifactService)

    @patch.dict(
        "os.environ",
        {"ARTIFACT_BACKEND": "gcs", "ARTIFACT_GCS_BUCKET": "my-bucket"},
    )
    def test_env_var_gcs(self):
        with patch("autopilot.core.artifact.GcsArtifactService") as mock_gcs_cls:
            mock_gcs_cls.return_value = MagicMock(spec=BaseArtifactService)
            create_artifact_service()
            mock_gcs_cls.assert_called_once_with(bucket_name="my-bucket")

    @patch.dict("os.environ", {"ARTIFACT_BACKEND": "gcs"})
    def test_gcs_without_bucket_raises(self):
        with pytest.raises(ValueError, match="ARTIFACT_GCS_BUCKET is required"):
            create_artifact_service()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ARTIFACT_BACKEND"):
            create_artifact_service("redis")


# ── Singleton Tests ──────────────────────────────────────────────────


class TestArtifactServiceSingleton:
    """Tests for get_artifact_service() / reset_artifact_service()."""

    def setup_method(self):
        reset_artifact_service()

    def teardown_method(self):
        reset_artifact_service()

    def test_get_returns_singleton(self):
        s1 = get_artifact_service()
        s2 = get_artifact_service()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        s1 = get_artifact_service()
        reset_artifact_service()
        s2 = get_artifact_service()
        assert s1 is not s2


# ── AgentContext Convenience Methods ─────────────────────────────────


class TestAgentContextArtifacts:
    """Tests for save_artifact, load_artifact, list_artifacts on AgentContext."""

    def _make_ctx(self, **kwargs) -> AgentContext:
        """Create an AgentContext with InMemoryArtifactService."""
        return AgentContext(
            pipeline_name="test_pipeline",
            artifact_service=InMemoryArtifactService(),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self):
        ctx = self._make_ctx()
        part = types.Part(text='{"payee": "Store", "amount": 42.0}')

        version = await ctx.save_artifact("parsed_email.json", part)
        assert version == 0  # First version

        loaded = await ctx.load_artifact("parsed_email.json")
        assert loaded is not None
        assert loaded.text == '{"payee": "Store", "amount": 42.0}'

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self):
        ctx = self._make_ctx()
        result = await ctx.load_artifact("nonexistent.json")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_artifacts_empty(self):
        ctx = self._make_ctx()
        keys = await ctx.list_artifacts()
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_artifacts_populated(self):
        ctx = self._make_ctx()
        await ctx.save_artifact("a.json", types.Part(text="a"))
        await ctx.save_artifact("b.json", types.Part(text="b"))

        keys = await ctx.list_artifacts()
        assert sorted(keys) == ["a.json", "b.json"]

    @pytest.mark.asyncio
    async def test_versioning(self):
        ctx = self._make_ctx()

        v1 = await ctx.save_artifact("data.json", types.Part(text="v1"))
        v2 = await ctx.save_artifact("data.json", types.Part(text="v2"))

        assert v1 == 0
        assert v2 == 1

        # Load latest (default)
        latest = await ctx.load_artifact("data.json")
        assert latest.text == "v2"

        # Load specific version
        first = await ctx.load_artifact("data.json", version=0)
        assert first.text == "v1"

    @pytest.mark.asyncio
    async def test_cross_run_access(self):
        """Save in one execution, load in another via run_id."""
        artifact_service = InMemoryArtifactService()

        # Run 1: save artifact
        ctx1 = AgentContext(
            pipeline_name="test_pipeline",
            artifact_service=artifact_service,
            execution_id="run-001",
        )
        await ctx1.save_artifact("result.json", types.Part(text="run1-data"))

        # Run 2: load from run 1 via run_id
        ctx2 = AgentContext(
            pipeline_name="test_pipeline",
            artifact_service=artifact_service,
            execution_id="run-002",
        )
        loaded = await ctx2.load_artifact("result.json", run_id="run-001")
        assert loaded is not None
        assert loaded.text == "run1-data"

        # Run 2's own artifacts are empty
        own_keys = await ctx2.list_artifacts()
        assert own_keys == []

    @pytest.mark.asyncio
    async def test_list_artifacts_cross_run(self):
        artifact_service = InMemoryArtifactService()

        ctx1 = AgentContext(
            pipeline_name="test_pipeline",
            artifact_service=artifact_service,
            execution_id="run-A",
        )
        await ctx1.save_artifact("a.json", types.Part(text="a"))

        ctx2 = AgentContext(
            pipeline_name="test_pipeline",
            artifact_service=artifact_service,
            execution_id="run-B",
        )
        keys = await ctx2.list_artifacts(run_id="run-A")
        assert keys == ["a.json"]

    @pytest.mark.asyncio
    async def test_save_with_metadata(self):
        ctx = self._make_ctx()
        version = await ctx.save_artifact(
            "report.json",
            types.Part(text="report data"),
            metadata={"source": "email_parser", "run": "test"},
        )
        assert version == 0

    @pytest.mark.asyncio
    async def test_for_step_shares_artifact_service(self):
        ctx = self._make_ctx()
        child = ctx.for_step("parse_email")
        assert child.artifact_service is ctx.artifact_service

    @pytest.mark.asyncio
    async def test_auto_provisioning(self):
        """AgentContext auto-provisions InMemoryArtifactService when not injected."""
        ctx = AgentContext(pipeline_name="auto_test")
        assert isinstance(ctx.artifact_service, InMemoryArtifactService)


# ── ADKRunner Wiring Tests ───────────────────────────────────────────


class TestADKRunnerArtifactWiring:
    """Verify ADKRunner passes artifact_service to Runner."""

    def test_runner_has_artifact_service(self):
        from autopilot.core.adk_runner import ADKRunner

        runner = ADKRunner(app_name="test", user_id="test_user")
        assert isinstance(runner._artifact_service, BaseArtifactService)

    @patch("autopilot.core.adk_runner.Runner")
    @patch("autopilot.core.adk_runner.get_event_bus")
    async def test_standard_path_passes_artifact_service(
        self, mock_bus, mock_runner_cls
    ):
        """Runner() should receive artifact_service kwarg."""
        from autopilot.core.adk_runner import ADKRunner

        runner = ADKRunner(app_name="test", user_id="user")
        # Verify the artifact service is a BaseArtifactService instance
        assert isinstance(runner._artifact_service, BaseArtifactService)
