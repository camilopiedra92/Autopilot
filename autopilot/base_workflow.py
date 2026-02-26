"""
BaseWorkflow — Smart base class for all workflows.

Every workflow in the platform extends this class. It provides:
  - Auto-loading of manifest.yaml (A2A-compatible metadata)
  - Auto-execution of pipeline.yaml (DSL pipeline) when present
  - Auto-discovery of .agent.yaml cards
  - Built-in execution timing, error handling, and observability
  - Standard interface for health checks and lifecycle hooks

Three workflow levels, one single path:
  1. Pure YAML:     manifest.yaml + pipeline.yaml → 0 Python
  2. Minimal:       manifest.yaml + workflow.py (override execute()) → ~10 lines
  3. Full Custom:   manifest.yaml + workflow.py + agents/ → as needed

Usage:
    # Simplest: just override execute()
    class MyWorkflow(BaseWorkflow):
        async def execute(self, trigger_data: dict) -> WorkflowResult:
            ...

    # Or define pipeline.yaml and let BaseWorkflow auto-run it.
"""

import inspect
import time
import structlog
from pathlib import Path
from typing import Any
from uuid import uuid4

from autopilot.models import (
    WorkflowManifest,
    WorkflowResult,
    WorkflowRun,
    RunStatus,
    TriggerType,
)

logger = structlog.get_logger(__name__)


class BaseWorkflow:
    """
    Smart base class for all workflows.

    Auto-loads manifest.yaml and pipeline.yaml from the workflow directory.
    Subclasses only need to override execute() for custom logic.

    Subclasses MAY override:
      - manifest (property): Only if dynamic manifest generation is needed
      - execute(trigger_data): Custom workflow logic (default: run pipeline.yaml)
      - strategy (property): Return OrchestrationStrategy (default: SEQUENTIAL)
      - build_dag(): Return a DAGRunner for DAG-based workflows
      - health_check(): Custom health validation
      - setup() / teardown(): Lifecycle hooks
    """

    def __init__(self):
        self._runs: list[WorkflowRun] = []
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._manifest: WorkflowManifest | None = None
        self._agent_cards: list | None = None
        # Auto-resolve workflow directory from the subclass file location
        self._workflow_dir: Path = Path(inspect.getfile(type(self))).parent

    # ── Manifest (Auto-loaded) ────────────────────────────────────────

    @property
    def manifest(self) -> WorkflowManifest:
        """
        Auto-load manifest from manifest.yaml in the workflow directory.

        Override only if you need a dynamic manifest (rare).
        """
        if self._manifest is None:
            from autopilot.config_loader import load_manifest

            self._manifest = load_manifest(self._workflow_dir)
        return self._manifest

    # ── Execute (Auto-runs pipeline.yaml or override) ─────────────────

    async def execute(self, trigger_data: dict[str, Any]) -> WorkflowResult:
        """
        Execute the workflow with the given trigger data.

        Default behavior:
          - If pipeline.yaml exists → auto-build and run DSL pipeline
          - Otherwise → raise NotImplementedError (subclass must override)

        Args:
            trigger_data: Data from the trigger (e.g., email body, webhook payload).

        Returns:
            WorkflowResult with status, data, and any errors.
        """
        pipeline_path = self._workflow_dir / "pipeline.yaml"
        if pipeline_path.exists():
            return await self._execute_dsl_pipeline(trigger_data, pipeline_path)

        raise NotImplementedError(
            f"Workflow '{self.manifest.name}' has no pipeline.yaml and "
            f"does not override execute(). Either create a pipeline.yaml "
            f"or implement execute() in your workflow class."
        )

    async def _execute_dsl_pipeline(
        self,
        trigger_data: dict[str, Any],
        pipeline_path: Path,
    ) -> WorkflowResult:
        """
        Load and execute a DSL pipeline from a YAML file.

        Bridges the DSL loader (Phase 6) with the BaseWorkflow contract,
        producing a standard WorkflowResult.
        """
        from autopilot.core.dsl_loader import load_workflow
        from autopilot.core.context import AgentContext

        try:
            pipeline = load_workflow(str(pipeline_path))
            ctx = AgentContext(
                pipeline_name=self.manifest.name,
                metadata={"source": "pipeline.yaml"},
            )
            result = await pipeline.execute(ctx, initial_input=trigger_data)

            # Transfer session events to long-term memory (workflow opt-in)
            if self.manifest.memory and ctx.session:
                try:
                    await ctx.memory.add_session_to_memory(ctx.session)
                except Exception as exc:
                    logger.warning(
                        "dsl_memory_transfer_failed",
                        workflow=self.manifest.name,
                        error=str(exc),
                    )

            return WorkflowResult(
                workflow_id=self.manifest.name,
                status=RunStatus.SUCCESS,
                data=result.state if hasattr(result, "state") else {"result": result},
            )
        except Exception as e:
            import traceback

            print("=== RAW DSL PIPELINE PIPELINE ERROR ===")
            traceback.print_exc()
            logger.error(
                "dsl_pipeline_failed",
                workflow=self.manifest.name,
                error=str(e),
            )
            return WorkflowResult(
                workflow_id=self.manifest.name,
                status=RunStatus.FAILED,
                error=str(e),
            )

    # ── Agent Cards (Auto-discovered) ─────────────────────────────────

    def get_agent_cards(self) -> list[Any]:
        """
        Auto-discover .agent.yaml cards from the agents/ directory.

        Override only if you need custom agent card loading logic.
        """
        if self._agent_cards is None:
            agents_dir = self._workflow_dir / "agents"
            if agents_dir.exists():
                try:
                    from autopilot.agents.agent_cards import discover_agent_cards

                    self._agent_cards = discover_agent_cards(agents_dir)
                except Exception as e:
                    logger.warning(
                        "agent_cards_discovery_failed",
                        workflow_dir=str(self._workflow_dir),
                        error=str(e),
                    )
                    self._agent_cards = []
            else:
                self._agent_cards = []
        return self._agent_cards

    # ── Orchestration Strategy (V3) ───────────────────────────────────

    @property
    def strategy(self):
        """
        Return the orchestration strategy for this workflow.

        Default is SEQUENTIAL (standard linear pipeline).
        Override to return ``OrchestrationStrategy.DAG`` and implement
        ``build_dag()`` for graph-based workflows.
        """
        from autopilot.core.orchestrator import OrchestrationStrategy

        return OrchestrationStrategy.SEQUENTIAL

    def build_dag(self):
        """
        Build and return a DAGRunner for this workflow.

        Only called when ``strategy`` returns ``OrchestrationStrategy.DAG``.
        Subclasses that use DAG orchestration MUST override this method.

        Returns:
            A ``DAGRunner`` instance ready for execution.

        Raises:
            NotImplementedError: If not overridden by a DAG workflow.
        """
        raise NotImplementedError(
            f"Workflow '{self.manifest.name}' declares DAG strategy "
            f"but does not implement build_dag()."
        )

    # ── Optional Lifecycle Hooks ──────────────────────────────────────

    async def setup(self) -> None:
        """Lifecycle hook — called once after registration.

        Hydrates run stats from durable backend on cold start.
        Registers EventBus subscribers for HITL resume and manual trigger.
        Override in subclasses for workflow-specific setup (call super!).
        """
        try:
            from autopilot.core.run_log import get_run_log_service

            svc = get_run_log_service()
            stats = await svc.get_stats(self.manifest.name)
            self._total_runs = stats.get("total", 0)
            self._successful_runs = stats.get("successful", 0)

            # Load most recent run for last_run property (uses efficient get_latest_run)
            latest = await svc.get_latest_run(self.manifest.name)
            if latest:
                self._runs = [latest]
            logger.debug(
                "run_log_hydrated",
                workflow=self.manifest.name,
                total_runs=self._total_runs,
                successful_runs=self._successful_runs,
            )
        except Exception:
            logger.debug("run_log_hydration_skipped", workflow=self.manifest.name)

        # Register EventBus subscribers for API HITL & triggers
        try:
            from autopilot.core.subscribers import get_subscriber_registry

            registry = get_subscriber_registry()
            registry.add(
                "api.hitl_resumed",
                self._on_hitl_resumed,
                source=f"BaseWorkflow:{self.manifest.name}",
            )
            registry.add(
                "api.workflow_triggered",
                self._on_manual_trigger,
                source=f"BaseWorkflow:{self.manifest.name}",
            )
            logger.debug(
                "api_subscribers_registered",
                workflow=self.manifest.name,
            )
        except Exception:
            logger.debug("api_subscribers_skipped", workflow=self.manifest.name)

    async def teardown(self) -> None:
        """Called when the platform shuts down. Override for cleanup."""
        pass

    async def _on_hitl_resumed(self, msg) -> None:
        """Handle HITL resume events from the API.

        Filters for this workflow, then re-runs with the human-override
        payload injected as trigger_data. The hitl_approved flag signals
        the pipeline that human approval was granted.
        """
        payload = msg.payload if hasattr(msg, "payload") else msg
        target_workflow = payload.get("workflow_id", "")

        if target_workflow != self.manifest.name:
            return  # Not for us — ignore

        run_id = payload.get("run_id", "")
        hitl_payload = payload.get("payload", {})

        resume_data = {
            **hitl_payload,
            "hitl_approved": True,
            "__resume_run_id__": run_id,
        }

        try:
            await self.run(TriggerType.MANUAL, resume_data)
        except Exception as exc:
            logger.error(
                "hitl_resume_failed",
                workflow=self.manifest.name,
                run_id=run_id,
                error=str(exc),
            )

    async def _on_manual_trigger(self, msg) -> None:
        """Handle manual workflow trigger events from the API.

        Filters for this workflow, then runs with the provided payload.
        """
        payload = msg.payload if hasattr(msg, "payload") else msg
        target_workflow = payload.get("workflow_id", "")

        if target_workflow != self.manifest.name:
            return  # Not for us — ignore

        trigger_payload = payload.get("payload", {})

        try:
            await self.run(TriggerType.MANUAL, trigger_payload)
        except Exception as exc:
            logger.error(
                "manual_trigger_failed",
                workflow=self.manifest.name,
                error=str(exc),
            )

    # ── Trigger Matching ──────────────────────────────────────────────

    def _matches_gmail_trigger(self, email_data: dict) -> bool:
        """
        Check if an email matches this workflow's GMAIL_PUSH trigger config.

        Uses the same sender/label matching logic that was previously in
        ``WorkflowRouter.route_gmail_push()``, but as an instance method
        so workflows can self-match when reacting to ``email.received`` events.

        Label matching checks both ``labelIds`` (raw Gmail IDs like
        ``Label_123``) and ``labelNames`` (resolved names like
        ``Bancolombia``), so manifests can specify human-readable names.

        Args:
            email_data: Dict containing 'sender'/'from', 'label_ids'/'labelIds',
                and optionally 'labelNames'.

        Returns:
            True if the email matches at least one GMAIL_PUSH trigger.
        """
        if not self.manifest.enabled:
            return False

        sender = (email_data.get("sender", "") or email_data.get("from", "")).lower()
        # Collect all label identifiers: IDs + resolved names
        email_label_ids = set(
            email_data.get("label_ids", []) or email_data.get("labelIds", [])
        )
        email_label_names = set(email_data.get("labelNames", []))
        all_email_labels = email_label_ids | email_label_names

        for trigger in self.manifest.triggers:
            if trigger.type != TriggerType.GMAIL_PUSH:
                continue

            # Match sender filter
            if trigger.filter and trigger.filter.lower() not in sender:
                continue

            # Match label IDs (checks against both raw IDs and resolved names)
            if trigger.label_ids and not set(trigger.label_ids).intersection(
                all_email_labels
            ):
                continue

            return True  # Match found

        return False

    async def health_check(self) -> bool:
        """Check if the workflow is healthy and ready to execute."""
        return True

    def get_health_info(self) -> dict:
        """
        Return a rich health-info dict for the platform /health endpoint.

        Override in subclasses to add workflow-specific details
        (e.g., connector status, ingestion mode, feature flags).
        The platform aggregates these dicts per-workflow.
        """
        last = self.last_run
        return {
            "status": "healthy",
            "version": self.manifest.version,
            "total_runs": self._total_runs,
            "success_rate": self.success_rate,
            "last_run": {
                "id": last.id,
                "status": last.status.value,
                "duration_ms": last.duration_ms,
            }
            if last
            else None,
        }

    def register_routes(self, app) -> None:
        """
        Register workflow-specific FastAPI routes on the app.

        Override this to add custom HTTP endpoints for your workflow.
        All workflow logic stays self-contained — the platform app
        simply calls this hook during startup.

        Args:
            app: The FastAPI application instance.
        """
        pass

    # ── Execution Wrapper ─────────────────────────────────────────────

    def _apply_setting_defaults(self, trigger_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply manifest setting defaults to trigger data.

        The workflow knows its own settings (declared in manifest.yaml).
        Before execution, it merges any default values into the trigger
        data — but ONLY when the key is NOT already present, so explicit
        caller values always take priority.

        This is the canonical place for setting resolution. No external
        component (router, CLI, tests) needs to know about or inject
        workflow-specific settings — the workflow is self-contained.

        Example manifest.yaml:
            settings:
              - key: auto_create
                type: boolean
                default: true

        Effect: `auto_create=True` is always available in the pipeline
        state, unless the caller explicitly passes `auto_create=False`.
        """
        enriched = dict(trigger_data)  # Shallow copy to avoid mutation
        for setting in self.manifest.settings:
            if setting.key not in enriched and setting.default is not None:
                enriched[setting.key] = setting.default
        return enriched

    async def run(
        self,
        trigger_type: TriggerType,
        trigger_data: dict[str, Any],
    ) -> WorkflowRun:
        """
        Run the workflow with full lifecycle management.

        This is THE single entry point for all workflow executions.
        It wraps execute() with:
          - Manifest setting default injection (self-contained config)
          - Timing and observability
          - Error handling
          - Run history tracking
        """
        run_id = f"run_{uuid4().hex[:12]}"
        from datetime import datetime, timezone

        # Apply manifest setting defaults before execution
        enriched_data = self._apply_setting_defaults(trigger_data)

        run = WorkflowRun(
            id=run_id,
            workflow_id=self.manifest.name,
            status=RunStatus.RUNNING,
            trigger_type=trigger_type,
            trigger_data=enriched_data,
            started_at=datetime.now(timezone.utc),
        )

        logger.info(
            "workflow_run_started",
            workflow=self.manifest.name,
            run_id=run_id,
            trigger_type=trigger_type.value,
        )

        start = time.monotonic()

        try:
            result = await self.execute(enriched_data)

            run.status = result.status
            run.result = result.data
            run.error = result.error

            if result.status == RunStatus.SUCCESS:
                self._successful_runs += 1

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
            logger.error(
                "workflow_run_failed",
                workflow=self.manifest.name,
                run_id=run_id,
                error=str(e),
            )

        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            run.duration_ms = round(elapsed_ms, 2)
            run.completed_at = datetime.now(timezone.utc)
            self._total_runs += 1

            # Keep last 100 runs in memory
            self._runs.append(run)
            if len(self._runs) > 100:
                self._runs = self._runs[-100:]

            # Persist to durable backend (fire-and-forget — errors logged, not raised)
            try:
                from autopilot.core.run_log import get_run_log_service

                await get_run_log_service().save_run(run)
            except Exception as exc:
                logger.warning(
                    "run_log_persist_failed",
                    run_id=run.id,
                    workflow=self.manifest.name,
                    error=str(exc),
                )

            logger.info(
                "workflow_run_completed",
                workflow=self.manifest.name,
                run_id=run_id,
                status=run.status.value,
                duration_ms=run.duration_ms,
            )

        return run

    # ── Stats ─────────────────────────────────────────────────────────

    @property
    def total_runs(self) -> int:
        return self._total_runs

    @property
    def success_rate(self) -> float:
        if self._total_runs == 0:
            return 0.0
        return round(self._successful_runs / self._total_runs * 100, 1)

    @property
    def last_run(self) -> WorkflowRun | None:
        return self._runs[-1] if self._runs else None

    @property
    def recent_runs(self) -> list[WorkflowRun]:
        return list(reversed(self._runs[-20:]))
