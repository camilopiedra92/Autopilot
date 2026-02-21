"""
WorkflowRegistry — Auto-discovers and manages all registered workflows.

The registry:
  1. Scans the `workflows/` directory for Python packages
  2. Supports three discovery modes (fallback chain):
     a. Classic:   __init__.py exports a `workflow` BaseWorkflow instance
     b. Auto-class: workflow.py contains a BaseWorkflow subclass → auto-instantiate
     c. Pure YAML:  manifest.yaml only → create BaseWorkflow() directly
  3. Provides CRUD operations: list, get, enable, disable
  4. Tracks run history per workflow

Discovery convention:
  workflows/
    bank_to_ynab/
      __init__.py           # Classic: exports workflow = BankToYnabWorkflow()
      manifest.yaml         # A2A-compatible metadata
      pipeline.yaml         # Optional: DSL pipeline definition
      workflow.py            # Optional: custom execute() logic
      ...
"""

from __future__ import annotations

import importlib
import inspect
import structlog
from pathlib import Path

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowInfo

logger = structlog.get_logger(__name__)


class WorkflowRegistry:
    """
    Discovers and manages all registered workflows.

    Usage:
        registry = WorkflowRegistry()
        registry.discover()  # Scans workflows/ directory

        workflow = registry.get("bank_to_ynab")
        all_workflows = registry.list_all()
    """

    def __init__(self, workflows_dir: str = "workflows"):
        self._workflows: dict[str, BaseWorkflow] = {}
        self._workflows_dir = Path(workflows_dir)

    # ── Discovery ─────────────────────────────────────────────────────

    def discover(self) -> list[str]:
        """
        Scan the workflows directory and register all valid workflows.

        Returns list of discovered workflow names.

        Supports three modes (checked in order):
          1. Classic:    __init__.py with `workflow` export
          2. Auto-class: workflow.py with BaseWorkflow subclass
          3. Pure YAML:  manifest.yaml only

        Directories starting with _ or . are skipped.
        """
        discovered = []

        if not self._workflows_dir.exists():
            logger.warning("workflows_dir_not_found", path=str(self._workflows_dir))
            return discovered

        for item in sorted(self._workflows_dir.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith("_") or item.name.startswith("."):
                continue

            try:
                self._load_workflow(item.name)
                discovered.append(item.name)
            except Exception as e:
                logger.error(
                    "workflow_discovery_failed",
                    workflow=item.name,
                    error=str(e),
                )

        logger.info(
            "workflows_discovered",
            count=len(discovered),
            workflows=discovered,
        )
        return discovered

    def _load_workflow(self, name: str) -> None:
        """
        Load a workflow using the fallback chain:
          1. Classic __init__.py export
          2. Auto-class from workflow.py
          3. Pure YAML from manifest.yaml
        """
        pkg_dir = self._workflows_dir / name

        # Priority 1: Classic __init__.py with `workflow` export
        if self._try_classic_load(name, pkg_dir):
            return

        # Priority 2: workflow.py with BaseWorkflow subclass
        if self._try_auto_class_load(name, pkg_dir):
            return

        # Priority 3: Pure YAML — manifest.yaml only
        if self._try_yaml_only_load(name, pkg_dir):
            return

        raise ImportError(
            f"No valid workflow found in {pkg_dir}. "
            f"Expected one of: __init__.py with `workflow` export, "
            f"workflow.py with BaseWorkflow subclass, or manifest.yaml."
        )

    def _try_classic_load(self, name: str, pkg_dir: Path) -> bool:
        """Priority 1: Classic __init__.py with `workflow` attribute."""
        if not (pkg_dir / "__init__.py").exists():
            return False

        try:
            module_path = f"workflows.{name}"
            module = importlib.import_module(module_path)

            if not hasattr(module, "workflow"):
                return False

            workflow = module.workflow
            if not isinstance(workflow, BaseWorkflow):
                raise TypeError(
                    f"workflows.{name}.workflow must be an instance of BaseWorkflow, "
                    f"got {type(workflow).__name__}"
                )

            self._register_workflow(workflow)
            logger.debug("workflow_loaded_classic", name=name)
            return True
        except ImportError:
            return False

    def _try_auto_class_load(self, name: str, pkg_dir: Path) -> bool:
        """Priority 2: workflow.py contains a BaseWorkflow subclass."""
        if not (pkg_dir / "workflow.py").exists():
            return False

        try:
            module_path = f"workflows.{name}.workflow"
            module = importlib.import_module(module_path)

            # Find the first BaseWorkflow subclass defined in this module
            for attr_name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseWorkflow)
                    and obj is not BaseWorkflow
                    and obj.__module__ == module.__name__
                ):
                    workflow = obj()
                    self._register_workflow(workflow)
                    logger.debug("workflow_loaded_auto_class", name=name, cls=attr_name)
                    return True

            return False
        except ImportError:
            return False

    def _try_yaml_only_load(self, name: str, pkg_dir: Path) -> bool:
        """Priority 3: Pure YAML — only manifest.yaml (+ optional pipeline.yaml)."""
        if not (pkg_dir / "manifest.yaml").exists():
            return False

        try:
            workflow = BaseWorkflow()
            workflow._workflow_dir = pkg_dir
            self._register_workflow(workflow)
            logger.debug("workflow_loaded_yaml_only", name=name)
            return True
        except Exception:
            return False

    def _register_workflow(self, workflow: BaseWorkflow) -> None:
        """Register a workflow instance internally."""
        self._workflows[workflow.manifest.name] = workflow
        logger.info(
            "workflow_registered",
            name=workflow.manifest.name,
            display_name=workflow.manifest.display_name,
            version=workflow.manifest.version,
            triggers=[t.type.value for t in workflow.manifest.triggers],
        )

    # ── Registration (manual) ─────────────────────────────────────────

    def register(self, workflow: BaseWorkflow) -> None:
        """Manually register a workflow instance."""
        self._register_workflow(workflow)

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, workflow_id: str) -> BaseWorkflow | None:
        """Get a workflow by its ID."""
        return self._workflows.get(workflow_id)

    def get_or_raise(self, workflow_id: str) -> BaseWorkflow:
        """Get a workflow by ID or raise KeyError."""
        wf = self._workflows.get(workflow_id)
        if wf is None:
            raise KeyError(f"Workflow '{workflow_id}' not found in registry.")
        return wf

    # ── Listing ───────────────────────────────────────────────────────

    def list_all(self) -> list[WorkflowInfo]:
        """List all registered workflows with summary info."""
        return [
            WorkflowInfo(
                name=wf.manifest.name,
                display_name=wf.manifest.display_name,
                description=wf.manifest.description,
                version=wf.manifest.version,
                icon=wf.manifest.icon,
                color=wf.manifest.color,
                enabled=wf.manifest.enabled,
                triggers=wf.manifest.triggers,
                tags=wf.manifest.tags,
                last_run=wf.last_run,
                total_runs=wf.total_runs,
                success_rate=wf.success_rate,
            )
            for wf in self._workflows.values()
        ]

    def get_all_workflows(self) -> list[BaseWorkflow]:
        """
        Get all workflow instances (Platform internal use).

        Returns:
            List of BaseWorkflow objects.
        """
        return list(self._workflows.values())

    def list_names(self) -> list[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())

    # ── Management ────────────────────────────────────────────────────

    def enable(self, workflow_id: str) -> None:
        """Enable a workflow."""
        wf = self.get_or_raise(workflow_id)
        wf.manifest.enabled = True
        logger.info("workflow_enabled", name=workflow_id)

    def disable(self, workflow_id: str) -> None:
        """Disable a workflow."""
        wf = self.get_or_raise(workflow_id)
        wf.manifest.enabled = False
        logger.info("workflow_disabled", name=workflow_id)

    @property
    def count(self) -> int:
        return len(self._workflows)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def setup_all(self) -> None:
        """Call setup() on all registered workflows."""
        for name, wf in self._workflows.items():
            try:
                await wf.setup()
                logger.info("workflow_setup_complete", name=name)
            except Exception as e:
                logger.error("workflow_setup_failed", name=name, error=str(e))

    async def teardown_all(self) -> None:
        """Call teardown() on all registered workflows."""
        for name, wf in self._workflows.items():
            try:
                await wf.teardown()
            except Exception as e:
                logger.error("workflow_teardown_failed", name=name, error=str(e))

    # ── Trigger Matching ──────────────────────────────────────────────

    def find_by_webhook_path(self, path: str) -> BaseWorkflow | None:
        """Find workflow that handles a given webhook path."""
        for wf in self._workflows.values():
            if not wf.manifest.enabled:
                continue
            for trigger in wf.manifest.triggers:
                if trigger.type.value == "webhook" and trigger.path == path:
                    return wf
        return None

    def find_by_gmail_filter(self, sender: str) -> list[BaseWorkflow]:
        """Find all workflows that handle Gmail push from a given sender."""
        results = []
        for wf in self._workflows.values():
            if not wf.manifest.enabled:
                continue
            for trigger in wf.manifest.triggers:
                if trigger.type.value == "gmail_push":
                    if trigger.filter and trigger.filter in sender:
                        results.append(wf)
        return results

    def find_scheduled(self) -> list[BaseWorkflow]:
        """Find all workflows with scheduled triggers."""
        results = []
        for wf in self._workflows.values():
            if not wf.manifest.enabled:
                continue
            for trigger in wf.manifest.triggers:
                if trigger.type.value == "scheduled":
                    results.append(wf)
        return results


# ── Singleton ─────────────────────────────────────────────────────────

_registry: WorkflowRegistry | None = None


def get_registry() -> WorkflowRegistry:
    """Get or create the global WorkflowRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = WorkflowRegistry()
    return _registry
