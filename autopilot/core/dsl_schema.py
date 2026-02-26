"""
DSL Schema — Pydantic models for declarative YAML workflow definitions.

Provides the typed schema that validates workflow YAML files before the
``DSLLoader`` instantiates Python objects.  Every field is documented
and constrained so that invalid YAML is caught at parse time, not at
runtime deep inside an executor.

Models:
  - DSLRetryPolicy: Loop retry parameters (max_iterations, condition_expr)
  - DSLStepDef: A single step (function, agent, loop, parallel, sequential)
  - DSLNodeDef: A single DAG node with dependencies
  - DSLWorkflowDef: Top-level workflow definition

Usage (typically called by DSLLoader, not directly)::

    import yaml
    from autopilot.core.dsl_schema import DSLWorkflowDef

    with open("workflow.yaml") as f:
        raw = yaml.safe_load(f)
    definition = DSLWorkflowDef(**raw)
"""

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DSLStepType(str, enum.Enum):
    """Supported step types in a DSL workflow definition."""

    FUNCTION = "function"
    """Plain Python function — auto-wrapped as FunctionalAgent."""

    AGENT = "agent"
    """BaseAgent subclass or factory returning one."""

    LOOP = "loop"
    """Retry loop: runs body until condition_expr passes or max_iterations exhausted."""

    PARALLEL = "parallel"
    """Concurrent execution of multiple child steps."""

    SEQUENTIAL = "sequential"
    """Nested sequential block of child steps."""


class DSLStrategy(str, enum.Enum):
    """Top-level orchestration strategy for the workflow."""

    SEQUENTIAL = "sequential"
    """Linear Pipeline: A → B → C."""

    DAG = "dag"
    """Directed Acyclic Graph: topological parallel execution."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step / Node Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DSLStepDef(BaseModel):
    """
    Definition of a single step in a sequential workflow.

    The ``type`` field determines how the step is interpreted:
      - ``function`` / ``agent``: ``ref`` is required (dotted import path).
      - ``loop``: ``body``, ``condition_expr``, and ``max_iterations`` required.
      - ``parallel`` / ``sequential``: ``children`` required.
    """

    name: str = Field(description="Unique step name within this workflow.")
    type: DSLStepType = Field(description="How to interpret this step.")
    ref: str | None = Field(
        default=None,
        description=(
            "Dotted import path to a callable or BaseAgent subclass. "
            "E.g. 'workflows.bank_to_ynab.agents.categorizer.create_categorizer'."
        ),
    )
    description: str = ""

    # Loop-specific
    body: "DSLStepDef | None" = Field(
        default=None,
        description="Inner step to loop (only for type=loop).",
    )
    condition_expr: str | None = Field(
        default=None,
        description=(
            "Python expression evaluated with 'state' in scope. "
            "Returns True to exit the loop. "
            "E.g. \"state.get('valid', False)\"."
        ),
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Max loop iterations before raising MaxRetriesExceededError.",
    )

    # Parallel / Sequential children
    children: "list[DSLStepDef] | None" = Field(
        default=None,
        description="Child steps (for type=parallel or type=sequential).",
    )


DSLStepDef.model_rebuild()


class DSLNodeDef(BaseModel):
    """
    Definition of a single node in a DAG workflow.

    Unlike ``DSLStepDef``, DAG nodes declare explicit ``dependencies``
    instead of implied sequential order.
    """

    name: str = Field(description="Unique node name within the DAG.")
    ref: str = Field(
        description="Dotted import path to a callable or BaseAgent subclass.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Node names that must complete before this node can run.",
    )
    description: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Top-level Workflow Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DSLWorkflowDef(BaseModel):
    """
    Top-level declarative workflow definition (parsed from YAML).

    Determines how the workflow is orchestrated:
      - ``strategy: sequential`` → uses ``steps`` → produces a ``Pipeline``.
      - ``strategy: dag`` → uses ``nodes`` → produces a ``DAGRunner``.
    """

    name: str = Field(description="Workflow identifier.")
    version: str = "1.0.0"
    description: str = ""
    strategy: DSLStrategy = DSLStrategy.SEQUENTIAL

    # Sequential mode
    steps: list[DSLStepDef] | None = Field(
        default=None,
        description="Ordered list of steps (required when strategy=sequential).",
    )

    # DAG mode
    nodes: list[DSLNodeDef] | None = Field(
        default=None,
        description="DAG node definitions (required when strategy=dag).",
    )

    # Optional initial state seed
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs seeded into the pipeline state before execution.",
    )

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, v, info):
        """Ensure steps is not empty when strategy is sequential."""
        strategy = info.data.get("strategy")
        if strategy == DSLStrategy.SEQUENTIAL and (v is None or len(v) == 0):
            from autopilot.errors import DSLValidationError

            raise DSLValidationError(
                "Strategy 'sequential' requires at least one step.",
                field="steps",
            )
        return v

    @field_validator("nodes")
    @classmethod
    def _validate_nodes(cls, v, info):
        """Ensure nodes is not empty when strategy is dag."""
        strategy = info.data.get("strategy")
        if strategy == DSLStrategy.DAG and (v is None or len(v) == 0):
            from autopilot.errors import DSLValidationError

            raise DSLValidationError(
                "Strategy 'dag' requires at least one node.",
                field="nodes",
            )
        return v
