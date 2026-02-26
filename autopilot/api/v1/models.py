"""
V1 API Response Models — Schema-first Pydantic models.

These models define the exact JSON shape returned by each
/api/v1/* endpoint. All are read-only data projections
composed from platform primitives (WorkflowManifest, AgentCard, etc).

Design principles:
  - Compose from existing platform types (WorkflowRun, AgentCard, TriggerConfig)
  - Never duplicate field definitions — import and reuse
  - Every model has Field descriptions for auto-generated OpenAPI docs
  - All models are immutable data projections (no methods, no side effects)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from autopilot.models import (
    AgentType,
    RunStatus,
    TriggerConfig,
    TriggerType,
    WorkflowRun,
)


# ── Pipeline Graph ───────────────────────────────────────────────────


class PipelineNode(BaseModel):
    """A single node (step) in the pipeline graph."""

    name: str = Field(description="Unique node identifier (e.g., 'email_parser')")
    type: str = Field(description="Node type: 'agent', 'function', 'parallel', 'loop'")
    ref: str = Field(
        default="", description="Dotted ref path (e.g., 'steps.parse_email')"
    )
    description: str = ""
    dependencies: list[str] = Field(
        default_factory=list, description="Node names this depends on"
    )
    layer: int = Field(
        default=0, description="Topological layer (0 = roots, 1 = first deps, ...)"
    )


class PipelineEdge(BaseModel):
    """A directed edge in the pipeline graph."""

    source: str = Field(description="Source node name")
    target: str = Field(description="Target node name")


class PipelineGraph(BaseModel):
    """Complete pipeline topology — nodes, edges, and layers for visualization.

    The `layers` field groups nodes by topological depth (Kahn's algorithm).
    Layer 0 = root nodes (no dependencies), Layer N depends on Layer N-1.
    Nodes within the same layer can execute concurrently (DAG strategy).
    """

    strategy: str = Field(
        description="Orchestration strategy: SEQUENTIAL, DAG, REACT, ROUTER"
    )
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    layers: list[list[str]] = Field(
        default_factory=list,
        description="Nodes grouped by topological layer [[roots], [layer1], ...]",
    )


# ── Workflow Detail ───────────────────────────────────────────────


class WorkflowDetail(BaseModel):
    """Enriched workflow info for the API overview.

    Composes data from:
      - WorkflowManifest (static metadata)
      - pipeline.yaml (strategy, step count)
      - RunLogService (total_runs, success_rate, last_run)
    """

    id: str
    display_name: str
    description: str
    version: str
    icon: str
    color: str
    enabled: bool
    triggers: list[TriggerConfig]
    tags: list[str]
    strategy: str = ""
    step_count: int = 0
    agent_count: int = 0
    total_runs: int = 0
    success_rate: float = 0.0
    last_run: WorkflowRun | None = None


# ── Token Economics & Metrics ────────────────────────────────────────


class TokenMetrics(BaseModel):
    """Token usage metrics for a single step or run.

    Extracted from ADKRunner LLM artifact `.llm.json` files in GCS.
    `est_cost_usd` is computed from the Gemini pricing model.
    """

    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    compression_events: int = 0
    est_cost_usd: float = 0.0


# ── Run Trace ────────────────────────────────────────────────────────


class RunStepTrace(BaseModel):
    """Trace data for a single step within a run.

    Composed from two GCS artifact types:
      - `{step}.json` — Pipeline/DAG node output (output, duration_ms)
      - `{step}.llm.json` — ADKRunner LLM response (optional companion)
    """

    name: str
    artifact_key: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0
    has_llm_response: bool = False
    llm_response: dict[str, Any] | None = None
    token_metrics: TokenMetrics | None = None


class RunTrace(BaseModel):
    """Full execution trace for a run — composed from WorkflowRun + GCS artifacts.

    The `run` field contains the durable run record from RunLogService.
    The `steps` field contains per-step artifact data loaded from GCS.
    """

    run: WorkflowRun
    steps: list[RunStepTrace] = Field(default_factory=list)


class PaginationMeta(BaseModel):
    """Cursor-based pagination metadata.

    `next_cursor` is an opaque token. Pass it as `start_after` query param
    on the next request. `None` means no more results.
    """

    next_cursor: str | None = None


class PaginatedRuns(BaseModel):
    """Paginated list of workflow runs with cursor metadata."""

    runs: list[WorkflowRun]
    meta: PaginationMeta


# ── Agent Card Response ──────────────────────────────────────────────


class AgentCardResponse(BaseModel):
    """Agent card enriched for API display.

    Flattens nested AgentCard fields (tools, guardrails, I/O schemas)
    into a flat structure suitable for frontend consumption.
    """

    name: str
    display_name: str
    type: AgentType
    description: str = ""
    model: str | None = None
    stage: int = 0
    tools: list[str] = Field(default_factory=list)
    guardrails_before: list[str] = Field(default_factory=list)
    guardrails_after: list[str] = Field(default_factory=list)
    input_schema: str | None = None
    output_schema: str | None = None


# ── Event ────────────────────────────────────────────────────────────


class EventItem(BaseModel):
    """EventBus message formatted for API display.

    Projection of `AgentMessage` with only the fields relevant
    to the event log and SSE stream.
    """

    topic: str
    sender: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""
    correlation_id: str = ""


# ── Phase 2: HITL Models ────────────────────────────────────────────


class PendingRunItem(BaseModel):
    """Minimal projection of a PAUSED run for the pending-action list.

    Only includes fields needed for a summary card:
    workflow context, timing, and the trigger that created it.
    """

    run_id: str = Field(description="Unique run identifier")
    workflow_id: str = Field(description="Parent workflow name")
    status: RunStatus = RunStatus.PAUSED
    trigger_type: TriggerType = TriggerType.MANUAL
    started_at: datetime | None = None
    paused_reason: str = Field(
        default="",
        description="Human-readable reason for the pause (e.g., 'Requires approval')",
    )


class ResumeRunRequest(BaseModel):
    """Payload sent to resume a PAUSED run.

    The `payload` dict is injected into the workflow's trigger_data
    as the human-override context (e.g., approved=True, notes='OK').
    """

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Human-override data merged into trigger_data on resume",
    )


class ResumeRunResponse(BaseModel):
    """Confirmation that a resume event was dispatched to EventBus."""

    run_id: str
    workflow_id: str
    status: str = Field(
        default="dispatched",
        description="Always 'dispatched' — actual resume happens async",
    )


# ── Phase 2: Manual Trigger Models ──────────────────────────────────


class TriggerWorkflowRequest(BaseModel):
    """Request body for manual workflow trigger.

    `payload` is passed as trigger_data to the workflow execution.
    Empty payload is valid (workflow uses manifest defaults).
    """

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom trigger data for the workflow",
    )


class TriggerWorkflowResponse(BaseModel):
    """Confirmation that a trigger event was dispatched."""

    workflow_id: str
    status: str = Field(
        default="dispatched",
        description="Always 'dispatched' — workflow runs async via EventBus",
    )
    trigger_type: str = "MANUAL"


# ── Phase 2: Copilot Models ─────────────────────────────────────────


class CopilotQuery(BaseModel):
    """Natural language question for the platform copilot.

    The query should be about workflow failures, statistics,
    event timelines, or run history. The copilot uses read-only
    tools to answer.
    """

    query: str = Field(
        ...,
        min_length=3,
        description="Natural language question (e.g., 'Which workflows failed today?')",
    )


class CopilotToolCall(BaseModel):
    """Record of a tool invocation made by the copilot during reasoning."""

    tool: str = Field(description="Tool name (e.g., 'get_recent_errors')")
    args: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = Field(
        default="",
        description="Brief summary of what the tool returned",
    )


class CopilotResponse(BaseModel):
    """Structured response from the platform copilot."""

    answer: str = Field(description="Synthesized natural language answer")
    tools_used: list[CopilotToolCall] = Field(
        default_factory=list,
        description="Tools invoked during reasoning (for transparency)",
    )
    iterations: int = Field(
        default=0,
        description="Number of ReAct iterations taken",
    )


# ── Phase 3: Cancel & Delete Run Models ─────────────────────────────


class CancelRunResponse(BaseModel):
    """Confirmation that a run was cancelled."""

    run_id: str
    workflow_id: str
    status: str = Field(
        default="cancelled",
        description="Always 'cancelled' after successful cancellation",
    )


class DeleteRunResponse(BaseModel):
    """Confirmation that a run was deleted from history."""

    run_id: str
    workflow_id: str
    deleted: bool = Field(
        default=True,
        description="Always True after successful deletion",
    )


# ── Phase 3: Platform Stats ─────────────────────────────────────────


class WorkflowStatsItem(BaseModel):
    """Per-workflow stats for the platform overview."""

    workflow_id: str
    display_name: str
    total_runs: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0
    enabled: bool = True


class PlatformStatsResponse(BaseModel):
    """Global platform statistics aggregated across all workflows."""

    total_workflows: int = Field(description="Total registered workflows")
    enabled_workflows: int = Field(description="Currently enabled workflows")
    total_runs: int = Field(description="Total runs across all workflows")
    total_successful: int = 0
    total_failed: int = 0
    global_success_rate: float = Field(description="Overall success rate (%)")
    top_workflow: str | None = Field(
        default=None,
        description="Workflow with the most runs",
    )
    workflows: list[WorkflowStatsItem] = Field(default_factory=list)
    bus_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="EventBus stats (published, delivered, errors)",
    )


# ── Phase 3: Workflow Toggle ────────────────────────────────────────


class WorkflowToggleRequest(BaseModel):
    """Request body for toggling a workflow's enabled state."""

    enabled: bool = Field(description="Set to true to enable, false to disable")


class WorkflowToggleResponse(BaseModel):
    """Confirmation of a workflow state change."""

    workflow_id: str
    enabled: bool
    status: str = Field(
        default="updated",
        description="Always 'updated' after successful toggle",
    )
