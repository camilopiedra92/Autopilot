# Phase 1A. Pydantic Response Models — Schema-First Dashboard API

> **Status**: ✅ COMPLETED  
> **Completed**: 2026-02-26  
> **Effort**: ~25 min  
> **Type**: NEW (API Layer)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 1A  
> **Depends on**: Phase 0 complete (RunLogService, Error Taxonomy, BaseWorkflow integration)

---

## Problem Statement

The Dashboard API needs **typed, documented response shapes** before any endpoint logic is written. Without pre-defined schemas:

- Endpoints return ad-hoc dicts — no IDE autocomplete, no contract validation, no auto-generated OpenAPI docs.
- Frontend consumers can't code-gen TypeScript types from the OpenAPI spec.
- Response shape drift between endpoints is invisible until production.

**Schema-First** (ARCHITECTURE.md §9 Rule 2): _All Pydantic models before logic._

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                 | Current                 | Target                                      |
| ----------------------- | ------------------------------------------- | ----------------------- | ------------------------------------------- |
| §9 Rule 2               | Schema-First — Pydantic models before logic | No dashboard models     | Complete response model hierarchy           |
| §1 Core Philosophy      | Headless API — JSON responses only          | V1 routes use raw dicts | Typed Pydantic models for all responses     |
| §3 Core Primitives      | Reuse existing platform types               | N/A                     | Compose from `WorkflowRun`, `AgentCard`     |
| §9.3 Error Taxonomy     | Use `AutoPilotError` subclasses             | N/A                     | `DashboardWorkflowNotFoundError` (Phase 0B) |

---

## Prerequisites

- Phase 0 fully complete (`RunLogService`, `RunStatus.PAUSED`, `DashboardError` hierarchy).
- These platform types must be importable:
  - `autopilot.models.AgentCard`, `AgentType`, `RunStatus`, `TriggerConfig`, `TriggerType`, `WorkflowRun`

**Verify prerequisites**:

```bash
python -c "from autopilot.models import AgentCard, AgentType, RunStatus, TriggerConfig, TriggerType, WorkflowRun; print('OK')"
python -c "from autopilot.errors import DashboardWorkflowNotFoundError, RunNotFoundError; print('OK')"
```

---

## Implementation

### Step 1: Create `autopilot/api/v1/dashboard_models.py` [NEW]

Create this file with the **complete** contents below. This is the entire file — do not add anything extra.

```python
"""
Dashboard API Response Models — Schema-first Pydantic models.

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
    AgentCard,
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
    ref: str = Field(default="", description="Dotted ref path (e.g., 'steps.parse_email')")
    description: str = ""
    dependencies: list[str] = Field(default_factory=list, description="Node names this depends on")
    layer: int = Field(default=0, description="Topological layer (0 = roots, 1 = first deps, ...)")


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

    strategy: str = Field(description="Orchestration strategy: SEQUENTIAL, DAG, REACT, ROUTER")
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    layers: list[list[str]] = Field(
        default_factory=list,
        description="Nodes grouped by topological layer [[roots], [layer1], ...]",
    )


# ── Dashboard Workflow ───────────────────────────────────────────────


class DashboardWorkflow(BaseModel):
    """Enriched workflow info for the dashboard overview.

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
    """Agent card enriched for dashboard display.

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
    """EventBus message formatted for dashboard display.

    Projection of `AgentMessage` with only the fields relevant
    to the dashboard's event log and SSE stream.
    """

    topic: str
    sender: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""
    correlation_id: str = ""
```

### Step 2: Verify compilation

```bash
python -c "from autopilot.api.v1.routes_models import DashboardWorkflow, PipelineGraph, RunTrace, AgentCardResponse, EventItem; print('OK')"
```

---

## Model Dependency Graph

```mermaid
graph TD
    subgraph Platform Types (existing)
        WR[WorkflowRun]
        AC[AgentCard]
        AT[AgentType]
        TC[TriggerConfig]
        RS[RunStatus]
    end

    subgraph Dashboard Models (new)
        DW[DashboardWorkflow]
        PG[PipelineGraph]
        PN[PipelineNode]
        PE[PipelineEdge]
        TM[TokenMetrics]
        RST[RunStepTrace]
        RT[RunTrace]
        PM[PaginationMeta]
        PR[PaginatedRuns]
        ACR[AgentCardResponse]
        EI[EventItem]
    end

    DW --> TC
    DW --> WR
    PG --> PN
    PG --> PE
    RST --> TM
    RT --> WR
    RT --> RST
    PR --> WR
    PR --> PM
    ACR --> AT
```

---

## Model-to-Endpoint Mapping

| Model               | Endpoint                                         | HTTP Method |
| ------------------- | ------------------------------------------------ | ----------- |
| `DashboardWorkflow` | `/api/v1/workflows`                    | GET         |
| `DashboardWorkflow` | `/api/v1/workflows/{id}`               | GET         |
| `PipelineGraph`     | `/api/v1/workflows/{id}/pipeline`      | GET         |
| `AgentCardResponse` | `/api/v1/workflows/{id}/agents`        | GET         |
| `PaginatedRuns`     | `/api/v1/workflows/{id}/runs`          | GET         |
| `RunTrace`          | `/api/v1/workflows/{id}/runs/{run_id}` | GET         |
| `EventItem`         | `/api/v1/events`                       | GET         |
| `EventItem`         | `/api/v1/events/stream` (SSE)          | GET         |

---

## Design Decisions

| Decision                                                    | Rationale                                                                                       |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Compose from existing platform types instead of duplicating | Prevents schema drift; `WorkflowRun` changes automatically propagate to `RunTrace`              |
| `PipelineGraph.layers` as `list[list[str]]`                 | Matches Kahn's algorithm output directly; frontend can render layers as swim lanes              |
| `TokenMetrics` as separate model                            | Reused in both `RunStepTrace` and future per-run aggregate metrics                              |
| `PaginationMeta` as wrapper                                 | Keeps pagination concerns separate from business data; extensible for future `total_count`      |
| `AgentCardResponse` flattens nested `AgentCard`             | Dashboard frontend doesn't need full `GuardrailConfig` / `ToolConfig` nesting — flat is simpler |
| `EventItem` as projection of `AgentMessage`                 | Only exposes dashboard-relevant fields; hides internal bus metadata                             |
| All fields have defaults                                    | Prevents 500 errors from missing optional data (e.g., new workflow with zero runs)              |
| `from __future__ import annotations`                        | Enables `X                                                                                      | None` syntax and lazy annotation evaluation per ADK Import Policy |

---

## Files Modified

| File                                   | Change                                | Lines      |
| -------------------------------------- | ------------------------------------- | ---------- |
| `autopilot/api/v1/dashboard_models.py` | **[NEW]** Complete response model set | ~175 lines |
