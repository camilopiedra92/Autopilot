# Phase 1B. Dashboard Router & Endpoints

> **Status**: ✅ COMPLETED  
> **Completed**: 2026-02-26  
> **Effort**: ~60 min  
> **Type**: NEW (API Layer)  
> **Parent**: [dashboard-implementation.md](../dashboard-implementation.md) § Phase 1B  
> **Depends on**: Phase 1A (Response Models), Phase 0 complete

---

## Problem Statement

The platform currently exposes basic workflow CRUD via `autopilot/api/v1/routes.py`. The Dashboard needs a richer API surface that provides:

- **Enriched workflow listing** — with pipeline strategy, step/agent counts, run stats, and success rates.
- **Pipeline topology** — nodes, edges, and topological layers for DAG visualization.
- **Agent card introspection** — `*.agent.yaml` files parsed and returned as JSON.
- **Durable run history** — cursor-paginated from `RunLogService` (not transient in-memory `self._runs`).
- **Full run traces** — composite of `WorkflowRun` metadata + GCS artifact data per step.
- **Event log** — EventBus history aggregated across topics.
- **SSE real-time stream** — live event feed with Edge-safe reconnection.
- **Health check** — platform status aggregation for monitoring.

All of this must use typed Pydantic models from Phase 1A, never raw dicts or untyped JSON.

---

## Architecture Alignment

| ARCHITECTURE.md Section | Requirement                                                | Current                 | Target                                                 |
| ----------------------- | ---------------------------------------------------------- | ----------------------- | ------------------------------------------------------ |
| §1 Core Philosophy      | Headless API — no frontend, JSON + Events                  | V1 routes only          | Dashboard API at `/api/v1/*`                 |
| §1 Core Philosophy      | **NEVER `asyncio.create_task`** in ephemeral compute       | N/A                     | SSE uses request-scoped async generator                |
| §1 Core Philosophy      | X-API-Key auth on all endpoints                            | V1 router has `Depends` | Inherited from parent V1 router                        |
| §5 Observability        | OTel spans on all endpoints                                | V1 routes lack OTel     | Every endpoint wraps in `tracer.start_as_current_span` |
| §6 Error Taxonomy       | Use `AutoPilotError` subclasses, never raw `HTTPException` | V1 uses `HTTPException` | Uses `DashboardWorkflowNotFoundError`                  |
| §9 Rule 2               | Schema-First — models before logic                         | Phase 1A models ready   | Endpoints return typed Pydantic models                 |
| §9.1 Observability      | `structlog.get_logger(__name__)`                           | Convention              | `logger` and `tracer` at module level                  |

---

## Prerequisites

- Phase 1A complete (`dashboard_models.py` importable).
- Phase 0 complete (`RunLogService`, `DashboardWorkflowNotFoundError`).

**Verify prerequisites**:

```bash
python -c "from autopilot.api.v1.routes_models import DashboardWorkflow, PipelineGraph, RunTrace, AgentCardResponse, EventItem; print('Models OK')"
python -c "from autopilot.core.run_log import get_run_log_service; print('RunLog OK')"
python -c "from autopilot.errors import DashboardWorkflowNotFoundError; print('Errors OK')"
```

---

## Implementation

### Step 1: Create `autopilot/api/v1/dashboard.py` [NEW]

Create this file with the **complete** contents below. This is the entire file — do not add anything extra.

```python
"""
Dashboard API — Read-only endpoints for the Autopilot Dashboard.

Provides enriched views of workflows, pipeline topologies, agent cards,
durable run history, and real-time EventBus streaming via SSE.

All endpoints are protected by X-API-Key (inherited from V1 router).
Uses structlog + OpenTelemetry per platform conventions.

Endpoint summary:
  GET /dashboard/workflows                         — All workflows with stats
  GET /dashboard/workflows/{id}                    — Full workflow detail
  GET /dashboard/workflows/{id}/pipeline           — Pipeline graph topology
  GET /dashboard/workflows/{id}/agents             — Agent cards
  GET /dashboard/workflows/{id}/runs               — Paginated run history
  GET /dashboard/workflows/{id}/runs/{run_id}      — Full run trace
  GET /dashboard/events                            — EventBus history
  GET /dashboard/events/stream                     — SSE live stream
  GET /dashboard/health                            — Platform health check
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from autopilot.api.v1.routes_models import (
    AgentCardResponse,
    DashboardWorkflow,
    EventItem,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    RunStepTrace,
    RunTrace,
)
from autopilot.core.artifact import get_artifact_service
from autopilot.core.bus import AgentMessage, get_event_bus
from autopilot.core.run_log import get_run_log_service
from autopilot.errors import DashboardWorkflowNotFoundError
from autopilot.models import AgentCard
from autopilot.registry import get_registry

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_api_key)])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers — Private, not exported
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_workflow(workflow_id: str):
    """Get workflow or raise DashboardWorkflowNotFoundError.

    Uses the typed error from the error taxonomy (Phase 0B) instead
    of a raw HTTPException, per ARCHITECTURE.md §6.
    """
    wf = get_registry().get(workflow_id)
    if not wf:
        raise DashboardWorkflowNotFoundError(
            f"Workflow '{workflow_id}' not found",
            detail=f"Available: {[w.name for w in get_registry().list_all()]}",
        )
    return wf


def _parse_pipeline_yaml(wf) -> dict[str, Any]:
    """Load and parse pipeline.yaml from the workflow directory.

    Returns empty dict if file doesn't exist (valid — not all workflows
    have a pipeline.yaml, e.g., conversational_assistant).
    """
    pipeline_path = Path(wf._workflow_dir) / "pipeline.yaml"
    if not pipeline_path.exists():
        return {}
    with open(pipeline_path) as f:
        return yaml.safe_load(f) or {}


def _build_pipeline_graph(pipeline_data: dict[str, Any]) -> PipelineGraph:
    """Convert pipeline.yaml data into a PipelineGraph with topological layers.

    Uses Kahn's algorithm to compute topological layers:
      - Layer 0 = root nodes (in_degree == 0)
      - Layer N = nodes whose dependencies are all in layers < N
      - Nodes in the same layer can execute concurrently (DAG strategy)

    For SEQUENTIAL pipelines, each step depends on the previous one,
    resulting in one node per layer.
    """
    strategy = pipeline_data.get("strategy", "SEQUENTIAL").upper()
    raw_steps = pipeline_data.get("steps", []) or pipeline_data.get("nodes", [])

    if not raw_steps:
        return PipelineGraph(strategy=strategy)

    # Build nodes
    nodes: list[PipelineNode] = []
    edges: list[PipelineEdge] = []
    deps_map: dict[str, list[str]] = {}

    for step in raw_steps:
        name = step.get("name", step.get("id", "unknown"))
        deps = step.get("depends_on", [])
        node = PipelineNode(
            name=name,
            type=step.get("type", "agent"),
            ref=step.get("ref", ""),
            description=step.get("description", ""),
            dependencies=deps,
        )
        nodes.append(node)
        deps_map[name] = deps

        for dep in deps:
            edges.append(PipelineEdge(source=dep, target=name))

    # Kahn's algorithm for topological layers
    in_degree: dict[str, int] = {n.name: 0 for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        in_degree[edge.target] += 1
        adj[edge.source].append(edge.target)

    layers: list[list[str]] = []
    queue = [name for name, degree in in_degree.items() if degree == 0]

    while queue:
        layers.append(sorted(queue))
        next_queue: list[str] = []
        for name in queue:
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        queue = next_queue

    # Assign layer to each node
    name_to_layer = {}
    for i, layer in enumerate(layers):
        for name in layer:
            name_to_layer[name] = i
    for node in nodes:
        node.layer = name_to_layer.get(node.name, 0)

    return PipelineGraph(strategy=strategy, nodes=nodes, edges=edges, layers=layers)


def _load_agent_cards(wf) -> list[AgentCardResponse]:
    """Load .agent.yaml files from the workflow's agents directory.

    Parses each YAML file into an AgentCard (platform model), then
    projects it into an AgentCardResponse (flat dashboard model).

    Parse errors are logged and skipped (never crash the endpoint).
    """
    agents_dir = Path(wf._workflow_dir) / "agents"
    cards: list[AgentCardResponse] = []

    if not agents_dir.exists():
        return cards

    for yaml_file in sorted(agents_dir.glob("*.agent.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}
            card = AgentCard.model_validate(data)
            cards.append(
                AgentCardResponse(
                    name=card.name,
                    display_name=card.display_name,
                    type=card.type,
                    description=card.description,
                    model=card.model,
                    stage=card.stage,
                    tools=[t.name for t in card.tools],
                    guardrails_before=card.guardrails.before_model,
                    guardrails_after=card.guardrails.after_model,
                    input_schema=card.input.schema_ref if card.input else None,
                    output_schema=card.output.schema_ref if card.output else None,
                )
            )
        except Exception as exc:
            logger.warning("agent_card_parse_error", file=str(yaml_file), error=str(exc))

    return cards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Endpoints — Read-only observability surface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/workflows")
async def list_dashboard_workflows() -> dict[str, Any]:
    """List all workflows with enriched dashboard metadata.

    For each workflow, includes:
      - Static manifest data (name, triggers, tags, icon, color)
      - Pipeline info (strategy, step count) from pipeline.yaml
      - Run stats (total, success rate) from RunLogService
      - Last run metadata (if any)
    """
    with tracer.start_as_current_span("dashboard.list_workflows"):
        registry = get_registry()
        run_log = get_run_log_service()
        workflows: list[dict] = []

        for info in registry.list_all():
            wf = registry.get(info.name)
            pipeline_data = _parse_pipeline_yaml(wf) if wf else {}
            strategy = pipeline_data.get("strategy", "SEQUENTIAL")
            steps = pipeline_data.get("steps", []) or pipeline_data.get("nodes", [])
            stats = await run_log.get_stats(info.name)
            total = stats.get("total", 0)
            successful = stats.get("successful", 0)

            dw = DashboardWorkflow(
                id=info.name,
                display_name=info.display_name,
                description=info.description,
                version=info.version,
                icon=info.icon,
                color=info.color,
                enabled=info.enabled,
                triggers=info.triggers,
                tags=info.tags,
                strategy=strategy,
                step_count=len(steps),
                agent_count=len(info.triggers),
                total_runs=total,
                success_rate=round(successful / total * 100, 1) if total > 0 else 0.0,
                last_run=info.last_run,
            )
            workflows.append(dw.model_dump(mode="json"))

        return {"workflows": workflows, "total": len(workflows)}


@router.get("/workflows/{workflow_id}")
async def get_dashboard_workflow(workflow_id: str) -> dict[str, Any]:
    """Get full workflow detail including manifest, pipeline graph, agents, and stats."""
    with tracer.start_as_current_span(
        "dashboard.get_workflow", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        pipeline_data = _parse_pipeline_yaml(wf)
        graph = _build_pipeline_graph(pipeline_data)
        agents = _load_agent_cards(wf)
        stats = await get_run_log_service().get_stats(workflow_id)

        return {
            "manifest": wf.manifest.model_dump(mode="json"),
            "pipeline": graph.model_dump(mode="json"),
            "agents": [a.model_dump(mode="json") for a in agents],
            "stats": stats,
        }


@router.get("/workflows/{workflow_id}/pipeline")
async def get_pipeline_graph(workflow_id: str) -> dict[str, Any]:
    """Get the pipeline graph topology (nodes, edges, layers).

    Returns the data structure needed to render a DAG visualization.
    For SEQUENTIAL workflows, each node is in its own layer.
    For DAG workflows, nodes without mutual dependencies share a layer.
    """
    with tracer.start_as_current_span(
        "dashboard.get_pipeline", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        pipeline_data = _parse_pipeline_yaml(wf)
        graph = _build_pipeline_graph(pipeline_data)
        return graph.model_dump(mode="json")


@router.get("/workflows/{workflow_id}/agents")
async def get_workflow_agents(workflow_id: str) -> dict[str, Any]:
    """Get agent cards for a workflow.

    Parses all `.agent.yaml` files in the workflow's `agents/` directory
    and returns them as flat `AgentCardResponse` objects.
    """
    with tracer.start_as_current_span(
        "dashboard.get_agents", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        agents = _load_agent_cards(wf)
        return {"agents": [a.model_dump(mode="json") for a in agents], "total": len(agents)}


@router.get("/workflows/{workflow_id}/runs")
async def list_workflow_runs(
    workflow_id: str, limit: int = 50, start_after: str | None = None
) -> dict[str, Any]:
    """List recent runs from durable RunLogService.

    Supports cursor-based pagination via the `start_after` query param.
    Returns runs newest-first with aggregate stats.
    """
    with tracer.start_as_current_span(
        "dashboard.list_runs", attributes={"workflow_id": workflow_id}
    ):
        _get_workflow(workflow_id)  # validate exists
        run_log = get_run_log_service()
        runs, next_cursor = await run_log.list_runs(
            workflow_id, limit=limit, start_after=start_after
        )
        stats = await run_log.get_stats(workflow_id)
        return {
            "workflow_id": workflow_id,
            "runs": [r.model_dump(mode="json") for r in runs],
            "meta": {"next_cursor": next_cursor},
            "stats": stats,
        }


@router.get("/workflows/{workflow_id}/runs/{run_id}")
async def get_run_trace(workflow_id: str, run_id: str) -> dict[str, Any]:
    """Get full run trace — run metadata + per-step artifact data from GCS.

    Loads the run from RunLogService, then fetches step artifacts from
    the ArtifactService (GCS in production). For each step:
      - Loads `{step}.json` — node output
      - Optionally loads `{step}.llm.json` — LLM response companion

    Artifact loading errors are logged and skipped (never crash the endpoint).
    """
    with tracer.start_as_current_span(
        "dashboard.get_run_trace",
        attributes={"workflow_id": workflow_id, "run_id": run_id},
    ):
        run_log = get_run_log_service()
        run = await run_log.get_run(workflow_id, run_id)
        if not run:
            raise DashboardWorkflowNotFoundError(f"Run '{run_id}' not found")

        # Load step artifacts from GCS
        steps: list[RunStepTrace] = []
        try:
            artifact_svc = get_artifact_service()
            execution_id = run_id  # Convention: run_id == artifact session_id
            keys = await artifact_svc.list_artifact_keys(
                app_name=workflow_id,
                user_id="default",
                session_id=execution_id,
            )
            for key in sorted(keys):
                if key.endswith(".llm.json"):
                    continue  # Handled as part of the parent step
                try:
                    artifact = await artifact_svc.load_artifact(
                        app_name=workflow_id,
                        user_id="default",
                        session_id=execution_id,
                        filename=key,
                    )
                    output = (
                        json.loads(artifact.text) if artifact and artifact.text else {}
                    )
                    step_name = key.replace(".json", "")

                    # Check for companion LLM response
                    llm_key = key.replace(".json", ".llm.json")
                    llm_response = None
                    if llm_key in keys:
                        llm_artifact = await artifact_svc.load_artifact(
                            app_name=workflow_id,
                            user_id="default",
                            session_id=execution_id,
                            filename=llm_key,
                        )
                        if llm_artifact and llm_artifact.text:
                            llm_response = json.loads(llm_artifact.text)

                    steps.append(
                        RunStepTrace(
                            name=step_name,
                            artifact_key=key,
                            output=output,
                            duration_ms=output.get("duration_ms", 0),
                            has_llm_response=llm_response is not None,
                            llm_response=llm_response,
                        )
                    )
                except Exception as exc:
                    logger.debug("artifact_load_error", key=key, error=str(exc))
        except Exception as exc:
            logger.warning(
                "artifact_listing_error", workflow_id=workflow_id, error=str(exc)
            )

        trace_data = RunTrace(run=run, steps=steps)
        return trace_data.model_dump(mode="json")


@router.get("/events")
async def get_events(topic: str = "*", limit: int = 50) -> dict[str, Any]:
    """Get recent events from EventBus history.

    Supports topic filtering:
      - `topic=*` (default): Aggregates across all topics, sorted newest first
      - `topic=pipeline.completed`: Only events matching that topic
    """
    with tracer.start_as_current_span("dashboard.get_events"):
        bus = get_event_bus()
        if topic == "*":
            # Aggregate history from all topics
            all_topics = list(getattr(bus, "_history", {}).keys())
            messages: list[AgentMessage] = []
            for t in all_topics:
                messages.extend(bus.history(t, limit=limit))
            messages.sort(key=lambda m: m.timestamp, reverse=True)
            messages = messages[:limit]
        else:
            messages = bus.history(topic, limit=limit)

        return {
            "events": [
                EventItem(
                    topic=m.topic,
                    sender=m.sender,
                    payload=m.payload,
                    timestamp=m.timestamp,
                    correlation_id=m.correlation_id,
                ).model_dump()
                for m in messages
            ],
            "total": len(messages),
        }


@router.get("/events/stream")
async def event_stream(request: Request):
    """SSE endpoint — real-time event stream with durable replay.

    Edge-safe design:
      - Uses request-scoped async generator (no asyncio.create_task)
      - Intentional disconnect after 5 minutes (Cloud Run LB safety)
      - Supports Last-Event-ID header for reconnection replay
      - 30-second keepalive heartbeats prevent idle timeouts
      - Custom `reconnect` event signals clients to reconnect cleanly

    SSE format per event:
      event: {topic}
      data: {AgentMessage JSON}
      id: {timestamp}
    """
    with tracer.start_as_current_span("dashboard.event_stream"):
        return StreamingResponse(
            _event_generator(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )


async def _event_generator(request: Request):
    """Yields SSE events. Edge-safe reconnection after 5 minutes.

    Two phases:
      1. Replay — if Last-Event-ID header present, replay missed events
         from EventBus history (in-memory) or Pub/Sub retained messages
      2. Live — subscribe to EventBus wildcard, yield events as they arrive

    The 5-minute intentional disconnect prevents Cloud Run load-balancer
    zombie connections. Clients receive a `reconnect` event and should
    auto-reconnect with EventSource, which will include Last-Event-ID.
    """
    bus = get_event_bus()
    last_event_id = request.headers.get("Last-Event-ID")

    # Phase 1: Replay missed events from Pub/Sub or in-memory
    if last_event_id:
        try:
            missed = await bus.replay("*", since=last_event_id)
            for msg in missed:
                yield f"event: {msg.topic}\ndata: {msg.model_dump_json()}\nid: {msg.timestamp}\n\n"
        except Exception as exc:
            logger.warning("sse_replay_failed", error=str(exc))

    # Phase 2: Live stream
    queue: asyncio.Queue[AgentMessage] = asyncio.Queue()

    async def handler(msg: AgentMessage) -> None:
        await queue.put(msg)

    sub = bus.subscribe("*", handler)
    import time

    start_time = time.time()
    MAX_CONNECTION_LIFETIME = 300  # 5 minutes

    try:
        while True:
            if await request.is_disconnected():
                break

            # Intentional disconnect for Edge LBs to drop TCP without erroring client
            if time.time() - start_time > MAX_CONNECTION_LIFETIME:
                logger.debug("sse_intentional_reconnect", reason="Edge LB safety")
                yield f"event: reconnect\ndata: \n\n"
                break

            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"event: {msg.topic}\ndata: {msg.model_dump_json()}\nid: {msg.timestamp}\n\n"
            except asyncio.TimeoutError:
                yield f"event: keepalive\ndata: \n\n"
    finally:
        bus.unsubscribe(sub)
        logger.debug("sse_client_disconnected")


@router.get("/health")
async def dashboard_health() -> dict[str, Any]:
    """Platform health aggregated for the dashboard.

    Returns:
      - Overall status
      - Workflow count (total and enabled)
      - EventBus stats (published, delivered, errors)
    """
    with tracer.start_as_current_span("dashboard.health"):
        registry = get_registry()
        workflows = registry.list_all()
        return {
            "status": "healthy",
            "workflows": {
                "total": len(workflows),
                "enabled": sum(1 for w in workflows if w.enabled),
            },
            "bus": get_event_bus().stats,
        }
```

### Step 2: Verify compilation

```bash
python -c "from autopilot.api.v1.routes import router; print(f'{len(router.routes)} routes OK')"
```

---

## Endpoint Reference

### `GET /api/v1/workflows`

Lists all registered workflows with enriched dashboard metadata.

**Response shape**:

```json
{
  "workflows": [
    {
      "id": "bank_to_ynab",
      "display_name": "Bank → YNAB",
      "description": "Parses bank emails and creates YNAB transactions",
      "version": "2.1.0",
      "icon": "💰",
      "color": "#4CAF50",
      "enabled": true,
      "triggers": [{"type": "gmail_push", "filter": "from:banco@example.com"}],
      "tags": ["finance", "automation"],
      "strategy": "DAG",
      "step_count": 9,
      "agent_count": 1,
      "total_runs": 142,
      "success_rate": 97.2,
      "last_run": { "id": "abc123", "status": "success", ... }
    }
  ],
  "total": 3
}
```

---

### `GET /api/v1/workflows/{workflow_id}`

Full workflow detail: manifest + pipeline graph + agent cards + run stats.

**Response shape**:

```json
{
  "manifest": { /* WorkflowManifest fields */ },
  "pipeline": { "strategy": "DAG", "nodes": [...], "edges": [...], "layers": [[...], [...]] },
  "agents": [{ "name": "email_parser", "type": "llm", "model": "gemini-3-flash-preview", ... }],
  "stats": { "total": 142, "successful": 138 }
}
```

---

### `GET /api/v1/workflows/{workflow_id}/pipeline`

Pipeline graph topology for visualization.

**Response shape**:

```json
{
  "strategy": "DAG",
  "nodes": [
    {
      "name": "email_parser",
      "type": "agent",
      "layer": 1,
      "dependencies": ["format_parser_prompt"]
    },
    {
      "name": "match_account",
      "type": "function",
      "layer": 2,
      "dependencies": ["email_parser"]
    },
    {
      "name": "researcher",
      "type": "agent",
      "layer": 2,
      "dependencies": ["email_parser"]
    }
  ],
  "edges": [
    { "source": "email_parser", "target": "match_account" },
    { "source": "email_parser", "target": "researcher" }
  ],
  "layers": [
    ["format_parser_prompt"],
    ["email_parser"],
    ["match_account", "researcher"]
  ]
}
```

---

### `GET /api/v1/workflows/{workflow_id}/agents`

Agent cards from `.agent.yaml` files.

---

### `GET /api/v1/workflows/{workflow_id}/runs?limit=50&start_after=cursor`

Cursor-paginated run history from RunLogService.

**Response shape**:

```json
{
  "workflow_id": "bank_to_ynab",
  "runs": [{ "id": "abc", "status": "success", "duration_ms": 5432, ... }],
  "meta": { "next_cursor": "2026-02-26T10:00:00Z" },
  "stats": { "total": 142, "successful": 138 }
}
```

---

### `GET /api/v1/workflows/{workflow_id}/runs/{run_id}`

Full execution trace — run metadata + per-step GCS artifacts.

---

### `GET /api/v1/events?topic=*&limit=50`

EventBus history aggregated across all topics or filtered by topic.

---

### `GET /api/v1/events/stream`

Server-Sent Events (SSE) real-time stream.

**Headers**:

- `Last-Event-ID`: Optional. Timestamp for reconnection replay.

**SSE format**:

```
event: pipeline.completed
data: {"topic":"pipeline.completed","sender":"bank_to_ynab","payload":{...},"timestamp":"..."}
id: 2026-02-26T10:00:00Z

event: keepalive
data:

event: reconnect
data:
```

---

### `GET /api/v1/health`

Platform health aggregation.

---

## Design Decisions

| Decision                                                   | Rationale                                                                                                  |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Sub-router at `/dashboard` (not top-level routes)          | Groups all dashboard endpoints under one prefix; easy to disable/swap                                      |
| Helper functions prefixed with `_` (private)               | Not part of the public API — internal implementation detail                                                |
| `_parse_pipeline_yaml` reads from filesystem               | `pipeline.yaml` is always local (co-located with workflow code); no need for async I/O                     |
| Kahn's algorithm for topological layers                    | Same algorithm used by `DAGRunner` — consistent layer numbering                                            |
| `_load_agent_cards` logs and skips parse errors            | A malformed `.agent.yaml` file should never crash the dashboard                                            |
| SSE uses `asyncio.Queue` + request-scoped generator        | No `asyncio.create_task` per ARCHITECTURE.md §1; queue naturally back-pressures                            |
| 5-minute SSE disconnect                                    | Cloud Run has a 60-minute max request timeout; 5-minute reconnect prevents LB zombie connections           |
| `Last-Event-ID` for SSE replay                             | Standard SSE reconnection mechanism; EventSource browsers send this automatically                          |
| 30-second keepalive heartbeat                              | Prevents Cloud Run idle timeouts (default 300s) and detects dead TCP connections                           |
| `get_events` aggregates across all topics when `topic=*`   | Dashboard event log needs a unified view; per-topic filtering is secondary                                 |
| Run trace loads artifacts by key naming convention         | `{step}.json` + `{step}.llm.json` pairing is established by `_artifact_persist.py` and `ADKRunner`         |
| `import time` inside `_event_generator` (not module level) | Keeps the import at function scope — this is a standard library module, but function-scope is conventional |

---

## Files Modified

| File                            | Change                              | Lines      |
| ------------------------------- | ----------------------------------- | ---------- |
| `autopilot/api/v1/dashboard.py` | **[NEW]** Complete dashboard router | ~350 lines |
