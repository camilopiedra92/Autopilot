"""
V1 API Routes — Unified endpoints for the AutoPilot platform.

Provides enriched views of workflows, pipeline topologies, agent cards,
durable run history, real-time EventBus streaming via SSE, HITL controls,
manual triggers, and the Copilot meta-agent.

All endpoints are protected by X-API-Key header.
Uses structlog + OpenTelemetry per platform conventions.

Endpoint summary:
  GET /workflows                               — All workflows with stats
  GET /workflows/{id}                          — Full workflow detail
  GET /workflows/{id}/pipeline                 — Pipeline graph topology
  GET /workflows/{id}/agents                   — Agent cards
  PATCH /workflows/{id}                        — Toggle workflow enable/disable
  GET /workflows/{id}/runs                     — Paginated run history (filterable)
  GET /workflows/{id}/runs/{run_id}            — Full run trace
  DELETE /workflows/{id}/runs/{run_id}         — Delete a run
  POST /workflows/{id}/runs/{run_id}/cancel    — Cancel a running run
  POST /workflows/{id}/runs/{run_id}/resume    — Resume a paused run
  GET /events                                  — EventBus history
  GET /events/stream                           — SSE live stream
  GET /health                                  — Platform health check
  GET /stats                                   — Global platform statistics
  GET /runs/pending-action                     — Paused runs awaiting HITL
  POST /workflows/{id}/trigger                 — Manually trigger a workflow
  POST /copilot/ask                            — Copilot observability agent
  GET /openapi.json                            — V1-only OpenAPI spec
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import yaml
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from autopilot.api.security import get_api_key
from autopilot.agents.copilot import CopilotAgent
from autopilot.api.v1.models import (
    AgentCardResponse,
    CancelRunResponse,
    CopilotQuery,
    CopilotResponse,
    CopilotToolCall,
    DeleteRunResponse,
    WorkflowDetail,
    EventItem,
    PendingRunItem,
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
    PlatformStatsResponse,
    ResumeRunRequest,
    ResumeRunResponse,
    RunStepTrace,
    RunTrace,
    TriggerWorkflowRequest,
    TriggerWorkflowResponse,
    WorkflowStatsItem,
    WorkflowToggleRequest,
    WorkflowToggleResponse,
)
from autopilot.core.artifact import get_artifact_service
from autopilot.core.bus import AgentMessage, get_event_bus
from autopilot.core.react import ReactRunner
from autopilot.core.run_log import get_run_log_service
from autopilot.errors import (
    APIError,
    WorkflowNotFoundError,
    RunNotFoundError,
    RunNotPausedError,
    RunNotCancellableError,
)
from autopilot.models import AgentCard, RunStatus
from autopilot.registry import get_registry

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_api_key)])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers — Private, not exported
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_workflow(workflow_id: str):
    """Get workflow or raise WorkflowNotFoundError.

    Uses the typed error from the error taxonomy (Phase 0B) instead
    of a raw HTTPException, per ARCHITECTURE.md §6.
    """
    wf = get_registry().get(workflow_id)
    if not wf:
        raise WorkflowNotFoundError(
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
    projects it into an AgentCardResponse (flat API model).

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
            logger.warning(
                "agent_card_parse_error", file=str(yaml_file), error=str(exc)
            )

    return cards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Endpoints — Observability & Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/workflows",
    tags=["Workflows"],
    summary="List all workflows",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def list_workflows() -> dict[str, Any]:
    """List all workflows with enriched metadata.

    For each workflow, includes:
      - Static manifest data (name, triggers, tags, icon, color)
      - Pipeline info (strategy, step count) from pipeline.yaml
      - Run stats (total, success rate) from RunLogService
      - Last run metadata (if any)
    """
    with tracer.start_as_current_span("api.list_workflows"):
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

            dw = WorkflowDetail(
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
            workflows.append(dw.model_dump(mode="json", exclude_none=True))

        return {"workflows": workflows, "total": len(workflows)}


@router.get(
    "/workflows/{workflow_id}",
    tags=["Workflows"],
    summary="Get full workflow detail",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
    },
)
async def get_workflow(workflow_id: str) -> dict[str, Any]:
    """Get full workflow detail including manifest, pipeline graph, agents, and stats."""
    with tracer.start_as_current_span(
        "api.get_workflow", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        pipeline_data = _parse_pipeline_yaml(wf)
        graph = _build_pipeline_graph(pipeline_data)
        agents = _load_agent_cards(wf)
        stats = await get_run_log_service().get_stats(workflow_id)

        return {
            "manifest": wf.manifest.model_dump(mode="json", exclude_none=True),
            "pipeline": graph.model_dump(mode="json", exclude_none=True),
            "agents": [a.model_dump(mode="json", exclude_none=True) for a in agents],
            "stats": stats,
        }


@router.get(
    "/workflows/{workflow_id}/pipeline",
    tags=["Workflows"],
    summary="Get pipeline graph topology",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
    },
)
async def get_pipeline_graph(workflow_id: str) -> dict[str, Any]:
    """Get the pipeline graph topology (nodes, edges, layers).

    Returns the data structure needed to render a DAG visualization.
    For SEQUENTIAL workflows, each node is in its own layer.
    For DAG workflows, nodes without mutual dependencies share a layer.
    """
    with tracer.start_as_current_span(
        "api.get_pipeline", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        pipeline_data = _parse_pipeline_yaml(wf)
        graph = _build_pipeline_graph(pipeline_data)
        return graph.model_dump(mode="json", exclude_none=True)


@router.get(
    "/workflows/{workflow_id}/agents",
    tags=["Workflows"],
    summary="Get agent cards for a workflow",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
    },
)
async def get_workflow_agents(workflow_id: str) -> dict[str, Any]:
    """Get agent cards for a workflow.

    Parses all `.agent.yaml` files in the workflow's `agents/` directory
    and returns them as flat `AgentCardResponse` objects.
    """
    with tracer.start_as_current_span(
        "api.get_agents", attributes={"workflow_id": workflow_id}
    ):
        wf = _get_workflow(workflow_id)
        agents = _load_agent_cards(wf)
        return {
            "agents": [a.model_dump(mode="json", exclude_none=True) for a in agents],
            "total": len(agents),
        }


@router.get(
    "/workflows/{workflow_id}/runs",
    tags=["Runs"],
    summary="List recent workflow runs",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
    },
)
async def list_workflow_runs(
    workflow_id: str,
    limit: int = 50,
    start_after: str | None = None,
    status: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """List recent runs from durable RunLogService.

    Supports cursor-based pagination via the `start_after` query param.
    Returns runs newest-first with aggregate stats.

    Optional filters:
      - `status`: Filter by RunStatus value (e.g., 'failed', 'success')
      - `since`: ISO 8601 datetime — only runs started after this time
    """
    with tracer.start_as_current_span(
        "api.list_runs", attributes={"workflow_id": workflow_id}
    ):
        _get_workflow(workflow_id)  # validate exists
        run_log = get_run_log_service()
        runs, next_cursor = await run_log.list_runs(
            workflow_id, limit=limit, start_after=start_after
        )

        # Apply in-memory filters (works for both backends)
        if status:
            runs = [r for r in runs if r.status.value == status.lower()]
        if since:
            try:
                # URL decoding turns '+' to ' ' in timezone offsets, restore it
                since_clean = since.replace(" ", "+")
                since_dt = datetime.fromisoformat(since_clean)
                # Ensure consistent comparison — make both tz-aware or strip tz
                runs = [
                    r
                    for r in runs
                    if r.started_at.replace(
                        tzinfo=r.started_at.tzinfo or since_dt.tzinfo
                    )
                    >= since_dt
                ]
            except ValueError:
                pass  # Invalid date format — skip filter silently

        stats = await run_log.get_stats(workflow_id)
        return {
            "workflow_id": workflow_id,
            "runs": [r.model_dump(mode="json", exclude_none=True) for r in runs],
            "meta": {"next_cursor": next_cursor},
            "stats": stats,
        }


@router.get(
    "/workflows/{workflow_id}/runs/{run_id}",
    tags=["Runs"],
    summary="Get full run execution trace",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow or run not found"},
    },
)
async def get_run_trace(workflow_id: str, run_id: str) -> dict[str, Any]:
    """Get full run trace — run metadata + per-step artifact data from GCS.

    Loads the run from RunLogService, then fetches step artifacts from
    the ArtifactService (GCS in production). For each step:
      - Loads `{step}.json` — node output
      - Optionally loads `{step}.llm.json` — LLM response companion

    Artifact loading errors are logged and skipped (never crash the endpoint).
    """
    with tracer.start_as_current_span(
        "api.get_run_trace",
        attributes={"workflow_id": workflow_id, "run_id": run_id},
    ):
        run_log = get_run_log_service()
        run = await run_log.get_run(workflow_id, run_id)
        if not run:
            raise WorkflowNotFoundError(f"Run '{run_id}' not found")

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
        return trace_data.model_dump(mode="json", exclude_none=True)


@router.get(
    "/events",
    tags=["Events"],
    summary="Get recent EventBus history",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def get_events(topic: str = "*", limit: int = 50) -> dict[str, Any]:
    """Get recent events from EventBus history.

    Supports topic filtering:
      - `topic=*` (default): Aggregates across all topics, sorted newest first
      - `topic=pipeline.completed`: Only events matching that topic
    """
    with tracer.start_as_current_span("api.get_events"):
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


@router.get(
    "/events/stream",
    tags=["Events"],
    summary="SSE live event stream",
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {"text/event-stream": {}},
        },
        401: {"description": "Invalid or missing API key"},
    },
)
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
    with tracer.start_as_current_span("api.event_stream"):
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
    import time

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
    start_time = time.time()
    MAX_CONNECTION_LIFETIME = 300  # 5 minutes

    try:
        while True:
            if await request.is_disconnected():
                break

            # Intentional disconnect for Edge LBs to drop TCP without erroring client
            if time.time() - start_time > MAX_CONNECTION_LIFETIME:
                logger.debug("sse_intentional_reconnect", reason="Edge LB safety")
                yield "event: reconnect\ndata: \n\n"
                break

            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"event: {msg.topic}\ndata: {msg.model_dump_json()}\nid: {msg.timestamp}\n\n"
            except asyncio.TimeoutError:
                yield "event: keepalive\ndata: \n\n"
    finally:
        bus.unsubscribe(sub)
        logger.debug("sse_client_disconnected")


@router.get(
    "/health",
    tags=["System"],
    summary="Platform health check",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def api_health() -> dict[str, Any]:
    """Platform health aggregated for observability.

    Returns:
      - Overall status
      - Workflow count (total and enabled)
      - EventBus stats (published, delivered, errors)
    """
    with tracer.start_as_current_span("api.health"):
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HITL & Manual Trigger Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/runs/pending-action",
    tags=["HITL"],
    summary="List runs awaiting human action",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def list_pending_runs() -> dict[str, Any]:
    """List all globally PAUSED runs awaiting human intervention.

    Queries RunLogService for runs with status=PAUSED across all workflows.
    Returns minimal projections (PendingRunItem) suitable for a summary card.
    """
    with tracer.start_as_current_span("api.list_pending_runs"):
        run_log = get_run_log_service()
        paused_runs = await run_log.get_pending_runs()

        items = [
            PendingRunItem(
                run_id=r.id,
                workflow_id=r.workflow_id,
                status=r.status,
                trigger_type=r.trigger_type,
                started_at=r.started_at,
            ).model_dump(mode="json", exclude_none=True)
            for r in paused_runs
        ]

        return {"pending": items, "total": len(items)}


@router.post(
    "/workflows/{workflow_id}/runs/{run_id}/resume",
    tags=["HITL"],
    summary="Resume a paused run",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow or run not found"},
        409: {"description": "Run is not in PAUSED state"},
    },
)
async def resume_run(
    workflow_id: str, run_id: str, body: ResumeRunRequest | None = None
) -> dict[str, Any]:
    """Resume a PAUSED run by dispatching an event to the EventBus.

    Validation chain:
      1. Workflow exists → WorkflowNotFoundError (404)
      2. Run exists → RunNotFoundError (404)
      3. Run is PAUSED → RunNotPausedError (409)

    The actual resume happens asynchronously — the workflow's
    _on_hitl_resumed() subscriber picks up the event.
    """
    with tracer.start_as_current_span(
        "api.resume_run",
        attributes={"workflow_id": workflow_id, "run_id": run_id},
    ):
        # 1. Workflow exists
        _get_workflow(workflow_id)

        # 2. Run exists
        run_log = get_run_log_service()
        run = await run_log.get_run(workflow_id, run_id)
        if not run:
            raise RunNotFoundError(
                f"Run '{run_id}' not found for workflow '{workflow_id}'"
            )

        # 3. Run is PAUSED
        if run.status != RunStatus.PAUSED:
            raise RunNotPausedError(
                f"Run '{run_id}' is {run.status.value}, not PAUSED",
                detail=f"Current status: {run.status.value}",
            )

        # Dispatch resume event via EventBus
        bus = get_event_bus()
        await bus.publish(
            "api.hitl_resumed",
            {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "payload": body.payload if body else {},
            },
            sender="v1_api",
        )

        logger.info(
            "hitl_resume_dispatched",
            workflow_id=workflow_id,
            run_id=run_id,
        )

        response = ResumeRunResponse(run_id=run_id, workflow_id=workflow_id)
        return response.model_dump()


@router.post(
    "/workflows/{workflow_id}/trigger",
    tags=["Workflows"],
    summary="Manually trigger a workflow",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
        500: {"description": "Workflow is disabled"},
    },
)
async def trigger_workflow(
    workflow_id: str, body: TriggerWorkflowRequest | None = None
) -> dict[str, Any]:
    """Manually trigger a workflow by dispatching an event to the EventBus.

    Checks that the workflow exists and is enabled before dispatching.
    The actual execution happens asynchronously — the workflow's
    _on_manual_trigger() subscriber picks up the event.
    """
    with tracer.start_as_current_span(
        "api.trigger_workflow",
        attributes={"workflow_id": workflow_id},
    ):
        wf = _get_workflow(workflow_id)

        if not wf.manifest.enabled:
            raise APIError(
                f"Workflow '{workflow_id}' is disabled",
                detail="Enable the workflow in manifest.yaml before triggering",
            )

        # Dispatch trigger event via EventBus
        bus = get_event_bus()
        await bus.publish(
            "api.workflow_triggered",
            {
                "workflow_id": workflow_id,
                "payload": body.payload if body else {},
            },
            sender="v1_api",
        )

        logger.info(
            "manual_trigger_dispatched",
            workflow_id=workflow_id,
        )

        response = TriggerWorkflowResponse(workflow_id=workflow_id)
        return response.model_dump()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cancel & Delete Run Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/workflows/{workflow_id}/runs/{run_id}/cancel",
    tags=["HITL"],
    summary="Cancel a running or pending run",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow or run not found"},
        409: {"description": "Run is not in a cancellable state"},
    },
)
async def cancel_run(workflow_id: str, run_id: str) -> dict[str, Any]:
    """Cancel a RUNNING or PENDING run.

    Validation chain:
      1. Workflow exists → WorkflowNotFoundError (404)
      2. Run exists → RunNotFoundError (404)
      3. Run is RUNNING or PENDING → RunNotCancellableError (409)

    Updates the run status to CANCELLED in RunLogService and
    publishes an `api.run_cancelled` event to the EventBus.
    """
    with tracer.start_as_current_span(
        "api.cancel_run",
        attributes={"workflow_id": workflow_id, "run_id": run_id},
    ):
        _get_workflow(workflow_id)

        run_log = get_run_log_service()
        run = await run_log.get_run(workflow_id, run_id)
        if not run:
            raise RunNotFoundError(
                f"Run '{run_id}' not found for workflow '{workflow_id}'"
            )

        cancellable = {RunStatus.RUNNING, RunStatus.PENDING}
        if run.status not in cancellable:
            raise RunNotCancellableError(
                f"Run '{run_id}' is {run.status.value}, not RUNNING or PENDING",
                detail=f"Current status: {run.status.value}. Only RUNNING/PENDING runs can be cancelled.",
            )

        # Update status
        run.status = RunStatus.CANCELLED
        await run_log.save_run(run)

        # Notify via EventBus
        bus = get_event_bus()
        await bus.publish(
            "api.run_cancelled",
            {"workflow_id": workflow_id, "run_id": run_id},
            sender="v1_api",
        )

        logger.info("run_cancelled", workflow_id=workflow_id, run_id=run_id)

        response = CancelRunResponse(run_id=run_id, workflow_id=workflow_id)
        return response.model_dump()


@router.delete(
    "/workflows/{workflow_id}/runs/{run_id}",
    tags=["Runs"],
    summary="Delete a run from history",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow or run not found"},
    },
)
async def delete_run(workflow_id: str, run_id: str) -> dict[str, Any]:
    """Delete a run from durable history.

    Removes the run from RunLogService and adjusts aggregate stats.
    This is a destructive operation — the run data cannot be recovered.
    """
    with tracer.start_as_current_span(
        "api.delete_run",
        attributes={"workflow_id": workflow_id, "run_id": run_id},
    ):
        _get_workflow(workflow_id)

        run_log = get_run_log_service()
        deleted = await run_log.delete_run(workflow_id, run_id)
        if not deleted:
            raise RunNotFoundError(
                f"Run '{run_id}' not found for workflow '{workflow_id}'"
            )

        logger.info("run_deleted", workflow_id=workflow_id, run_id=run_id)

        response = DeleteRunResponse(run_id=run_id, workflow_id=workflow_id)
        return response.model_dump()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Global Stats & Workflow Toggle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/stats",
    tags=["System"],
    summary="Get global platform statistics",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def get_platform_stats() -> dict[str, Any]:
    """Get aggregated statistics across all workflows.

    Returns:
      - Workflow counts (total, enabled)
      - Run totals and global success rate
      - Per-workflow breakdown
      - Top workflow by run count
      - EventBus stats
    """
    with tracer.start_as_current_span("api.get_platform_stats"):
        registry = get_registry()
        run_log = get_run_log_service()
        workflows = registry.list_all()

        total_runs = 0
        total_successful = 0
        top_workflow = None
        top_runs = 0
        wf_items: list[WorkflowStatsItem] = []

        for info in workflows:
            stats = await run_log.get_stats(info.name)
            wf_total = stats.get("total", 0)
            wf_successful = stats.get("successful", 0)
            wf_failed = wf_total - wf_successful

            total_runs += wf_total
            total_successful += wf_successful

            if wf_total > top_runs:
                top_runs = wf_total
                top_workflow = info.name

            wf_items.append(
                WorkflowStatsItem(
                    workflow_id=info.name,
                    display_name=info.display_name,
                    total_runs=wf_total,
                    successful=wf_successful,
                    failed=wf_failed,
                    success_rate=round(wf_successful / wf_total * 100, 1)
                    if wf_total > 0
                    else 0.0,
                    enabled=info.enabled,
                )
            )

        total_failed = total_runs - total_successful
        global_rate = (
            round(total_successful / total_runs * 100, 1) if total_runs > 0 else 0.0
        )

        response = PlatformStatsResponse(
            total_workflows=len(workflows),
            enabled_workflows=sum(1 for w in workflows if w.enabled),
            total_runs=total_runs,
            total_successful=total_successful,
            total_failed=total_failed,
            global_success_rate=global_rate,
            top_workflow=top_workflow,
            workflows=wf_items,
            bus_stats=get_event_bus().stats,
        )
        return response.model_dump()


@router.patch(
    "/workflows/{workflow_id}",
    tags=["Workflows"],
    summary="Toggle workflow enabled state",
    responses={
        401: {"description": "Invalid or missing API key"},
        404: {"description": "Workflow not found"},
    },
)
async def toggle_workflow(
    workflow_id: str, body: WorkflowToggleRequest
) -> dict[str, Any]:
    """Toggle a workflow's enabled/disabled state at runtime.

    This updates the in-memory manifest only — it does NOT modify
    the `manifest.yaml` file on disk. The change persists until
    the next container restart.
    """
    with tracer.start_as_current_span(
        "api.toggle_workflow",
        attributes={"workflow_id": workflow_id, "enabled": body.enabled},
    ):
        wf = _get_workflow(workflow_id)
        wf.manifest.enabled = body.enabled

        logger.info(
            "workflow_toggled",
            workflow_id=workflow_id,
            enabled=body.enabled,
        )

        response = WorkflowToggleResponse(
            workflow_id=workflow_id,
            enabled=body.enabled,
        )
        return response.model_dump()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OpenAPI v1 Export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get(
    "/openapi.json",
    tags=["System"],
    summary="Export v1-only OpenAPI specification",
    responses={401: {"description": "Invalid or missing API key"}},
)
async def get_v1_openapi(request: Request) -> dict[str, Any]:
    """Return the OpenAPI spec filtered to v1 endpoints only.

    Useful for code generation, SDK building, and API documentation
    tools that need the v1 subset without system/webhook routes.
    """
    from copy import deepcopy

    app = request.app
    full_spec = deepcopy(app.openapi())

    # Filter paths to only /api/v1/* endpoints
    v1_paths = {
        path: ops
        for path, ops in full_spec.get("paths", {}).items()
        if path.startswith("/api/v1")
    }
    full_spec["paths"] = v1_paths
    full_spec["info"]["title"] = "AutoPilot API v1"
    full_spec["info"]["description"] = (
        "V1 API endpoints for the AutoPilot headless platform."
    )

    return full_spec


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Copilot — Platform observability meta-agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post(
    "/copilot/ask",
    tags=["Copilot"],
    summary="Ask the platform copilot",
    responses={
        401: {"description": "Invalid or missing API key"},
    },
)
async def ask_copilot(body: CopilotQuery) -> dict[str, Any]:
    """Ask the platform copilot a natural language question.

    The copilot uses read-only tools to query platform state
    (workflow stats, errors, runs, events) and synthesizes
    a structured response.

    Runs synchronously within the request — typical latency < 500ms.
    """
    with tracer.start_as_current_span(
        "api.copilot.ask",
        attributes={"query_length": len(body.query)},
    ):
        agent = CopilotAgent()
        runner = ReactRunner(
            name="copilot_runner",
            agent=agent,
            max_iterations=5,
        )

        try:
            result = await runner.execute(
                initial_input={"query": body.query},
            )

            answer = result.state.get("answer", "Unable to process your query.")
            tools_used_raw = result.state.get("tools_used", [])
            iterations = result.state.get("iterations", 0)

            tools_used = [
                CopilotToolCall(**tc) if isinstance(tc, dict) else tc
                for tc in tools_used_raw
            ]

            response = CopilotResponse(
                answer=answer,
                tools_used=tools_used,
                iterations=iterations,
            )

            logger.info(
                "copilot_query_completed",
                query_length=len(body.query),
                tools_used=len(tools_used),
                iterations=iterations,
            )

            return response.model_dump()

        except Exception as exc:
            logger.error("copilot_query_failed", error=str(exc))
            response = CopilotResponse(
                answer=f"Sorry, I encountered an error: {exc}",
                tools_used=[],
                iterations=0,
            )
            return response.model_dump()
