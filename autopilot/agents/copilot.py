"""
Platform Copilot Agent — LLM-powered platform observability meta-agent.

A ReAct agent that answers natural language questions about workflows,
failures, run history, and event timelines using read-only tools.

Architecture:
  - Uses ReactRunner with max_iterations=5
  - 4 read-only tools (no side effects — safe for observability use)
  - Keyword-based tool routing for reliable, zero-cost intent matching
  - Synthesized answer returned as structured response

Design decisions:
  - Synchronous execution within request (conversational pattern, fast read-only tools)
  - Tools query RunLogService, EventBus, and WorkflowRegistry
  - All tools are read-only — never mutate state
  - Runs in-process, no EventBus dispatch needed (unlike HITL/trigger)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from autopilot.core.agent import BaseAgent
from autopilot.core.bus import get_event_bus
from autopilot.core.context import AgentContext
from autopilot.core.run_log import get_run_log_service
from autopilot.registry import get_registry

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Read-only tools — Zero side effects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def get_workflow_stats(workflow_id: str | None = None) -> dict[str, Any]:
    """Get run stats for a specific workflow or all workflows.

    Returns total runs, success/failure counts, and success rate.
    If workflow_id is None, aggregates across all workflows.
    """
    run_log = get_run_log_service()
    registry = get_registry()

    if workflow_id:
        stats = await run_log.get_stats(workflow_id)
        return {"workflow_id": workflow_id, **stats}

    # Aggregate across all workflows
    all_stats: list[dict] = []
    for info in registry.list_all():
        stats = await run_log.get_stats(info.name)
        all_stats.append({"workflow_id": info.name, **stats})

    return {"workflows": all_stats, "total_workflows": len(all_stats)}


async def get_recent_errors(
    workflow_id: str | None = None, limit: int = 10
) -> dict[str, Any]:
    """Get recent failed runs with error details.

    Scans run history for FAILED status and extracts error messages.
    If workflow_id is None, scans across all workflows.
    """
    run_log = get_run_log_service()
    registry = get_registry()
    errors: list[dict] = []

    workflow_ids = (
        [workflow_id] if workflow_id else [w.name for w in registry.list_all()]
    )

    for wf_id in workflow_ids:
        runs, _ = await run_log.list_runs(wf_id, limit=limit)
        for run in runs:
            if run.status.value == "failed":
                errors.append(
                    {
                        "run_id": run.id,
                        "workflow_id": run.workflow_id,
                        "error": run.error or "Unknown error",
                        "started_at": (
                            run.started_at.isoformat() if run.started_at else None
                        ),
                        "trigger_type": run.trigger_type.value,
                    }
                )

    errors.sort(key=lambda e: e.get("started_at") or "", reverse=True)
    return {"errors": errors[:limit], "total": len(errors[:limit])}


async def get_run_history(workflow_id: str, limit: int = 10) -> dict[str, Any]:
    """Get recent run history for a specific workflow.

    Returns the most recent runs with status, timing, and trigger info.
    """
    run_log = get_run_log_service()
    runs, _ = await run_log.list_runs(workflow_id, limit=limit)

    return {
        "workflow_id": workflow_id,
        "runs": [
            {
                "run_id": r.id,
                "status": r.status.value,
                "duration_ms": r.duration_ms,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "error": r.error,
                "trigger_type": r.trigger_type.value,
            }
            for r in runs
        ],
        "total": len(runs),
    }


async def get_event_timeline(topic: str = "*", limit: int = 20) -> dict[str, Any]:
    """Get recent EventBus events for timeline display.

    Supports topic filtering or wildcard (*) for all events.
    """
    bus = get_event_bus()

    if topic == "*":
        all_topics = list(getattr(bus, "_history", {}).keys())
        messages = []
        for t in all_topics:
            messages.extend(bus.history(t, limit=limit))
        messages.sort(key=lambda m: m.timestamp, reverse=True)
        messages = messages[:limit]
    else:
        messages = bus.history(topic, limit=limit)

    return {
        "events": [
            {
                "topic": m.topic,
                "sender": m.sender,
                "timestamp": m.timestamp,
                "payload_keys": list(m.payload.keys()) if m.payload else [],
            }
            for m in messages
        ],
        "total": len(messages),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool registry — maps name → callable for the ReAct agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COPILOT_TOOLS: dict[str, Any] = {
    "get_workflow_stats": {
        "fn": get_workflow_stats,
        "description": "Get run statistics (total, success, failures) for a workflow or all workflows",
        "params": {"workflow_id": "Optional workflow ID, omit for all workflows"},
    },
    "get_recent_errors": {
        "fn": get_recent_errors,
        "description": "Get recent failed runs with error details",
        "params": {
            "workflow_id": "Optional workflow ID, omit for all workflows",
            "limit": "Max errors to return (default: 10)",
        },
    },
    "get_run_history": {
        "fn": get_run_history,
        "description": "Get recent run history for a specific workflow",
        "params": {
            "workflow_id": "Required workflow ID",
            "limit": "Max runs to return (default: 10)",
        },
    },
    "get_event_timeline": {
        "fn": get_event_timeline,
        "description": "Get recent EventBus events with optional topic filter",
        "params": {
            "topic": "Event topic filter or '*' for all (default: '*')",
            "limit": "Max events to return (default: 20)",
        },
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CopilotAgent — ReAct agent that uses tools to answer questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CopilotAgent(BaseAgent):
    """Platform observability agent — answers questions using read-only tools.

    Implements a ReAct loop:
      1. Reason — analyze the query and determine which tools to call
      2. Act — execute the selected tools
      3. Observe — examine results and decide if more info is needed
      4. Synthesize — produce a final answer

    The agent tracks tool calls for transparency in the response.
    Sets `react_finished=True` when the answer is ready.
    """

    def __init__(self):
        super().__init__(name="copilot", description="Platform observability agent")
        self._tool_calls: list = []
        self._iterations: int = 0

    async def run(self, ctx: AgentContext, input: dict[str, Any]) -> dict[str, Any]:
        """Execute one iteration of the ReAct loop.

        Iteration 1: Route query to tools, execute them, collect results.
        Iteration 2+: Synthesize answer from collected results.
        """
        from autopilot.api.v1.models import CopilotToolCall

        query = input.get("query", ctx.state.get("query", ""))
        self._iterations += 1

        # If we already have tool results, synthesize the final answer
        if ctx.state.get("tool_results"):
            answer = self._synthesize_answer(query, ctx.state["tool_results"])
            return {
                "answer": answer,
                "tools_used": [tc.model_dump() for tc in self._tool_calls],
                "iterations": self._iterations,
                "react_finished": True,
            }

        # First iteration — determine and execute tools
        tools_to_call = self._route_query(query)
        tool_results: dict[str, Any] = {}

        for tool_name, tool_args in tools_to_call:
            tool_info = COPILOT_TOOLS.get(tool_name)
            if not tool_info:
                continue

            try:
                result = await tool_info["fn"](**tool_args)
                tool_results[tool_name] = result

                # Record tool call for transparency
                result_summary = self._summarize_result(tool_name, result)
                self._tool_calls.append(
                    CopilotToolCall(
                        tool=tool_name,
                        args=tool_args,
                        result_summary=result_summary,
                    )
                )

                ctx.logger.info(
                    "copilot_tool_called",
                    tool=tool_name,
                    args=tool_args,
                )
            except Exception as exc:
                logger.warning(
                    "copilot_tool_error",
                    tool=tool_name,
                    error=str(exc),
                )
                tool_results[tool_name] = {"error": str(exc)}

        # If no tools were matched, provide a direct answer
        if not tool_results:
            return {
                "answer": (
                    "I can help you with workflow statistics, recent errors, "
                    "run history, and event timelines. Try asking something like "
                    "'Which workflows failed today?' or 'Show me stats for all workflows'."
                ),
                "tools_used": [],
                "iterations": self._iterations,
                "react_finished": True,
            }

        # Store results for next iteration to synthesize
        return {"tool_results": tool_results}

    def _route_query(self, query: str) -> list[tuple[str, dict]]:
        """Route the query to appropriate tools based on intent analysis.

        Uses keyword matching for reliable, zero-cost routing.
        Returns a list of (tool_name, args) tuples.
        """
        q = query.lower()
        tools: list[tuple[str, dict]] = []

        # Extract workflow_id if mentioned
        registry = get_registry()
        workflow_ids = [w.name for w in registry.list_all()]
        mentioned_workflow = None
        for wf_id in workflow_ids:
            if wf_id in q:
                mentioned_workflow = wf_id
                break

        # Error / failure queries
        if any(kw in q for kw in ["error", "fail", "crash", "broken", "issue"]):
            args: dict[str, Any] = {}
            if mentioned_workflow:
                args["workflow_id"] = mentioned_workflow
            tools.append(("get_recent_errors", args))

        # Stats / performance queries
        if any(
            kw in q
            for kw in ["stat", "success", "rate", "performance", "count", "how many"]
        ):
            args = {}
            if mentioned_workflow:
                args["workflow_id"] = mentioned_workflow
            tools.append(("get_workflow_stats", args))

        # History / runs queries
        if any(kw in q for kw in ["history", "run", "recent", "last", "today"]):
            if mentioned_workflow:
                tools.append(("get_run_history", {"workflow_id": mentioned_workflow}))
            else:
                tools.append(("get_workflow_stats", {}))

        # Event / timeline queries
        if any(kw in q for kw in ["event", "timeline", "stream", "bus"]):
            tools.append(("get_event_timeline", {}))

        # Default: show stats if nothing matched
        if not tools:
            args = {}
            if mentioned_workflow:
                args["workflow_id"] = mentioned_workflow
            tools.append(("get_workflow_stats", args))

        # Deduplicate while preserving order
        seen = set()
        unique: list[tuple[str, dict]] = []
        for tool_name, tool_args in tools:
            key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append((tool_name, tool_args))

        return unique

    def _summarize_result(self, tool_name: str, result: dict) -> str:
        """Create a brief summary of a tool result for the response."""
        if tool_name == "get_workflow_stats":
            if "workflows" in result:
                return f"{result.get('total_workflows', 0)} workflows found"
            total = result.get("total", 0)
            return f"{total} total runs"

        if tool_name == "get_recent_errors":
            total = result.get("total", 0)
            return f"{total} recent errors found"

        if tool_name == "get_run_history":
            total = result.get("total", 0)
            return f"{total} recent runs"

        if tool_name == "get_event_timeline":
            total = result.get("total", 0)
            return f"{total} recent events"

        return "completed"

    def _synthesize_answer(self, query: str, tool_results: dict) -> str:
        """Synthesize a natural language answer from tool results.

        Uses template-based synthesis for fast, predictable responses.
        """
        parts: list[str] = []

        for tool_name, result in tool_results.items():
            if "error" in result:
                parts.append(f"⚠️ Error from {tool_name}: {result['error']}")
                continue

            if tool_name == "get_workflow_stats":
                if "workflows" in result:
                    wfs = result["workflows"]
                    for wf in wfs:
                        total = wf.get("total", 0)
                        success = wf.get("successful", 0)
                        failed = total - success
                        rate = round(success / total * 100, 1) if total > 0 else 0
                        parts.append(
                            f"**{wf['workflow_id']}**: {total} runs, "
                            f"{success} success, {failed} failed ({rate}% success rate)"
                        )
                else:
                    total = result.get("total", 0)
                    success = result.get("successful", 0)
                    failed = total - success
                    rate = round(success / total * 100, 1) if total > 0 else 0
                    wf_id = result.get("workflow_id", "unknown")
                    parts.append(
                        f"**{wf_id}**: {total} runs, "
                        f"{success} success, {failed} failed ({rate}% success rate)"
                    )

            elif tool_name == "get_recent_errors":
                errors = result.get("errors", [])
                if not errors:
                    parts.append("✅ No recent errors found.")
                else:
                    parts.append(f"Found **{len(errors)}** recent errors:")
                    for err in errors[:5]:
                        parts.append(
                            f"  - `{err['workflow_id']}` run `{err['run_id']}`: "
                            f"{err['error'][:100]}"
                        )

            elif tool_name == "get_run_history":
                runs = result.get("runs", [])
                if not runs:
                    parts.append("No recent runs found.")
                else:
                    parts.append(
                        f"Last **{len(runs)}** runs for `{result.get('workflow_id', 'unknown')}`:"
                    )
                    for r in runs[:5]:
                        status_emoji = "✅" if r["status"] == "success" else "❌"
                        duration = (
                            f" ({r['duration_ms']:.0f}ms)"
                            if r.get("duration_ms")
                            else ""
                        )
                        parts.append(
                            f"  - {status_emoji} `{r['run_id']}` — {r['status']}{duration}"
                        )

            elif tool_name == "get_event_timeline":
                events = result.get("events", [])
                if not events:
                    parts.append("No recent events.")
                else:
                    parts.append(f"Last **{len(events)}** events:")
                    for e in events[:5]:
                        parts.append(
                            f"  - `{e['topic']}` from {e.get('sender', 'unknown')} "
                            f"at {e.get('timestamp', 'N/A')}"
                        )

        return "\n".join(parts) if parts else "No data available for your query."
