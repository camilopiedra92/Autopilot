"""
RouterRunner — Dynamic LLM-based workflow routing.

Allows a router agent to analyze an initial input and dynamically select the
best-fit sub-workflow (Pipeline, DAG, or ReAct loop) to handle the request.
"""

import time
from typing import Any

import structlog

from autopilot.core.context import AgentContext
from autopilot.core.agent import BaseAgent
from autopilot.core.pipeline import PipelineExecutionResult

logger = structlog.get_logger(__name__)


class RouterRunner:
    """
    Executes a router agent to select a route, then delegates execution.

    The router agent must output a dictionary containing a 'route' key
    that matches one of the defined routes.
    """

    def __init__(
        self,
        name: str,
        router_agent: BaseAgent,
        routes: dict[str, Any],
        default_route: str | None = None,
    ):
        """
        Args:
            name: Runner instance name.
            router_agent: The LLM agent responsible for picking the route.
            routes: Mapping of route string keys to executable engines (e.g. Pipeline, DAGRunner).
            default_route: Fallback route key if the router fails to pick a valid one.
        """
        self.name = name
        self.router_agent = router_agent
        self.routes = routes
        self.default_route = default_route

    async def execute(
        self,
        ctx: AgentContext | None = None,
        *,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineExecutionResult:
        """
        Execute the routing process and delegate to the selected target.
        """
        if ctx is None:
            ctx = AgentContext(pipeline_name=self.name)
        elif not ctx.pipeline_name:
            ctx.pipeline_name = self.name

        if initial_input:
            ctx.update_state(initial_input)

        result = PipelineExecutionResult(execution_id=ctx.execution_id)
        start = time.monotonic()

        ctx.logger.info(
            "router_started", runner=self.name, routes=list(self.routes.keys())
        )
        await ctx.publish("router.started", {"runner": self.name})

        try:
            # Step 1: Execute the router agent to determine intent
            route_decision = await self.router_agent.invoke(ctx, ctx.state)
            if route_decision:
                ctx.update_state(route_decision)

            selected_route = route_decision.get("route", self.default_route)

            if not selected_route or selected_route not in self.routes:
                raise ValueError(
                    f"RouterAgent '{self.router_agent.name}' returned invalid route: '{selected_route}'. "
                    f"Available routes: {list(self.routes.keys())}"
                )

            ctx.logger.info("route_selected", runner=self.name, route=selected_route)
            await ctx.publish(
                "router.route_selected", {"runner": self.name, "route": selected_route}
            )

            # Step 2: Retrieve and execute the target engine
            target_engine = self.routes[selected_route]

            # The target engine should implement execute(ctx, initial_input=...)
            sub_result = await target_engine.execute(ctx, initial_input=ctx.state)

            # Step 3: Merge result states
            ctx.update_state(sub_result.state)
            result.steps_completed = [
                f"routed_to_{selected_route}"
            ] + sub_result.steps_completed

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            ctx.logger.error("router_failed", runner=self.name, error=str(exc))
            await ctx.publish("router.failed", {"runner": self.name, "error": str(exc)})
            raise
        finally:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            result.duration_ms = elapsed
            result.state = dict(ctx.state)

            if result.success:
                ctx.logger.info(
                    "router_completed", runner=self.name, duration_ms=elapsed
                )
                await ctx.publish(
                    "router.completed", {"runner": self.name, "duration_ms": elapsed}
                )

        return result

    def __repr__(self) -> str:
        return f"<RouterRunner {self.name!r} routes={list(self.routes.keys())}>"
