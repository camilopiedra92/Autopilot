"""
ReactRunner — Dynamic Reason-Act-Observe loop engine.

Unlike a static Pipeline, the ReactRunner gives the LLM autonomy to reason
about its current state, choose tools to act on the environment, observe the
results, and decide when its primary goal has been met.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from autopilot.core.context import AgentContext
from autopilot.core.agent import BaseAgent
from autopilot.core.pipeline import PipelineExecutionResult
from autopilot.errors import MaxRetriesExceededError

logger = structlog.get_logger(__name__)


class ReactRunner:
    """
    Executes a reasoning loop over a single autonomous agent.

    The loop continues until the state signals `react_finished = True` or the 
    max_iterations limit is reached.
    """

    def __init__(self, name: str, agent: BaseAgent, max_iterations: int = 10):
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations

    async def execute(
        self,
        ctx: AgentContext | None = None,
        *,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineExecutionResult:
        """
        Run the ReAct loop.
        """
        if ctx is None:
            ctx = AgentContext(pipeline_name=self.name)
        else:
            if not ctx.pipeline_name:
                ctx.pipeline_name = self.name

        if initial_input:
            ctx.update_state(initial_input)

        result = PipelineExecutionResult(execution_id=ctx.execution_id)
        start = time.monotonic()

        ctx.logger.info("react_started", runner=self.name, agent=self.agent.name)
        await ctx.emit("react_started", {"runner": self.name, "max_iterations": self.max_iterations})

        try:
            for iteration in range(1, self.max_iterations + 1):
                ctx.logger.info("react_iteration_started", iteration=iteration)
                await ctx.emit("react_iteration_started", {"iteration": iteration})

                # Let the agent reason and act
                output = await self.agent.invoke(ctx, ctx.state)
                
                if output:
                    ctx.update_state(output)
                
                result.steps_completed.append(f"react_iter_{iteration}")

                ctx.logger.info("react_iteration_completed", iteration=iteration)
                await ctx.emit("react_iteration_completed", {"iteration": iteration})

                # Check for termination sequence signaling the goal is met
                if ctx.state.get("react_finished") is True or (isinstance(output, dict) and output.get("react_finished") is True):
                    ctx.logger.info("react_goal_met", iteration=iteration)
                    await ctx.emit("react_goal_met", {"iteration": iteration})
                    break

            else:
                # Loop exhausted without 'react_finished' == True
                raise MaxRetriesExceededError(
                    f"ReactRunner '{self.name}' exhausted {self.max_iterations} iterations without completion.",
                    iterations=self.max_iterations
                )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            ctx.logger.error("react_failed", runner=self.name, error=str(exc))
            await ctx.emit("react_failed", {"runner": self.name, "error": str(exc)})
            raise
        finally:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            result.duration_ms = elapsed
            result.state = dict(ctx.state)

            if result.success:
                ctx.logger.info("react_completed", runner=self.name, duration_ms=elapsed)
                await ctx.emit("react_completed", {"runner": self.name, "duration_ms": elapsed})

        return result

    def __repr__(self) -> str:
        return f"<ReactRunner {self.name!r} agent={self.agent.name!r}>"
