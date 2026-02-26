"""
Pipeline — Declarative workflow composition engine.

Provides a typed, fluent pipeline builder and executor that:
  - Orchestrates BaseAgent steps in sequence (DAG support ready)
  - Propagates AgentContext (logger, events, state) through every step
  - Supports mixed step types: LLM agents, code functions, ADK agents
  - Auto-wraps plain functions and ADK agents into the BaseAgent contract
  - Emits lifecycle events for real-time observability

Usage:
    from autopilot.core import Pipeline, PipelineBuilder, AgentContext

    pipeline = (
        PipelineBuilder("bank_to_ynab")
        .step(email_parser)              # BaseAgent or ADK LlmAgent
        .step(match_account)             # plain function → auto-wrapped
        .step(categorizer)
        .step(merge_and_validate)
        .build()
    )

    ctx = AgentContext(pipeline_name="bank_to_ynab")
    result = await pipeline.execute(ctx, initial_input={"email_body": raw})
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import structlog
from opentelemetry import trace

from autopilot.core._artifact_persist import persist_node_artifact
from autopilot.core.context import AgentContext
from autopilot.core.agent import BaseAgent, FunctionalAgent, ADKAgent

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PipelineExecutionResult:
    """
    Structured result from a Pipeline execution.

    Attributes:
        execution_id: Correlation ID for this run.
        state: Final accumulated pipeline state.
        steps_completed: Names of steps that ran successfully.
        duration_ms: Total pipeline execution time.
        success: Whether all steps completed without errors.
        error: Error message if the pipeline failed.
    """

    execution_id: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    steps_completed: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    paused: bool = False
    error: str | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline — Sequential execution engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Pipeline:
    """
    A named, ordered sequence of BaseAgent steps.

    The pipeline manages execution flow:
      1. Creates an AgentContext (or uses one provided).
      2. Feeds the initial input into the state.
      3. Runs each step, passing accumulated state as input.
      4. Merges each step's output back into the state.
      5. Emits lifecycle events at each transition.

    Manages sequential execution flow with full observability.
    """

    def __init__(self, name: str, steps: list[BaseAgent]):
        self.name = name
        self.steps = steps

    async def execute(
        self,
        ctx: AgentContext | None = None,
        *,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineExecutionResult:
        """
        Execute the full pipeline.

        Args:
            ctx: Execution context. If None, a new one is created.
            initial_input: Initial state/input for the pipeline.

        Returns:
            PipelineExecutionResult with final state and metadata.
        """
        # Initialize context
        if ctx is None:
            ctx = AgentContext(pipeline_name=self.name)
        else:
            # Ensure pipeline name is set
            if not ctx.pipeline_name:
                ctx.pipeline_name = self.name

        # Seed state
        if initial_input:
            ctx.update_state(initial_input)

        # Initialize ADK session lazily
        await ctx.ensure_session()

        result = PipelineExecutionResult(execution_id=ctx.execution_id)
        start = time.monotonic()

        with tracer.start_as_current_span(
            "pipeline.execute",
            attributes={
                "pipeline_name": self.name,
                "execution_id": ctx.execution_id,
                "step_count": len(self.steps),
            },
        ) as span:
            ctx.logger.info(
                "pipeline_started",
                pipeline=self.name,
                steps=[s.name for s in self.steps],
                initial_keys=list(ctx.state.keys()),
            )

        await ctx.publish(
            "pipeline.started",
            {
                "pipeline": self.name,
                "step_count": len(self.steps),
            },
        )

        try:
            # For resuming HITL, we track which steps were already done
            completed_in_state = ctx.state.get("__steps_completed__", [])

            for step in self.steps:
                if step.name in completed_in_state:
                    ctx.logger.debug("step_skipped_resuming", step=step.name)
                    result.steps_completed.append(step.name)
                    continue

                step_start = time.monotonic()

                ctx.logger.info("step_started", step=step.name)
                await ctx.publish("stage.started", {"step": step.name})

                # Each step receives the FULL accumulated state as input
                step_output = await step.invoke(ctx, ctx.state)

                # Merge output back into state
                if step_output:
                    ctx.update_state(step_output)

                step_elapsed = round((time.monotonic() - step_start) * 1000, 2)
                result.steps_completed.append(step.name)

                # Keep state updated with completed steps for future resumes
                if "__steps_completed__" not in ctx.state:
                    ctx.state["__steps_completed__"] = []
                if step.name not in ctx.state["__steps_completed__"]:
                    ctx.state["__steps_completed__"].append(step.name)

                ctx.logger.info(
                    "step_completed",
                    step=step.name,
                    duration_ms=step_elapsed,
                    output_keys=list(step_output.keys()) if step_output else [],
                )
                await ctx.publish(
                    "stage.completed",
                    {
                        "step": step.name,
                        "duration_ms": step_elapsed,
                    },
                )

                # ── Persist step output as a versioned artifact ───────
                if step_output:
                    await persist_node_artifact(
                        ctx,
                        engine_name=self.name,
                        node_name=step.name,
                        output=step_output,
                        duration_ms=step_elapsed,
                    )

                # Human-In-The-Loop Pause Check
                if ctx.state.get("hitl_requested") is True:
                    ctx.logger.info("pipeline_paused_for_hitl", step=step.name)
                    await ctx.publish("pipeline.paused", {"step": step.name})
                    result.paused = True
                    # Clear the flag so it doesn't immediately re-pause when resumed
                    ctx.state["hitl_requested"] = False
                    break

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            ctx.logger.error(
                "pipeline_failed",
                pipeline=self.name,
                step=step.name if "step" in dir() else "unknown",
                error=str(exc),
            )
            await ctx.publish(
                "pipeline.failed",
                {
                    "pipeline": self.name,
                    "error": str(exc),
                },
            )
            raise

        finally:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            result.duration_ms = elapsed
            result.state = dict(ctx.state)

            if result.success:
                ctx.logger.info(
                    "pipeline_completed",
                    pipeline=self.name,
                    duration_ms=elapsed,
                    steps_completed=result.steps_completed,
                )
                await ctx.publish(
                    "pipeline.completed",
                    {
                        "pipeline": self.name,
                        "duration_ms": elapsed,
                        "steps_completed": result.steps_completed,
                    },
                )

            span.set_attribute("duration_ms", elapsed)
            span.set_attribute("success", result.success)
            span.set_attribute("paused", result.paused)

        return result

    def __repr__(self) -> str:
        step_names = ", ".join(s.name for s in self.steps)
        return f"<Pipeline {self.name!r} steps=[{step_names}]>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PipelineBuilder — Fluent API for pipeline construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


StepLike = Union[BaseAgent, Callable[..., Any], Any]
"""
Type alias for things that can be added as pipeline steps:
  - BaseAgent subclass (direct)
  - Callable (auto-wrapped as FunctionalAgent)
  - ADK LlmAgent or similar (auto-wrapped as ADKAgent)
"""


class PipelineBuilder:
    """
    Fluent builder for constructing Pipeline instances.

    Supports three step types via auto-detection:
      1. BaseAgent subclass → used directly
      2. Plain function → wrapped as FunctionalAgent
      3. ADK agent (has .name + .instruction) → wrapped as ADKAgent

    Usage:
        pipeline = (
            PipelineBuilder("my_pipeline")
            .step(parser_agent)          # BaseAgent
            .step(match_account)         # function
            .step(create_categorizer())  # ADK LlmAgent
            .build()
        )
    """

    def __init__(self, name: str):
        self.name = name
        self._steps: list[BaseAgent] = []

    def step(self, step_like: StepLike) -> "PipelineBuilder":
        """
        Add a step to the pipeline.

        Auto-detects the type and wraps it appropriately.
        """
        self._steps.append(_wrap_step(step_like))
        return self

    def loop(
        self,
        body: StepLike,
        *,
        condition: Callable[[dict], bool],
        max_iterations: int = 3,
        name: str | None = None,
    ) -> "PipelineBuilder":
        """
        Add a loop step that retries body until ``condition(state) → True``.

        Mirrors ADK's ``LoopAgent`` pattern.  The body is auto-wrapped
        if it's a plain function or ADK agent.

        Args:
            body: Agent, function, or ADK agent to loop.
            condition: ``fn(state_dict) → bool``; True = exit loop (success).
            max_iterations: Max loop iterations before raising MaxRetriesExceededError.
            name: Optional name for the loop adapter.
        """
        from autopilot.core.agent import LoopAgentAdapter

        wrapped_body = _wrap_step(body)
        resolved_name = name or f"loop_{wrapped_body.name}"
        self._steps.append(
            LoopAgentAdapter(
                resolved_name,
                wrapped_body,
                condition=condition,
                max_iterations=max_iterations,
            )
        )
        return self

    def parallel(
        self,
        *branches: StepLike,
        name: str | None = None,
    ) -> "PipelineBuilder":
        """
        Add a parallel step that runs all branches concurrently.

        Mirrors ADK's ``ParallelAgent`` pattern.  Each branch is
        auto-wrapped if it's a plain function or ADK agent.

        Args:
            *branches: Agents, functions, or ADK agents to run in parallel.
            name: Optional name for the parallel adapter.
        """
        from autopilot.core.agent import ParallelAgentAdapter

        wrapped = [_wrap_step(b) for b in branches]
        resolved_name = name or f"parallel_{'_'.join(b.name for b in wrapped)}"
        self._steps.append(
            ParallelAgentAdapter(
                resolved_name,
                branches=wrapped,
            )
        )
        return self

    def build(self) -> Pipeline:
        """Build and return the Pipeline."""
        if not self._steps:
            raise ValueError(f"Pipeline '{self.name}' has no steps.")
        return Pipeline(self.name, list(self._steps))

    def __repr__(self) -> str:
        return f"<PipelineBuilder {self.name!r} steps={len(self._steps)}>"


# ── Helpers ──────────────────────────────────────────────────────────


def _is_adk_agent(obj: Any) -> bool:
    """
    Heuristic to detect a Google ADK agent.

    ADK agents have `name` and `instruction` attributes but are NOT
    BaseAgent subclasses (those are our own).
    """
    return (
        hasattr(obj, "name")
        and hasattr(obj, "instruction")
        and not isinstance(obj, BaseAgent)
    )


def _wrap_step(step_like: StepLike) -> BaseAgent:
    """
    Normalize any StepLike into a BaseAgent instance.

    Supports:
      - BaseAgent subclass → used directly
      - Callable → wrapped as FunctionalAgent
      - ADK agent (has .name + .instruction) → wrapped as ADKAgent
    """
    if isinstance(step_like, BaseAgent):
        return step_like
    elif callable(step_like) and not _is_adk_agent(step_like):
        return FunctionalAgent(step_like)
    elif _is_adk_agent(step_like):
        return ADKAgent(step_like)
    else:
        raise TypeError(
            f"Cannot add step of type {type(step_like).__name__}. "
            f"Expected BaseAgent, callable, or ADK agent."
        )
