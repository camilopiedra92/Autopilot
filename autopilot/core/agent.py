"""
BaseAgent — Strictly typed agent interface for the Edge-Native AI Platform.

The foundational contract for all agents in the platform.  Every agent
declares its Input and Output as Pydantic models (or plain types) via
Python generics, enabling:
  - Compile-time type safety
  - Automatic validation (Pydantic)
  - Self-documenting pipelines
  - Composability: Agents are units of work in a Pipeline

Types of agents:
  1. BaseAgent[In, Out]  — Subclass to implement custom agents.
  2. FunctionalAgent     — Wraps a plain function (sync or async) as an agent.
  3. ADK Bridge          — Wraps a Google ADK LlmAgent as a BaseAgent-compatible step.

Usage:
    class MyParser(BaseAgent[RawEmail, ParsedEmail]):
        async def run(self, ctx, input) -> ParsedEmail:
            ...

    # Or functional:
    @functional_agent
    def match_account(card_suffix: str) -> MatchedAccount:
        ...
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import time
from typing import Any, Callable, Generic, TypeVar, get_type_hints

import structlog
from opentelemetry import trace

from autopilot.core.context import AgentContext

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# ── Generic Type Variables ───────────────────────────────────────────
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BaseAgent — The core agent contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaseAgent(abc.ABC, Generic[InputT, OutputT]):
    """
    Strictly typed agent interface.

    Every agent in the platform implements this contract.
    - `InputT`:  The Pydantic model or type the agent expects.
    - `OutputT`: The Pydantic model or type the agent produces.

    The Pipeline engine calls `invoke()` (which wraps `run()` with
    observability, timing, and error handling), never `run()` directly.
    """

    def __init__(self, name: str, *, description: str = ""):
        self.name = name
        self.description = description

    @abc.abstractmethod
    async def run(self, ctx: AgentContext, input: InputT) -> OutputT:
        """
        Execute the agent's core logic.

        Implement this in subclasses.  The pipeline engine guarantees
        that `input` is validated and `ctx` is fully initialized.

        Args:
            ctx: Rich execution context (logger, state, events).
            input: Typed input data for this agent.

        Returns:
            Typed output data produced by this agent.
        """
        ...

    async def invoke(self, ctx: AgentContext, input: InputT) -> OutputT:
        """
        Platform entry point — wraps `run()` with observability.

        Called by the Pipeline engine.  Emits events, records timing,
        and catches exceptions with structured error handling.
        """
        with tracer.start_as_current_span(
            f"agent.invoke",
            attributes={
                "agent_name": self.name,
                "agent_type": self.__class__.__name__,
                "execution_id": ctx.execution_id,
            }
        ) as span:
            step_ctx = ctx.for_step(self.name)
            step_ctx.logger.info("agent_started", agent=self.name)

            await ctx.emit("agent_started", {"agent": self.name})

            start = time.monotonic()
            try:
                result = await self.run(step_ctx, input)
                elapsed = round((time.monotonic() - start) * 1000, 2)

                step_ctx.logger.info(
                    "agent_completed",
                    agent=self.name,
                    duration_ms=elapsed,
                )
                await ctx.emit("agent_completed", {
                    "agent": self.name,
                    "duration_ms": elapsed,
                })
                return result

            except Exception as exc:
                elapsed = round((time.monotonic() - start) * 1000, 2)
                step_ctx.logger.error(
                    "agent_failed",
                    agent=self.name,
                    duration_ms=elapsed,
                    error=str(exc),
                )
                await ctx.emit("agent_failed", {
                    "agent": self.name,
                    "duration_ms": elapsed,
                    "error": str(exc),
                })
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise
            
            finally:
                if 'elapsed' in locals():
                    span.set_attribute("duration_ms", elapsed)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FunctionalAgent — Wraps a plain function as an agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FunctionalAgent(BaseAgent[dict, dict]):
    """
    Wraps a plain Python function (sync or async) as a BaseAgent.

    The function's parameters are automatically extracted from the
    pipeline state dict.  The function's return value is merged back
    into the state.

    Usage:
        agent = FunctionalAgent(my_function)
        # or
        agent = FunctionalAgent(my_function, name="custom_name")
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str = "",
    ):
        resolved_name = name or func.__name__
        super().__init__(resolved_name, description=description or func.__doc__ or "")
        self._func = func
        self._is_async = asyncio.iscoroutinefunction(func)
        
        # Cache parameter names and types for fast state unpacking and auto-hydration
        sig = inspect.signature(func)
        self._params = {}
        self._has_var_keyword = False
        
        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                self._has_var_keyword = True
            else:
                self._params[param_name] = param.annotation

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """Extract function args from state and execute."""
        # Build kwargs from the pipeline state
        kwargs = {}
        
        # If function accepts **kwargs, pass the full accumulated state
        if self._has_var_keyword:
            kwargs.update(ctx.state)
            kwargs.update(input)
            
        for param_name, param_type in self._params.items():
            if param_name in input:
                val = input[param_name]
            elif param_name in ctx.state:
                val = ctx.state[param_name]
            else:
                continue

            # Auto-hydrate Pydantic models from dict based on type hint
            if isinstance(val, dict) and param_type is not inspect.Parameter.empty:
                # Get the actual type if it's an Optional/Union
                import typing as _typing
                origin = _typing.get_origin(param_type)
                args = _typing.get_args(param_type)
                
                # Check if the type is a Union (like dict | None)
                from types import UnionType
                if origin is UnionType or origin is _typing.Union:
                    if type(None) in args: # Typical Optional check
                        effective_type = next((t for t in args if t is not type(None)), None)
                    else:
                        effective_type = param_type
                else:
                    effective_type = param_type
                    
                import inspect as _inspect
                if _inspect.isclass(effective_type) and hasattr(effective_type, "model_validate"):
                    val = effective_type.model_validate(val)
                    
            kwargs[param_name] = val

        # Execute
        if self._is_async:
            result = await self._func(**kwargs)
        else:
            result = self._func(**kwargs)

        # Normalize return to dict
        if result is None:
            return {}
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {self.name: result}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADKAgent — Bridges Google ADK LlmAgent into BaseAgent contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ADKAgent(BaseAgent[dict, dict]):
    """
    Wraps a Google ADK LlmAgent (or any ADK agent) as a BaseAgent.

    Bridges the ADK's Runner/Session model into our typed pipeline.
    The wrapped agent is run via PipelineRunner and its output
    is returned as a dict ready for state merging.
    """

    def __init__(self, adk_agent: Any, *, name: str | None = None):
        resolved_name = name or getattr(adk_agent, "name", "adk_agent")
        super().__init__(resolved_name, description=getattr(adk_agent, "description", ""))
        self._adk_agent = adk_agent

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """
        Run the ADK agent through the PipelineRunner.

        Uses the existing PipelineRunner._run_adk_agent path so that
        all ADK features (tools, memory, structured output) work.
        """
        from autopilot.agents.pipeline_runner import get_pipeline_runner

        runner = get_pipeline_runner(ctx.pipeline_name or "autopilot")

        # Build the message from the state
        message = input.get("message", "")
        if not message:
            import json
            message = json.dumps(input, default=str)

        result = await runner.run(
            pipeline=self._adk_agent,
            message=message,
            initial_state=ctx.state,
            stream_session_id=ctx._stream_id,
        )

        output_key = getattr(self._adk_agent, "output_key", None)

        # Strategy: prefer ADK session state (canonical for output_key),
        # fall back to text-parsed JSON.  ADK natively stores structured
        # output in state[output_key] when output_schema is used, which
        # is far more reliable than regex-parsing the response text.
        if output_key and output_key in result.state:
            return {output_key: result.state[output_key]}

        base_output = result.parsed_json or {"output": result.final_text}
        if output_key:
            return {output_key: base_output}
        return base_output


# ── Decorator shorthand ──────────────────────────────────────────────

def functional_agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
) -> FunctionalAgent | Callable[..., FunctionalAgent]:
    """
    Decorator to turn a plain function into a FunctionalAgent.

    Usage:
        @functional_agent
        def match_account(card_suffix: str) -> MatchedAccount:
            ...

        # or with options:
        @functional_agent(name="matcher")
        def match_account(card_suffix: str) -> MatchedAccount:
            ...
    """
    if func is not None:
        return FunctionalAgent(func, name=name, description=description)

    def wrapper(f: Callable[..., Any]) -> FunctionalAgent:
        return FunctionalAgent(f, name=name, description=description)
    return wrapper


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  V3 Workflow Adapters — ADK-aligned composition primitives
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SequentialAgentAdapter(BaseAgent[dict, dict]):
    """
    Executes a list of child agents sequentially, accumulating state.

    Mirrors Google ADK's ``SequentialAgent(sub_agents=[...])``.
    Each child receives the full accumulated state as input and its
    output is merged back into the state before the next child runs.
    All children share the same AgentContext (execution_id, logger, events).

    Usage::

        seq = SequentialAgentAdapter(
            "parse_and_validate",
            children=[parser_agent, validator_agent],
        )
    """

    def __init__(
        self,
        name: str,
        children: list[BaseAgent],
        *,
        description: str = "",
    ):
        super().__init__(name, description=description)
        if not children:
            raise ValueError(f"SequentialAgentAdapter '{name}' requires at least one child.")
        self.children = children

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """Run children in order, accumulating state."""
        state = dict(input)

        for child in self.children:
            child_output = await child.invoke(ctx, state)
            if child_output:
                state.update(child_output)

            await ctx.emit("sequence_step_completed", {
                "adapter": self.name,
                "child": child.name,
            })

        return state


class LoopAgentAdapter(BaseAgent[dict, dict]):
    """
    Loops a body agent until an exit condition is met or max iterations exhausted.

    Mirrors Google ADK's ``LoopAgent(max_iterations=N)`` + escalate pattern.
    After each iteration, the ``condition`` callable receives the accumulated
    state dict and returns True to exit the loop (success) or False to continue.

    Raises ``MaxRetriesExceededError`` if all iterations are exhausted
    without the condition returning True.

    Usage::

        loop = LoopAgentAdapter(
            "retry_parse",
            body=parser_agent,
            condition=lambda state: state.get("valid", False),
            max_iterations=3,
        )
    """

    def __init__(
        self,
        name: str,
        body: BaseAgent,
        *,
        condition: Callable[[dict], bool],
        max_iterations: int = 3,
        description: str = "",
    ):
        super().__init__(name, description=description)
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """Execute body in a loop until condition passes or max_iterations reached."""
        from autopilot.errors import MaxRetriesExceededError

        state = dict(input)

        for iteration in range(1, self.max_iterations + 1):
            body_output = await self.body.invoke(ctx, state)
            if body_output:
                state.update(body_output)

            await ctx.emit("loop_iteration", {
                "adapter": self.name,
                "iteration": iteration,
                "max_iterations": self.max_iterations,
            })

            if self.condition(state):
                ctx.logger.info(
                    "loop_exit_condition_met",
                    adapter=self.name,
                    iteration=iteration,
                )
                return state

        raise MaxRetriesExceededError(
            f"LoopAgentAdapter '{self.name}' exhausted {self.max_iterations} iterations "
            f"without exit condition passing.",
            iterations=self.max_iterations,
        )


class ParallelAgentAdapter(BaseAgent[dict, dict]):
    """
    Executes multiple branch agents concurrently and merges their outputs.

    Mirrors Google ADK's ``ParallelAgent(sub_agents=[...])``.
    All branches receive the same input snapshot. Their outputs are
    merged into a single dict (last-write-wins for key conflicts).

    If any branch raises, the entire parallel step fails immediately.

    Usage::

        par = ParallelAgentAdapter(
            "fetch_all",
            branches=[api1_fetcher, api2_fetcher],
        )
    """

    def __init__(
        self,
        name: str,
        branches: list[BaseAgent],
        *,
        description: str = "",
    ):
        super().__init__(name, description=description)
        if not branches:
            raise ValueError(f"ParallelAgentAdapter '{name}' requires at least one branch.")
        self.branches = branches

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """Run all branches concurrently and merge results."""
        # Snapshot input so branches don't mutate each other's view
        frozen_input = dict(input)

        async def _run_branch(branch: BaseAgent) -> dict:
            result = await branch.invoke(ctx, frozen_input)
            await ctx.emit("parallel_branch_completed", {
                "adapter": self.name,
                "branch": branch.name,
            })
            return result or {}

        results = await asyncio.gather(
            *[_run_branch(b) for b in self.branches]
        )

        # Merge all branch outputs (last-write-wins)
        merged: dict[str, Any] = {}
        for result in results:
            merged.update(result)

        return merged


class FallbackAgentAdapter(BaseAgent[dict, dict]):
    """
    Tries a primary agent, and if it fails, falls back to a secondary agent.

    Useful for routing around rate limits or transient model errors by falling
    back from a faster/cheaper model to a larger, more robust model.

    Usage::

        fallback = FallbackAgentAdapter(
            "robust_parser",
            primary=fast_agent,
            fallback=strong_agent,
        )
    """

    def __init__(
        self,
        name: str,
        primary: BaseAgent,
        fallback: BaseAgent,
        *,
        description: str = "",
    ):
        super().__init__(name, description=description)
        self.primary = primary
        self.fallback = fallback

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        """Run primary agent, catching exceptions to trigger fallback."""
        try:
            ctx.logger.info("fallback_attempting_primary", agent=self.name, primary=self.primary.name)
            return await self.primary.invoke(ctx, input)
        except Exception as exc:
            ctx.logger.warning(
                "fallback_primary_failed",
                agent=self.name,
                primary=self.primary.name,
                fallback=self.fallback.name,
                error=str(exc),
            )
            await ctx.emit("fallback_triggered", {
                "adapter": self.name,
                "failed_agent": self.primary.name,
                "fallback_agent": self.fallback.name,
                "error": str(exc),
            })
            # Try fallback
            return await self.fallback.invoke(ctx, input)
