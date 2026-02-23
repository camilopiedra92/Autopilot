"""
Platform Agent Factory — Standardized agent creation.

This module provides a factory function `create_platform_agent` to instantiate
ADK `LlmAgent`s with standard platform configuration:
  - Automatic observability callbacks (logging, tracing)
  - Consistent error handling wrapper
  - Standardized model defaults
  - Dual-layer rate limiting (ADK HttpRetryOptions + proactive token-bucket)
"""

from __future__ import annotations

from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.genai import types

from autopilot.agents.callbacks import (
    before_model_logger,
    after_model_logger,
    create_chained_before_callback,
    create_chained_after_callback,
)
from autopilot.agents.context_cache import PLATFORM_CACHE_CONTEXT_ATTR
from autopilot.agents.rate_limiter import get_model_rate_limit_callback
from autopilot.agents.tool_callbacks import before_tool_logger, after_tool_logger
from autopilot.core.agent import BaseAgent, FallbackAgentAdapter, ADKAgent

# ADK-native retry for transient 429s (Layer 1 — reactive)
_DEFAULT_RETRY = types.HttpRetryOptions(initial_delay=1, attempts=3)

# ── Spanish day/month names for temporal context ────────────────────
_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS_ES = [
    "",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]


class _SafeFormatMap(dict):
    """Dict subclass for ``str.format_map()`` that preserves missing keys.

    ``defaultdict(str)`` would replace ``{missing}`` with ``""``, breaking
    instruction placeholders that haven't been set in the session state yet.
    This class returns ``"{key}"`` for any missing key, keeping unresolved
    placeholders intact for later resolution.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _with_temporal_context(
    instruction: str,
) -> Callable:
    """Wrap a static instruction into an ADK InstructionProvider.

    Returns a callable ``(ReadonlyContext) -> str`` that:

    1. Prepends the current date/time (Colombia TZ, Spanish locale).
    2. Resolves ``{key}`` placeholders from ``ReadonlyContext.state``,
       replicating ADK's native state injection (which is disabled when
       using callable instructions per ADK docs).

    ADK calls this function on **every LLM invocation**, so the timestamp is
    always accurate — no stale data, no tech debt.

    This is the canonical ADK approach:
    ``LlmAgent(instruction=callable)`` instead of a plain string.
    """
    import datetime
    import zoneinfo

    _tz = zoneinfo.ZoneInfo("America/Bogota")

    def provider(ctx) -> str:
        now = datetime.datetime.now(_tz)
        dt = (
            f"{_DAYS_ES[now.weekday()]} {now.day} de "
            f"{_MONTHS_ES[now.month]} de {now.year}, "
            f"{now.strftime('%H:%M')} (hora Colombia)"
        )
        base = f"[Fecha y hora actual: {dt}]\n\n{instruction}"

        # Resolve {key} placeholders from session state.
        #
        # ADK disables its native state injection for callable instructions
        # (see: https://google.github.io/adk-docs/sessions/state).
        # We replicate the same behavior here so workflows can use
        # {key} placeholders in instructions alongside the dynamic datetime.
        # ReadonlyContext.state → MappingProxyType(session.state).
        if ctx is not None and hasattr(ctx, "state"):
            state = ctx.state  # property → MappingProxyType[str, Any]
            if hasattr(state, "get"):
                safe = _SafeFormatMap(state)
                try:
                    base = base.format_map(safe)
                except (KeyError, ValueError, IndexError):
                    pass  # Malformed placeholders — leave instruction as-is

        return base

    return provider


def create_platform_agent(
    name: str,
    instruction: str,
    model: str = "gemini-3-flash-preview",
    fallback_model: str | None = None,
    description: str | None = None,
    tools: list[Callable | Any] | None = None,
    output_key: str | None = None,
    output_schema: type | None = None,
    temperature: float | None = None,
    cache_context: bool = False,
    **kwargs,
) -> LlmAgent | BaseAgent:
    """
    Create a standard LlmAgent with platform observability and defaults.

    Args:
        name: Unique name of the agent (used in logs/traces).
        instruction: System prompt/instruction for the agent.
        model: Model ID to use (default: gemini-3-flash-preview).
        fallback_model: Optional fallback model if the primary fails (e.g., gemini-2.0-pro-exp).
        description: Description of what the agent does.
        tools: List of tools (functions or Tool instances) available to the agent.
        output_key: Session state key to write the result to.
        output_schema: Pydantic model for structured output. When set, ADK
            activates Gemini native JSON mode (``response_schema`` +
            ``response_mime_type=application/json``) — the LLM produces
            constrained JSON that ADK validates via ``model_validate_json()``.
            Agent transfers are also disabled for full isolation.
        temperature: Sampling temperature (optional).
        cache_context: If True, enables Gemini context caching for this agent.
            Caches system instructions + tool schemas server-side for reuse
            across invocations. Best for agents with long, static instructions.
        **kwargs: Additional arguments passed to LlmAgent constructor.

    Returns:
        Configured LlmAgent instance.
    """
    # Configure generation config if parameters provided
    gen_config = kwargs.pop("generate_content_config", None)

    # Wrap the instruction in a temporal context provider so every agent
    # knows the current date/time.  Uses ADK's native InstructionProvider
    # pattern: LlmAgent(instruction=callable) — called on every LLM invocation.
    instruction_provider = _with_temporal_context(instruction)

    # Auto-resolve tools passed as strings via the platform registry
    resolved_tools = []
    if tools:
        string_tools = [t for t in tools if isinstance(t, str)]
        other_tools = [t for t in tools if not isinstance(t, str)]

        if string_tools:
            from autopilot.core.tools import get_tool_registry

            resolved_from_strings = get_tool_registry().to_adk_tools(names=string_tools)
            resolved_tools.extend(resolved_from_strings)

        resolved_tools.extend(other_tools)

    # Inject ADK-native HttpRetryOptions for transient 429s (Layer 1)
    if gen_config is None:
        gen_config = types.GenerateContentConfig(
            http_options=types.HttpOptions(retry_options=_DEFAULT_RETRY),
        )
    elif not getattr(getattr(gen_config, "http_options", None), "retry_options", None):
        if gen_config.http_options is None:
            gen_config.http_options = types.HttpOptions(retry_options=_DEFAULT_RETRY)
        elif gen_config.http_options.retry_options is None:
            gen_config.http_options.retry_options = _DEFAULT_RETRY

    if temperature is not None:
        gen_config.temperature = temperature

    # Merge callbacks with platform defaults
    # Chain: rate_limiter (Layer 2, optional) → user guardrails → platform logger
    rate_limiter_cb = get_model_rate_limit_callback()
    user_before_model = kwargs.pop("before_model_callback", None)

    before_chain = []
    if rate_limiter_cb:
        before_chain.append(rate_limiter_cb)
    if user_before_model:
        before_chain.append(user_before_model)
    before_chain.append(before_model_logger)
    final_before_model = create_chained_before_callback(*before_chain)

    user_after_model = kwargs.pop("after_model_callback", None)
    if user_after_model:
        # Platform logger first (latency), then user callback (guardrails)
        final_after_model = create_chained_after_callback(
            after_model_logger, user_after_model
        )
    else:
        final_after_model = after_model_logger

    user_before_tool = kwargs.pop("before_tool_callback", None)
    final_before_tool = before_tool_logger
    if user_before_tool:
        final_before_tool = create_chained_before_callback(
            user_before_tool, before_tool_logger
        )

    user_after_tool = kwargs.pop("after_tool_callback", None)
    final_after_tool = after_tool_logger
    if user_after_tool:
        final_after_tool = create_chained_after_callback(
            after_tool_logger, user_after_tool
        )

    # When output_schema is set, ADK activates Gemini native JSON mode and
    # disables tools. We also disable agent transfers for full isolation —
    # schema-constrained agents should ONLY produce structured output.
    if output_schema is not None:
        kwargs.setdefault("disallow_transfer_to_parent", True)
        kwargs.setdefault("disallow_transfer_to_peers", True)

    primary_agent = LlmAgent(
        name=name,
        model=model,
        instruction=instruction_provider,
        description=description,
        tools=resolved_tools,
        output_key=output_key,
        output_schema=output_schema,
        # Merged callbacks
        before_model_callback=final_before_model,
        after_model_callback=final_after_model,
        before_tool_callback=final_before_tool,
        after_tool_callback=final_after_tool,
        # Config
        generate_content_config=gen_config,
        **kwargs,
    )

    # Mark agent for context caching if opted in
    if cache_context:
        setattr(primary_agent, PLATFORM_CACHE_CONTEXT_ATTR, True)

    if not fallback_model:
        return primary_agent

    # Create the fallback LlmAgent matching all configuration except the model
    fallback_adk_agent = LlmAgent(
        name=f"{name}_fallback",
        model=fallback_model,
        instruction=instruction_provider,
        description=f"Fallback for {name}",
        tools=resolved_tools,
        output_key=output_key,
        output_schema=output_schema,
        before_model_callback=final_before_model,
        after_model_callback=final_after_model,
        before_tool_callback=final_before_tool,
        after_tool_callback=final_after_tool,
        generate_content_config=gen_config,
        **kwargs,
    )

    # Mark fallback agent for context caching if opted in
    if cache_context:
        setattr(fallback_adk_agent, PLATFORM_CACHE_CONTEXT_ATTR, True)

    # Wrap both in ADKAgent and return the Fallback adapter
    return FallbackAgentAdapter(
        name=f"{name}_with_fallback",
        primary=ADKAgent(primary_agent),
        fallback=ADKAgent(fallback_adk_agent),
        description=f"Fallback wrapper for {name} ({model} -> {fallback_model})",
    )
