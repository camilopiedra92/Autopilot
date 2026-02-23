# Autopilot Rules

> [!IMPORTANT]
> These rules must be followed by **Autopilot** and all agentic sessions.

## 1. Architecture Compliance

- **ALWAYS** check `docs/ARCHITECTURE.md` before:
  - Creating a new agent.
  - Modifying the platform architecture.
  - Adding a new workflow.
  - Creating or modifying tools.

- **Objective**: Ensure alignment with "Edge" standards, Google ADK patterns, the Factory pattern, and **multi-strategy orchestration** (Sequential, DAG, ReAct, Router).

## 2. Agent Creation & Guardrails

- **NEVER** instantiate `LlmAgent` directly; **ALWAYS** use `create_platform_agent` from `autopilot.agents.base`.
- **ALWAYS** create a corresponding `.agent.yaml` "Agent Card" for every new agent.
- **NEVER** implement guardrail callbacks directly in workflow code. **ALWAYS** use platform guard factories (`autopilot.agents.guardrails`) and compose them via `create_chained_before_callback` / `create_chained_after_callback`.
- **Domain-Specific Guards**: Put workflow-specific guards in `<workflow>/agents/guardrails.py` using the factory pattern. Do not mix them with platform guards.

## 3. Tool Ecosystem & Inter-Agent Communication (A2A)

- **ALWAYS** use the `AgentBus` (`ctx.publish()` / `ctx.subscribe()`) for decoupled async messaging between agents — never use imperative state passing.
- **ALWAYS** inject tools into agents via string references (e.g., `tools=["search_web", "ynab.get_accounts"]`). The platform auto-resolves them from the `ToolRegistry`. For MCP tools, attach via `MCPRegistry().get_all_toolsets()`.
- **ALWAYS** use the `@tool` decorator to register reusable **workflow-specific** tool functions (from `autopilot.core.tools`).
- **NEVER** use `@tool` to wrap methods that belong to Platform Connectors (like YNAB, Gmail). The platform **lazily auto-resolves** connector tools when agents reference them by `connector.method` name (e.g., `tools=["ynab.get_categories_string"]`).
- **ALWAYS** use `ToolAuthConfig` + `get_auth_manager()` for tool credentials — **NEVER** hardcode API keys.
- **ALWAYS** add `tool_context: ToolContext` parameter when a tool needs access to session state, memory, or auth.
- **ALWAYS** use `@long_running_tool` for operations that need async/batch processing or human approval.
- **ALWAYS** use `get_callback_manager()` to register lifecycle hooks (rate limits, audit logs, auth checks) at the platform level.
- **ALWAYS** set `output_key` on LLM agents to leverage ADK session state for structured output. When the agent produces structured data, **ALWAYS** also set `output_schema` (Pydantic model) — this activates Gemini native JSON mode (`response_schema` + `response_mime_type=application/json`) and auto-disables agent transfers for full isolation. Never rely on fragile text parsing as the primary output path.
- **PREFER** LLM-friendly tool signatures with explicit named parameters over opaque `dict` parameters. When a connector method takes a single `dict` (e.g., `create_task(payload: dict)`), create an `_simple` wrapper with explicit args (e.g., `create_task_simple(content, due_string, priority)`) and use that in agent tool lists. LLMs cannot infer dict schema from a `dict` type hint.

## 4. Declarative Pipelines (DSL)

- **ALWAYS** use Pydantic `BaseModel` type hints in pure Python functions (inside `steps.py`) meant to be used as `type: function` in `pipeline.yaml`. The DSL engine auto-hydrates them from the execution state.

## 5. Communication

- When proposing architectural changes, reference `docs/ARCHITECTURE.md` to justify your decisions.
- Use "Edge Design" terminology (e.g., "Edge Agent", "Platform Factory", "Connector Pattern").
