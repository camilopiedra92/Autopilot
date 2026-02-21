---
description: Create a new workflow for Autopilot
---

// turbo-all

1.  **Read Skill**: Read `.agent/skills/create_autopilot_workflow/SKILL.md` via `view_file`.
2.  **Analyze Request**: Identify the workflow name (`snake_case`), goal, trigger type (`manual` | `webhook` | `gmail_push` | `scheduled`), and whether custom Python logic is needed.
3.  **Choose Level**:
    - **Pure YAML** (default): Pipeline of functions, no custom logic → omit `--custom`
    - **Custom Python**: Complex `execute()` logic → use `--custom`
4.  **Run Scaffold**:

    ```bash
    # Pure YAML (0 Python):
    python3 -m autopilot.cli create-workflow <name> --display-name "<Display Name>" --icon "<Icon>" --trigger <trigger>

    # Custom Python:
    python3 -m autopilot.cli create-workflow <name> --display-name "<Display Name>" --icon "<Icon>" --trigger <trigger> --custom
    ```

5.  **Wait**: Ensure the CLI completes successfully.
6.  **Refine Manifest** (`workflows/<name>/manifest.yaml`):
    - Update `triggers` with correct filters/paths per user request.
    - Add `settings` for secrets or config needed.
    - Update `description` to be clear for A2A discovery.
7.  **Implement Logic** (depends on level chosen):
    - **Pure YAML**: Edit `workflows/<name>/steps.py` (add step functions) and `workflows/<name>/pipeline.yaml` (wire steps).
    - **Custom**: Edit `workflows/<name>/workflow.py` — implement `execute()`. Do NOT add `manifest` property or `__init__.py` — both are auto-handled.
8.  **Register Tools** (if workflow uses external APIs or reusable functions):
    - **CRITICAL**: Use `@tool(tags=[...])` decorator in `steps.py` ONLY for workflow-specific reusable functions. **NEVER** wrap Platform Connectors (like YNAB, Gmail) with `@tool`; they are **lazily auto-resolved** when agents reference them (e.g., `tools=["ynab.get_categories_string"]`).
    - Add `tool_context: ToolContext` parameter if the tool needs session state, memory, or auth.
    - Configure `ToolAuthConfig` + `get_auth_manager().register(...)` for tools that need credentials.
    - Add lifecycle callbacks via `get_callback_manager()` for rate limiting or audit logging.
    - Inject tools into agents: pass tools as a list of string references (e.g., `tools=["my_custom_tool"]`).
9.  **Create Agents** (optional, for Full level):
    - Create `workflows/<name>/agents/` and add `*.agent.yaml` card files.
    - Use `create_platform_agent()` factory — **never** instantiate `LlmAgent` directly.
    - Attach Guardrails via `create_chained_before_callback` and `create_chained_after_callback`. Platform guards (e.g., `input_length_guard`, `uuid_format_guard`) should be used over custom ones when applicable. Use `<workflow>/agents/guardrails.py` for domain-specific guards.
    - Inject registered tools simply via string names (e.g., `tools=["custom_tool", "search_web"]`) and consider MCP integration using `MCPRegistry().get_all_toolsets()`.
10. **Notify User**: Inform the user the workflow is created. Remind them to restart the platform for auto-discovery.
