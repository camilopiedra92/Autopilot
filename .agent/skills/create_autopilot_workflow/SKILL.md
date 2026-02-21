---
name: create_autopilot_workflow
description: Expert ability to create, configure, and implement new Autopilot workflows following the Three-File Architecture.
---

# Create Autopilot Workflow

> **Read `docs/ARCHITECTURE.md` § Separation of Concerns before using this skill.**

This skill creates workflows for the Autopilot platform following the **Three-File Architecture**: `manifest.yaml` (WHO), `pipeline.yaml` (WHAT), `workflow.py` (HOW).

---

## Step 1 — Define the Concept

Before writing any code:

1. **Name**: `snake_case` only (e.g., `invoice_processor`, `customer_onboarding`)
2. **Class Name**: `PascalCase` + `Workflow` suffix (e.g., `InvoiceProcessorWorkflow`)
3. **Trigger type**: How is this activated? (`gmail_push` | `webhook` | `scheduled` | `manual`)
4. **Level**: Choose before scaffolding:

| Level         | Files                                          | When to Use                                       |
| ------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Pure YAML** | `manifest.yaml` + `pipeline.yaml` + `steps.py` | Simple: chain existing functions, no custom logic |
| **Custom**    | `manifest.yaml` + `workflow.py`                | Need imperative `execute()` logic                 |
| **Full**      | Above + `agents/` + `*.agent.yaml`             | Multi-agent orchestration with LLMs               |

---

## Step 2 — Scaffold via CLI

```bash
# Pure YAML workflow (0 Python in the workflow itself):
python3 -m autopilot.cli create-workflow <name> \
  --display-name "<Display Name>" --icon "<Icon>" --trigger <trigger>

# Custom Python workflow (generates workflow.py):
python3 -m autopilot.cli create-workflow <name> \
  --display-name "<Display Name>" --icon "<Icon>" --trigger <trigger> --custom
```

Trigger options: `manual` (default) | `webhook` | `gmail_push` | `scheduled`

---

## Step 3 — Configure `manifest.yaml` (WHO am I?)

> ⚠️ **A2A-compatible metadata ONLY.** Never put execution logic here.

This is the workflow's **public identity** for the A2A discovery protocol:

```yaml
name: invoice_processor # Unique snake_case ID
display_name: "Invoice Processor" # Dashboard label
description: "Parses invoices from email" # A2A discovery description
version: "1.0.0"
icon: "📄"
color: "#4285f4"

triggers:
  - type: gmail_push
    filter: "invoices@company.com"
    description: "Triggered by invoice emails"

settings:
  - key: api_key
    type: secret
    description: "API key for invoice service"
    required: true

agents:
  cards_dir: agents/

tags:
  - finance
  - automation
```

**Fields reference:**

| Field              | Purpose                     | Required |
| ------------------ | --------------------------- | -------- |
| `name`             | Unique identifier           | ✅       |
| `display_name`     | Human-readable UI label     | ✅       |
| `description`      | What this workflow does     | ✅       |
| `version`          | Semver                      | ✅       |
| `triggers`         | Activation method           | ✅       |
| `settings`         | Config / secrets needed     | Optional |
| `agents.cards_dir` | Path to `.agent.yaml` cards | Optional |
| `tags`             | Categorization              | Optional |

---

## Step 4 — Define Execution

### Option A: `pipeline.yaml` (WHAT do I do? — Declarative)

If the workflow is a sequence of steps, define them declaratively:

```yaml
name: invoice_pipeline
strategy: sequential # sequential | dag

steps:
  - name: parse_email
    type: function
    ref: workflows.invoice_processor.steps.parse_email
    description: "Extract invoice data from email body"

  - name: validate
    type: function
    ref: workflows.invoice_processor.steps.validate_invoice

  - name: categorize
    type: agent # LLM agent
    ref: workflows.invoice_processor.agents.categorizer

  - name: save_result
    type: function
    ref: workflows.invoice_processor.steps.save_to_db
```

Step types: `function` | `agent` | `loop` | `parallel`

`BaseWorkflow.execute()` auto-runs this if it exists. **No Python needed.**

> 💡 **Pydantic Auto-Hydration**: Pure functions in `steps.py` (referenced by `type: function`) should use strict Pydantic `BaseModel` type hints. The declarative pipeline engine will strictly hydrate dictionaries from the state into those models for you, and fail fast with `ValidationError` if data is missing or invalid. **Never use `Optional` fallbacks or `**state` extraction\*\* when a model is required.

### Option B: `workflow.py` (HOW do I do it? — Imperative)

For complex logic that DSL can't express:

```python
from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowResult, RunStatus


class InvoiceProcessorWorkflow(BaseWorkflow):
    """manifest.yaml and agent cards are auto-loaded."""

    async def execute(self, trigger_data):
        email_body = trigger_data.get("body", "")
        cards = self.get_agent_cards()  # Auto-discovered

        # Use AgentBus for event publishing if necessary
        # await self.ctx.publish("invoice.received", {"email": email_body})

        result = await invoke_pipeline(email_body, cards)

        return WorkflowResult(
            workflow_id=self.manifest.name,   # Auto-loaded
            status=RunStatus.SUCCESS,
            data=result,
        )
```

**You do NOT need to implement:**

- ~~`manifest` property~~ → auto-loaded from `manifest.yaml`
- ~~`get_agent_cards()`~~ → auto-discovered from `agents/`
- ~~`__init__.py` with `workflow = ...`~~ → registry auto-discovers

---

## Step 5 — Add Agents (if needed)

Only for `Full` level workflows:

1. Create `workflows/<name>/agents/` directory
2. Add `.agent.yaml` cards (auto-discovered by `BaseWorkflow.get_agent_cards()`):

```yaml
# agents/categorizer.agent.yaml
name: invoice_categorizer
description: "Categorizes invoices by type"
model: gemini-2.0-flash-exp
input_schema:
  type: object
  properties:
    invoice_text: { type: string }
```

3. Create agents using the factory (**never instantiate LlmAgent directly**) and ensure Guardrails are attached:

```python
from autopilot.agents.base import create_platform_agent
from autopilot.agents.callbacks import create_chained_before_callback, create_chained_after_callback
from autopilot.agents.guardrails import input_length_guard, amount_sanity_guard

agent = create_platform_agent(
    name="categorizer",
    instruction="Categorize this invoice...",
    model="gemini-2.0-flash-exp",
    # Pass tools simply as string references. The platform auto-resolves them!
    tools=["my_custom_tool", "search_web", "ynab.get_accounts"],
    # ALWAYS chain guardrails
    before_model_callback=create_chained_before_callback(input_length_guard(min_chars=10)),
    after_model_callback=create_chained_after_callback(amount_sanity_guard(max_amount=100_000)),
)
```

---

## Step 6 — Register Tools (if needed)

For workflows that need custom, reusable tool functions (e.g., custom parsers, data formatting, custom APIs):

> ⚠️ **DO NOT wrap Platform Connectors.** If interacting with YNAB, Gmail, PubSub, etc., the platform **lazily auto-resolves** connector tools when agents reference them by `connector.method` name (e.g., `tools=["ynab.get_categories_string"]`). Use `@tool` ONLY for custom workflow-specific logic.

### Register workflow-specific logic with `@tool`

```python
# workflows/<name>/steps.py
from autopilot.core.tools import tool

@tool(tags=["finance", "invoices"])
async def lookup_vendor(vendor_name: str) -> dict:
    """Look up vendor details from the database."""
    return {"vendor": vendor_name, "status": "active"}
```

### ToolContext-aware tools (access session state, auth)

```python
from google.adk.tools import ToolContext
from autopilot.core.tools import tool

@tool(tags=["finance"])
async def create_payment(amount: float, tool_context: ToolContext) -> dict:
    """Create payment with session-aware auth."""
    api_key = tool_context.state.get("payment_api_key")
    # tool_context is auto-injected by ADK — excluded from user-facing params
    ...
```

### Configure tool auth

```python
from autopilot.core.tools import ToolAuthConfig, get_auth_manager

get_auth_manager().register(ToolAuthConfig(
    tool_name="create_payment",
    auth_type="api_key",
    credential_key="PAYMENT_API_KEY",
))
```

### Add lifecycle callbacks (rate limiting, audit)

```python
from autopilot.core.tools import get_callback_manager, create_rate_limit_callback

mgr = get_callback_manager()
mgr.register_before(
    create_rate_limit_callback(max_calls=10, window_seconds=60),
    tools=["create_payment"],
)
```

| Capability             | Import                         | Purpose                          |
| ---------------------- | ------------------------------ | -------------------------------- |
| `@tool`                | `autopilot.core.tools`         | Register + auto-extract metadata |
| `ToolContext` param    | `google.adk.tools.ToolContext` | Session state, auth, memory      |
| `ToolAuthConfig`       | `autopilot.core.tools`         | Declarative credential config    |
| `get_callback_manager` | `autopilot.core.tools`         | Before/after lifecycle hooks     |
| `@long_running_tool`   | `autopilot.core.tools`         | Async/batch operations           |

---

## Step 7 — Registration

**Fully automatic.** No action needed. The `WorkflowRegistry` discovers workflows via:

| Priority | What it looks for                          | Action                            |
| -------- | ------------------------------------------ | --------------------------------- |
| 1        | `__init__.py` with `workflow` export       | Uses it (backward-compatible)     |
| 2        | `workflow.py` with `BaseWorkflow` subclass | Auto-instantiates                 |
| 3        | `manifest.yaml` only                       | Creates `BaseWorkflow()` directly |

Just restart the platform — your workflow is live.

---

## Anti-Patterns (NEVER DO)

| ❌ Don't                                           | ✅ Do Instead                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| Put pipeline steps in `manifest.yaml`              | Use `pipeline.yaml` for execution logic                                  |
| Override `manifest` property in subclass           | Let `BaseWorkflow` auto-load it                                          |
| Write `__init__.py` with `workflow = MyWorkflow()` | Let registry auto-discover                                               |
| Create `dsl_example.yaml` files                    | Use the `pipeline.yaml` convention                                       |
| Instantiate `LlmAgent(...)` directly               | Use `create_platform_agent()` factory                                    |
| Import `get_tool_registry()` inside agents         | Pass tools as string references (`tools=["name"]`)                       |
| Implement callback logic directly in agents        | Use `create_chained_*_callback` and guard factories                      |
| Hardcode API keys in tool functions                | Use `ToolAuthConfig` + `get_auth_manager()`                              |
| Skip tool registration for reusable functions      | Use `@tool` decorator with descriptive tags                              |
| Wrap Connector methods with `@tool`                | Pass as string refs — lazy auto-resolved (`tools=["ynab.get_accounts"]`) |
