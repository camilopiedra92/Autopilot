# Autopilot Workflow Development Guide

This document explains the core architecture of workflows in Autopilot and provides a complete, step-by-step guide to creating new ones, following best practices and the Edge-First architecture.

## 🏗️ Architecture: The Four Pillars of a Workflow

Every workflow in Autopilot is built upon a modular, loosely-coupled architecture. This separation of concerns ensures that business logic is testable, execution is declarative, and the workflow identity is self-documenting.

When you look inside a workflow directory (e.g., `workflows/bank_to_ynab`), you will find four main files that define its behavior. These are not redundant; each serves a singular, specialized purpose.

### 1. `manifest.yaml` (Identity & Configuration)

This is the **"Passport"** of the workflow. It defines _what_ the workflow is, what initiates it, and what settings it requires.

- **Purpose**: Defines metadata (name, description, icon), triggers (e.g., `gmail_push`, `webhook`, `manual`), and required user settings/secrets (e.g., `ynab_access_token`).
- **Why it exists**: Allows the platform and UI dashboards to discover, display, and configure the workflow without executing any Python code.

### 2. `pipeline.yaml` (Execution Graph / DAG)

This is the **"Blueprint"** of the workflow. It defines the execution order using a Directed Acyclic Graph (DAG) strategy.

- **Purpose**: Describes the declarative topology of steps. You define nodes and their dependencies (`dependencies: [node_a]`). The execution engine automatically parallelizes nodes that have no dependent relationship.
- **Why it exists**: To separate _control flow_ from _business logic_. You can add parallelism or reorder steps entirely without touching a single line of Python code.

### 3. `steps.py` (Business Logic)

These are the **"Builder's Tools"**. This file contains the pure Python functions that each node in `pipeline.yaml` executes.

- **Purpose**: Contains stateless functions. Each function receives the accumulated pipeline state as input arguments and returns a dictionary that is merged into the state for the next step.
- **Why it exists**: Purity. Because these functions don't know about the Event Bus, the DAG engine, or the orchestration, they are trivial to test in isolation (`tests/test_steps.py`).

### 4. `workflow.py` (Orchestration & Event Lifecycle)

This is the **"Conductor"**. It connects the outside world (Event Bus) to the internal pipeline execution.

- **Purpose**: It extends `BaseWorkflow` and implements the `setup()` method to subscribe to events (e.g., `email.received`, `transaction.created`). When an event arrives, this class parses it, checks if it matches the workflow's `manifest.yaml` triggers, and kicks off the `run()` pipeline.
- **Why it exists**: To act as the bridge between asynchronous, system-wide events (PubSub) and the deterministic execution defined in your `pipeline.yaml`.

---

## 🛠️ How to Create a New Workflow (Step-by-Step)

Follow this guide to build a new workflow natively, correctly, and without workarounds.

### Step 1: Scaffold the Directory

Create a new directory for your workflow inside `workflows/`:

```bash
mkdir -p workflows/my_new_workflow/{agents,tests}
```

### Step 2: Define `manifest.yaml`

Create `workflows/my_new_workflow/manifest.yaml`. Define its identity, how it gets triggered, and what configurations it needs.

```yaml
name: my_new_workflow
display_name: "My New Workflow"
description: "Brief description of what this workflow accomplishes."
version: "1.0.0"
icon: "🚀"
color: "#FF5733"

triggers:
  - type: webhook
    path: /my-webhook
    description: "Trigger via HTTP POST"

settings:
  - key: important_api_key
    type: secret
    required: true
    description: "API Key for external service"

agents:
  cards_dir: agents/ # Discovers any *.agent.yaml files here
```

### Step 3: Implement Pure Logic in `steps.py`

Create `workflows/my_new_workflow/steps.py`. Write stateless functions for each stage of your process. Ensure type hints and proper logging (`structlog`).

```python
import structlog

logger = structlog.get_logger(__name__)

def parse_input_data(**state) -> dict:
    """Step 1: Parse the initial input from the webhook."""
    raw_payload = state.get("payload", {})
    # Return what gets added to the pipeline state
    return {"parsed_message": raw_payload.get("message", "").strip()}

def enrich_data(parsed_message: str) -> dict:
    """Step 2: Enrich the message. Dependencies are mapped by argument name."""
    logger.info("enriching_data", message=parsed_message)
    return {"enriched_result": f"Enriched: {parsed_message}"}

async def publish_result(ctx, enriched_result: str) -> dict:
    """Step 3: Publish to EventBus. The 'ctx' object is injected."""
    await ctx.publish("my_workflow.completed", {"result": enriched_result})
    return {"status": "published"}
```

_Rule: Never call other steps directly. Let the pipeline engine chain them._

### Step 4: Define the Topology in `pipeline.yaml`

Create `workflows/my_new_workflow/pipeline.yaml`. Link the functions you just wrote into a DAG.

```yaml
name: my_new_workflow
strategy: dag

nodes:
  - name: parse_input_data
    ref: workflows.my_new_workflow.steps.parse_input_data

  - name: enrich_data
    ref: workflows.my_new_workflow.steps.enrich_data
    dependencies: [parse_input_data]

  - name: publish_result
    ref: workflows.my_new_workflow.steps.publish_result
    dependencies: [enrich_data]
```

### Step 5: Wire up Events in `workflow.py`

Create `workflows/my_new_workflow/workflow.py`. Extend `BaseWorkflow` to connect your pipeline to the outside world.

```python
import structlog
from autopilot.base_workflow import BaseWorkflow
from autopilot.models import TriggerType

logger = structlog.get_logger(__name__)

class MyNewWorkflow(BaseWorkflow):
    """Entry point and orchestrator for My New Workflow."""

    async def setup(self) -> None:
        from autopilot.core.subscribers import get_subscriber_registry

        registry = get_subscriber_registry()
        # React to webhook events specifically routed to this workflow
        registry.register(
            "webhook.my_new_workflow.received",
            self._on_webhook_received,
            name="my_new_workflow_webhook_trigger"
        )

    async def _on_webhook_received(self, msg) -> None:
        """Trigger the pipeline when the webhook event fires."""
        payload = msg.payload if hasattr(msg, "payload") else msg

        logger.info("webhook_received", workflow=self.manifest.name)

        # Pass the initial state into the pipeline execution
        trigger_payload = {
            "source": "webhook",
            "payload": payload,
        }

        # Execute the pipeline defined in pipeline.yaml
        await self.run(TriggerType.WEBHOOK, trigger_payload)
```

### Step 6: Define any Agents (Optional)

If your workflow uses LLMs natively, define them in `workflows/my_new_workflow/agents/`.
Use the standard `.agent.yaml` definition cards and link them in the `pipeline.yaml` just like normal steps.

### Step 7: Test in Isolation (Crucial)

Because `steps.py` is entirely decoupled, you can and should write pure Python `pytest` functions to validate your logic without booting up the DAG engine or the Event Bus.

Create `workflows/my_new_workflow/tests/test_steps.py`:

```python
from workflows.my_new_workflow.steps import parse_input_data

def test_parse_input_data():
    state = {"payload": {"message": " hello padding "}}
    result = parse_input_data(**state)
    assert result["parsed_message"] == "hello padding"
```

## Summary Checklist for New Workflows

- [ ] `manifest.yaml` created with triggers and settings.
- [ ] `steps.py` implemented as pure, stateless functions.
- [ ] `pipeline.yaml` references `steps.py` and defines `dependencies`.
- [ ] `workflow.py` inherits `BaseWorkflow` and implements `setup()` subscriptions.
- [ ] `tests/test_steps.py` covers core logic paths.
