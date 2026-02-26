#!/usr/bin/env python3
"""
Autopilot CLI — Management tools for the autopilot platform.

Usage:
    python -m autopilot.cli create-workflow <name> [--display-name "My Workflow"]
    python -m autopilot.cli create-workflow <name> --custom  # Include workflow.py

Scaffolded workflows include:
  - manifest.yaml (A2A metadata & triggers)
  - pipeline.yaml + steps.py (Pure YAML mode) OR workflow.py (Custom mode)
  - @tool decorator for registering reusable tool functions
"""

import argparse
import sys
import re
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
WORKFLOWS_DIR = BASE_DIR / "workflows"


def to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def create_workflow(
    name: str,
    display_name: str = None,
    description: str = None,
    icon: str = None,
    trigger: str = None,
    custom: bool = False,
):
    """Scaffold a new workflow."""

    # 1. Validate name (snake_case only)
    if not re.match(r"^[a-z0-9_]+$", name):
        print(
            f"Error: Workflow name '{name}' must be snake_case (lowercase, numbers, underscores)."
        )
        sys.exit(1)

    target_dir = WORKFLOWS_DIR / name

    if target_dir.exists():
        print(f"Error: Workflow directory '{target_dir}' already exists.")
        sys.exit(1)

    print(f"Creating workflow '{name}'...")

    # 2. Default values
    class_name = to_pascal_case(name) + "Workflow"
    display_name = display_name or name.replace("_", " ").title()
    description = description or f"Description for {display_name}"
    icon = icon or "⚡"
    trigger = trigger or "manual"

    # 3. Create directory structure
    target_dir.mkdir(parents=True)

    # 4. Create manifest.yaml (always)
    trigger_yaml = _trigger_yaml(trigger)
    manifest_content = f"""name: {name}
display_name: "{display_name}"
description: "{description}"
version: "1.0.0"
icon: "{icon}"
color: "#6366f1"

triggers:
{trigger_yaml}

settings: []

agents:
  cards_dir: agents/

tags:
  - new
"""
    (target_dir / "manifest.yaml").write_text(manifest_content)

    # 5. Create workflow.py (only with --custom flag)
    if custom:
        workflow_content = f'''"""
{display_name} Workflow — Custom execution logic.

manifest.yaml is auto-loaded by BaseWorkflow.
"""


from typing import Any

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowResult, RunStatus


class {class_name}(BaseWorkflow):
    """{display_name} workflow."""

    async def execute(self, trigger_data: dict[str, Any]) -> WorkflowResult:
        """Implement your workflow logic here."""
        return WorkflowResult(
            workflow_id=self.manifest.name,
            status=RunStatus.SUCCESS,
            data={{"message": "{display_name} executed successfully"}},
        )
'''
        (target_dir / "workflow.py").write_text(workflow_content)
    else:
        # Pure YAML mode: create a minimal pipeline.yaml instead
        pipeline_content = f"""# Pipeline definition for {display_name}
# Auto-executed by BaseWorkflow — no Python needed.
strategy: sequential

steps:
  - name: process
    type: function
    ref: workflows.{name}.steps.process_input
    description: "Process the input data."
"""
        (target_dir / "pipeline.yaml").write_text(pipeline_content)

        # Create steps.py with placeholder function using @tool
        steps_content = f'''"""
{display_name} — Pipeline step functions.

Pure functions auto-wrapped as FunctionalAgents by the DSL loader.
Use @tool decorator to register reusable tools for LLM agents.
"""

from pydantic import BaseModel
from autopilot.core.tools import tool

class InputData(BaseModel):
    user_id: str
    message: str

def process_input(input_data: InputData) -> dict:
    """
    Process input data. Replace with your logic.
    The platform strictly auto-hydrates Pydantic models in pure functions based on type hints.
    A ValidationError will be raised upstream if required fields are missing.
    """
    return {{"processed": True, "user": input_data.user_id}}


# Example: Register a reusable tool for LLM agents
# NOTE: ONLY use @tool for workflow-specific custom logic.
# NEVER wrap Platform Connectors (like YNAB, Gmail) with @tool;
# the platform lazily auto-resolves those when agents reference them.
#
# @tool(tags=["{name}"])
# async def my_tool(param: str) -> dict:
#     """Description of what this tool does."""
#     return {{"result": param}}
#
# For tools that need session state or auth:
# from google.adk.tools import ToolContext
# @tool(tags=["{name}"])
# async def auth_tool(param: str, tool_context: ToolContext) -> dict:
#     api_key = tool_context.state.get("API_KEY")
#     ...
#
# Remember to pass tools as strings (e.g., tools=["my_tool", "ynab.get_accounts"])
# when creating agents. The platform auto-resolves them via the registry.
# Also consider attaching platform/domain guardrails directly to the agent callbacks
# and using ctx.publish("my_topic", data) for decoupled AgentBus communication.
'''
        (target_dir / "steps.py").write_text(steps_content)

    # 6. Print summary
    print(f"✅ Workflow '{name}' created at {target_dir}")
    print()

    if custom:
        print("📁 Files created:")
        print("  ├── manifest.yaml    (A2A metadata & triggers)")
        print("  └── workflow.py      (custom execute() logic)")
        print()
        print("Next steps:")
        print(f"  1. Implement execute() in: workflows/{name}/workflow.py")
        print(f"  2. Edit metadata in: workflows/{name}/manifest.yaml")
    else:
        print("📁 Files created:")
        print("  ├── manifest.yaml    (A2A metadata & triggers)")
        print("  ├── pipeline.yaml    (DSL pipeline definition)")
        print("  └── steps.py         (pipeline step functions + @tool examples)")
        print()
        print("Next steps:")
        print(f"  1. Add step functions in: workflows/{name}/steps.py")
        print(f"  2. Wire them in: workflows/{name}/pipeline.yaml")
        print("  3. Use @tool decorator for reusable tools (see steps.py examples)")

    print(
        f"  {4 if not custom else 3}. Restart the platform — it auto-discovers your workflow!"
    )


def _trigger_yaml(trigger: str) -> str:
    """Generate YAML trigger config."""
    triggers = {
        "manual": '  - type: manual\n    description: "Manual trigger"',
        "webhook": '  - type: webhook\n    path: /process\n    description: "HTTP webhook trigger"',
        "gmail_push": (
            '  - type: gmail_push\n    filter: "sender@example.com"'
            '\n    description: "Gmail push notification"'
        ),
        "scheduled": '  - type: scheduled\n    cron: "0 */6 * * *"\n    description: "Runs every 6 hours"',
    }
    return triggers.get(trigger, triggers["manual"])


def main():
    parser = argparse.ArgumentParser(description="Autopilot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-workflow command
    create_parser = subparsers.add_parser(
        "create-workflow", help="Create a new workflow"
    )
    create_parser.add_argument("name", help="Workflow name (snake_case)")
    create_parser.add_argument("--display-name", help="Display name for the workflow")
    create_parser.add_argument("--description", help="Description of the workflow")
    create_parser.add_argument("--icon", help="Emoji icon for the workflow")
    create_parser.add_argument(
        "--trigger",
        choices=["manual", "webhook", "gmail_push", "scheduled"],
        default="manual",
        help="Primary trigger type",
    )
    create_parser.add_argument(
        "--custom",
        action="store_true",
        help="Generate workflow.py for custom execute() logic (otherwise pure YAML)",
    )

    args = parser.parse_args()

    if args.command == "create-workflow":
        create_workflow(
            args.name,
            args.display_name,
            args.description,
            args.icon,
            args.trigger,
            args.custom,
        )


if __name__ == "__main__":
    main()
