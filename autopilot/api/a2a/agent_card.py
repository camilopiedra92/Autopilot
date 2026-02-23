"""
Agent Card Builder — Constructs the A2A AgentCard from WorkflowRegistry.

The AgentCard is the platform's public identity in multi-agent ecosystems.
Each registered workflow becomes an A2A Skill, enabling external agents
to discover and invoke specific Autopilot capabilities.
"""

from __future__ import annotations

import os

from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentProvider,
    AgentSkill,
)

from autopilot.version import VERSION, APP_NAME
from autopilot.registry import WorkflowRegistry


def build_agent_card(registry: WorkflowRegistry) -> AgentCard:
    """Build the platform-level AgentCard from registered workflows.

    Each workflow in the registry becomes an AgentSkill with:
      - id = workflow name (snake_case)
      - name = workflow display_name
      - description = workflow description
      - tags = workflow tags

    Args:
        registry: The WorkflowRegistry with discovered workflows.

    Returns:
        A2A AgentCard ready to serve at /.well-known/agent-card.json.
    """
    base_url = os.getenv("A2A_BASE_URL", "http://localhost:8080")

    skills = [
        AgentSkill(
            id=info.name,
            name=info.display_name,
            description=info.description or f"Workflow: {info.display_name}",
            tags=info.tags,
        )
        for info in registry.list_all()
        if info.enabled
    ]

    return AgentCard(
        name=APP_NAME,
        description=(
            "Multi-workflow AI automation platform powered by Google ADK. "
            "Invoke workflows as A2A tasks via message/send."
        ),
        url=base_url,
        version=VERSION,
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
        ),
        provider=AgentProvider(
            organization="AutoPilot",
            url=base_url,
        ),
        skills=skills,
    )
