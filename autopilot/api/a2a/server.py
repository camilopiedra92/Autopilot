"""
A2A Server Integration — Mounts A2A protocol routes on the FastAPI app.

Uses the official a2a-sdk's A2AFastAPIApplication to wire:
  - GET /.well-known/agent-card.json  → agent discovery (unauthenticated)
  - POST /                            → JSON-RPC (message/send, message/stream, tasks/get)
"""

import structlog
from fastapi import FastAPI

from a2a.server.apps import A2AFastAPIApplication

from autopilot.api.a2a.agent_card import build_agent_card
from autopilot.api.a2a.request_handler import AutopilotA2ARequestHandler
from autopilot.registry import WorkflowRegistry

logger = structlog.get_logger(__name__)


def mount_a2a_routes(app: FastAPI, registry: WorkflowRegistry) -> None:
    """Mount A2A protocol endpoints on the FastAPI app.

    Wires the A2A SDK's FastAPI application into the existing app,
    serving the agent card at /.well-known/agent-card.json and
    the JSON-RPC endpoint at /a2a.

    Args:
        app: The FastAPI application.
        registry: The WorkflowRegistry with discovered workflows.
    """
    agent_card = build_agent_card(registry)
    handler = AutopilotA2ARequestHandler(registry)
    a2a_app = A2AFastAPIApplication(agent_card, handler)

    # Mount on the existing app — agent card at well-known, RPC at /a2a
    a2a_app.add_routes_to_app(
        app,
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
    )

    logger.info(
        "a2a_protocol_mounted",
        agent_card_url="/.well-known/agent-card.json",
        rpc_url="/a2a",
        skills_count=len(agent_card.skills),
        skill_ids=[s.id for s in agent_card.skills],
    )
