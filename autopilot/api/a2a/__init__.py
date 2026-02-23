"""
A2A Protocol Server — Agent-to-Agent protocol implementation.

Exposes Autopilot as a discoverable, invocable agent in multi-agent
ecosystems via the Google A2A Protocol specification.

Endpoints:
  - GET /.well-known/agent-card.json → Agent discovery (unauthenticated)
  - POST /a2a → JSON-RPC (message/send, message/stream, tasks/get)
"""

from autopilot.api.a2a.agent_card import build_agent_card
from autopilot.api.a2a.request_handler import AutopilotA2ARequestHandler
from autopilot.api.a2a.server import mount_a2a_routes

__all__ = [
    "build_agent_card",
    "AutopilotA2ARequestHandler",
    "mount_a2a_routes",
]
