"""
MemoryService — ADK-native re-exports + 12-Factor factory.

Aligned with Google ADK's ``BaseMemoryService`` contract.  All custom
abstractions (Observation, TF-IDF, ChromaDB) have been eliminated.

Three backends (selected via ``MEMORY_BACKEND`` env var):
  - ``memory``    → ADK InMemoryMemoryService (keyword matching, zero deps)
  - ``firestore`` → FirestoreVectorMemoryService (Firestore + Gemini embeddings)
  - ``vertexai``  → VertexAiMemoryBankService (Vertex AI Memory Bank)

Usage:
    from autopilot.core.memory import create_memory_service

    service = create_memory_service()  # Reads MEMORY_BACKEND env var
"""

from __future__ import annotations

import os

import structlog

# ── ADK Re-exports (public API only) ──────────────────────────────────
from google.adk.memory import BaseMemoryService, InMemoryMemoryService

logger = structlog.get_logger(__name__)

# ── Factory (12-Factor, same pattern as session/bus) ──────────────────

_memory_service: BaseMemoryService | None = None


def create_memory_service() -> BaseMemoryService:
    """Create a MemoryService based on MEMORY_BACKEND env var.

    | MEMORY_BACKEND | Backend                       | Cost                  |
    |----------------|-------------------------------|-----------------------|
    | ``memory``     | ADK InMemoryMemoryService     | Free                  |
    | ``firestore``  | FirestoreVectorMemoryService  | ~$0.15/month          |
    | ``vertexai``   | VertexAiMemoryBankService     | Free (Express)        |

    Returns:
        A configured BaseMemoryService instance.

    Raises:
        ValueError: If MEMORY_BACKEND is unknown.
    """
    backend = os.getenv("MEMORY_BACKEND", "memory").lower().strip()

    if backend == "memory":
        logger.info("memory_backend_selected", backend="memory")
        return InMemoryMemoryService()

    if backend == "firestore":
        from autopilot.core.memory_firestore import FirestoreVectorMemoryService

        logger.info("memory_backend_selected", backend="firestore")
        return FirestoreVectorMemoryService.from_env()

    if backend == "vertexai":
        from google.adk.memory import VertexAiMemoryBankService

        agent_engine_id = os.environ.get("MEMORY_AGENT_ENGINE_ID", "")
        if not agent_engine_id:
            raise ValueError(
                "MEMORY_AGENT_ENGINE_ID is required when MEMORY_BACKEND=vertexai"
            )
        logger.info(
            "memory_backend_selected",
            backend="vertexai",
            agent_engine_id=agent_engine_id,
        )
        return VertexAiMemoryBankService(agent_engine_id=agent_engine_id)

    raise ValueError(
        f"Unknown MEMORY_BACKEND: {backend!r}. "
        "Supported: 'memory', 'firestore', 'vertexai'."
    )


def get_memory_service() -> BaseMemoryService:
    """Get or create the singleton MemoryService."""
    global _memory_service
    if _memory_service is None:
        _memory_service = create_memory_service()
    return _memory_service


def reset_memory_service() -> None:
    """Reset the singleton (for tests)."""
    global _memory_service
    _memory_service = None
