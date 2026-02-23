"""
SessionService — ADK-native session management with factory selection.

The platform uses Google ADK's native Session lifecycle directly:
  - ``BaseSessionService``: ABC with create/get/list/delete/append_event
  - ``InMemorySessionService``: Dict-backed implementation (dev/test)
  - ``FirestoreSessionService``: Durable Firestore backend (production)
  - ``Session``: Pydantic model (id, app_name, user_id, state, events)

Backend selection via ``SESSION_BACKEND`` env var (12-Factor)::

    from autopilot.core.session import create_session_service

    service = create_session_service()             # reads env var
    service = create_session_service("firestore")  # explicit override
"""

from __future__ import annotations

import logging
import os

from google.adk.sessions import (
    BaseSessionService,
    InMemorySessionService,
    Session,
)

logger = logging.getLogger("autopilot.core.session")


def create_session_service(
    backend: str | None = None,
) -> BaseSessionService:
    """Factory for creating the appropriate session backend.

    Backend selection follows 12-Factor App (Factor III: Config):
      - ``"memory"`` (default): In-memory sessions for dev/test
      - ``"firestore"``: Cloud Firestore for production (durable, serverless)

    Args:
        backend: Override backend choice. Defaults to ``SESSION_BACKEND``
                 env var, falling back to ``"memory"``.

    Returns:
        A ``BaseSessionService`` implementation.
    """
    backend = backend or os.getenv("SESSION_BACKEND", "memory")
    logger.info("session_backend_selected", extra={"backend": backend})

    if backend == "firestore":
        from autopilot.core.session_firestore import FirestoreSessionService

        return FirestoreSessionService.from_env()

    return InMemorySessionService()


__all__ = [
    "BaseSessionService",
    "InMemorySessionService",
    "Session",
    "create_session_service",
]
