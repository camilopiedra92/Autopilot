"""
SessionService — ADK-native session management (re-exports).

The platform uses Google ADK's native Session lifecycle directly:
  - ``BaseSessionService``: ABC with create/get/list/delete/append_event
  - ``InMemorySessionService``: Dict-backed implementation (dev/test)
  - ``Session``: Pydantic model (id, app_name, user_id, state, events)

No custom wrappers or abstractions — agents interact with
``Session.state`` dict directly, exactly as ADK intended.
"""

from google.adk.sessions import (
    BaseSessionService,
    InMemorySessionService,
    Session,
)

__all__ = [
    "BaseSessionService",
    "InMemorySessionService",
    "Session",
]
