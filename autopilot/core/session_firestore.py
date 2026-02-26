"""
FirestoreSessionService — Durable session backend using Google Cloud Firestore.

ADK-native ``BaseSessionService`` implementation for production Cloud Run
deployments.  Provides persistent session state across container restarts
with serverless, pay-per-use scaling — no infra to manage.

Firestore document hierarchy::

    autopilot_sessions/{app_name}/
    ├── _app_state                              → { state: {...} }
    └── users/{user_id}/
        ├── _user_state                         → { state: {...} }
        └── sessions/{session_id}               → {
                state: {...},
                events: [...],
                last_update_time: float
            }

Backend selection follows 12-Factor config (``SESSION_BACKEND`` env var)::

    from autopilot.core.session import create_session_service

    service = create_session_service()          # reads SESSION_BACKEND
    service = create_session_service("firestore")  # explicit
"""

import logging
import time
from typing import Any, Optional
import uuid

from google.cloud import firestore
from typing_extensions import override

from google.adk.sessions import _session_util
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

logger = logging.getLogger("autopilot.core.session_firestore")

_ROOT_COLLECTION = "autopilot_sessions"
_APP_STATE_DOC = "_app_state"
_USER_STATE_DOC = "_user_state"


def _merge_state(
    app_state: dict[str, Any],
    user_state: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """Merge app, user, and session states into a single state dict."""
    merged = dict(session_state)
    for key, value in app_state.items():
        merged[State.APP_PREFIX + key] = value
    for key, value in user_state.items():
        merged[State.USER_PREFIX + key] = value
    return merged


def _session_from_doc(
    doc_data: dict[str, Any],
    app_name: str,
    user_id: str,
    session_id: str,
    app_state: dict[str, Any],
    user_state: dict[str, Any],
    *,
    include_events: bool = True,
    config: GetSessionConfig | None = None,
) -> Session:
    """Build an ADK Session from a Firestore document dict."""
    session_state = doc_data.get("state", {})
    merged_state = _merge_state(app_state, user_state, session_state)

    events: list[Event] = []
    if include_events:
        raw_events = doc_data.get("events", [])
        events = [Event.model_validate(e) for e in raw_events]

        if config:
            if config.after_timestamp:
                events = [e for e in events if e.timestamp >= config.after_timestamp]
            if config.num_recent_events:
                events = events[-config.num_recent_events :]

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
        events=events,
        last_update_time=doc_data.get("last_update_time", 0.0),
    )


class FirestoreSessionService(BaseSessionService):
    """Durable session backend using Google Cloud Firestore.

    Production-grade ADK ``BaseSessionService`` for Cloud Run deployments.
    Uses Firestore's native async client for fully non-blocking I/O.

    Attributes:
        client: Firestore ``AsyncClient`` instance.
        root_collection: Top-level Firestore collection name.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        root_collection: str = _ROOT_COLLECTION,
    ) -> None:
        self.client = firestore.AsyncClient(project=project)
        self.root_collection = root_collection

    @classmethod
    def from_env(cls) -> "FirestoreSessionService":
        """Create from environment — zero-config on Cloud Run.

        Reads ``GOOGLE_CLOUD_PROJECT`` (auto-set on Cloud Run).
        """
        import os

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        return cls(project=project)

    # ── Collection / Document References ─────────────────────────────

    def _app_ref(self, app_name: str) -> firestore.AsyncDocumentReference:
        """Ref to the app document (parent of app_state and users)."""
        return self.client.collection(self.root_collection).document(app_name)

    def _app_state_ref(self, app_name: str) -> firestore.AsyncDocumentReference:
        """Ref to the app-level state document."""
        return self._app_ref(app_name).collection("config").document(_APP_STATE_DOC)

    def _user_state_ref(
        self, app_name: str, user_id: str
    ) -> firestore.AsyncDocumentReference:
        """Ref to the user-level state document."""
        return (
            self._app_ref(app_name)
            .collection("users")
            .document(user_id)
            .collection("config")
            .document(_USER_STATE_DOC)
        )

    def _session_ref(
        self, app_name: str, user_id: str, session_id: str
    ) -> firestore.AsyncDocumentReference:
        """Ref to a specific session document."""
        return (
            self._app_ref(app_name)
            .collection("users")
            .document(user_id)
            .collection("sessions")
            .document(session_id)
        )

    def _sessions_collection(
        self, app_name: str, user_id: str
    ) -> firestore.AsyncCollectionReference:
        """Ref to the sessions subcollection for a user."""
        return (
            self._app_ref(app_name)
            .collection("users")
            .document(user_id)
            .collection("sessions")
        )

    # ── Helpers ──────────────────────────────────────────────────────

    async def _get_app_state(self, app_name: str) -> dict[str, Any]:
        """Read app-level state, returning empty dict if missing."""
        doc = await self._app_state_ref(app_name).get()
        return doc.to_dict().get("state", {}) if doc.exists else {}

    async def _get_user_state(self, app_name: str, user_id: str) -> dict[str, Any]:
        """Read user-level state, returning empty dict if missing."""
        doc = await self._user_state_ref(app_name, user_id).get()
        return doc.to_dict().get("state", {}) if doc.exists else {}

    # ── BaseSessionService ABC ───────────────────────────────────────

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else str(uuid.uuid4())
        )

        # Check for duplicate
        existing = await self._session_ref(app_name, user_id, session_id).get()
        if existing.exists:
            raise AlreadyExistsError(f"Session with id {session_id} already exists.")

        # Extract state deltas (app: / user: / session)
        state_deltas = _session_util.extract_state_delta(state or {})
        app_state_delta = state_deltas["app"]
        user_state_delta = state_deltas["user"]
        session_state = state_deltas["session"]

        # Upsert app state
        if app_state_delta:
            await self._app_state_ref(app_name).set(
                {"state": app_state_delta}, merge=True
            )

        # Upsert user state
        if user_state_delta:
            await self._user_state_ref(app_name, user_id).set(
                {"state": user_state_delta}, merge=True
            )

        # Create session document
        now = time.time()
        session_doc = {
            "state": session_state or {},
            "events": [],
            "last_update_time": now,
        }
        await self._session_ref(app_name, user_id, session_id).set(session_doc)

        logger.debug(
            "session_created",
            extra={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        # Return session with merged state
        app_state = await self._get_app_state(app_name)
        user_state = await self._get_user_state(app_name, user_id)
        merged = _merge_state(app_state, user_state, session_state or {})

        return Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=merged,
            events=[],
            last_update_time=now,
        )

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        doc = await self._session_ref(app_name, user_id, session_id).get()
        if not doc.exists:
            return None

        app_state = await self._get_app_state(app_name)
        user_state = await self._get_user_state(app_name, user_id)

        return _session_from_doc(
            doc.to_dict(),
            app_name,
            user_id,
            session_id,
            app_state,
            user_state,
            config=config,
        )

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        sessions: list[Session] = []
        app_state = await self._get_app_state(app_name)

        if user_id is not None:
            user_state = await self._get_user_state(app_name, user_id)
            docs = self._sessions_collection(app_name, user_id).stream()
            async for doc in docs:
                sessions.append(
                    _session_from_doc(
                        doc.to_dict(),
                        app_name,
                        user_id,
                        doc.id,
                        app_state,
                        user_state,
                        include_events=False,
                    )
                )
        else:
            # List all users under the app
            users_col = self._app_ref(app_name).collection("users")
            async for user_doc in users_col.list_documents():
                uid = user_doc.id
                user_state = await self._get_user_state(app_name, uid)
                docs = self._sessions_collection(app_name, uid).stream()
                async for doc in docs:
                    sessions.append(
                        _session_from_doc(
                            doc.to_dict(),
                            app_name,
                            uid,
                            doc.id,
                            app_state,
                            user_state,
                            include_events=False,
                        )
                    )

        return ListSessionsResponse(sessions=sessions)

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        await self._session_ref(app_name, user_id, session_id).delete()
        logger.debug(
            "session_deleted",
            extra={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
            },
        )

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        app_name = session.app_name
        user_id = session.user_id
        session_id = session.id

        # Trim temp state before persisting
        event = self._trim_temp_delta_state(event)

        @firestore.async_transactional
        async def _transactional_append(transaction):
            ref = self._session_ref(app_name, user_id, session_id)
            snapshot = await ref.get(transaction=transaction)
            if not snapshot.exists:
                logger.warning(
                    "append_event_session_not_found",
                    extra={"session_id": session_id},
                )
                return

            doc_data = snapshot.to_dict()

            # Update state deltas
            if event.actions and event.actions.state_delta:
                state_deltas = _session_util.extract_state_delta(
                    event.actions.state_delta
                )
                app_state_delta = state_deltas["app"]
                user_state_delta = state_deltas["user"]
                session_state_delta = state_deltas["session"]

                if app_state_delta:
                    app_state_ref = self._app_state_ref(app_name)
                    app_doc = await app_state_ref.get(transaction=transaction)
                    current = (
                        app_doc.to_dict().get("state", {}) if app_doc.exists else {}
                    )
                    current.update(app_state_delta)
                    transaction.set(app_state_ref, {"state": current}, merge=True)

                if user_state_delta:
                    user_state_ref = self._user_state_ref(app_name, user_id)
                    user_doc = await user_state_ref.get(transaction=transaction)
                    current = (
                        user_doc.to_dict().get("state", {}) if user_doc.exists else {}
                    )
                    current.update(user_state_delta)
                    transaction.set(user_state_ref, {"state": current}, merge=True)

                if session_state_delta:
                    session_state = doc_data.get("state", {})
                    session_state.update(session_state_delta)
                    doc_data["state"] = session_state

            # Append serialized event
            events_list = doc_data.get("events", [])
            events_list.append(event.model_dump(mode="json"))
            doc_data["events"] = events_list
            doc_data["last_update_time"] = event.timestamp

            transaction.set(ref, doc_data)

        transaction = self.client.transaction()
        await _transactional_append(transaction)

        # Also update the in-memory session object
        await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp

        return event

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the Firestore client."""
        self.client.close()

    async def __aenter__(self) -> "FirestoreSessionService":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"FirestoreSessionService("
            f"project={self.client.project!r}, "
            f"collection={self.root_collection!r})"
        )
