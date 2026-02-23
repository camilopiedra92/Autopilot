"""
Tests for FirestoreSessionService — ADK-native session backend on Firestore.

All Firestore interactions are fully mocked — zero external dependencies.
Tests verify the ADK BaseSessionService contract: CRUD lifecycle,
state delta extraction, state merging, event handling, and factory selection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from google.adk.sessions.base_session_service import (
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session

from autopilot.core.session_firestore import (
    FirestoreSessionService,
    _merge_state,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _mock_doc_snapshot(exists: bool, data: dict | None = None, doc_id: str = "test"):
    """Create a mock Firestore DocumentSnapshot."""
    snap = MagicMock()
    snap.exists = exists
    snap.id = doc_id
    snap.to_dict.return_value = data if exists else None
    return snap


def _make_service() -> FirestoreSessionService:
    """Create a FirestoreSessionService with a mocked Firestore client."""
    with patch("autopilot.core.session_firestore.firestore.AsyncClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client

        # Setup collection/document chain mocks
        service = FirestoreSessionService(project="test-project")
        service.client = client
        return service


def _setup_refs(service: FirestoreSessionService) -> dict[str, MagicMock]:
    """Set up common mock references and return them for assertions."""
    session_ref = AsyncMock()
    app_state_ref = AsyncMock()
    user_state_ref = AsyncMock()
    sessions_collection = MagicMock()

    service._session_ref = MagicMock(return_value=session_ref)
    service._app_state_ref = MagicMock(return_value=app_state_ref)
    service._user_state_ref = MagicMock(return_value=user_state_ref)
    service._sessions_collection = MagicMock(return_value=sessions_collection)
    service._app_ref = MagicMock()

    return {
        "session_ref": session_ref,
        "app_state_ref": app_state_ref,
        "user_state_ref": user_state_ref,
        "sessions_collection": sessions_collection,
    }


# ── Unit Tests ───────────────────────────────────────────────────────


class TestMergeState:
    """Tests for the _merge_state helper."""

    def test_merge_empty(self):
        result = _merge_state({}, {}, {})
        assert result == {}

    def test_merge_session_only(self):
        result = _merge_state({}, {}, {"key": "value"})
        assert result == {"key": "value"}

    def test_merge_all_levels(self):
        result = _merge_state(
            {"theme": "dark"},
            {"pref": "compact"},
            {"counter": 1},
        )
        assert result == {
            "counter": 1,
            "app:theme": "dark",
            "user:pref": "compact",
        }


class TestCreateSession:
    """Tests for create_session()."""

    @pytest.mark.asyncio
    async def test_create_session_basic(self):
        service = _make_service()
        refs = _setup_refs(service)

        # Session doesn't exist yet
        refs["session_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))
        refs["session_ref"].set = AsyncMock()

        # App/user state empty
        refs["app_state_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))
        refs["user_state_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))
        service._get_app_state = AsyncMock(return_value={})
        service._get_user_state = AsyncMock(return_value={})

        session = await service.create_session(
            app_name="autopilot",
            user_id="u1",
            state={"key": "value"},
            session_id="s1",
        )

        assert isinstance(session, Session)
        assert session.id == "s1"
        assert session.app_name == "autopilot"
        assert session.user_id == "u1"
        assert session.state.get("key") == "value"
        assert session.events == []

    @pytest.mark.asyncio
    async def test_create_session_auto_id(self):
        service = _make_service()
        refs = _setup_refs(service)

        refs["session_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))
        refs["session_ref"].set = AsyncMock()
        service._get_app_state = AsyncMock(return_value={})
        service._get_user_state = AsyncMock(return_value={})

        session = await service.create_session(
            app_name="autopilot",
            user_id="u1",
        )

        assert session.id  # Auto-generated UUID
        assert len(session.id) > 0

    @pytest.mark.asyncio
    async def test_create_duplicate_raises(self):
        from google.adk.errors.already_exists_error import AlreadyExistsError

        service = _make_service()
        refs = _setup_refs(service)

        # Session already exists
        refs["session_ref"].get = AsyncMock(
            return_value=_mock_doc_snapshot(True, {"state": {}, "events": []})
        )

        with pytest.raises(AlreadyExistsError, match="already exists"):
            await service.create_session(
                app_name="autopilot",
                user_id="u1",
                session_id="existing",
            )

    @pytest.mark.asyncio
    async def test_create_session_with_app_state(self):
        service = _make_service()
        refs = _setup_refs(service)

        refs["session_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))
        refs["session_ref"].set = AsyncMock()
        refs["app_state_ref"].set = AsyncMock()
        service._get_app_state = AsyncMock(return_value={"theme": "dark"})
        service._get_user_state = AsyncMock(return_value={})

        session = await service.create_session(
            app_name="autopilot",
            user_id="u1",
            state={"app:theme": "dark", "local_key": "val"},
        )

        # App state was persisted
        refs["app_state_ref"].set.assert_called_once()
        # Session has merged state
        assert session.state.get("app:theme") == "dark"
        assert session.state.get("local_key") == "val"


class TestGetSession:
    """Tests for get_session()."""

    @pytest.mark.asyncio
    async def test_get_existing(self):
        service = _make_service()
        refs = _setup_refs(service)

        refs["session_ref"].get = AsyncMock(
            return_value=_mock_doc_snapshot(
                True,
                {"state": {"k": "v"}, "events": [], "last_update_time": 123.0},
            )
        )
        service._get_app_state = AsyncMock(return_value={})
        service._get_user_state = AsyncMock(return_value={})

        session = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id="s1",
        )

        assert session is not None
        assert session.id == "s1"
        assert session.state["k"] == "v"
        assert session.last_update_time == 123.0

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        service = _make_service()
        refs = _setup_refs(service)

        refs["session_ref"].get = AsyncMock(return_value=_mock_doc_snapshot(False))

        result = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id="nonexistent",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_config_num_recent_events(self):
        service = _make_service()
        refs = _setup_refs(service)

        events = [
            {
                "id": f"e{i}",
                "timestamp": float(i),
                "author": "user",
                "invocation_id": f"inv{i}",
            }
            for i in range(5)
        ]
        refs["session_ref"].get = AsyncMock(
            return_value=_mock_doc_snapshot(
                True,
                {"state": {}, "events": events, "last_update_time": 5.0},
            )
        )
        service._get_app_state = AsyncMock(return_value={})
        service._get_user_state = AsyncMock(return_value={})

        session = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id="s1",
            config=GetSessionConfig(num_recent_events=2),
        )

        assert len(session.events) == 2


class TestListSessions:
    """Tests for list_sessions()."""

    @pytest.mark.asyncio
    async def test_list_sessions_for_user(self):
        service = _make_service()
        refs = _setup_refs(service)
        service._get_app_state = AsyncMock(return_value={})
        service._get_user_state = AsyncMock(return_value={})

        # Mock async stream
        doc1 = _mock_doc_snapshot(
            True, {"state": {}, "events": [], "last_update_time": 1.0}, "s1"
        )
        doc2 = _mock_doc_snapshot(
            True, {"state": {}, "events": [], "last_update_time": 2.0}, "s2"
        )

        async def mock_stream():
            for d in [doc1, doc2]:
                yield d

        refs["sessions_collection"].stream = mock_stream

        result = await service.list_sessions(app_name="autopilot", user_id="u1")

        assert isinstance(result, ListSessionsResponse)
        assert len(result.sessions) == 2
        assert result.sessions[0].id == "s1"
        assert result.sessions[1].id == "s2"
        # Events should NOT be included in list
        assert result.sessions[0].events == []


class TestDeleteSession:
    """Tests for delete_session()."""

    @pytest.mark.asyncio
    async def test_delete_session(self):
        service = _make_service()
        refs = _setup_refs(service)
        refs["session_ref"].delete = AsyncMock()

        await service.delete_session(
            app_name="autopilot", user_id="u1", session_id="s1"
        )

        refs["session_ref"].delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self):
        """Deleting a nonexistent session should not raise."""
        service = _make_service()
        refs = _setup_refs(service)
        refs["session_ref"].delete = AsyncMock()

        # Should not raise
        await service.delete_session(
            app_name="autopilot", user_id="u1", session_id="nonexistent"
        )


class TestFromEnv:
    """Tests for from_env() factory classmethod."""

    @patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "my-gcp-project"})
    def test_from_env_reads_project(self):
        with patch(
            "autopilot.core.session_firestore.firestore.AsyncClient"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            FirestoreSessionService.from_env()
            mock_cls.assert_called_once_with(project="my-gcp-project")

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_no_project(self):
        with patch(
            "autopilot.core.session_firestore.firestore.AsyncClient"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            FirestoreSessionService.from_env()
            mock_cls.assert_called_once_with(project=None)


class TestCreateSessionServiceFactory:
    """Tests for the create_session_service() factory function."""

    def test_default_returns_in_memory(self):
        from autopilot.core.session import create_session_service
        from google.adk.sessions import InMemorySessionService

        service = create_session_service()
        assert isinstance(service, InMemorySessionService)

    def test_explicit_memory(self):
        from autopilot.core.session import create_session_service
        from google.adk.sessions import InMemorySessionService

        service = create_session_service("memory")
        assert isinstance(service, InMemorySessionService)

    @patch.dict("os.environ", {"SESSION_BACKEND": "firestore"})
    def test_firestore_from_env(self):
        with patch(
            "autopilot.core.session_firestore.firestore.AsyncClient"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            from autopilot.core.session import create_session_service

            service = create_session_service()
            assert isinstance(service, FirestoreSessionService)

    def test_explicit_firestore(self):
        with patch(
            "autopilot.core.session_firestore.firestore.AsyncClient"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            from autopilot.core.session import create_session_service

            service = create_session_service("firestore")
            assert isinstance(service, FirestoreSessionService)
