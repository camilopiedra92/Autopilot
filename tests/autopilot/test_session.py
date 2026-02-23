"""
Tests for session module — ADK-native SessionService re-exports.

Verifies that the platform re-exports are the exact ADK types
and that ADK's InMemorySessionService lifecycle works correctly.
"""

import pytest

from autopilot.core.session import (
    BaseSessionService,
    InMemorySessionService,
    Session,
)


class TestADKReExports:
    """Verify that ADK session types are correctly re-exported."""

    def test_base_session_service_is_adk(self):
        from google.adk.sessions import (
            BaseSessionService as ADKBaseSessionService,
        )

        assert BaseSessionService is ADKBaseSessionService

    def test_in_memory_session_service_is_adk(self):
        from google.adk.sessions import (
            InMemorySessionService as ADKInMemorySessionService,
        )

        assert InMemorySessionService is ADKInMemorySessionService

    def test_session_is_adk(self):
        from google.adk.sessions import Session as ADKSession

        assert Session is ADKSession


class TestSessionLifecycle:
    """Tests for ADK InMemorySessionService create/get/list/delete lifecycle."""

    @pytest.mark.asyncio
    async def test_create_and_get_session(self):
        service = InMemorySessionService()
        session = await service.create_session(
            app_name="autopilot",
            user_id="u1",
            state={"key": "value"},
        )
        assert isinstance(session, Session)
        assert session.app_name == "autopilot"
        assert session.user_id == "u1"
        assert session.state.get("key") == "value"

        retrieved = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id=session.id,
        )
        assert retrieved is not None
        assert retrieved.id == session.id

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        service = InMemorySessionService()
        await service.create_session(app_name="autopilot", user_id="u1")
        await service.create_session(app_name="autopilot", user_id="u1")

        result = await service.list_sessions(app_name="autopilot", user_id="u1")
        assert len(result.sessions) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self):
        service = InMemorySessionService()
        session = await service.create_session(app_name="autopilot", user_id="u1")
        await service.delete_session(
            app_name="autopilot",
            user_id="u1",
            session_id=session.id,
        )
        retrieved = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id=session.id,
        )
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_returns_none(self):
        service = InMemorySessionService()
        result = await service.get_session(
            app_name="autopilot",
            user_id="u1",
            session_id="nonexistent",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_session_state_is_plain_dict(self):
        """Session.state is a plain dict — no wrappers, no async."""
        service = InMemorySessionService()
        session = await service.create_session(
            app_name="autopilot",
            user_id="u1",
            state={"initial": True},
        )
        # Direct dict operations — this is the ADK way
        session.state["new_key"] = "new_value"
        assert session.state["new_key"] == "new_value"
        assert session.state.get("initial") is True
        assert len(session.state) == 2
