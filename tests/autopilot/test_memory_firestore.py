"""
Tests for FirestoreVectorMemoryService — ADK-native memory backend on Firestore + Gemini.

All Firestore and Gemini interactions are fully mocked — zero external dependencies.
Tests verify the ADK BaseMemoryService contract: add_session_to_memory,
add_events_to_memory, search_memory, idempotent upserts, batching, and factory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip(
    "google.cloud.firestore", reason="google-cloud-firestore not installed"
)

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.cloud.firestore_v1.vector import Vector
from google.genai.types import Content, Part

from autopilot.core.memory_firestore import (
    FirestoreVectorMemoryService,
    _event_to_text,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_event(
    event_id: str = "e1",
    text: str = "Hello world",
    author: str = "user",
    timestamp: float = 1000.0,
) -> Event:
    """Create an ADK Event with text content."""
    return Event(
        id=event_id,
        author=author,
        timestamp=timestamp,
        invocation_id="inv1",
        content=Content(parts=[Part(text=text)]),
    )


def _make_event_no_text(event_id: str = "e_empty") -> Event:
    """Create an ADK Event without text content (e.g., tool call)."""
    return Event(
        id=event_id,
        author="tool",
        timestamp=2000.0,
        invocation_id="inv1",
    )


def _make_session(
    app_name: str = "test_app",
    user_id: str = "u1",
    session_id: str = "s1",
    events: list[Event] | None = None,
) -> Session:
    """Create an ADK Session with events."""
    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state={},
        events=events or [],
    )


def _make_service(
    embedding_response: list[list[float]] | None = None,
) -> tuple[FirestoreVectorMemoryService, MagicMock, MagicMock]:
    """Create a FirestoreVectorMemoryService with mocked Firestore + Gemini.

    Returns:
        (service, mock_db, mock_genai) tuple for assertions.
    """
    mock_db = MagicMock()
    mock_genai = MagicMock()

    # Default embedding response: 768-dim vector per text
    default_embeddings = embedding_response or [[0.1] * 768]

    # Mock async embed_content
    mock_embedding_results = MagicMock()
    mock_embedding_results.embeddings = [
        MagicMock(values=emb) for emb in default_embeddings
    ]
    mock_genai.aio.models.embed_content = AsyncMock(return_value=mock_embedding_results)

    service = FirestoreVectorMemoryService(
        db=mock_db,
        genai=mock_genai,
    )

    return service, mock_db, mock_genai


def _mock_doc_snapshot(data: dict, doc_id: str = "d1") -> MagicMock:
    """Create a mock Firestore document snapshot."""
    snap = MagicMock()
    snap.id = doc_id
    snap.to_dict.return_value = data
    return snap


# ── Unit Tests ───────────────────────────────────────────────────────


class TestEventToText:
    """Tests for the _event_to_text helper."""

    def test_extracts_text(self):
        event = _make_event(text="Hello world")
        assert _event_to_text(event) == "Hello world"

    def test_multipart_text(self):
        event = Event(
            id="e1",
            author="user",
            timestamp=1.0,
            invocation_id="inv1",
            content=Content(parts=[Part(text="Hello"), Part(text="world")]),
        )
        assert _event_to_text(event) == "Hello world"

    def test_no_content_returns_none(self):
        event = _make_event_no_text()
        assert _event_to_text(event) is None

    def test_empty_parts_returns_none(self):
        event = Event(
            id="e1",
            author="user",
            timestamp=1.0,
            invocation_id="inv1",
            content=Content(parts=[]),
        )
        assert _event_to_text(event) is None


class TestAddSessionToMemory:
    """Tests for add_session_to_memory()."""

    @pytest.mark.asyncio
    async def test_embeds_and_stores_events(self):
        events = [
            _make_event("e1", "Hello"),
            _make_event("e2", "World"),
        ]
        service, mock_db, mock_genai = _make_service(
            embedding_response=[[0.1] * 768, [0.2] * 768]
        )
        session = _make_session(events=events)

        # Mock collection chain
        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection
        mock_doc_ref = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        # Mock batch
        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_db.batch.return_value = mock_batch

        await service.add_session_to_memory(session)

        # Gemini was called once with both texts
        mock_genai.aio.models.embed_content.assert_called_once()
        call_args = mock_genai.aio.models.embed_content.call_args
        assert call_args.kwargs["model"] == "gemini-embedding-001"
        assert len(call_args.kwargs["contents"]) == 2

        # Batch commit was called
        mock_batch.commit.assert_called_once()

        # Two documents were set in the batch
        assert mock_batch.set.call_count == 2

    @pytest.mark.asyncio
    async def test_idempotent_upsert(self):
        """Re-adding the same session uses same event IDs (upsert, no duplicates)."""
        events = [_make_event("e1", "Hello")]
        service, mock_db, mock_genai = _make_service()
        session = _make_session(events=events)

        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection
        mock_doc_ref = MagicMock()
        mock_collection.document.return_value = mock_doc_ref

        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_db.batch.return_value = mock_batch

        # Call twice
        await service.add_session_to_memory(session)
        await service.add_session_to_memory(session)

        # Each call uses batch.set (upsert) — same doc ID = no duplication
        assert mock_batch.set.call_count == 2  # 1 event × 2 calls
        # Document ID is derived from event.id
        mock_collection.document.assert_called_with("e1")

    @pytest.mark.asyncio
    async def test_skips_events_without_text(self):
        """Events without text content (tool calls) are skipped."""
        events = [
            _make_event("e1", "Hello"),
            _make_event_no_text("e2"),
        ]
        service, mock_db, mock_genai = _make_service()
        session = _make_session(events=events)

        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection
        mock_collection.document.return_value = MagicMock()

        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_db.batch.return_value = mock_batch

        await service.add_session_to_memory(session)

        # Only 1 text was embedded (e2 has no text)
        call_args = mock_genai.aio.models.embed_content.call_args
        assert len(call_args.kwargs["contents"]) == 1

    @pytest.mark.asyncio
    async def test_empty_session_is_noop(self):
        """Session with no events does nothing."""
        service, mock_db, mock_genai = _make_service()
        session = _make_session(events=[])

        await service.add_session_to_memory(session)

        mock_genai.aio.models.embed_content.assert_not_called()
        mock_db.batch.assert_not_called()


class TestAddEventsToMemory:
    """Tests for add_events_to_memory()."""

    @pytest.mark.asyncio
    async def test_incremental_delta_append(self):
        events = [_make_event("e3", "Delta event")]
        service, mock_db, mock_genai = _make_service()

        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection
        mock_collection.document.return_value = MagicMock()

        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_db.batch.return_value = mock_batch

        await service.add_events_to_memory(
            app_name="test_app",
            user_id="u1",
            events=events,
            session_id="s1",
        )

        # Embedding was called
        mock_genai.aio.models.embed_content.assert_called_once()

        # Batch set includes session_id in the document data
        set_call = mock_batch.set.call_args
        doc_data = set_call[0][1]
        assert doc_data["session_id"] == "s1"
        assert doc_data["text"] == "Delta event"
        assert isinstance(doc_data["embedding"], Vector)


class TestSearchMemory:
    """Tests for search_memory()."""

    @pytest.mark.asyncio
    async def test_returns_memory_entries(self):
        service, mock_db, mock_genai = _make_service()

        # Mock find_nearest result
        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection

        doc1 = _mock_doc_snapshot(
            {"text": "Remember this", "author": "user", "timestamp": 1.0}
        )
        doc2 = _mock_doc_snapshot(
            {"text": "Also this", "author": "assistant", "timestamp": 2.0}
        )

        mock_query = MagicMock()

        async def mock_stream():
            for d in [doc1, doc2]:
                yield d

        mock_query.stream = mock_stream
        mock_collection.find_nearest.return_value = mock_query

        result = await service.search_memory(
            app_name="test_app",
            user_id="u1",
            query="What do you remember?",
        )

        assert isinstance(result, SearchMemoryResponse)
        assert len(result.memories) == 2
        assert isinstance(result.memories[0], MemoryEntry)
        assert result.memories[0].content.parts[0].text == "Remember this"
        assert result.memories[0].author == "user"
        assert result.memories[1].content.parts[0].text == "Also this"
        assert result.memories[1].author == "assistant"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        service, mock_db, mock_genai = _make_service()

        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection

        mock_query = MagicMock()

        async def mock_stream():
            return
            yield  # noqa: RET504  — make it an async generator

        mock_query.stream = mock_stream
        mock_collection.find_nearest.return_value = mock_query

        result = await service.search_memory(
            app_name="test_app",
            user_id="u1",
            query="Something",
        )

        assert isinstance(result, SearchMemoryResponse)
        assert result.memories == []

    @pytest.mark.asyncio
    async def test_scoped_to_user(self):
        """Search queries the correct app_name/user_id collection path."""
        service, mock_db, mock_genai = _make_service()

        mock_root = MagicMock()
        mock_app = MagicMock()
        mock_users = MagicMock()
        mock_user = MagicMock()
        mock_memories = MagicMock()

        mock_db.collection.return_value = mock_root
        mock_root.document.return_value = mock_app
        mock_app.collection.return_value = mock_users
        mock_users.document.return_value = mock_user
        mock_user.collection.return_value = mock_memories

        mock_query = MagicMock()

        async def mock_stream():
            return
            yield

        mock_query.stream = mock_stream
        mock_memories.find_nearest.return_value = mock_query

        await service.search_memory(
            app_name="my_app",
            user_id="specific_user",
            query="test",
        )

        # Verify collection path
        mock_db.collection.assert_called_with("autopilot_memory")
        mock_root.document.assert_called_with("my_app")
        mock_app.collection.assert_called_with("users")
        mock_users.document.assert_called_with("specific_user")
        mock_user.collection.assert_called_with("memories")


class TestEmbeddingBatching:
    """Tests for embedding batch efficiency."""

    @pytest.mark.asyncio
    async def test_multiple_events_single_api_call(self):
        """Multiple events should be embedded in a single batch API call."""
        events = [
            _make_event("e1", "First message"),
            _make_event("e2", "Second message"),
            _make_event("e3", "Third message"),
        ]
        service, mock_db, mock_genai = _make_service(
            embedding_response=[[0.1] * 768, [0.2] * 768, [0.3] * 768]
        )
        session = _make_session(events=events)

        mock_collection = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value = mock_collection
        mock_collection.document.return_value = MagicMock()

        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_db.batch.return_value = mock_batch

        await service.add_session_to_memory(session)

        # Single API call with all 3 texts
        mock_genai.aio.models.embed_content.assert_called_once()
        call_args = mock_genai.aio.models.embed_content.call_args
        assert len(call_args.kwargs["contents"]) == 3


class TestFromEnv:
    """Tests for from_env() factory classmethod."""

    @patch.dict(
        "os.environ",
        {
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "MEMORY_EMBEDDING_MODEL": "custom-model",
            "MEMORY_EMBEDDING_DIMENSIONALITY": "128",
            "MEMORY_SEARCH_LIMIT": "10",
        },
    )
    def test_reads_all_env_vars(self):
        with (
            patch("autopilot.core.memory_firestore.firestore.AsyncClient") as mock_fs,
            patch("autopilot.core.memory_firestore.GenAIClient") as mock_genai_cls,
        ):
            mock_fs.return_value = MagicMock()
            mock_genai_cls.return_value = MagicMock()

            service = FirestoreVectorMemoryService.from_env()

            mock_fs.assert_called_once_with(project="my-project")
            assert service.embedding_model == "custom-model"
            assert service.dimensionality == 128
            assert service.search_limit == 10

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults(self):
        with (
            patch("autopilot.core.memory_firestore.firestore.AsyncClient") as mock_fs,
            patch("autopilot.core.memory_firestore.GenAIClient") as mock_genai_cls,
        ):
            mock_fs.return_value = MagicMock()
            mock_genai_cls.return_value = MagicMock()

            service = FirestoreVectorMemoryService.from_env()

            mock_fs.assert_called_once_with(project=None)
            assert service.embedding_model == "gemini-embedding-001"
            assert service.dimensionality == 768
            assert service.search_limit == 20


class TestCreateMemoryServiceFactory:
    """Tests for the create_memory_service() factory with firestore backend."""

    @patch.dict("os.environ", {"MEMORY_BACKEND": "firestore"})
    def test_firestore_from_env(self):
        with (
            patch("autopilot.core.memory_firestore.firestore.AsyncClient") as mock_fs,
            patch("autopilot.core.memory_firestore.GenAIClient") as mock_genai_cls,
        ):
            mock_fs.return_value = MagicMock()
            mock_genai_cls.return_value = MagicMock()

            from autopilot.core.memory import create_memory_service

            service = create_memory_service()
            assert isinstance(service, FirestoreVectorMemoryService)

    def test_default_returns_in_memory(self):
        from google.adk.memory import InMemoryMemoryService

        from autopilot.core.memory import create_memory_service

        service = create_memory_service()
        assert isinstance(service, InMemoryMemoryService)

    def test_unknown_backend_raises(self):
        from autopilot.core.memory import create_memory_service

        with (
            patch.dict("os.environ", {"MEMORY_BACKEND": "unknown"}),
            pytest.raises(ValueError, match="Unknown MEMORY_BACKEND"),
        ):
            create_memory_service()
