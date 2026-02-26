"""
FirestoreVectorMemoryService — Durable semantic memory using Firestore + Gemini Embeddings.

ADK-native ``BaseMemoryService`` implementation.  Stores conversation events as
embedded vectors in Firestore, enabling semantic recall across sessions and
container restarts.

Firestore document hierarchy::

    autopilot_memory/{app_name}/users/{user_id}/memories/{event_id}
        → { text, embedding: Vector([...]), author, timestamp, session_id }

Backend selection (12-Factor)::

    MEMORY_BACKEND=firestore  →  FirestoreVectorMemoryService

Configuration env vars:

    MEMORY_EMBEDDING_MODEL          — Embedding model (default: gemini-embedding-001)
    MEMORY_EMBEDDING_DIMENSIONALITY — Vector dimensions (default: 768)
    MEMORY_SEARCH_LIMIT             — Max search results (default: 20)
"""

import logging
import os
from collections.abc import Sequence
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from google.genai import Client as GenAIClient
from google.genai.types import EmbedContentConfig
from typing_extensions import override

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.genai.types import Content, Part

logger = logging.getLogger("autopilot.core.memory_firestore")

_ROOT_COLLECTION = "autopilot_memory"
_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
_DEFAULT_DIMENSIONALITY = 768
_DEFAULT_SEARCH_LIMIT = 20


def _event_to_text(event: Event) -> str | None:
    """Extract plain text from an ADK Event's Content parts.

    Returns None if the event has no text content (e.g., tool calls).
    """
    if not event.content or not event.content.parts:
        return None
    texts = [p.text for p in event.content.parts if p.text]
    return " ".join(texts) if texts else None


class FirestoreVectorMemoryService(BaseMemoryService):
    """Durable semantic memory using Firestore vectors + Gemini embeddings.

    Production-grade ADK ``BaseMemoryService`` for Cloud Run deployments.
    Events are embedded via the Gemini API and stored with Firestore
    ``Vector`` fields for native cosine-similarity search.

    Attributes:
        db: Firestore ``AsyncClient`` instance.
        genai: Gemini ``Client`` for embedding generation.
        root_collection: Top-level Firestore collection name.
        embedding_model: Gemini embedding model ID.
        dimensionality: Output vector dimensions.
        search_limit: Max results from ``search_memory()``.
    """

    def __init__(
        self,
        *,
        db: firestore.AsyncClient,
        genai: GenAIClient,
        root_collection: str = _ROOT_COLLECTION,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        dimensionality: int = _DEFAULT_DIMENSIONALITY,
        search_limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> None:
        self.db = db
        self.genai = genai
        self.root_collection = root_collection
        self.embedding_model = embedding_model
        self.dimensionality = dimensionality
        self.search_limit = search_limit

    @classmethod
    def from_env(cls) -> "FirestoreVectorMemoryService":
        """Create from environment — zero-config on Cloud Run.

        Reads:
            ``GOOGLE_CLOUD_PROJECT``            — GCP project (auto-set on Cloud Run)
            ``MEMORY_EMBEDDING_MODEL``          — Embedding model (default: gemini-embedding-001)
            ``MEMORY_EMBEDDING_DIMENSIONALITY`` — Vector dimensions (default: 768)
            ``MEMORY_SEARCH_LIMIT``             — Max search results (default: 20)
        """
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        embedding_model = os.getenv("MEMORY_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
        dimensionality = int(
            os.getenv("MEMORY_EMBEDDING_DIMENSIONALITY", str(_DEFAULT_DIMENSIONALITY))
        )
        search_limit = int(os.getenv("MEMORY_SEARCH_LIMIT", str(_DEFAULT_SEARCH_LIMIT)))

        logger.info(
            "firestore_vector_memory_init",
            extra={
                "project": project,
                "embedding_model": embedding_model,
                "dimensionality": dimensionality,
                "search_limit": search_limit,
            },
        )

        return cls(
            db=firestore.AsyncClient(project=project),
            genai=GenAIClient(),
            embedding_model=embedding_model,
            dimensionality=dimensionality,
            search_limit=search_limit,
        )

    # ── Collection References ────────────────────────────────────────

    def _memories_collection(
        self, app_name: str, user_id: str
    ) -> firestore.AsyncCollectionReference:
        """Ref to the memories subcollection for a specific user."""
        return (
            self.db.collection(self.root_collection)
            .document(app_name)
            .collection("users")
            .document(user_id)
            .collection("memories")
        )

    # ── Embedding ────────────────────────────────────────────────────

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the Gemini API (async).

        Returns a list of embedding vectors, one per input text.
        """
        if not texts:
            return []

        response = await self.genai.aio.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=EmbedContentConfig(output_dimensionality=self.dimensionality),
        )

        return [e.values for e in response.embeddings]

    # ── BaseMemoryService ABC ────────────────────────────────────────

    @override
    async def add_session_to_memory(self, session: Any) -> None:
        """Embed all session events and upsert into Firestore.

        Uses the event ``id`` as the document ID for idempotent upserts —
        re-adding the same session does not create duplicates.
        """
        events = session.events or []
        await self._store_events(
            app_name=session.app_name,
            user_id=session.user_id,
            events=events,
            session_id=session.id,
        )

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Any | None = None,
    ) -> None:
        """Embed delta events and append to Firestore (incremental)."""
        await self._store_events(
            app_name=app_name,
            user_id=user_id,
            events=events,
            session_id=session_id,
        )

    @override
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Embed query and search via Firestore ``find_nearest()``."""
        embeddings = await self._embed_texts([query])
        if not embeddings:
            return SearchMemoryResponse(memories=[])

        query_vector = Vector(embeddings[0])
        collection = self._memories_collection(app_name, user_id)

        results = collection.find_nearest(
            vector_field="embedding",
            query_vector=query_vector,
            limit=self.search_limit,
            distance_measure=DistanceMeasure.COSINE,
        )

        memories: list[MemoryEntry] = []
        async for doc in results.stream():
            data = doc.to_dict()
            text = data.get("text", "")
            memories.append(
                MemoryEntry(
                    content=Content(parts=[Part(text=text)]),
                    author=data.get("author"),
                    timestamp=str(data.get("timestamp", "")),
                )
            )

        logger.debug(
            "memory_search_complete",
            extra={
                "app_name": app_name,
                "user_id": user_id,
                "query_length": len(query),
                "results": len(memories),
            },
        )

        return SearchMemoryResponse(memories=memories)

    # ── Internal Helpers ─────────────────────────────────────────────

    async def _store_events(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
    ) -> None:
        """Embed and upsert events into Firestore.

        Filters out events without text content, batches embedding calls,
        and writes each event as a document with the event ID as the
        document key (idempotent upsert).
        """
        # Extract text from events, keeping index alignment
        event_texts: list[tuple[Event, str]] = []
        for event in events:
            text = _event_to_text(event)
            if text:
                event_texts.append((event, text))

        if not event_texts:
            return

        # Batch embed all texts in a single API call
        texts = [t for _, t in event_texts]
        embeddings = await self._embed_texts(texts)

        # Write to Firestore (batch for efficiency)
        collection = self._memories_collection(app_name, user_id)
        batch = self.db.batch()

        for (event, text), embedding in zip(event_texts, embeddings):
            doc_id = event.id or str(id(event))
            doc_ref = collection.document(doc_id)

            doc_data: dict[str, Any] = {
                "text": text,
                "embedding": Vector(embedding),
                "author": event.author or "unknown",
                "timestamp": event.timestamp,
            }
            if session_id:
                doc_data["session_id"] = session_id

            batch.set(doc_ref, doc_data)

        await batch.commit()

        logger.debug(
            "memory_events_stored",
            extra={
                "app_name": app_name,
                "user_id": user_id,
                "events_stored": len(event_texts),
                "events_skipped": len(list(events)) - len(event_texts),
            },
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the Firestore client."""
        self.db.close()

    async def __aenter__(self) -> "FirestoreVectorMemoryService":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"FirestoreVectorMemoryService("
            f"project={self.db.project!r}, "
            f"collection={self.root_collection!r}, "
            f"model={self.embedding_model!r}, "
            f"dims={self.dimensionality})"
        )
