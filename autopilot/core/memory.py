"""
MemoryService — Long-term semantic memory for agents.

Provides a vector-like store where agents can record observations and
later retrieve the most semantically relevant ones.  Aligned with
Google ADK's ``MemoryService`` / ``InMemoryMemoryService`` pattern.

Two implementations:
  - InMemoryMemoryService: Pure-Python TF-IDF + cosine similarity (zero deps).
  - Drop-in replacement for ChromaDB / Vertex AI Search in prod.

Usage:
    memory = InMemoryMemoryService()
    await memory.add_observation("User prefers dark mode", {"source": "prefs"})
    results = await memory.search_relevant("theme preference", top_k=3)
"""

from __future__ import annotations

import abc
import math
import uuid
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

import structlog

try:
    import chromadb
    import chromadb.config
except ImportError:
    chromadb = None

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Observation — A single memory record
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Observation:
    """
    A single piece of long-term memory.

    Observations are freeform text with optional metadata (source agent,
    workflow, tags) and a timestamp for temporal ordering.

    Attributes:
        text: The observation content (natural language).
        metadata: Arbitrary key-value pairs (agent, workflow, tags, etc.).
        timestamp: When the observation was recorded.
        relevance_score: Populated by search — how relevant to the query (0–1).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    relevance_score: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BaseMemoryService — Abstract contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaseMemoryService(abc.ABC):
    """
    Abstract contract for long-term agent memory.

    Implementations handle observation storage and semantic retrieval.
    The interface is intentionally minimal — add / search — so that
    backends ranging from in-memory dicts to ChromaDB or Vertex AI
    can be swapped transparently.
    """

    @abc.abstractmethod
    async def add_observation(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """
        Record an observation in the memory store.

        Args:
            text: Natural language content to remember.
            metadata: Optional key-value context (agent name, tags, etc.).

        Returns:
            The created Observation with timestamp.
        """
        ...

    @abc.abstractmethod
    async def search_relevant(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Observation]:
        """
        Retrieve the most semantically relevant observations for a query.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of Observations ordered by descending relevance_score.
        """
        ...

    @abc.abstractmethod
    async def count(self) -> int:
        """Return the total number of stored observations."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  InMemoryMemoryService — Pure-Python TF-IDF + cosine similarity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InMemoryMemoryService(BaseMemoryService):
    """
    In-memory semantic memory using TF-IDF + cosine similarity.

    Zero external dependencies — uses only Python stdlib math.
    Suitable for dev/test and low-volume production workloads.

    For high-volume or production use, swap with a ChromaDB or
    Vertex AI Search implementation that implements BaseMemoryService.

    Algorithm:
      1. All texts (stored + query) are tokenized into lowercased words.
      2. TF-IDF vectors are computed on the full corpus + query.
      3. Cosine similarity ranks stored observations against the query.
      4. Top-k results are returned, sorted by descending relevance.
    """

    def __init__(self) -> None:
        self._observations: list[Observation] = []

    async def add_observation(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        obs = Observation(text=text, metadata=dict(metadata or {}))
        self._observations.append(obs)
        logger.debug(
            "memory_observation_added",
            text_preview=text[:80],
            total_observations=len(self._observations),
        )
        return obs

    async def search_relevant(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Observation]:
        if not self._observations or not query.strip():
            return []

        # Collect all texts: observations + query
        corpus_texts = [obs.text for obs in self._observations]
        all_texts = corpus_texts + [query]

        # Compute TF-IDF vectors
        vectors = _tfidf_vectorize(all_texts)
        query_vector = vectors[-1]
        obs_vectors = vectors[:-1]

        # Score each observation
        scored: list[tuple[float, int]] = []
        for idx, obs_vec in enumerate(obs_vectors):
            score = _cosine_similarity(query_vector, obs_vec)
            if score > 0:
                scored.append((score, idx))

        # Sort by descending relevance, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[Observation] = []

        for score, idx in scored[:top_k]:
            obs = self._observations[idx]
            # Return a copy with relevance_score populated
            results.append(
                Observation(
                    text=obs.text,
                    metadata=dict(obs.metadata),
                    timestamp=obs.timestamp,
                    relevance_score=round(score, 4),
                )
            )

        return results

    async def count(self) -> int:
        return len(self._observations)

    def __repr__(self) -> str:
        return f"<InMemoryMemoryService observations={len(self._observations)}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TF-IDF Helpers — Pure Python, zero dependencies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _tokenize(text: str) -> list[str]:
    """Lowercase split with basic punctuation stripping."""
    import re

    return re.findall(r"[a-záéíóúñü\d]+", text.lower())


def _tfidf_vectorize(texts: list[str]) -> list[dict[str, float]]:
    """
    Compute TF-IDF vectors for a list of texts.

    Returns a list of sparse vectors (dicts: term → TF-IDF weight),
    one per input text.
    """
    n_docs = len(texts)
    tokenized = [_tokenize(t) for t in texts]

    # Document frequency: how many documents contain each term
    df: Counter = Counter()
    for tokens in tokenized:
        unique = set(tokens)
        for term in unique:
            df[term] += 1

    # Compute TF-IDF for each document
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec: dict[str, float] = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((1 + n_docs) / (1 + df[term])) + 1  # smoothed IDF
            vec[term] = tf_val * idf_val
        vectors.append(vec)

    return vectors


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    # Dot product
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ChromaMemoryService — ChromaDB Vector Search implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ChromaMemoryService(BaseMemoryService):
    """
    ChromaDB-backed semantic memory.

    Provides production-ready, edge-compatible vector search.
    Documents are automatically embedded by Chroma's default embedding function
    (all-MiniLM-L6-v2) or a supplied custom embedding function.
    """

    def __init__(
        self,
        collection_name: str = "autopilot_memory",
        persist_directory: str | None = None,
    ):
        if chromadb is None:
            raise ImportError(
                "chromadb is required for ChromaMemoryService. Install with `pip install chromadb`."
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._collection_name = collection_name

    async def add_observation(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        obs_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        meta = dict(metadata or {})
        meta["timestamp"] = timestamp.isoformat()

        # Flatten and sanitize metadata (Chroma requires string, int, float, or bool)
        sanitized_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized_meta[k] = v
            else:
                sanitized_meta[k] = str(v)

        # ChromaDB sdk is currently synchronous, so we run it directly
        # In a high-throughput async system, this should be offloaded to a threadpool.
        self._collection.add(documents=[text], metadatas=[sanitized_meta], ids=[obs_id])

        obs = Observation(text=text, metadata=meta, timestamp=timestamp)
        logger.debug(
            "chroma_observation_added",
            id=obs_id,
            text_preview=text[:80],
            collection=self._collection_name,
        )
        return obs

    async def search_relevant(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Observation]:
        if not query.strip() or await self.count() == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=top_k)

        observations: list[Observation] = []
        if not results["documents"] or not results["documents"][0]:
            return observations

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = (
            results["distances"][0] if results["distances"] else [0.0] * len(docs)
        )

        for doc, meta, dist in zip(docs, metas, distances):
            meta_dict = dict(meta or {})

            # Parse previously serialized timestamp if present
            timestamp_str = meta_dict.pop("timestamp", None)
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            # Cosine distance to similarity (rough heuristic for Chroma L2/Cosine)
            # Chroma default is L2 squared distance.
            similarity_score = max(0.0, 1.0 - dist)

            observations.append(
                Observation(
                    text=doc,
                    metadata=meta_dict,
                    timestamp=ts,
                    relevance_score=round(similarity_score, 4),
                )
            )

        return observations

    async def count(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return f"<ChromaMemoryService collection='{self._collection_name}'>"
