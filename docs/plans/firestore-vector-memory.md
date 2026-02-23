# N7. Firestore Vector Memory Service — Durable Semantic Recall

> **Status**: ✅ Implemented
> **ADK Contract**: `BaseMemoryService` (same ABC as `InMemoryMemoryService`, `VertexAiMemoryBankService`)
> **Cost**: Firestore (already provisioned) + Gemini Embedding API (~$0.00004/1000 chars)
> **New Dependencies**: Zero (uses existing `google-cloud-firestore` + `google.genai`)

## Problem

Context window compression (`SlidingWindow`) prevents token overflow for long-running conversations, but older events are compressed out of the LLM's context. The current `InMemoryMemoryService` retains these events for cross-session recall, but is **process-scoped** — lost on Cloud Run scale-to-zero or restart.

## Solution

A new `FirestoreVectorMemoryService(BaseMemoryService)` that:

1. **Embeds** conversation events via `google.genai` (`gemini-embedding-001`)
2. **Stores** embeddings + text in Firestore with native `Vector` fields
3. **Searches** via Firestore's `find_nearest()` (cosine similarity)
4. **Persists** across restarts — survives scale-to-zero

This follows the **exact same pattern** as `FirestoreSessionService` (N4) — a custom ADK ABC implementation backed by Firestore, selected via 12-Factor env var.

## Firestore Data Isolation (Best Practices)

The platform uses a **single Firestore database** (`(default)`) with **isolated root collections** per data domain. This is Google's recommended pattern — one database, one billing, zero data mixing:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Firestore Database (default)                  │
│                                                                  │
│  ┌─────────────────────────────────┐  ┌────────────────────────┐│
│  │ autopilot_sessions/             │  │ autopilot_memory/      ││
│  │   └── {app_name}/              │  │   └── {app_name}/      ││
│  │       └── users/{user_id}/     │  │       └── users/{uid}/ ││
│  │           └── sessions/{sid}   │  │           └── memories/ ││
│  │               → state, events  │  │               → text,  ││
│  │                                 │  │                 embed, ││
│  │                                 │  │                 vector ││
│  │  DOMAIN: Session persistence    │  │  DOMAIN: Semantic      ││
│  │  SERVICE: FirestoreSessionSvc   │  │  recall (NEW)          ││
│  │  BACKEND: SESSION_BACKEND=      │  │  SERVICE: Firestore    ││
│  │           firestore             │  │  VectorMemorySvc       ││
│  │                                 │  │  BACKEND: MEMORY_      ││
│  │                                 │  │  BACKEND=firestore     ││
│  └─────────────────────────────────┘  └────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│       GCS Bucket (separate)      │
│  antigravity-bank-ynab-artifacts │
│    └── {app_name}/default/       │
│        └── {execution_id}/       │
│            → .json, .llm.json    │
│                                  │
│  DOMAIN: Versioned artifacts     │
│  SERVICE: GcsArtifactService     │
│  BACKEND: ARTIFACT_BACKEND=gcs   │
└──────────────────────────────────┘
```

**Isolation guarantees:**

| Concern             | Isolation Mechanism                                                        |
| ------------------- | -------------------------------------------------------------------------- |
| Sessions vs Memory  | Different root collections (`autopilot_sessions/` vs `autopilot_memory/`)  |
| Memory vs Artifacts | Different storage systems (Firestore vs GCS)                               |
| Per-app scoping     | Both use `{app_name}/` as first-level partition                            |
| Per-user scoping    | Both use `users/{user_id}/` — ADK's 3-tier model                           |
| Cross-query safety  | Firestore queries are collection-scoped — impossible to accidentally cross |

## ADK Alignment

The service implements `BaseMemoryService` — the same abstract contract that ADK's built-in backends use:

```python
class BaseMemoryService(ABC):
    @abstractmethod
    async def add_session_to_memory(self, session: Session) -> None: ...

    async def add_events_to_memory(self, *, app_name, user_id, events, session_id=None, custom_metadata=None) -> None: ...

    @abstractmethod
    async def search_memory(self, *, app_name, user_id, query) -> SearchMemoryResponse: ...
```

Our implementation provides **all three methods**:

| Method                           | Behavior                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `add_session_to_memory(session)` | Embed all session events → upsert into Firestore with `Vector` field                                   |
| `add_events_to_memory(events)`   | Embed delta events → append to Firestore (incremental, no full re-ingest)                              |
| `search_memory(query)`           | Embed query → `find_nearest()` on Firestore → return `SearchMemoryResponse` with `MemoryEntry` objects |

## Proposed Changes

### [NEW] `autopilot/core/memory_firestore.py`

```python
"""
FirestoreVectorMemoryService — Durable semantic memory using Firestore + Gemini Embeddings.

ADK-native BaseMemoryService implementation. Stores conversation events as
embedded vectors in Firestore, enabling semantic recall across sessions and
container restarts.

Firestore document hierarchy:

    autopilot_memory/{app_name}/users/{user_id}/memories/{event_id}
        → { text, embedding: Vector([...]), author, timestamp, session_id }

Backend selection (12-Factor):

    MEMORY_BACKEND=firestore  →  FirestoreVectorMemoryService
"""
```

**Internal design:**

- **Embedding**: Uses `google.genai.Client().models.embed_content(model='gemini-embedding-001', ...)` with `output_dimensionality=768` (Google's first recommended MRL dimension — optimal trade-off between accuracy and storage for conversational recall)
- **Storage**: Each event → one Firestore document with `embedding: Vector([...])` field
- **Deduplication**: Event `id` is the document ID — `add_session_to_memory()` upserts idempotently
- **Batching**: Events are embedded in batches (Gemini API supports batch `embed_content`) to minimize API calls
- **Search**: `collection.find_nearest(vector_field='embedding', query_vector=query_embedding, limit=20, distance_measure=COSINE)`

**Key decisions:**

| Decision        | Choice                              | Why                                                               |
| --------------- | ----------------------------------- | ----------------------------------------------------------------- |
| Embedding model | `gemini-embedding-001`              | Free tier, same API key as LLM calls                              |
| Dimensions      | 768                                 | Google's first recommended MRL dimension for gemini-embedding-001 |
| Distance        | Cosine                              | Standard for text similarity, normalized                          |
| Limit           | 20 results                          | Reasonable context injection without bloating prompt              |
| Index           | Firestore single-field vector index | Auto-created, no manual provisioning needed                       |

### [MODIFY] `autopilot/core/memory.py`

Add `firestore` backend to the factory:

```python
def create_memory_service() -> BaseMemoryService:
    backend = os.getenv("MEMORY_BACKEND", "memory").lower().strip()

    if backend == "memory":
        return InMemoryMemoryService()

    if backend == "firestore":
        from autopilot.core.memory_firestore import FirestoreVectorMemoryService
        return FirestoreVectorMemoryService.from_env()

    if backend == "vertexai":
        # ... existing code ...
```

### [NEW] `tests/autopilot/test_memory_firestore.py`

Unit tests with fully mocked Firestore + Gemini (same pattern as `test_session_firestore.py`):

- `test_add_session_to_memory` — embeds events, writes to Firestore with Vector
- `test_add_session_idempotent` — re-adding same session doesn't duplicate
- `test_add_events_to_memory` — incremental delta append
- `test_search_memory_returns_entries` — embeds query, calls `find_nearest`, returns `MemoryEntry` list
- `test_search_memory_empty_results` — returns empty when no matches
- `test_search_memory_filters_by_user` — scoped to `app_name/user_id`
- `test_embedding_batching` — multiple events → single batch API call
- `test_from_env_factory` — reads env vars correctly

### [MODIFY] Documentation & Deployment

| File                                      | Change                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| `.env.example`                            | Add `MEMORY_BACKEND=firestore` option, `MEMORY_EMBEDDING_DIMENSIONALITY` var |
| `ARCHITECTURE.md`                         | Update memory backend table — add `firestore` row                            |
| `Dockerfile`                              | Add `MEMORY_BACKEND=firestore` to env docs                                   |
| `.github/workflows/ci.yml`                | Add `MEMORY_BACKEND=firestore` to deploy `--set-env-vars`                    |
| `.agent/workflows/deploy_to_cloud_run.md` | Add `MEMORY_BACKEND=firestore` to manual deploy command                      |
| `docs/plans/Implementation Plan`          | Add N7 entry                                                                 |

### [PROVISION] Firestore Vector Index

A single-field vector index is needed on the `embedding` field:

```bash
gcloud firestore indexes composite create \
  --collection-group=memories \
  --query-scope=COLLECTION \
  --field-config=vector-config='{"dimension":"768","flat": {}}',field-path=embedding
```

> [!NOTE]
> This is a **one-time** operation. Firestore manages the index automatically after creation.

## Configuration (12-Factor)

| Variable                          | Default                | Purpose                                              |
| --------------------------------- | ---------------------- | ---------------------------------------------------- |
| `MEMORY_BACKEND`                  | `memory`               | Backend selection: `memory`, `firestore`, `vertexai` |
| `MEMORY_EMBEDDING_MODEL`          | `gemini-embedding-001` | Embedding model (override if needed)                 |
| `MEMORY_EMBEDDING_DIMENSIONALITY` | `768`                  | Vector dimensions (Google recommended)               |
| `MEMORY_SEARCH_LIMIT`             | `20`                   | Max results from `search_memory()`                   |

## Backend Selection Matrix (Updated)

| Environment    | Backend                        | `MEMORY_BACKEND`           | Why                                   |
| -------------- | ------------------------------ | -------------------------- | ------------------------------------- |
| **Unit Tests** | `InMemoryMemoryService`        | Unset → `memory` (default) | Zero deps, deterministic, instant     |
| **Local Dev**  | `InMemoryMemoryService`        | Unset → `memory` (default) | Stateless pipelines — persistence n/a |
| **Cloud Run**  | `FirestoreVectorMemoryService` | `MEMORY_BACKEND=firestore` | Durable, semantic search, low cost    |
| **Cloud Run**  | `VertexAiMemoryBankService`    | `MEMORY_BACKEND=vertexai`  | Managed, semantic search, higher cost |

## Verification Plan

```bash
# New tests only
python -m pytest tests/autopilot/test_memory_firestore.py -xvs

# Full regression
python -m pytest tests/ -x
```

## Cost Analysis

| Component                | Cost                   | Volume                        |
| ------------------------ | ---------------------- | ----------------------------- |
| Gemini Embedding API     | ~$0.00004 / 1000 chars | ~100 events/day = ~$0.004/day |
| Firestore writes         | $0.18 / 100k writes    | ~100 writes/day = negligible  |
| Firestore reads (search) | $0.06 / 100k reads     | ~50 queries/day = negligible  |
| **Total**                | **< $0.15/month**      | Conservative estimate         |
