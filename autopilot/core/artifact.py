"""
ArtifactService — ADK-native re-exports + 12-Factor factory.

Aligned with Google ADK's ``BaseArtifactService`` contract. Provides
versioned storage for pipeline run outputs (parsed emails, transaction
JSONs, categorization results) enabling cross-run debugging, audit
trails, and replay capabilities.

Two backends (selected via ``ARTIFACT_BACKEND`` env var):
  - ``memory``  → ADK InMemoryArtifactService (zero deps, dev/test)
  - ``gcs``     → GcsArtifactService (Google Cloud Storage, production)

Usage::

    from autopilot.core.artifact import create_artifact_service

    service = create_artifact_service()  # Reads ARTIFACT_BACKEND env var
"""

import os

import structlog

# ── ADK Re-exports (public API only) ──────────────────────────────────
from google.adk.artifacts import (
    BaseArtifactService,
    InMemoryArtifactService,
    GcsArtifactService,
)

logger = structlog.get_logger(__name__)

# ── Factory (12-Factor, same pattern as session/memory/bus) ───────────

_artifact_service: BaseArtifactService | None = None


def create_artifact_service(
    backend: str | None = None,
) -> BaseArtifactService:
    """Create an ArtifactService based on ARTIFACT_BACKEND env var.

    | ARTIFACT_BACKEND | Backend                     | Cost              |
    |------------------|-----------------------------|-------------------|
    | ``memory``       | InMemoryArtifactService     | Free              |
    | ``gcs``          | GcsArtifactService          | ~$0.02/GB/month   |

    Args:
        backend: Override backend choice. Defaults to ``ARTIFACT_BACKEND``
                 env var, falling back to ``"memory"``.

    Returns:
        A configured BaseArtifactService instance.

    Raises:
        ValueError: If ARTIFACT_BACKEND is unknown or GCS bucket not set.
    """
    backend = (backend or os.getenv("ARTIFACT_BACKEND", "memory")).lower().strip()

    if backend == "memory":
        logger.info("artifact_backend_selected", backend="memory")
        return InMemoryArtifactService()

    if backend == "gcs":
        bucket_name = os.environ.get("ARTIFACT_GCS_BUCKET", "")
        if not bucket_name:
            raise ValueError(
                "ARTIFACT_GCS_BUCKET is required when ARTIFACT_BACKEND=gcs"
            )
        logger.info(
            "artifact_backend_selected",
            backend="gcs",
            bucket=bucket_name,
        )
        return GcsArtifactService(bucket_name=bucket_name)

    raise ValueError(
        f"Unknown ARTIFACT_BACKEND: {backend!r}. Supported: 'memory', 'gcs'."
    )


def get_artifact_service() -> BaseArtifactService:
    """Get or create the singleton ArtifactService."""
    global _artifact_service
    if _artifact_service is None:
        _artifact_service = create_artifact_service()
    return _artifact_service


def reset_artifact_service() -> None:
    """Reset the singleton (for tests)."""
    global _artifact_service
    _artifact_service = None


__all__ = [
    "BaseArtifactService",
    "InMemoryArtifactService",
    "GcsArtifactService",
    "create_artifact_service",
    "get_artifact_service",
    "reset_artifact_service",
]
