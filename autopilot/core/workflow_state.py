"""
WorkflowStateService — Cross-run KV store for workflows.

Thin wrapper over ``ArtifactService`` that provides a clean dict-like API
for persistent workflow state.  Each workflow gets its own namespace
(``app_name=workflow_name``), and all keys share a fixed
``session_id="persistent"`` so data survives across pipeline executions.

Two usage patterns::

    # 1. Standalone (uses the global ArtifactService singleton)
    state = WorkflowStateService("polymarket_btc")

    # 2. From AgentContext (auto-scoped to the current workflow)
    data = await ctx.workflow_state.get("risk_state")

Backend is inherited from ``ARTIFACT_BACKEND`` env var — no extra config.
"""

import json

import structlog

logger = structlog.get_logger(__name__)

_SESSION_ID = "persistent"
_USER_ID = "default"


class WorkflowStateService:
    """Cross-run KV store scoped to a single workflow.

    All values are JSON-serializable dicts.  Under the hood each key
    becomes a ``{key}.json`` artifact stored via the platform's
    ``ArtifactService`` with a fixed ``session_id="persistent"``.

    Mutations are fire-and-forget: errors are logged but never raised.
    Reads return ``None`` on any failure (missing key, parse error, etc.).
    """

    def __init__(
        self,
        workflow_name: str,
        *,
        artifact_service=None,
    ) -> None:
        self._workflow_name = workflow_name
        self._artifact_service = artifact_service

    @property
    def _service(self):
        """Lazy-resolve the ArtifactService singleton."""
        if self._artifact_service is None:
            from autopilot.core.artifact import get_artifact_service

            self._artifact_service = get_artifact_service()
        return self._artifact_service

    # ── Read ──────────────────────────────────────────────────────────

    async def get(self, key: str) -> dict | None:
        """Load a JSON dict by key, or ``None`` if not found."""
        filename = f"{key}.json"
        try:
            artifact = await self._service.load_artifact(
                app_name=self._workflow_name,
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                filename=filename,
            )
            if artifact and artifact.text:
                return json.loads(artifact.text)
        except Exception as e:
            logger.debug(
                "workflow_state_get_failed",
                workflow=self._workflow_name,
                key=key,
                error=str(e),
            )
        return None

    # ── Write ─────────────────────────────────────────────────────────

    async def put(self, key: str, data: dict) -> None:
        """Save a JSON dict under the given key (fire-and-forget)."""
        from google.genai import types

        filename = f"{key}.json"
        try:
            artifact = types.Part(text=json.dumps(data, ensure_ascii=False))
            await self._service.save_artifact(
                app_name=self._workflow_name,
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                filename=filename,
                artifact=artifact,
            )
        except Exception as e:
            logger.warning(
                "workflow_state_put_failed",
                workflow=self._workflow_name,
                key=key,
                error=str(e),
            )

    # ── Delete ────────────────────────────────────────────────────────

    async def delete(self, key: str) -> None:
        """Delete a key by saving an empty artifact (fire-and-forget)."""
        from google.genai import types

        filename = f"{key}.json"
        try:
            await self._service.save_artifact(
                app_name=self._workflow_name,
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                filename=filename,
                artifact=types.Part(text=""),
            )
        except Exception as e:
            logger.warning(
                "workflow_state_delete_failed",
                workflow=self._workflow_name,
                key=key,
                error=str(e),
            )

    # ── List ──────────────────────────────────────────────────────────

    async def list_keys(self) -> list[str]:
        """List all state keys for this workflow."""
        try:
            filenames = await self._service.list_artifact_keys(
                app_name=self._workflow_name,
                user_id=_USER_ID,
                session_id=_SESSION_ID,
            )
            return [
                f.removesuffix(".json")
                for f in (filenames or [])
                if f.endswith(".json")
            ]
        except Exception as e:
            logger.debug(
                "workflow_state_list_failed",
                workflow=self._workflow_name,
                error=str(e),
            )
            return []


__all__ = ["WorkflowStateService"]
