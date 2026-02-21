"""
TodoistConnector — Async-only Todoist integration block.

Wraps the Todoist REST API v2 as a platform connector. Provides project collection,
task lookup, and task creation — all async.

Usage:
    from autopilot.connectors import get_connector_registry
    todoist = get_connector_registry().get("todoist")
    projects = await todoist.client.get_projects()
"""

from __future__ import annotations

import os
import time
import structlog
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)

TODOIST_API_URL_REST = "https://api.todoist.com/api/v1"


# ── Exceptions ───────────────────────────────────────────────────────

from autopilot.errors import ConnectorError as TodoistError, ConnectorRateLimitError as TodoistRateLimitError


# ── TTL Cache ────────────────────────────────────────────────────────

class TTLCache:
    """Simple thread-safe TTL cache for Todoist data."""

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, any]] = {}

    def get(self, key: str):
        if key in self._store:
            ts, data = self._store[key]
            if time.monotonic() - ts < self._ttl:
                return data
            del self._store[key]
        return None

    def set(self, key: str, value):
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str | None = None):
        if key:
            self._store.pop(key, None)
        else:
            self._store.clear()


# ── Async Todoist Client ─────────────────────────────────────────────

class AsyncTodoistClient:
    """
    Production-grade async Todoist API client.

    Features:
    - httpx.AsyncClient with HTTP/2 and connection pooling
    - TTL-based caching for projects
    - Automatic retry with exponential backoff on rate limits
    - Structured logging for every API call
    """

    def __init__(self, access_token: str):
        self._token = access_token
        self._client = httpx.AsyncClient(
            base_url=TODOIST_API_URL_REST,
            http2=True,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0, read=20.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self._projects_cache = TTLCache(ttl_seconds=300)

    async def _request(self, method: str, path: str, **kwargs) -> any:
        """Execute an async API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise TodoistError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise TodoistError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            logger.warning("todoist_rate_limited", path=path, latency_ms=round(latency_ms))
            raise TodoistRateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise TodoistError("Authentication failed — check your access token", detail="401")
        if resp.status_code >= 400:
            raise TodoistError(f"API error: {resp.status_code} — {resp.text}", detail=str(resp.status_code))

        logger.debug(
            "todoist_request",
            method=method,
            path=path,
            status=resp.status_code,
            latency_ms=round(latency_ms),
            http_version=resp.http_version,
        )
        
        if resp.status_code == 204:
            return None
        return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def get_projects(self) -> list[dict]:
        """Fetch all projects (cached)."""
        cached = self._projects_cache.get("projects")
        if cached is not None:
            logger.debug("todoist_projects_cache_hit")
            return cached

        data = await self._request("GET", "/projects")
        projects = data.get("results", data) if isinstance(data, dict) else data
        self._projects_cache.set("projects", projects)
        logger.info("todoist_projects_fetched", count=len(projects))
        return projects

    async def get_projects_string(self) -> str:
        """Returns formatted project list for AI tool consumption."""
        projects = await self.get_projects()
        output = [
            f"Project: {p['name']} (ID: {p['id']})"
            for p in projects
        ]
        return "\n".join(output)

    async def project_exists(self, project_id: str) -> bool:
        """Check if a project UUID actually exists."""
        projects = await self.get_projects()
        return any(p["id"] == project_id for p in projects)

    async def get_active_tasks(self, project_id: str | None = None) -> list[dict]:
        """Fetch active tasks, optionally filtered by project."""
        params = {}
        if project_id:
            params["project_id"] = project_id
            
        data = await self._request("GET", "/tasks", params=params)
        return data.get("results", data) if isinstance(data, dict) else data

    async def get_active_tasks_string(self, project_id: str | None = None) -> str:
        """Returns formatted active tasks for AI consumption."""
        tasks = await self.get_active_tasks(project_id=project_id)
        output = []
        for t in tasks:
            due = t.get("due", {})
            due_str = due.get("string", "No due date") if due else "No due date"
            output.append(
                f"Task: {t['content']} (ID: {t['id']}) | "
                f"Due: {due_str} | Priority: {t['priority']}"
            )
        return "\n".join(output)

    async def get_task(self, task_id: str) -> dict:
        """Retrieve a specific task by ID."""
        return await self._request("GET", f"/tasks/{task_id}")

    async def task_exists(self, task_id: str) -> bool:
        """Check if a task ID actually exists."""
        try:
            await self.get_task(task_id)
            return True
        except TodoistError as e:
            if e.detail == "404":
                return False
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def create_task(self, transaction_payload: dict) -> dict:
        """
        Create a task in Todoist.
        
        Args:
            transaction_payload: A dictionary matching the Todoist API task schema.
                For example: {"content": "Buy milk", "project_id": "123", "due_string": "tomorrow"}
        """
        logger.info(
            "todoist_creating_task",
            content=transaction_payload.get("content", "unknown"),
            project_id=transaction_payload.get("project_id", "inbox"),
        )
        data = await self._request("POST", "/tasks", json=transaction_payload)
        logger.info("todoist_task_created", task_id=data.get("id"))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def update_task(self, task_id: str, payload: dict) -> dict:
        """Update an existing task."""
        logger.info(
            "todoist_updating_task",
            task_id=task_id,
            keys_updated=list(payload.keys()),
        )
        data = await self._request("POST", f"/tasks/{task_id}", json=payload)
        logger.info("todoist_task_updated", task_id=data.get("id"))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def close_task(self, task_id: str) -> bool:
        """Mark a task as complete."""
        logger.info("todoist_closing_task", task_id=task_id)
        await self._request("POST", f"/tasks/{task_id}/close")
        logger.info("todoist_task_closed", task_id=task_id)
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        logger.info("todoist_deleting_task", task_id=task_id)
        await self._request("DELETE", f"/tasks/{task_id}")
        logger.info("todoist_task_deleted", task_id=task_id)
        return True

    # ── Sections ───────────────────────────────────────────────────

    async def get_sections(self, project_id: str | None = None) -> list[dict]:
        """Fetch sections, optionally filtered by project."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        data = await self._request("GET", "/sections", params=params)
        return data.get("results", data) if isinstance(data, dict) else data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def create_section(self, name: str, project_id: str, order: int | None = None) -> dict:
        """Create a new section within a project."""
        logger.info("todoist_creating_section", name=name, project_id=project_id)
        payload = {"name": name, "project_id": project_id}
        if order is not None:
            payload["order"] = order
        data = await self._request("POST", "/sections", json=payload)
        logger.info("todoist_section_created", section_id=data.get("id"))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def delete_section(self, section_id: str) -> bool:
        """Delete a section."""
        logger.info("todoist_deleting_section", section_id=section_id)
        await self._request("DELETE", f"/sections/{section_id}")
        logger.info("todoist_section_deleted", section_id=section_id)
        return True

    # ── Labels ─────────────────────────────────────────────────────

    async def get_labels(self) -> list[dict]:
        """Fetch all user labels."""
        data = await self._request("GET", "/labels")
        return data.get("results", data) if isinstance(data, dict) else data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def create_label(self, name: str, color: str | None = None, is_favorite: bool = False) -> dict:
        """Create a new personal label."""
        logger.info("todoist_creating_label", name=name)
        payload = {"name": name, "is_favorite": is_favorite}
        if color:
            payload["color"] = color
        data = await self._request("POST", "/labels", json=payload)
        logger.info("todoist_label_created", label_id=data.get("id"))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def delete_label(self, label_id: str) -> bool:
        """Delete a personal label."""
        logger.info("todoist_deleting_label", label_id=label_id)
        await self._request("DELETE", f"/labels/{label_id}")
        logger.info("todoist_label_deleted", label_id=label_id)
        return True

    # ── Comments ───────────────────────────────────────────────────

    async def get_comments(self, task_id: str | None = None, project_id: str | None = None) -> list[dict]:
        """Fetch comments for a given task or project."""
        if not task_id and not project_id:
            raise TodoistError("Must provide either task_id or project_id to fetch comments.")
        params = {}
        if task_id:
            params["task_id"] = task_id
        if project_id:
            params["project_id"] = project_id
            
        data = await self._request("GET", "/comments", params=params)
        return data.get("results", data) if isinstance(data, dict) else data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(TodoistRateLimitError),
    )
    async def create_comment(self, content: str, task_id: str | None = None, project_id: str | None = None) -> dict:
        """Create a comment on a task or project."""
        if not task_id and not project_id:
            raise TodoistError("Must provide either task_id or project_id to create a comment.")
            
        logger.info(
            "todoist_creating_comment", 
            task_id=task_id, 
            project_id=project_id
        )
        
        payload = {"content": content}
        if task_id:
            payload["task_id"] = task_id
        if project_id:
            payload["project_id"] = project_id
            
        data = await self._request("POST", "/comments", json=payload)
        logger.info("todoist_comment_created", comment_id=data.get("id"))
        return data

    async def close(self):
        """Close the underlying HTTP/2 connection pool."""
        await self._client.aclose()


# ── Connector ────────────────────────────────────────────────────────

class TodoistConnector(BaseConnector):
    """
    Todoist integration block — manage projects and tasks.

    Provides an async-only client via `self.client`.
    """

    @property
    def name(self) -> str:
        return "todoist"

    @property
    def icon(self) -> str:
        return "✅"

    @property
    def description(self) -> str:
        return "Manage Todoist projects and tasks"

    def __init__(self):
        self._client: AsyncTodoistClient | None = None

    @property
    def client(self) -> AsyncTodoistClient:
        """Get the async Todoist client. Lazy-initializes on first access."""
        if self._client is None:
            token = os.environ.get("TODOIST_API_TOKEN", "")
            if not token:
                raise TodoistError("TODOIST_API_TOKEN environment variable is not set")
            self._client = AsyncTodoistClient(access_token=token)
        return self._client

    async def setup(self) -> None:
        """Pre-initialize the client."""
        _ = self.client

    async def teardown(self) -> None:
        """Close the HTTP/2 connection pool."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """Check Todoist API connectivity."""
        try:
            await self.client.get_projects()
            return True
        except Exception:
            return False
