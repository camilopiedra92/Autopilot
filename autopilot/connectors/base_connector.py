"""
BaseConnector — Abstract base class for all integration blocks.

Every connector in the platform extends this class. Connectors are
reusable integration pieces (like Autopilot nodes) that workflows can
declaratively depend on.

Usage:
    class MyConnector(BaseConnector):
        name = "my_service"
        icon = "🔌"
        description = "Connects to My Service API"

        async def setup(self) -> None:
            self._client = MyServiceClient()

        async def health_check(self) -> bool:
            return await self._client.ping()
"""

from __future__ import annotations

import structlog
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ConnectorInfo(BaseModel):
    """Summary info for listing connectors in the dashboard/API."""

    name: str
    icon: str
    description: str
    healthy: bool = True
    connected_workflows: list[str] = Field(default_factory=list)


class BaseConnector(ABC):
    """
    Abstract base class for all integration connectors.

    Subclasses MUST define:
      - name: str — Unique identifier (e.g. "gmail", "ynab")
      - icon: str — Emoji for display
      - description: str — What this connector does

    Subclasses MAY override:
      - setup(): One-time initialization (auth, client creation)
      - teardown(): Cleanup (close connections)
      - health_check(): Verify the connection is alive
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique connector identifier (e.g. 'gmail', 'ynab')."""
        ...

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji icon for display (e.g. '📧', '💰')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this connector does."""
        ...

    # ── Lifecycle Hooks ──────────────────────────────────────────────

    async def setup(self) -> None:
        """Called once when the connector is registered. Override for initialization."""
        pass

    async def teardown(self) -> None:
        """Called when the platform shuts down. Override for cleanup."""
        pass

    async def health_check(self) -> bool:
        """Check if the connector is healthy and ready to use."""
        return True

    # ── Info ──────────────────────────────────────────────────────────

    def get_info(self, connected_workflows: list[str] | None = None) -> ConnectorInfo:
        """Return summary info for this connector."""
        return ConnectorInfo(
            name=self.name,
            icon=self.icon,
            description=self.description,
            connected_workflows=connected_workflows or [],
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
