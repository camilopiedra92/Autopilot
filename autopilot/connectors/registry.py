"""
ConnectorRegistry — Auto-discovery and management for integration blocks.

The registry maintains all available connectors and provides lifecycle
management (setup, teardown, health checks). Similar in spirit to
WorkflowRegistry but for reusable integration pieces.

Note: This registry uses a process-global singleton pattern. It assumes a
single-process execution environment (like Cloud Run or standard uvicorn workers).
It is NOT thread-safe for concurrent writes during runtime, but safe for
asyncio event loops within a single process.

Usage:
    from autopilot.connectors import get_connector_registry

    registry = get_connector_registry()
    registry.register(GmailConnector())
    registry.register(YNABConnector())

    gmail = registry.get("gmail")
    emails = gmail.get_unread_emails("alertas@bancolombia.com.co")

    # Health check all connectors
    health = await registry.health_check_all()
    # {"gmail": True, "ynab": True, "pubsub": False}
"""

import structlog

from autopilot.connectors.base_connector import BaseConnector, ConnectorInfo

logger = structlog.get_logger(__name__)


class ConnectorRegistry:
    """
    Registry for all available integration connectors.

    Provides:
      - Registration and lookup by name
      - Lifecycle management (setup_all, teardown_all)
      - Health check aggregation
      - Listing for dashboard/API
    """

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        """Register a connector by its name."""
        if connector.name in self._connectors:
            logger.warning(
                "connector_already_registered",
                name=connector.name,
                replacing=True,
            )
        self._connectors[connector.name] = connector
        logger.info(
            "connector_registered",
            name=connector.name,
            icon=connector.icon,
            description=connector.description,
        )

    def get(self, name: str) -> BaseConnector:
        """Get a connector by name. Raises KeyError if not found."""
        if name not in self._connectors:
            available = list(self._connectors.keys())
            raise KeyError(f"Connector '{name}' not found. Available: {available}")
        return self._connectors[name]

    def list_all(self) -> list[ConnectorInfo]:
        """List all registered connectors with their info."""
        return [c.get_info() for c in self._connectors.values()]

    async def setup_all(self) -> None:
        """Initialize all registered connectors."""
        for name, connector in self._connectors.items():
            try:
                await connector.setup()
                logger.info("connector_setup_complete", name=name)
            except Exception as e:
                logger.error("connector_setup_failed", name=name, error=str(e))

    async def teardown_all(self) -> None:
        """Graceful shutdown of all connectors."""
        for name, connector in self._connectors.items():
            try:
                await connector.teardown()
                logger.info("connector_teardown_complete", name=name)
            except Exception as e:
                logger.error("connector_teardown_failed", name=name, error=str(e))

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all connectors."""
        results = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = await connector.health_check()
            except Exception:
                results[name] = False
        return results

    @property
    def names(self) -> list[str]:
        """List of registered connector names."""
        return list(self._connectors.keys())

    def __len__(self) -> int:
        return len(self._connectors)

    def __contains__(self, name: str) -> bool:
        return name in self._connectors


# ── Singleton ────────────────────────────────────────────────────────

_registry: ConnectorRegistry | None = None


def get_connector_registry() -> ConnectorRegistry:
    """Singleton accessor for the ConnectorRegistry."""
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
        _auto_register_connectors(_registry)
    return _registry


def _auto_register_connectors(registry: ConnectorRegistry) -> None:
    """Auto-register all built-in connectors."""
    from autopilot.connectors.gmail_connector import GmailConnector
    from autopilot.connectors.ynab_connector import YNABConnector
    from autopilot.connectors.pubsub_connector import PubSubConnector
    from autopilot.connectors.todoist_connector import TodoistConnector
    from autopilot.connectors.airtable_connector import AirtableConnector
    from autopilot.connectors.telegram_connector import TelegramConnector

    registry.register(GmailConnector())
    registry.register(YNABConnector())
    registry.register(PubSubConnector())
    registry.register(TodoistConnector())
    registry.register(AirtableConnector())
    registry.register(TelegramConnector())

    logger.info(
        "connectors_auto_registered",
        count=len(registry),
        names=registry.names,
    )
