"""
Connectors — Reusable integration blocks (Autopilot-style nodes).

Each connector wraps an external service (Gmail, YNAB, Pub/Sub, etc.)
and can be composed into any workflow. The ConnectorRegistry auto-discovers
and manages all available connectors.

Usage:
    from autopilot.connectors import get_connector_registry

    registry = get_connector_registry()
    gmail = registry.get("gmail")
    ynab = registry.get("ynab")
"""

from autopilot.connectors.base_connector import BaseConnector, ConnectorInfo
from autopilot.connectors.registry import ConnectorRegistry, get_connector_registry
from autopilot.connectors.gmail_connector import GmailConnector
from autopilot.connectors.ynab_connector import YNABConnector, AsyncYNABClient
from autopilot.connectors.pubsub_connector import PubSubConnector
from autopilot.connectors.todoist_connector import TodoistConnector, AsyncTodoistClient
from autopilot.connectors.airtable_connector import (
    AirtableConnector,
    AsyncAirtableClient,
)
from autopilot.connectors.telegram_connector import (
    TelegramConnector,
    AsyncTelegramClient,
)
from autopilot.connectors.polymarket_connector import (
    PolymarketConnector,
    AsyncPolymarketClient,
)

__all__ = [
    "BaseConnector",
    "ConnectorInfo",
    "ConnectorRegistry",
    "get_connector_registry",
    "GmailConnector",
    "YNABConnector",
    "AsyncYNABClient",
    "PubSubConnector",
    "TodoistConnector",
    "AsyncTodoistClient",
    "AirtableConnector",
    "AsyncAirtableClient",
    "TelegramConnector",
    "AsyncTelegramClient",
    "PolymarketConnector",
    "AsyncPolymarketClient",
]
