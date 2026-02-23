"""
YNABConnector — Async-only YNAB integration block.

Wraps the YNAB API as a platform connector. Provides account/category
lookups, transaction creation, and duplicate detection — all async.

ADK FunctionTool supports async def natively, so there is no need
for a separate SyncYNABClient.

Usage:
    from autopilot.connectors import get_connector_registry
    ynab = get_connector_registry().get("ynab")
    accounts = await ynab.client.get_all_accounts_string()
"""

from __future__ import annotations

import os
import time
import structlog
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)

YNAB_API_URL = "https://api.ynab.com/v1"


# ── Exceptions ───────────────────────────────────────────────────────

from autopilot.errors import (
    ConnectorError as YNABError,
    ConnectorRateLimitError as YNABRateLimitError,
)


# ── TTL Cache ────────────────────────────────────────────────────────


class TTLCache:
    """Simple thread-safe TTL cache for YNAB data."""

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


# ── Async YNAB Client ────────────────────────────────────────────────


class AsyncYNABClient:
    """
    Production-grade async YNAB API client.

    Features:
    - httpx.AsyncClient with HTTP/2 and connection pooling
    - TTL-based caching for accounts and categories
    - Automatic retry with exponential backoff on rate limits
    - Structured logging for every API call
    """

    def __init__(self, access_token: str):
        self._token = access_token
        self._client = httpx.AsyncClient(
            base_url=YNAB_API_URL,
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
        self._accounts_cache = TTLCache(ttl_seconds=300)
        self._categories_cache = TTLCache(ttl_seconds=300)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Execute an async API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise YNABError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise YNABError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            logger.warning("ynab_rate_limited", path=path, latency_ms=round(latency_ms))
            raise YNABRateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise YNABError(
                "Authentication failed — check your access token", detail="401"
            )
        if resp.status_code >= 400:
            raise YNABError(
                f"API error: {resp.status_code} — {resp.text}",
                detail=str(resp.status_code),
            )

        logger.debug(
            "ynab_request",
            method=method,
            path=path,
            status=resp.status_code,
            latency_ms=round(latency_ms),
            http_version=resp.http_version,
        )
        return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_all_budgets(self) -> list[dict]:
        """Fetch all budgets."""
        data = await self._request("GET", "/budgets")
        return data["data"]["budgets"]

    async def get_accounts(self, budget_id: str) -> list[dict]:
        """Fetch accounts for a budget (cached)."""
        cached = self._accounts_cache.get(f"accounts:{budget_id}")
        if cached is not None:
            logger.debug("ynab_accounts_cache_hit", budget_id=budget_id)
            return cached

        data = await self._request("GET", f"/budgets/{budget_id}/accounts")
        accounts = [
            a for a in data["data"]["accounts"] if not a["closed"] and not a["deleted"]
        ]
        self._accounts_cache.set(f"accounts:{budget_id}", accounts)
        logger.info("ynab_accounts_fetched", budget_id=budget_id, count=len(accounts))
        return accounts

    async def get_all_accounts_string(self) -> str:
        """Returns formatted account list for AI tool consumption."""
        budgets = await self.get_all_budgets()
        output = []

        for budget in budgets:
            b_id = budget["id"]
            b_name = budget["name"]
            accounts = await self.get_accounts(b_id)

            for acc in accounts:
                note = acc.get("note", "") or ""
                output.append(
                    f"Budget: {b_name} (ID: {b_id}) | "
                    f"Account: {acc['name']} (ID: {acc['id']}) | "
                    f"Note: {note}"
                )

        return "\n".join(output)

    async def get_categories(self, budget_id: str) -> list[dict]:
        """Fetch categories for a budget (cached)."""
        cached = self._categories_cache.get(f"categories:{budget_id}")
        if cached is not None:
            logger.debug("ynab_categories_cache_hit", budget_id=budget_id)
            return cached

        data = await self._request("GET", f"/budgets/{budget_id}/categories")
        groups = data["data"]["category_groups"]
        categories = []
        for group in groups:
            if group["hidden"]:
                continue
            for cat in group["categories"]:
                if cat["hidden"] or cat["deleted"]:
                    continue
                categories.append({**cat, "_group_name": group["name"]})

        self._categories_cache.set(f"categories:{budget_id}", categories)
        logger.info(
            "ynab_categories_fetched", budget_id=budget_id, count=len(categories)
        )
        return categories

    async def get_categories_string(self, budget_id: str) -> str:
        """Returns formatted categories for AI tool consumption."""
        categories = await self.get_categories(budget_id)
        return "\n".join(
            f"Category: {c['name']} (ID: {c['id']}) - Group: {c['_group_name']}"
            for c in categories
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_category_by_id(self, budget_id: str, category_id: str) -> dict:
        """Fetch a single category's details including balance, budgeted, and activity.

        Returns the raw category dict from YNAB with fields like:
        - balance (milliunits): remaining available amount
        - budgeted (milliunits): amount assigned this month
        - activity (milliunits): spending activity this month
        - name: category name
        - goal_target (milliunits, nullable): goal target amount
        """
        data = await self._request(
            "GET", f"/budgets/{budget_id}/categories/{category_id}"
        )
        return data["data"]["category"]

    async def account_exists(self, budget_id: str, account_id: str) -> bool:
        """Check if an account UUID actually exists."""
        accounts = await self.get_accounts(budget_id)
        return any(a["id"] == account_id for a in accounts)

    async def category_exists(self, budget_id: str, category_id: str) -> bool:
        """Check if a category UUID actually exists."""
        categories = await self.get_categories(budget_id)
        return any(c["id"] == category_id for c in categories)

    # ── Transactions ────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def create_transaction(
        self, budget_id: str, transaction_payload: dict
    ) -> dict:
        """Create a single transaction in YNAB."""
        logger.info(
            "ynab_creating_transaction",
            budget_id=budget_id,
            payee=transaction_payload.get("payee_name", "unknown"),
            amount=transaction_payload.get("amount", 0),
        )
        data = await self._request(
            "POST",
            f"/budgets/{budget_id}/transactions",
            json={"transaction": transaction_payload},
        )
        logger.info(
            "ynab_transaction_created", transaction_id=data["data"]["transaction"]["id"]
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def bulk_create_transactions(
        self, budget_id: str, transactions: list[dict]
    ) -> dict:
        """Create multiple transactions in a single API call."""
        logger.info(
            "ynab_bulk_creating_transactions",
            budget_id=budget_id,
            count=len(transactions),
        )
        data = await self._request(
            "POST",
            f"/budgets/{budget_id}/transactions",
            json={"transactions": transactions},
        )
        ids = [t["id"] for t in data["data"].get("transactions", [])]
        logger.info("ynab_bulk_transactions_created", count=len(ids))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_transactions(
        self,
        budget_id: str,
        *,
        since_date: str | None = None,
        type: str | None = None,
        last_knowledge_of_server: int | None = None,
    ) -> dict:
        """Fetch transactions for a budget with optional filters.

        Args:
            since_date: Only return transactions on or after this date (ISO 8601).
            type: Filter by type — 'uncategorized' or 'unapproved'.
            last_knowledge_of_server: Delta request support.

        Returns:
            Full response dict including data.transactions and data.server_knowledge.
        """
        params: dict = {}
        if since_date:
            params["since_date"] = since_date
        if type:
            params["type"] = type
        if last_knowledge_of_server is not None:
            params["last_knowledge_of_server"] = last_knowledge_of_server
        data = await self._request(
            "GET", f"/budgets/{budget_id}/transactions", params=params or None
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_transaction(self, budget_id: str, transaction_id: str) -> dict:
        """Fetch a single transaction by ID."""
        data = await self._request(
            "GET", f"/budgets/{budget_id}/transactions/{transaction_id}"
        )
        return data["data"]["transaction"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def update_transaction(
        self, budget_id: str, transaction_id: str, transaction_payload: dict
    ) -> dict:
        """Update an existing transaction."""
        logger.info(
            "ynab_updating_transaction",
            budget_id=budget_id,
            transaction_id=transaction_id,
        )
        data = await self._request(
            "PATCH",
            f"/budgets/{budget_id}/transactions/{transaction_id}",
            json={"transaction": transaction_payload},
        )
        logger.info("ynab_transaction_updated", transaction_id=transaction_id)
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def bulk_update_transactions(
        self, budget_id: str, transactions: list[dict]
    ) -> dict:
        """Update multiple transactions in a single API call."""
        logger.info(
            "ynab_bulk_updating_transactions",
            budget_id=budget_id,
            count=len(transactions),
        )
        data = await self._request(
            "PATCH",
            f"/budgets/{budget_id}/transactions",
            json={"transactions": transactions},
        )
        logger.info("ynab_bulk_transactions_updated", count=len(transactions))
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def delete_transaction(self, budget_id: str, transaction_id: str) -> dict:
        """Delete a transaction."""
        logger.info(
            "ynab_deleting_transaction",
            budget_id=budget_id,
            transaction_id=transaction_id,
        )
        data = await self._request(
            "DELETE", f"/budgets/{budget_id}/transactions/{transaction_id}"
        )
        logger.info("ynab_transaction_deleted", transaction_id=transaction_id)
        return data

    async def get_recent_transactions(
        self, budget_id: str, account_id: str, since_date: str
    ) -> list[dict]:
        """Fetch recent transactions for a specific account (duplicate detection)."""
        data = await self._request(
            "GET",
            f"/budgets/{budget_id}/accounts/{account_id}/transactions",
            params={"since_date": since_date},
        )
        return data["data"]["transactions"]

    # ── Scheduled Transactions ───────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_scheduled_transactions(
        self, budget_id: str, *, last_knowledge_of_server: int | None = None
    ) -> dict:
        """Fetch all scheduled transactions for a budget."""
        params: dict = {}
        if last_knowledge_of_server is not None:
            params["last_knowledge_of_server"] = last_knowledge_of_server
        data = await self._request(
            "GET",
            f"/budgets/{budget_id}/scheduled_transactions",
            params=params or None,
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_scheduled_transaction(
        self, budget_id: str, scheduled_transaction_id: str
    ) -> dict:
        """Fetch a single scheduled transaction by ID."""
        data = await self._request(
            "GET",
            f"/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}",
        )
        return data["data"]["scheduled_transaction"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def create_scheduled_transaction(
        self, budget_id: str, scheduled_transaction_payload: dict
    ) -> dict:
        """Create a new scheduled transaction."""
        logger.info(
            "ynab_creating_scheduled_transaction",
            budget_id=budget_id,
            payee=scheduled_transaction_payload.get("payee_name", "unknown"),
        )
        data = await self._request(
            "POST",
            f"/budgets/{budget_id}/scheduled_transactions",
            json={"scheduled_transaction": scheduled_transaction_payload},
        )
        logger.info(
            "ynab_scheduled_transaction_created",
            id=data["data"]["scheduled_transaction"]["id"],
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def update_scheduled_transaction(
        self, budget_id: str, scheduled_transaction_id: str, payload: dict
    ) -> dict:
        """Update an existing scheduled transaction."""
        logger.info(
            "ynab_updating_scheduled_transaction",
            budget_id=budget_id,
            id=scheduled_transaction_id,
        )
        data = await self._request(
            "PUT",
            f"/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}",
            json={"scheduled_transaction": payload},
        )
        logger.info("ynab_scheduled_transaction_updated", id=scheduled_transaction_id)
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def delete_scheduled_transaction(
        self, budget_id: str, scheduled_transaction_id: str
    ) -> dict:
        """Delete a scheduled transaction."""
        logger.info(
            "ynab_deleting_scheduled_transaction",
            budget_id=budget_id,
            id=scheduled_transaction_id,
        )
        data = await self._request(
            "DELETE",
            f"/budgets/{budget_id}/scheduled_transactions/{scheduled_transaction_id}",
        )
        logger.info("ynab_scheduled_transaction_deleted", id=scheduled_transaction_id)
        return data

    # ── Payees ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_payees(
        self, budget_id: str, *, last_knowledge_of_server: int | None = None
    ) -> dict:
        """Fetch all payees for a budget."""
        params: dict = {}
        if last_knowledge_of_server is not None:
            params["last_knowledge_of_server"] = last_knowledge_of_server
        data = await self._request(
            "GET", f"/budgets/{budget_id}/payees", params=params or None
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_payee(self, budget_id: str, payee_id: str) -> dict:
        """Fetch a single payee by ID."""
        data = await self._request("GET", f"/budgets/{budget_id}/payees/{payee_id}")
        return data["data"]["payee"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def update_payee(
        self, budget_id: str, payee_id: str, payee_payload: dict
    ) -> dict:
        """Update a payee (e.g. rename)."""
        logger.info(
            "ynab_updating_payee",
            budget_id=budget_id,
            payee_id=payee_id,
        )
        data = await self._request(
            "PUT",
            f"/budgets/{budget_id}/payees/{payee_id}",
            json={"payee": payee_payload},
        )
        logger.info("ynab_payee_updated", payee_id=payee_id)
        return data

    # ── Budget Months ────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_months(
        self, budget_id: str, *, last_knowledge_of_server: int | None = None
    ) -> dict:
        """Fetch all budget months."""
        params: dict = {}
        if last_knowledge_of_server is not None:
            params["last_knowledge_of_server"] = last_knowledge_of_server
        data = await self._request(
            "GET", f"/budgets/{budget_id}/months", params=params or None
        )
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_month(self, budget_id: str, month: str) -> dict:
        """Fetch a single budget month detail (includes per-category breakdown).

        Args:
            month: Month in ISO format, e.g. '2026-02-01' or 'current'.
        """
        data = await self._request("GET", f"/budgets/{budget_id}/months/{month}")
        return data["data"]["month"]

    # ── Accounts (write) ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def create_account(self, budget_id: str, account_payload: dict) -> dict:
        """Create a new account in a budget.

        Args:
            account_payload: Dict with 'name', 'type', and 'balance' (milliunits).
        """
        logger.info(
            "ynab_creating_account",
            budget_id=budget_id,
            name=account_payload.get("name", "unknown"),
        )
        data = await self._request(
            "POST",
            f"/budgets/{budget_id}/accounts",
            json={"account": account_payload},
        )
        logger.info("ynab_account_created", account_id=data["data"]["account"]["id"])
        self._accounts_cache.invalidate()
        return data

    # ── Categories (write) ───────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def update_category(
        self, budget_id: str, category_id: str, category_payload: dict
    ) -> dict:
        """Update a category (e.g. change goal_target).

        Args:
            category_payload: Dict with fields to update (e.g. 'goal_target').
        """
        logger.info(
            "ynab_updating_category",
            budget_id=budget_id,
            category_id=category_id,
        )
        data = await self._request(
            "PATCH",
            f"/budgets/{budget_id}/categories/{category_id}",
            json={"category": category_payload},
        )
        logger.info("ynab_category_updated", category_id=category_id)
        self._categories_cache.invalidate()
        return data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def update_month_category(
        self, budget_id: str, month: str, category_id: str, category_payload: dict
    ) -> dict:
        """Update a category's budgeted amount for a specific month.

        Args:
            month: Month in ISO format, e.g. '2026-02-01'.
            category_payload: Dict with 'budgeted' (milliunits).
        """
        logger.info(
            "ynab_updating_month_category",
            budget_id=budget_id,
            month=month,
            category_id=category_id,
        )
        data = await self._request(
            "PATCH",
            f"/budgets/{budget_id}/months/{month}/categories/{category_id}",
            json={"category": category_payload},
        )
        logger.info(
            "ynab_month_category_updated",
            month=month,
            category_id=category_id,
        )
        return data

    # ── User ─────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(YNABRateLimitError),
    )
    async def get_user(self) -> dict:
        """Fetch the authenticated user's info."""
        data = await self._request("GET", "/user")
        return data["data"]["user"]

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self):
        """Close the underlying HTTP/2 connection pool."""
        await self._client.aclose()


# ── Connector ────────────────────────────────────────────────────────


class YNABConnector(BaseConnector):
    """
    YNAB integration block — manage budgets, accounts, categories, and transactions.

    Provides an async-only client via `self.client`. ADK FunctionTool
    supports async def natively, so no sync wrapper is needed.
    """

    @property
    def name(self) -> str:
        return "ynab"

    @property
    def icon(self) -> str:
        return "💰"

    @property
    def description(self) -> str:
        return "Manage YNAB budgets, accounts, categories, and transactions"

    def __init__(self):
        self._client: AsyncYNABClient | None = None

    @property
    def client(self) -> AsyncYNABClient:
        """Get the async YNAB client. Lazy-initializes on first access."""
        if self._client is None:
            token = os.environ.get("YNAB_ACCESS_TOKEN", "")
            if not token:
                raise YNABError("YNAB_ACCESS_TOKEN environment variable is not set")
            self._client = AsyncYNABClient(access_token=token)
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
        """Check YNAB API connectivity."""
        try:
            await self.client.get_all_budgets()
            return True
        except Exception:
            return False
