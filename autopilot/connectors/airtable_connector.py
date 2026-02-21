"""
AirtableConnector — Async-only Airtable integration block.

Wraps the Airtable REST API as a platform connector. Provides record fetching,
creation, updating, and deletion — all async.

Usage:
    from autopilot.connectors import get_connector_registry
    airtable = get_connector_registry().get("airtable")
    records = await airtable.client.get_records("base_id", "table_id")
"""

from __future__ import annotations

import os
import time
import structlog
import httpx
from typing import Any, AsyncIterator, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from autopilot.connectors.base_connector import BaseConnector

logger = structlog.get_logger(__name__)

AIRTABLE_API_URL = "https://api.airtable.com/v0"

# ── Exceptions ───────────────────────────────────────────────────────

from autopilot.errors import ConnectorError as AirtableError, ConnectorRateLimitError as AirtableRateLimitError


# ── Async Airtable Client ─────────────────────────────────────────────

class AsyncAirtableClient:
    """
    Production-grade async Airtable API client.

    Features:
    - httpx.AsyncClient with HTTP/2 and connection pooling
    - Support for pagination in `get_records`
    - Automatic retry with exponential backoff on HTTP 429
    - Structured logging for every API call
    """

    def __init__(self, access_token: str):
        self._token = access_token
        self._client = httpx.AsyncClient(
            base_url=AIRTABLE_API_URL,
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

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Execute an async API request with error handling."""
        start = time.monotonic()
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise AirtableError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise AirtableError(f"Request timed out: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code == 429:
            logger.warning("airtable_rate_limited", path=path, latency_ms=round(latency_ms))
            raise AirtableRateLimitError("Rate limit exceeded")
        if resp.status_code == 401:
            raise AirtableError("Authentication failed — check your access token", detail="401")
        if resp.status_code >= 400:
            raise AirtableError(f"API error: {resp.status_code} — {resp.text}", detail=str(resp.status_code))

        logger.debug(
            "airtable_request",
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
        retry=retry_if_exception_type(AirtableRateLimitError),
    )
    async def get_records(
        self, base_id: str, table_id_or_name: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch records from a given base and table.
        Automatically handles pagination if an `offset` is returned.
        
        Args:
            base_id: The ID of the Airtable base.
            table_id_or_name: The ID or name of the table.
            params: Optional query parameters (e.g. view, formula, sort, fields).
        """
        all_records = []
        current_params = params.copy() if params else {}
        path = f"/{base_id}/{table_id_or_name}"

        while True:
            logger.debug("airtable_fetching_records", base_id=base_id, table_id=table_id_or_name)
            data = await self._request("GET", path, params=current_params)
            
            records = data.get("records", [])
            all_records.extend(records)
            
            offset = data.get("offset")
            if not offset:
                break
            
            current_params["offset"] = offset

        logger.info("airtable_records_fetched", base_id=base_id, table_id=table_id_or_name, count=len(all_records))
        return all_records

    async def get_records_string(
        self, base_id: str, table_id_or_name: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Returns formatted string output of records for easy AI consumption."""
        records = await self.get_records(base_id, table_id_or_name, params)
        output = [
            f"Record ID: {record['id']} | Fields: {record.get('fields', {})}"
            for record in records
        ]
        return "\\n".join(output) if output else "No records found."

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(AirtableRateLimitError),
    )
    async def create_records(self, base_id: str, table_id_or_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create up to 10 records per request in Airtable.
        
        Args:
            base_id: The ID of the Airtable base.
            table_id_or_name: The ID or name of the table.
            records: List of dictionaries, e.g. [{"fields": {"Name": "Task 1", "Status": "Todo"}}]
        """
        if not records:
            return []
            
        path = f"/{base_id}/{table_id_or_name}"
        logger.info("airtable_creating_records", base_id=base_id, table_id=table_id_or_name, count=len(records))
        
        created_records = []
        # Airtable limits to 10 records per create request, so we chunk them
        for i in range(0, len(records), 10):
            chunk = records[i:i+10]
            payload = {"records": chunk}
            data = await self._request("POST", path, json=payload)
            response_records = data.get("records", [])
            created_records.extend(response_records)
            
        logger.info("airtable_records_created", base_id=base_id, table_id=table_id_or_name, total_created=len(created_records))
        return created_records

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(AirtableRateLimitError),
    )
    async def update_records(self, base_id: str, table_id_or_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update up to 10 records per request via PATCH.
        
        Args:
            base_id: The ID of the Airtable base.
            table_id_or_name: The ID or name of the table.
            records: List of dictionaries containing the record ID and fields to update.
                     Example: [{"id": "rec123", "fields": {"Status": "Done"}}]
        """
        if not records:
            return []
            
        path = f"/{base_id}/{table_id_or_name}"
        logger.info("airtable_updating_records", base_id=base_id, table_id=table_id_or_name, count=len(records))
        
        updated_records = []
        # Airtable limits to 10 records per update request
        for i in range(0, len(records), 10):
            chunk = records[i:i+10]
            payload = {"records": chunk}
            data = await self._request("PATCH", path, json=payload)
            response_records = data.get("records", [])
            updated_records.extend(response_records)
            
        logger.info("airtable_records_updated", base_id=base_id, table_id=table_id_or_name, total_updated=len(updated_records))
        return updated_records

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(AirtableRateLimitError),
    )
    async def delete_records(self, base_id: str, table_id_or_name: str, record_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Delete up to 10 records per request.
        
        Args:
            base_id: The ID of the Airtable base.
            table_id_or_name: The ID or name of the table.
            record_ids: List of record UUIDs to delete.
        """
        if not record_ids:
            return []
            
        path = f"/{base_id}/{table_id_or_name}"
        logger.info("airtable_deleting_records", base_id=base_id, table_id=table_id_or_name, count=len(record_ids))
        
        deleted_records = []
        # Airtable limits to 10 records per delete request via querystring array
        for i in range(0, len(record_ids), 10):
            chunk = record_ids[i:i+10]
            # Construct standard querystring: records[]=rec1&records[]=rec2
            params = [("records[]", rid) for rid in chunk]
            data = await self._request("DELETE", path, params=params)
            response_records = data.get("records", [])
            deleted_records.extend(response_records)
            
        logger.info("airtable_records_deleted", base_id=base_id, table_id=table_id_or_name, total_deleted=len(deleted_records))
        return deleted_records

    async def close(self):
        """Close the underlying HTTP/2 connection pool."""
        await self._client.aclose()


# ── Connector ────────────────────────────────────────────────────────

class AirtableConnector(BaseConnector):
    """
    Airtable integration block — manage bases, tables, and records.

    Provides an async-only client via `self.client`.
    """

    @property
    def name(self) -> str:
        return "airtable"

    @property
    def icon(self) -> str:
        return "📊"

    @property
    def description(self) -> str:
        return "Manage Airtable bases, tables, and records"

    def __init__(self):
        self._client: AsyncAirtableClient | None = None

    @property
    def client(self) -> AsyncAirtableClient:
        """Get the async Airtable client. Lazy-initializes on first access."""
        if self._client is None:
            token = os.environ.get("AIRTABLE_PERSONAL_ACCESS_TOKEN", "")
            if not token:
                raise AirtableError("AIRTABLE_PERSONAL_ACCESS_TOKEN environment variable is not set")
            self._client = AsyncAirtableClient(access_token=token)
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
        """
        Check Airtable API connectivity.
        NOTE: Airtable doesn't have an empty ping endpoint, but requesting WhoAmI
        with our token confirms network and auth validity.
        """
        try:
            # We use an ad-hoc request against meta/whoami to confirm token validity
            # See: https://airtable.com/developers/web/api/get-whoami
            await self.client._request("GET", "https://api.airtable.com/v0/meta/whoami")
            return True
        except Exception:
            return False
