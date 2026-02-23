"""
Shared pytest fixtures for the Bank→YNAB v4 test suite.

Provides:
  - settings: mocked Settings with test values
  - sample_parsed_email: ParsedEmail fixture
  - sample_matched_account: MatchedAccount fixture
  - sample_categorized_tx: CategorizedTransaction fixture
  - sample_transaction: complete Transaction fixture
  - mock_ynab_client: mocked AsyncYNABClient
  - golden_emails: loaded golden test fixtures
  - make_callback_context: factory for mocked CallbackContext
  - make_llm_request: factory for mocked LlmRequest
  - make_llm_response: factory for mocked LlmResponse
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Load .env FIRST so real keys are available for integration tests,
# then set fallback defaults for CI (where no .env exists).
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set default environment variables only when real keys are not present."""
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    if not os.environ.get("GOOGLE_GENAI_API_KEY"):
        monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key")
    if not os.environ.get("YNAB_ACCESS_TOKEN"):
        monkeypatch.setenv("YNAB_ACCESS_TOKEN", "test-token")


# ── Pydantic Model Fixtures ──────────────────────────────────────────


@pytest.fixture
def sample_parsed_email():
    """A ParsedEmail fixture representing a standard restaurant purchase."""
    from workflows.bank_to_ynab.models import ParsedEmail

    return ParsedEmail(
        date="2026-02-18",
        payee="Restaurante El Cielo",
        amount=-50000,
        card_suffix="52e0",
        memo="Compra con tarjeta terminada en 52e0",
        raw_email_snippet="Bancolombia le informa compra por $50.000...",
    )


@pytest.fixture
def sample_matched_account():
    """A MatchedAccount fixture with valid UUID-format IDs."""
    from workflows.bank_to_ynab.models import MatchedAccount

    return MatchedAccount(
        budget_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        budget_name="Mi Presupuesto",
        account_id="f1e2d3c4-b5a6-7890-abcd-ef0987654321",
        account_name="Visa Infinite Bancolombia",
        match_confidence="high",
        match_reasoning="Card suffix 52e0 found in account notes",
    )


@pytest.fixture
def sample_categorized_tx():
    """A CategorizedTransaction fixture for a dining purchase."""
    from workflows.bank_to_ynab.models import CategorizedTransaction

    return CategorizedTransaction(
        category_id="c1d2e3f4-a5b6-7890-abcd-ef1122334455",
        category_name="Dining Out",
        category_reasoning="Payee 'Restaurante El Cielo' matches dining category",
    )


@pytest.fixture
def sample_transaction(
    sample_parsed_email, sample_matched_account, sample_categorized_tx
):
    """A complete Transaction fixture, merging all pipeline stage outputs."""
    from workflows.bank_to_ynab.models import Transaction

    return Transaction(
        date=sample_parsed_email.date,
        payee=sample_parsed_email.payee,
        amount=sample_parsed_email.amount,
        memo=sample_parsed_email.memo,
        budget_id=sample_matched_account.budget_id,
        account_id=sample_matched_account.account_id,
        category_id=sample_categorized_tx.category_id,
        match_reasoning=sample_matched_account.match_reasoning,
        category_reasoning=sample_categorized_tx.category_reasoning,
        match_confidence=sample_matched_account.match_confidence,
    )


# ── Golden Test Data ─────────────────────────────────────────────────


@pytest.fixture
def golden_emails():
    """Loads the golden email test fixtures from tests/fixtures/golden_emails.json."""
    fixtures_path = Path(__file__).parent / "fixtures" / "golden_emails.json"
    with open(fixtures_path) as f:
        return json.load(f)


# ── Mock YNAB Clients ────────────────────────────────────────────────


MOCK_BUDGETS = [
    {"id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "name": "Mi Presupuesto"}
]

MOCK_ACCOUNTS = [
    {
        "id": "f1e2d3c4-b5a6-7890-abcd-ef0987654321",
        "name": "Visa Infinite Bancolombia",
        "note": "terminada en 52e0",
        "closed": False,
        "deleted": False,
    },
    {
        "id": "a2b3c4d5-e6f7-8901-bcde-f12345678901",
        "name": "Mastercard Gold",
        "note": "terminada en 1234",
        "closed": False,
        "deleted": False,
    },
]

MOCK_CATEGORIES = [
    {
        "id": "c1d2e3f4-a5b6-7890-abcd-ef1122334455",
        "name": "Dining Out",
        "hidden": False,
        "deleted": False,
        "_group_name": "Everyday Expenses",
    },
    {
        "id": "d2e3f4a5-b6c7-8901-cdef-012345678901",
        "name": "Groceries",
        "hidden": False,
        "deleted": False,
        "_group_name": "Everyday Expenses",
    },
    {
        "id": "e3f4a5b6-c7d8-9012-def0-123456789012",
        "name": "Transportation",
        "hidden": False,
        "deleted": False,
        "_group_name": "Everyday Expenses",
    },
]


@pytest.fixture
def mock_ynab_client():
    """Mocked AsyncYNABClient with realistic test data."""
    client = AsyncMock()
    client.get_all_budgets = AsyncMock(return_value=MOCK_BUDGETS)
    client.get_accounts = AsyncMock(return_value=MOCK_ACCOUNTS)
    client.get_categories = AsyncMock(return_value=MOCK_CATEGORIES)
    client.get_all_accounts_string = AsyncMock(
        return_value="\n".join(
            f"Budget: {MOCK_BUDGETS[0]['name']} (ID: {MOCK_BUDGETS[0]['id']}) | "
            f"Account: {a['name']} (ID: {a['id']}) | Note: {a['note']}"
            for a in MOCK_ACCOUNTS
        )
    )
    client.get_categories_string = AsyncMock(
        return_value="\n".join(
            f"Category: {c['name']} (ID: {c['id']}) - Group: {c['_group_name']}"
            for c in MOCK_CATEGORIES
        )
    )
    client.account_exists = AsyncMock(return_value=True)
    client.category_exists = AsyncMock(return_value=True)
    client.create_transaction = AsyncMock(
        return_value={"data": {"transaction": {"id": "tx-mock-001"}}}
    )
    client.get_category_by_id = AsyncMock(
        return_value={
            "id": "c1d2e3f4-a5b6-7890-abcd-ef1122334455",
            "name": "Dining Out",
            "budgeted": 500000,  # 500 currency units in milliunits
            "activity": -320000,  # -320 currency units spent
            "balance": 180000,  # 180 currency units remaining
            "goal_target": None,
            "hidden": False,
            "deleted": False,
        }
    )
    client.get_recent_transactions = AsyncMock(return_value=[])
    # New expanded API methods
    client.get_transactions = AsyncMock(
        return_value={"data": {"transactions": [], "server_knowledge": 100}}
    )
    client.get_transaction = AsyncMock(
        return_value={"id": "tx-mock-001", "payee_name": "Test", "amount": -50000}
    )
    client.bulk_create_transactions = AsyncMock(
        return_value={"data": {"transactions": [], "transaction_ids": []}}
    )
    client.update_transaction = AsyncMock(
        return_value={"data": {"transaction": {"id": "tx-mock-001"}}}
    )
    client.bulk_update_transactions = AsyncMock(
        return_value={"data": {"transactions": []}}
    )
    client.delete_transaction = AsyncMock(
        return_value={"data": {"transaction": {"id": "tx-mock-001", "deleted": True}}}
    )
    client.get_scheduled_transactions = AsyncMock(
        return_value={"data": {"scheduled_transactions": [], "server_knowledge": 50}}
    )
    client.get_scheduled_transaction = AsyncMock(
        return_value={"id": "stx-mock-001", "payee_name": "Netflix"}
    )
    client.create_scheduled_transaction = AsyncMock(
        return_value={"data": {"scheduled_transaction": {"id": "stx-mock-001"}}}
    )
    client.update_scheduled_transaction = AsyncMock(
        return_value={"data": {"scheduled_transaction": {"id": "stx-mock-001"}}}
    )
    client.delete_scheduled_transaction = AsyncMock(
        return_value={"data": {"scheduled_transaction": {"id": "stx-mock-001"}}}
    )
    client.get_payees = AsyncMock(
        return_value={"data": {"payees": [], "server_knowledge": 75}}
    )
    client.get_payee = AsyncMock(
        return_value={"id": "payee-mock-001", "name": "Amazon"}
    )
    client.update_payee = AsyncMock(
        return_value={"data": {"payee": {"id": "payee-mock-001"}}}
    )
    client.get_months = AsyncMock(
        return_value={"data": {"months": [], "server_knowledge": 88}}
    )
    client.get_month = AsyncMock(
        return_value={"month": "2026-02-01", "income": 5000000, "categories": []}
    )
    client.create_account = AsyncMock(
        return_value={"data": {"account": {"id": "acc-mock-001"}}}
    )
    client.update_category = AsyncMock(
        return_value={"data": {"category": {"id": "cat-mock-001"}}}
    )
    client.update_month_category = AsyncMock(
        return_value={"data": {"category": {"id": "cat-mock-001"}}}
    )
    client.get_user = AsyncMock(return_value={"id": "user-mock-001"})
    client.close = AsyncMock()
    return client


# ── Callback Context Factories ───────────────────────────────────────


@pytest.fixture
def make_callback_context():
    """Factory fixture: creates a mocked CallbackContext."""

    def _make(agent_name: str = "test_agent"):
        ctx = MagicMock()
        ctx.agent_name = agent_name
        return ctx

    return _make


@pytest.fixture
def make_llm_request():
    """Factory fixture: creates a mocked LlmRequest with user text."""

    def _make(user_text: str):
        from google.genai import types
        from google.adk.models import LlmRequest

        content = types.Content(
            role="user",
            parts=[types.Part(text=user_text)],
        )
        req = MagicMock(spec=LlmRequest)
        req.contents = [content]
        req.tools = []
        return req

    return _make


@pytest.fixture
def make_llm_response():
    """Factory fixture: creates a mocked LlmResponse with model text."""

    def _make(model_text: str):
        from google.genai import types
        from google.adk.models import LlmResponse

        content = types.Content(
            role="model",
            parts=[types.Part(text=model_text)],
        )
        resp = MagicMock(spec=LlmResponse)
        resp.content = content
        return resp

    return _make


# ── Mock Telegram Client ─────────────────────────────────────────────


@pytest.fixture
def mock_telegram_client():
    """Mocked AsyncTelegramClient for notification tests."""
    client = AsyncMock()
    client.send_message = AsyncMock(return_value={"message_id": 42})
    client.close = AsyncMock()
    return client
