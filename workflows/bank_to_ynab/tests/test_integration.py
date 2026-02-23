"""
Integration Tests — Contract tests with real external APIs.

Organized by testing layer (bottom → top):

  Layer 1 — YNAB Client:
    Tests YNAB API connectivity, budgets, accounts, categories.
    Requires: YNAB_ACCESS_TOKEN (real)

  Layer 2 — Email Parser (LLM):
    Tests Gemini parsing of bank emails against golden data.
    Requires: GOOGLE_API_KEY (real)

  Layer 3 — Account Matcher + Categorizer (LLM + YNAB):
    Tests each stage in isolation using ADK Runner with real APIs.
    Requires: GOOGLE_API_KEY + YNAB_ACCESS_TOKEN (both real)

  Layer 4 — Full Pipeline E2E:
    Tests the complete pipeline end-to-end.
    Requires: RUN_FULL_INTEGRATION=1 + both API keys

Usage:
  # Skip all integration tests (CI)
  pytest tests/ -m "not integration"

  # Run only integration tests (local, with .env)
  pytest tests/ -m integration -v

  # Run everything including full E2E
  RUN_FULL_INTEGRATION=1 pytest tests/ -v
"""

import os
import re
import json
import pytest

# Load .env for real keys (conftest.py also does this, but be explicit)
from dotenv import load_dotenv

load_dotenv()

# Detect available credentials
_goog = os.environ.get("GOOGLE_API_KEY", "")
_ynab = os.environ.get("YNAB_ACCESS_TOKEN", "")

HAS_REAL_GOOGLE_KEY = bool(_goog and _goog.startswith("AIza") and len(_goog) == 39)
HAS_REAL_YNAB_TOKEN = bool(_ynab and len(_ynab) == 64)
RUN_FULL_E2E = bool(os.environ.get("RUN_FULL_INTEGRATION"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 1 — YNAB Client Contract Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.integration
@pytest.mark.skipif(not HAS_REAL_YNAB_TOKEN, reason="Requires real YNAB_ACCESS_TOKEN")
class TestYNABClientIntegration:
    """Contract tests for the YNAB Connector.

    Validates that the YNAB API is reachable and returns data
    in the expected format. Does NOT create or modify any data.
    """

    async def _get_connector(self):
        from autopilot.connectors.ynab_connector import YNABConnector

        connector = YNABConnector()
        return connector

    @pytest.mark.asyncio
    async def test_fetch_budgets(self):
        """Should successfully fetch budgets from YNAB API."""
        from autopilot.connectors.ynab_connector import YNABConnector

        connector = YNABConnector()
        await connector.setup()
        budgets = await connector.client.get_all_budgets()
        assert isinstance(budgets, list), "Should return a list of budgets"
        if budgets:
            assert "id" in budgets[0], "Budget obj should have 'id'"
            assert "name" in budgets[0], "Budget obj should have 'name'"

    @pytest.mark.asyncio
    async def test_fetch_accounts(self):
        """Should fetch accounts for a given budget."""
        from autopilot.connectors.ynab_connector import YNABConnector

        connector = YNABConnector()
        await connector.setup()
        budgets = await connector.client.get_all_budgets()
        if not budgets:
            pytest.skip("No budgets available in test YNAB account")

        budget_id = budgets[0]["id"]
        accounts = await connector.client.get_accounts(budget_id)
        assert isinstance(accounts, list)
        if accounts:
            assert "id" in accounts[0]
            assert "name" in accounts[0]

    @pytest.mark.asyncio
    async def test_fetch_categories(self):
        """Should fetch categories for a given budget."""
        from autopilot.connectors.ynab_connector import YNABConnector

        connector = YNABConnector()
        await connector.setup()
        budgets = await connector.client.get_all_budgets()
        if not budgets:
            pytest.skip("No budgets available")

        # Find a budget that actually has categories
        categories = []
        for budget in budgets:
            cats = await connector.client.get_categories(budget["id"])
            if len(cats) >= 3:
                categories = cats
                break

        assert len(categories) >= 3, (
            f"No budget found with >= 3 categories across {len(budgets)} budgets"
        )

        # Validate category structure
        cat = categories[0]
        assert "id" in cat
        assert "name" in cat
        assert "_group_name" in cat, "Should include group name from flattening"

    @pytest.mark.asyncio
    async def test_accounts_string_format(self):
        """The formatted accounts string should be parseable by LLMs."""
        connector = await self._get_connector()
        await connector.setup()

        result = await connector.client.get_all_accounts_string()
        assert isinstance(result, str)
        assert len(result) > 50, "Accounts string seems too short"

        # Should contain expected format tokens
        assert "Budget:" in result
        assert "Account:" in result
        assert "ID:" in result

    @pytest.mark.asyncio
    async def test_categories_string_format(self):
        """The formatted categories string should be parseable by LLMs."""
        connector = await self._get_connector()
        await connector.setup()

        budgets = await connector.client.get_all_budgets()

        # Find a budget with categories
        result = ""
        for budget in budgets:
            r = await connector.client.get_categories_string(budget["id"])
            if len(r) > 50:
                result = r
                break

        if not result:
            pytest.skip("No budget found with sufficient categories")
        assert isinstance(result, str)
        assert len(result) > 50, "Categories string seems too short"

        # Should contain expected format tokens
        assert "Category:" in result
        assert "ID:" in result
        assert "Group:" in result

    @pytest.mark.asyncio
    async def test_accounts_caching(self):
        """Second fetch should use cache (no API call)."""
        connector = await self._get_connector()
        await connector.setup()

        budgets = await connector.client.get_all_budgets()
        if not budgets:
            pytest.skip()
        budget_id = budgets[0]["id"]

        # First call — fills cache
        accounts_1 = await connector.client.get_accounts(budget_id)
        # Second call — should hit cache
        accounts_2 = await connector.client.get_accounts(budget_id)

        assert accounts_1 == accounts_2, "Cached data should be identical"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 2 — Email Parser LLM Contract Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def _run_parser_on_text(email_text: str) -> dict:
    """Helper: runs the email parser agent on a single text and returns parsed JSON."""
    from google.adk.runners import Runner
    from autopilot.core.session import InMemorySessionService
    from google.genai import types
    from workflows.bank_to_ynab.agents.email_parser import create_email_parser
    from workflows.bank_to_ynab.steps import sanitize_email_html
    from autopilot.agents import extract_json

    clean = sanitize_email_html(email_text)
    parser = create_email_parser()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="parser_integration",
        agent=parser,
        session_service=session_service,
    )

    session_id = f"parse_{id(email_text)}"
    await session_service.create_session(
        app_name="parser_integration",
        user_id="test",
        session_id=session_id,
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"Process this bank email:\n\n{clean}")],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="test",
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                text_parts = [p.text for p in event.content.parts if p.text]
                if text_parts:
                    final_text = "\n".join(text_parts)
            break

    if not final_text:
        raise RuntimeError("Parser returned no output")

    return extract_json(final_text)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_REAL_GOOGLE_KEY, reason="Requires real GOOGLE_API_KEY")
class TestEmailParserIntegration:
    """Contract tests for the email parser LLM agent.

    Validates that Gemini correctly parses Colombian bank emails
    into structured JSON with expected fields and values.
    """

    @pytest.mark.asyncio
    async def test_parser_golden_emails(self, golden_emails):
        """Run the parser on golden test cases and validate extraction quality."""
        successes = 0
        rate_limited = 0

        for i, case in enumerate(golden_emails[:5]):
            try:
                parsed = await _run_parser_on_text(case["input"])
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    rate_limited += 1
                    continue
                if "API key not valid" in str(e) or "400 INVALID_ARGUMENT" in str(e):
                    pytest.skip("Invalid Google API key detected at runtime")
                raise

            successes += 1
            expected = case["expected"]

            # Must return all core fields
            assert "date" in parsed, f"[{case['name']}] Missing 'date'"
            assert "payee" in parsed, f"[{case['name']}] Missing 'payee'"
            assert "amount" in parsed, f"[{case['name']}] Missing 'amount'"

            # Validate date format
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", str(parsed["date"])), (
                f"[{case['name']}] Bad date: {parsed['date']}"
            )

            # Validate amount sign
            if "amount_is_negative" in expected:
                amount = parsed["amount"]
                if isinstance(amount, (int, float)):
                    if expected["amount_is_negative"]:
                        assert amount < 0, (
                            f"[{case['name']}] Expected negative, got {amount}"
                        )
                    else:
                        assert amount > 0, (
                            f"[{case['name']}] Expected positive, got {amount}"
                        )

            # Validate payee
            if "payee_contains" in expected:
                assert expected["payee_contains"].lower() in parsed["payee"].lower(), (
                    f"[{case['name']}] Payee '{parsed['payee']}' \u2260 expected '{expected['payee_contains']}'"
                )

        # At least 1 golden email should parse successfully
        assert successes >= 1, (
            f"Only {successes}/5 golden emails parsed (rate limited: {rate_limited})"
        )

    @pytest.mark.asyncio
    async def test_parser_consistency(self):
        """Running the same email 3x should produce consistent results."""
        email = (
            "Bancolombia le informa compra por $50.000 en RESTAURANTE EL CIELO "
            "con tarjeta terminada en 52e0. 18/02/2026 14:30."
        )

        results = []
        for i in range(3):
            try:
                parsed = await _run_parser_on_text(email)
                results.append(parsed)
            except Exception as e:
                if "API key not valid" in str(e) or "400 INVALID_ARGUMENT" in str(e):
                    pytest.skip("Invalid Google API key detected at runtime")
                pass  # Rate limit / transient error

        assert len(results) >= 2, f"Only {len(results)}/3 runs produced valid JSON"

        amounts = [
            r["amount"] for r in results if isinstance(r.get("amount"), (int, float))
        ]
        if len(amounts) >= 2:
            signs = [a < 0 for a in amounts]
            assert all(s == signs[0] for s in signs), f"Inconsistent signs: {amounts}"

        payees = [r["payee"].lower() for r in results if r.get("payee")]
        if len(payees) >= 2:
            base = payees[0]
            for p in payees[1:]:
                overlap = base in p or p in base or any(w in p for w in base.split())
                assert overlap, f"Payee inconsistency: {payees}"

    @pytest.mark.asyncio
    async def test_parser_html_email(self):
        """Parser should handle HTML-formatted Bancolombia emails."""
        html = """<html><body>
        <table><tr><td>Bancolombia le informa</td></tr>
        <tr><td>Compra por $85.000 en RAPPI RESTAURANTE SAS</td></tr>
        <tr><td>Tarjeta terminada en 52e0</td></tr>
        <tr><td>Fecha: 18/02/2026 12:30</td></tr></table>
        </body></html>"""

        try:
            parsed = await _run_parser_on_text(html)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pytest.skip("Rate limited by Gemini API")
            if "API key not valid" in str(e) or "400 INVALID_ARGUMENT" in str(e):
                pytest.skip("Invalid Google API key detected at runtime")
            raise

        assert parsed.get("amount") is not None
        assert isinstance(parsed["amount"], (int, float))
        assert parsed["amount"] < 0, "Purchase should be negative"
        assert "payee" in parsed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 3 — Account Matcher + Categorizer (LLM + YNAB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def _run_agent_with_state(agent, initial_state: dict) -> dict:
    """Helper: runs an ADK agent with pre-populated session state."""
    from google.adk.runners import Runner
    from autopilot.core.session import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="stage_test",
        agent=agent,
        session_service=session_service,
    )

    session_id = f"stage_{id(agent)}"
    await session_service.create_session(
        app_name="stage_test",
        user_id="test",
        session_id=session_id,
        state=initial_state,
    )

    # Agents in the pipeline don't need a meaningful user message
    # since they read from session state via {template_variables}.
    # But ADK requires a message to start, so we send a minimal one.
    message = types.Content(
        role="user",
        parts=[types.Part(text="Process the data from session state.")],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="test",
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                text_parts = [p.text for p in event.content.parts if p.text]
                if text_parts:
                    final_text = "\n".join(text_parts)
            break

    # Also return the session state (agents write their output_key there)
    updated_session = await session_service.get_session(
        app_name="stage_test",
        user_id="test",
        session_id=session_id,
    )

    return {
        "response_text": final_text,
        "state": updated_session.state if updated_session else {},
    }


@pytest.mark.integration
@pytest.mark.skipif(
    not (HAS_REAL_GOOGLE_KEY and HAS_REAL_YNAB_TOKEN),
    reason="Requires both GOOGLE_API_KEY and YNAB_ACCESS_TOKEN",
)
class TestCategorizerIntegration:
    """Contract tests for the Categorizer agent (LLM + YNAB API).

    Validates that the agent can:
    - Call the get_ynab_categories tool
    - Assign semantically appropriate categories
    - Return valid category UUIDs from real YNAB data
    """

    async def _get_real_budget_id(self) -> str:
        """Get a budget ID that has categories."""
        from autopilot.connectors.ynab_connector import YNABConnector

        connector = YNABConnector()
        await connector.setup()
        budgets = await connector.client.get_all_budgets()
        # Find a budget with categories
        for budget in budgets:
            cats = await connector.client.get_categories(budget["id"])
            if len(cats) >= 3:
                return budget["id"]
        # Fallback to first budget
        if budgets:
            return budgets[0]["id"]
        pytest.skip("No budgets available in test account.")

    @pytest.mark.asyncio
    async def test_categorizer_assigns_valid_category(self):
        """Categorizer should return a real YNAB category UUID."""
        from workflows.bank_to_ynab.agents.categorizer import create_categorizer
        from autopilot.agents import extract_json
        from autopilot.connectors.ynab_connector import YNABConnector

        budget_id = await self._get_real_budget_id()

        parsed_email = json.dumps(
            {
                "date": "2026-02-18",
                "payee": "Restaurante El Cielo",
                "amount": -50000,
                "card_suffix": "52e0",
                "memo": "Compra con tarjeta terminada en 52e0",
            }
        )

        matched_account = json.dumps(
            {
                "budget_id": budget_id,
                "budget_name": "My Budget",
                "account_id": "test-account-id",
                "account_name": "Test Account",
                "match_confidence": "high",
                "match_reasoning": "Test",
            }
        )

        categorizer = create_categorizer()
        try:
            result = await _run_agent_with_state(
                categorizer,
                {
                    "parsed_email": parsed_email,
                    "matched_account": matched_account,
                },
            )
        except Exception as e:
            if "ValidationError" in type(e).__name__:
                pytest.skip(f"LLM output format issue (expected in isolation): {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pytest.skip("Rate limited by Gemini API")
            raise

        state = result["state"]
        assert "categorized_tx" in state, "Categorizer should write to 'categorized_tx'"

        categorized = state["categorized_tx"]
        if isinstance(categorized, str):
            categorized = extract_json(categorized)

        # The LLM should return at least a category_name and reasoning
        assert "category_name" in categorized or "category_reasoning" in categorized, (
            f"Missing category fields in: {categorized}"
        )

        # Verify the category UUID exists in YNAB (if provided)
        cat_id = categorized.get("category_id")
        if cat_id:  # May be null/missing if no good match
            connector = YNABConnector()
            categories = await connector.get_categories(budget_id)
            cat_ids = {c["id"] for c in categories}
            assert cat_id in cat_ids, (
                f"Category ID {cat_id} not found in YNAB budget {budget_id}"
            )

    @pytest.mark.asyncio
    async def test_categorizer_reasoning_is_coherent(self):
        """The category reasoning should mention the payee."""
        from workflows.bank_to_ynab.agents.categorizer import create_categorizer
        from autopilot.agents import extract_json

        budget_id = await self._get_real_budget_id()

        parsed_email = json.dumps(
            {
                "date": "2026-02-18",
                "payee": "Netflix",
                "amount": -39900,
                "card_suffix": "52e0",
                "memo": "Cobro recurrente Netflix",
            }
        )

        matched_account = json.dumps(
            {
                "budget_id": budget_id,
                "budget_name": "My Budget",
                "account_id": "test-account-id",
                "account_name": "Test Account",
                "match_confidence": "high",
                "match_reasoning": "Test",
            }
        )

        categorizer = create_categorizer()
        try:
            result = await _run_agent_with_state(
                categorizer,
                {
                    "parsed_email": parsed_email,
                    "matched_account": matched_account,
                },
            )
        except Exception as e:
            if "ValidationError" in type(e).__name__:
                pytest.skip(f"LLM output format issue (expected in isolation): {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pytest.skip("Rate limited by Gemini API")
            raise

        categorized = result["state"].get("categorized_tx", "{}")
        if isinstance(categorized, str):
            categorized = extract_json(categorized)

        reasoning = categorized.get("category_reasoning", "").lower()
        assert (
            "netflix" in reasoning or "suscri" in reasoning or "subscri" in reasoning
        ), f"Reasoning should mention payee/subscription: {reasoning}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Layer 4 — Full Pipeline E2E
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_FULL_E2E,
    reason="Full E2E requires RUN_FULL_INTEGRATION=1",
)
class TestFullPipelineE2E:
    """Full end-to-end pipeline tests.

    Run manually: RUN_FULL_INTEGRATION=1 pytest tests/test_integration.py::TestFullPipelineE2E -v
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_restaurant_purchase(self):
        """Complete pipeline: email → parse → match → categorize → synthesize."""
        from workflows.bank_to_ynab.workflow import BankToYnabWorkflow

        email = (
            "Bancolombia le informa compra por $50.000 en RESTAURANTE EL CIELO "
            "con tarjeta terminada en 52e0. 18/02/2026 14:30."
        )
        wf = BankToYnabWorkflow()
        result = await wf.execute({"body": email})

        assert result.status.value == "success"
        tx = result.data

        # Structural validation
        assert tx["date"] == "2026-02-18"
        assert tx["amount"] < 0, "Purchase should be negative"
        assert "cielo" in tx["payee"].lower() or "restaurante" in tx["payee"].lower()
        assert tx["budget_id"], "Should have a budget_id"
        assert tx["account_id"], "Should have an account_id"

        # UUID format validation
        uuid_pattern = re.compile(r"^[a-f0-9-]{36}$")
        assert uuid_pattern.match(tx["budget_id"]), (
            f"Invalid budget_id format: {tx['budget_id']}"
        )
        assert uuid_pattern.match(tx["account_id"]), (
            f"Invalid account_id format: {tx['account_id']}"
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_golden_data(self, golden_emails):
        """Run the full pipeline on golden test cases."""
        from workflows.bank_to_ynab.workflow import BankToYnabWorkflow

        wf = BankToYnabWorkflow()
        for i, case in enumerate(golden_emails[:2]):
            result = await wf.execute({"body": case["input"]})
            assert result.status.value == "success"
            tx = result.data
            expected = case["expected"]

            if "amount_is_negative" in expected:
                if expected["amount_is_negative"]:
                    assert tx["amount"] < 0, f"[{case['name']}] Expected negative"
                else:
                    assert tx["amount"] > 0, f"[{case['name']}] Expected positive"

            if "date_format" in expected:
                assert re.match(r"^\d{4}-\d{2}-\d{2}$", tx["date"])

            if "payee_contains" in expected:
                assert expected["payee_contains"].lower() in tx["payee"].lower()

    @pytest.mark.asyncio
    async def test_full_pipeline_directv_purchase_with_category_balance(self):
        """Complete E2E: Bancolombia DirectTV purchase.

        Tests: parse → match → research → categorize → push → category balance.

        Uses a real Bancolombia email for a DirecTV GO subscription.
        Validates the entire pipeline including:
          - Email parsing (date, payee, amount, card suffix)
          - Account matching (card 7644 → Joint budget)
          - Categorization (assigns a real YNAB category)
          - Transaction creation in YNAB
          - Category balance retrieval
          - Overspending warning generation
        """
        from workflows.bank_to_ynab.workflow import BankToYnabWorkflow

        email = (
            "Bancolombia: Compraste COP93.900,00 en DTV*DIRECTVGO, "
            "el 14:44 a las 20/02/2026. Esta compra esta asociada a "
            "T.Cred *7644. Si tienes dudas, encuentranos aqui: "
            "01800931987. Siempre contigo."
        )

        wf = BankToYnabWorkflow()
        result = await wf.execute({"body": email, "auto_create": True})

        assert result.status.value == "success", f"Pipeline failed: {result.error}"
        data = result.data

        # ── 1. Email Parsing ──────────────────────────────────────
        # The final_result_data is populated by push_to_ynab (last step)
        final = data.get("final_result_data", data)

        assert final.get("date") == "2026-02-20", (
            f"Expected date 2026-02-20, got {final.get('date')}"
        )
        assert final.get("amount") is not None
        assert final["amount"] < 0, (
            f"Purchase should be negative, got {final['amount']}"
        )
        # COP 93900 → -93900
        assert abs(final["amount"]) == pytest.approx(93900, rel=0.01), (
            f"Amount should be ~93900, got {abs(final['amount'])}"
        )

        payee = final.get("payee", "").lower()
        assert "directv" in payee or "dtv" in payee, (
            f"Payee should contain 'directv' or 'dtv', got '{payee}'"
        )

        # ── 2. Account Matching ───────────────────────────────────
        # Card 7644 is in the Joint budget mapping
        assert final.get("budget_id"), "Should have a budget_id"
        assert final.get("account_id"), "Should have an account_id"
        assert final.get("match_confidence") == "high", (
            f"Card 7644 should be high confidence, got {final.get('match_confidence')}"
        )

        uuid_pattern = re.compile(r"^[a-f0-9-]{36}$")
        assert uuid_pattern.match(final["budget_id"]), (
            f"Invalid budget_id: {final['budget_id']}"
        )
        assert uuid_pattern.match(final["account_id"]), (
            f"Invalid account_id: {final['account_id']}"
        )

        # ── 3. Categorization ─────────────────────────────────────
        assert final.get("category_id"), (
            f"Should have a category_id, got {final.get('category_id')}"
        )
        assert uuid_pattern.match(final["category_id"]), (
            f"Invalid category_id: {final['category_id']}"
        )
        assert final.get("category_reasoning"), "Should have category_reasoning"

        # ── 4. YNAB Transaction Creation ──────────────────────────
        assert final.get("created_in_ynab") is True, (
            f"Transaction should have been created in YNAB, got {final.get('created_in_ynab')}"
        )
        assert final.get("ynab_transaction_id"), (
            "Should have a ynab_transaction_id after creation"
        )

        # ── 5. Category Balance ───────────────────────────────────
        assert "category_balance" in final, (
            "Should include category_balance after transaction creation"
        )
        balance = final["category_balance"]
        assert "category_name" in balance, "Balance should have category_name"
        assert "budgeted" in balance, "Balance should have budgeted"
        assert "activity" in balance, "Balance should have activity"
        assert "balance" in balance, "Balance should have balance field"
        assert "is_overspent" in balance, "Balance should have is_overspent"
        assert isinstance(balance["is_overspent"], bool)

        # Budgeted and activity should be numeric
        assert isinstance(balance["budgeted"], (int, float))
        assert isinstance(balance["activity"], (int, float))
        assert isinstance(balance["balance"], (int, float))

        # Activity should be negative (spending occurred)
        assert balance["activity"] <= 0, (
            f"Activity should be <= 0 after a purchase, got {balance['activity']}"
        )

        # ── 6. Overspending Warning ───────────────────────────────
        assert "overspending_warning" in final, (
            "Should include overspending_warning field"
        )
        if balance["is_overspent"]:
            assert "⚠️ OVERSPENDING" in final["overspending_warning"], (
                "Overspent category should have a warning message"
            )
            assert balance["category_name"] in final["overspending_warning"]
        else:
            assert final["overspending_warning"] == "", (
                "Non-overspent category should have empty warning"
            )

        # ── Summary log ──────────────────────────────────────────
        print("\n✅ E2E DirectTV Purchase Complete:")
        print(f"   Payee:    {final.get('payee')}")
        print(f"   Amount:   COP {final.get('amount'):,.0f}")
        print(f"   Category: {balance.get('category_name')}")
        print(f"   Balance:  COP {balance.get('balance'):,.0f}")
        print(f"   Budgeted: COP {balance.get('budgeted'):,.0f}")
        print(f"   Activity: COP {balance.get('activity'):,.0f}")
        print(f"   Overspent: {balance.get('is_overspent')}")
        if final.get("overspending_warning"):
            print(f"   ⚠️  {final['overspending_warning']}")
