"""
Pipeline step functions — Pure code stages for the Bank→YNAB pipeline.

Each function receives accumulated pipeline state as keyword arguments
and returns a dict that gets merged into the next step's state.

Steps:
  1. sanitize_email_html  — Strip HTML to clean text
  2. format_parser_prompt  — Prepare prompt for the email parser agent
  3. match_account         — Deterministic O(1) card→account lookup
  4. format_categorizer_input — Prepare context for categorizer agent
  5. synthesize_transaction — Merge all upstream outputs into final Transaction
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.core.context import AgentContext
import structlog
from pathlib import Path

from workflows.bank_to_ynab.models.transaction import (
    AccountMapping,
    ParsedEmail,
    MatchedAccount,
    CategorizedTransaction,
    Transaction,
    EnrichedPayee,
)

logger = structlog.get_logger(__name__)


# ── UUID validation pattern ──────────────────────────────────────────
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


# ── Account Mappings (loaded once, O(1) lookup) ─────────────────────


def _load_account_mappings() -> list[AccountMapping]:
    """Load account mappings from the workflow data directory."""
    mappings_file = Path(__file__).parent / "data" / "account_mappings.json"
    if not mappings_file.exists():
        return []

    try:
        data = json.loads(mappings_file.read_text(encoding="utf-8"))
        budgets = data.get("budgets", {})

        flat_mappings = []
        for budget_key, budget_data in budgets.items():
            budget_id = budget_data.get("budgetID")
            account_map = budget_data.get("accountMap", {})

            for card_suffix, account_id in account_map.items():
                flat_mappings.append(
                    AccountMapping(
                        card_suffix=card_suffix,
                        budget_id=budget_id,
                        budget_name=budget_key.capitalize(),
                        account_id=account_id,
                        account_name=f"{budget_key.capitalize()} ({card_suffix})",
                    )
                )
        return flat_mappings
    except Exception as e:
        logger.error("account_mappings_load_failed", error=str(e))
        return []


_ACCOUNT_MAPPINGS = _load_account_mappings()
_CARD_LOOKUP: dict[str, AccountMapping] = {
    m.card_suffix.lower(): m for m in _ACCOUNT_MAPPINGS
}


def lookup_account_by_card(card_suffix: str) -> AccountMapping | None:
    """O(1) deterministic lookup of a card suffix → YNAB account."""
    if not card_suffix:
        return None
    return _CARD_LOOKUP.get(card_suffix.lower())


# ── Step: HTML sanitizer ─────────────────────────────────────────────


def sanitize_email_html(html: str) -> str:
    """
    Aggressively strip HTML to get clean text from Bancolombia emails.
    Uses BeautifulSoup for robust parsing and structure preservation.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning(
            "beautifulsoup4_not_installed", detail="Falling back to simple stripping"
        )
        return re.sub(r"<[^>]+>", " ", html).strip()

    # Skip BS4 for plain text (no HTML tags or entities) — avoids MarkupResemblesLocatorWarning
    if "<" not in html and "&" not in html:
        return html.strip()

    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style elements
    for script in soup(["script", "style", "meta", "noscript"]):
        script.decompose()

    # Get text with separator to preserve table/block structure
    text = soup.get_text(separator="\n", strip=True)

    # Post-processing cleanup
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    # Collapse internal multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ── Step 1: Prepare parser prompt ────────────────────────────────────


def format_parser_prompt(**state) -> dict:
    """Prepare the prompt for the email parser agent using raw payload data."""
    email_body = state.get("body", "")
    if not email_body:
        raise ValueError("No email body provided")

    clean = sanitize_email_html(email_body)
    return {"message": f"Parse this bank email:\n\n{clean}"}


# ── Step 2: Deterministic account matching (code, 0 LLM calls) ──────


def match_account(parsed_email: ParsedEmail) -> dict:
    """
    Pure code account matcher — O(1) dict lookup from card mapper.

    If the card is in the mapper → instant match with 'high' confidence.
    If not → returns a 'low' confidence result (no YNAB creation).
    """
    # Sanitize: keep only alphanumeric characters to handle '*7644' or ' 7644 '
    card_suffix = re.sub(r"[^a-zA-Z0-9]", "", parsed_email.card_suffix)

    mapping = lookup_account_by_card(card_suffix)

    if mapping:
        logger.info(
            "account_matched_deterministic",
            card_suffix=card_suffix,
            account_name=mapping.account_name,
            budget_name=mapping.budget_name,
        )
        matched = MatchedAccount(
            budget_id=mapping.budget_id,
            budget_name=mapping.budget_name,
            account_id=mapping.account_id,
            account_name=mapping.account_name,
            match_confidence="high",
            match_reasoning=f"Deterministic: card '{card_suffix}' → {mapping.account_name} ({mapping.budget_name})",
        )
    else:
        logger.warning("account_not_in_mapper", card_suffix=card_suffix)
        matched = MatchedAccount(
            budget_id="",
            budget_name="",
            account_id="",
            account_name="",
            match_confidence="low",
            match_reasoning=f"Card '{card_suffix}' not found in mapper.",
        )

    return {"matched_account": matched.model_dump(), **matched.model_dump()}


# ── Step 3: Prepare researcher context ──────────────────────────────


def format_researcher_input(parsed_email: ParsedEmail) -> dict:
    """Prepare raw payee for the Web Researcher."""
    return {"payee": parsed_email.payee}


# ── Step 4: Prepare categorizer context ──────────────────────────────


def format_categorizer_input(
    parsed_email: ParsedEmail, enriched_payee: EnrichedPayee | None = None
) -> dict:
    """Ensure categorizer has the right message in state and uses enriched name if available."""
    # We pass the clean name so the categorizer has a much easier job
    payee = enriched_payee.clean_name if enriched_payee else parsed_email.payee

    msg_parts = [f"Categorize this transaction for merchant: {payee}"]

    # Add more context from the web researcher to help with categorization
    if enriched_payee:
        if enriched_payee.establishment_type:
            msg_parts.append(
                f"Type of establishment: {enriched_payee.establishment_type}"
            )
        if enriched_payee.website:
            msg_parts.append(f"Website: {enriched_payee.website}")
        if enriched_payee.location:
            msg_parts.append(f"Location: {enriched_payee.location}")

    return {"message": "\n".join(msg_parts)}


# ── Step 4: Synthesize + Validate (deterministic, 0 LLM calls) ──────


def synthesize_transaction(
    parsed_email: ParsedEmail,
    matched_account: MatchedAccount,
    categorized_tx: CategorizedTransaction,
    enriched_payee: EnrichedPayee | None = None,
) -> dict:
    """
    Pipeline step: merge all upstream outputs into a final Transaction.

    Reads from accumulated pipeline state, reconstructs typed models,
    performs quality checks, and returns the Transaction as a dict.
    """

    # Memo left empty — payee + account already convey all useful info.
    # parsed_email.memo is just a bank card reference (redundant with account).
    final_memo = ""

    # Determine final enriched name / details
    final_payee_name = (
        enriched_payee.clean_name if enriched_payee else parsed_email.payee
    )
    establishment_type = (
        (enriched_payee.establishment_type or "Unknown")
        if enriched_payee
        else "Unknown"
    )
    website = (enriched_payee.website or "") if enriched_payee else ""
    location = (enriched_payee.location or "") if enriched_payee else ""

    transaction = Transaction(
        date=parsed_email.date,
        payee=final_payee_name,  # We map the clean, resolved entity name to YNAB
        amount=parsed_email.amount,
        memo=final_memo,
        is_successful=parsed_email.is_successful,
        budget_id=matched_account.budget_id,
        account_id=matched_account.account_id,
        category_id=categorized_tx.category_id,
        enriched_name=final_payee_name,
        establishment_type=establishment_type,
        website=website,
        location=location,
        match_reasoning=matched_account.match_reasoning,
        category_reasoning=categorized_tx.category_reasoning,
        match_confidence=matched_account.match_confidence,
    )

    return {"transaction": transaction.model_dump(), **transaction.model_dump()}


# ── Step 5: Push to YNAB (if auto_create) ───────────────────────────


async def push_to_ynab(**state) -> dict:
    """
    Pipeline step: automatically create the transaction in YNAB.
    Requires 'auto_create' to be in initial_input/state.

    After successful creation, fetches the updated category balance
    and includes an overspending warning if applicable.
    """
    # The transaction was produced in synthesize_transaction
    tx_data = state.get("transaction", {})
    if not tx_data:
        return {"final_result_data": {}}

    transaction = Transaction(**tx_data)
    auto_create = state.get("auto_create", False)

    result_data = {
        "date": transaction.date,
        "payee": transaction.payee,
        "amount": transaction.amount,
        "memo": transaction.memo,
        "budget_id": transaction.budget_id,
        "account_id": transaction.account_id,
        "category_id": transaction.category_id,
        "match_reasoning": transaction.match_reasoning,
        "category_reasoning": transaction.category_reasoning,
        "match_confidence": transaction.match_confidence,
        "is_successful": transaction.is_successful,
    }

    if not transaction.is_successful:
        logger.warning(
            "transaction_skipped_not_successful",
            payee=transaction.payee,
            memo=transaction.memo,
        )
        result_data["created_in_ynab"] = False
        result_data["skip_reason"] = (
            "Transaction was not successful (declined/failed/reversed)"
        )

    elif not transaction.budget_id or not transaction.account_id:
        logger.warning(
            "transaction_skipped_no_account_mapped",
            payee=transaction.payee,
            memo=transaction.memo,
        )
        result_data["created_in_ynab"] = False
        result_data["skip_reason"] = "No YNAB account mapped for this card"

    elif auto_create:
        from autopilot.connectors import get_connector_registry

        ynab = get_connector_registry().get("ynab")
        ynab_result = await ynab.client.create_transaction(
            budget_id=transaction.budget_id,
            transaction_payload={
                "account_id": transaction.account_id,
                "date": transaction.date,
                "amount": transaction.amount_milliunits,
                "payee_name": transaction.payee,
                "memo": transaction.memo,
                "category_id": transaction.category_id,
                "flag_color": "purple",
            },
        )
        result_data["ynab_transaction_id"] = (
            ynab_result.get("data", {}).get("transaction", {}).get("id")
        )
        result_data["created_in_ynab"] = True

        # ── Fetch category balance after transaction creation ────
        result_data.update(
            await _fetch_category_balance(
                ynab, transaction.budget_id, transaction.category_id
            )
        )

    return {"final_result_data": result_data}


async def _fetch_category_balance(
    ynab, budget_id: str, category_id: str | None
) -> dict:
    """Fetch the post-transaction category balance and build warning if overspent.

    Returns a dict with 'category_balance' and 'overspending_warning' keys.
    Gracefully returns empty data if category_id is missing or the API call fails.
    """
    from workflows.bank_to_ynab.models.transaction import CategoryBalance

    if not category_id:
        return {}

    try:
        cat_data = await ynab.client.get_category_by_id(budget_id, category_id)
    except Exception as exc:
        logger.warning(
            "category_balance_fetch_failed",
            category_id=category_id,
            error=str(exc),
        )
        return {}

    # Convert milliunits → currency units
    balance_units = cat_data.get("balance", 0) / 1000
    budgeted_units = cat_data.get("budgeted", 0) / 1000
    activity_units = cat_data.get("activity", 0) / 1000
    cat_name = cat_data.get("name", "Unknown")

    cat_balance = CategoryBalance(
        category_name=cat_name,
        budgeted=budgeted_units,
        activity=activity_units,
        balance=balance_units,
        is_overspent=balance_units < 0,
    )

    warning = ""
    if cat_balance.is_overspent:
        overspent_amount = abs(cat_balance.balance)
        warning = f"⚠️ OVERSPENDING: {cat_name} is ${overspent_amount:,.0f} over budget"
        logger.warning(
            "category_overspent",
            category=cat_name,
            balance=balance_units,
            overspent_by=overspent_amount,
        )

    logger.info(
        "category_balance_fetched",
        category=cat_name,
        budgeted=budgeted_units,
        activity=activity_units,
        balance=balance_units,
        is_overspent=cat_balance.is_overspent,
    )

    return {
        "category_balance": cat_balance.model_dump(),
        "overspending_warning": warning,
    }


# ── Step 6: Publish transaction event to AgentBus ───────────────────


async def publish_transaction_event(ctx: "AgentContext", **state) -> dict:
    """
    Pipeline step: publish a typed ``transaction.created`` event to the AgentBus.

    Reactive subscribers (Telegram notifier, Airtable logger, anomaly detector,
    etc.) receive the event concurrently and independently via the bus's
    dead-letter isolation pattern.

    Uses ``ctx`` injection (auto-provided by FunctionalAgent when the parameter
    is annotated as ``AgentContext``).
    """
    from autopilot.core.context import AgentContext  # noqa: F811
    from workflows.bank_to_ynab.models.events import TransactionEvent

    result = state.get("final_result_data", {})
    if not result:
        ctx.logger.info("publish_skipped", reason="no final_result_data")
        return {}

    event = TransactionEvent.from_pipeline_state(result)
    await ctx.publish("transaction.created", event.model_dump())

    ctx.logger.info(
        "transaction_event_published",
        topic="transaction.created",
        payee=event.payee,
        amount=event.amount,
    )
    return {"event_published": True}
