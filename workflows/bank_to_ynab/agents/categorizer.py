"""
Agent 3 — Categorizer
Assigns the best YNAB category to a transaction based on the payee/merchant.
Reads from session state {parsed_email} + {matched_account}, writes to {categorized_tx}.

Uses `ynab.get_categories_string` tool to fetch real YNAB categories at runtime.
The platform's ToolRegistry lazily resolves connector tools, so this works
regardless of app startup order.

Guardrails:
  - after_model: semantic_coherence_guard (payee↔category) + uuid_format_guard (platform)
"""

import json
from pathlib import Path
from google.adk.agents import LlmAgent

from workflows.bank_to_ynab.models.transaction import CategorizedTransaction
from autopilot.agents.base import create_platform_agent
from autopilot.agents.guardrails import uuid_format_guard
from workflows.bank_to_ynab.agents.guardrails import semantic_coherence_guard
from autopilot.agents.callbacks import create_chained_after_callback


# ── Load coherence rules once at import time ─────────────────────────
_RULES_PATH = Path(__file__).parent.parent / "data" / "payee_category_rules.json"
_COHERENCE_RULES: list[dict] = (
    json.loads(_RULES_PATH.read_text(encoding="utf-8")) if _RULES_PATH.exists() else []
)


CATEGORIZER_INSTRUCTION = """
You are a YNAB transaction categorization expert.

Here is the parsed transaction from the bank email:
{parsed_email}

Here is the matched YNAB account:
{matched_account}

Your job is to assign the most appropriate YNAB category.

YOUR FIRST ACTION MUST BE to call the 'ynab.get_categories_string' tool using the budget_id from the matched account above. Do NOT respond with text first.

After getting the tool results:
1. Analyze the payee/merchant name to determine the best category
2. Use common sense for categorization:
   - Restaurants, cafés → "Dining Out" / "Restaurantes"  
   - Supermarkets, grocery stores → "Groceries" / "Mercado"
   - Gas stations → "Transportation" / "Gasolina"
   - Online subscriptions (Netflix, Spotify) → "Subscriptions" / "Suscripciones"
   - If uncertain, prefer a broader category over a wrong specific one

CRITICAL RULES:
- You MUST call ynab.get_categories_string FIRST — NEVER invent category UUIDs
- If no category fits well, set category_id to null and explain in reasoning
"""


def create_categorizer(model_name: str = "gemini-3-flash-preview") -> LlmAgent:
    """Creates the categorizer agent with semantic coherence guardrails.

    Uses `ynab.get_categories_string` tool (auto-resolved by platform)
    to fetch real categories from the YNAB API.

    Guardrails:
      - after_model: semantic_coherence_guard (payee↔category validation)
      - after_model: uuid_format_guard (category_id UUID validation)
    """
    return create_platform_agent(
        name="categorizer",
        model=model_name,
        instruction=CATEGORIZER_INSTRUCTION,
        description="Assigns YNAB categories to transactions based on payee/merchant analysis.",
        output_key="categorized_tx",
        output_schema=CategorizedTransaction,
        tools=["ynab.get_categories_string"],
        after_model_callback=create_chained_after_callback(
            semantic_coherence_guard(rules=_COHERENCE_RULES),
            uuid_format_guard(fields=("category_id",)),
        ),
    )
