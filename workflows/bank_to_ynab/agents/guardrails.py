"""
Workflow-specific guardrails for Bank→YNAB.

Domain guards that depend on workflow-specific fields (payee, category_name).
Generic guards (input_length, prompt_injection, uuid_format, amount_sanity)
live in autopilot.agents.guardrails.
"""

import re
from typing import Optional

import structlog
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

from autopilot.agents.guardrails import _extract_response_text

logger = structlog.get_logger(__name__)


def semantic_coherence_guard(
    rules: list[dict],
):
    """
    Returns an after_model_callback that validates semantic coherence
    between a payee name and the assigned category.

    Domain-specific to financial transaction workflows that produce
    JSON with "payee" and "category_name" fields.

    Uses keyword rules to detect obvious mismatches (e.g., Netflix → Groceries).
    When incoherent, returns an LlmResponse asking the model to re-categorize.

    Args:
        rules: List of rule dicts, each with keys:
            - payee_keywords: list[str]
            - expected_categories: list[str]
            - bad_categories: list[str]
    """
    def _guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        response_text = _extract_response_text(llm_response)
        if not response_text:
            return None

        # Extract payee and category_name from response JSON
        payee_match = re.search(r'"payee"\s*:\s*"([^"]+)"', response_text)
        category_match = re.search(r'"category_name"\s*:\s*"([^"]+)"', response_text)

        if not payee_match or not category_match:
            payee = ""
            category_name = ""
            if hasattr(callback_context, "state") and callback_context.state:
                payee = str(callback_context.state.get("payee", ""))
            if category_match:
                category_name = category_match.group(1)
            elif payee_match:
                payee = payee_match.group(1)

            if not payee or not category_name:
                return None

        payee = payee_match.group(1) if payee_match else payee
        category_name = category_match.group(1) if category_match else category_name
        payee_lower = payee.lower().strip()
        category_lower = category_name.lower().strip()

        for rule in rules:
            payee_keywords = rule.get("payee_keywords", [])
            expected_cats = rule.get("expected_categories", [])
            bad_cats = rule.get("bad_categories", [])

            payee_matches = any(kw in payee_lower for kw in payee_keywords)
            if not payee_matches:
                continue

            is_bad = any(bad in category_lower for bad in bad_cats)
            if is_bad:
                suggested = [kw.title() for kw in expected_cats[:3]]
                logger.warning(
                    "guardrail_semantic_coherence_mismatch",
                    agent=callback_context.agent_name,
                    payee=payee,
                    category=category_name,
                    suggested=suggested,
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(
                            text=f"⚠️ Semantic mismatch: Payee '{payee}' was assigned "
                                 f"category '{category_name}', which seems incorrect. "
                                 f"Expected category types: {', '.join(suggested)}. "
                                 f"Please re-categorize."
                        )],
                    )
                )

            # Good match — no need to check further rules
            is_good = any(exp in category_lower for exp in expected_cats)
            if is_good:
                break

        return None

    return _guard
