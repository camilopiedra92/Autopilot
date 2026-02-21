"""
Platform Guardrails — Reusable guardrail primitives for ADK agents.

Provides factory functions that return ADK-compatible callback functions
for common input/output validation concerns. Workflow-specific guardrails
(e.g., semantic coherence, amount limits) remain in each workflow.

Usage:
    from autopilot.agents.guardrails import (
        input_length_guard,
        prompt_injection_guard,
        uuid_format_guard,
    )

    agent = LlmAgent(
        ...,
        before_model_callback=create_chained_before_callback(
            input_length_guard(min_chars=10),
            prompt_injection_guard(),
            before_model_logger,
        ),
        after_model_callback=create_chained_after_callback(
            after_model_logger,
            uuid_format_guard(),
        ),
    )
"""

from __future__ import annotations

import re
import structlog
from typing import Optional, Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

logger = structlog.get_logger(__name__)

# Type aliases for ADK callbacks
BeforeCallback = Callable[
    [CallbackContext, LlmRequest], Optional[LlmResponse]
]
AfterCallback = Callable[
    [CallbackContext, LlmResponse], Optional[LlmResponse]
]

# ── Default injection patterns ────────────────────────────────────────
DEFAULT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"forget\s+(all\s+)?your\s+rules",
    r"system\s*prompt",
    r"jailbreak",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Input Guard Factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def input_length_guard(
    min_chars: int = 10,
    message: str = "⚠️ Input too short to process. Please provide more content.",
) -> BeforeCallback:
    """
    Returns a before_model_callback that rejects inputs shorter than min_chars.

    Args:
        min_chars: Minimum character count after stripping whitespace.
        message: Response message when input is rejected.
    """
    def _guard(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        last_message = _extract_last_user_message(llm_request)

        if len(last_message.strip()) < min_chars:
            logger.warning(
                "guardrail_input_too_short",
                agent=callback_context.agent_name,
                length=len(last_message.strip()),
                min_chars=min_chars,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=message)],
                )
            )
        return None

    return _guard


def prompt_injection_guard(
    patterns: list[str] | None = None,
    message: str = "🛡️ Request blocked by security guardrail.",
) -> BeforeCallback:
    """
    Returns a before_model_callback that detects prompt injection attempts.

    Args:
        patterns: Regex patterns to detect. Defaults to common injection patterns.
        message: Response message when injection is detected.
    """
    active_patterns = patterns or DEFAULT_INJECTION_PATTERNS

    def _guard(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        last_message = _extract_last_user_message(llm_request)

        for pattern in active_patterns:
            if re.search(pattern, last_message, re.IGNORECASE):
                logger.warning(
                    "guardrail_injection_detected",
                    agent=callback_context.agent_name,
                    pattern=pattern,
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=message)],
                    )
                )
        return None

    return _guard



def uuid_format_guard(
    fields: tuple[str, ...] = ("budget_id", "account_id", "category_id"),
) -> AfterCallback:
    """
    Returns an after_model_callback that validates UUIDs in specified JSON fields.

    Args:
        fields: JSON field names to check for valid UUID format.
    """
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    field_pattern = '|'.join(re.escape(f) for f in fields)

    def _guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        response_text = _extract_response_text(llm_response)
        if not response_text:
            return None

        id_fields = re.findall(
            rf'"(?:{field_pattern})"\s*:\s*"([^"]+)"', response_text
        )
        for field_value in id_fields:
            if field_value and not re.match(uuid_pattern, field_value):
                logger.warning(
                    "guardrail_invalid_uuid",
                    agent=callback_context.agent_name,
                    value=field_value,
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(
                            text=f"⚠️ Invalid UUID detected: '{field_value}'. "
                                 f"IDs must be real UUIDs from the API. "
                                 f"Please call the appropriate tool to get a valid ID."
                        )],
                    )
                )
        return None

    return _guard


def amount_sanity_guard(
    max_amount: float,
) -> AfterCallback:
    """
    Returns an after_model_callback that blocks transactions with excessive amounts.

    Generic guard — each workflow provides its domain-specific max_amount.

    Args:
        max_amount: Maximum allowed absolute amount (required).
    """
    def _guard(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        response_text = _extract_response_text(llm_response)
        if not response_text:
            return None

        amount_match = re.search(r'"amount"\s*:\s*(-?[\d.]+)', response_text)
        if amount_match:
            amount = abs(float(amount_match.group(1)))
            if amount > max_amount:
                logger.warning(
                    "guardrail_amount_too_large",
                    agent=callback_context.agent_name,
                    amount=amount,
                    max_allowed=max_amount,
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(
                            text=f"⚠️ Transaction amount ${amount:,.0f} exceeds the safety limit of "
                                 f"${max_amount:,.0f}. Please verify this is correct."
                        )],
                    )
                )
        return None

    return _guard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_last_user_message(llm_request: LlmRequest) -> str:
    """Extract the last user message text from an LLM request."""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                for part in content.parts:
                    if part.text:
                        return part.text
    return ""


def _extract_response_text(llm_response: LlmResponse) -> str:
    """Extract all text from an LLM response."""
    text = ""
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                text += part.text
    return text
