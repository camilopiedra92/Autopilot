"""
TransactionEvent — Typed event model for the ``transaction.created`` topic.

Pydantic model that defines the schema for transaction events published
to the AgentBus.  Provides a ``from_pipeline_state()`` factory to build
the event from the ``push_to_ynab`` step's ``final_result_data`` output.
"""

from pydantic import BaseModel
from typing import Any


class TransactionEvent(BaseModel):
    """Typed payload for ``transaction.created`` AgentBus events."""

    # Core transaction fields
    payee: str = ""
    amount: float = 0
    date: str = ""
    memo: str = ""

    # YNAB identifiers
    budget_id: str = ""
    account_id: str = ""
    category_id: str | None = None

    # Matching & categorization
    match_confidence: str = ""
    match_reasoning: str = ""
    category_reasoning: str = ""

    # YNAB result
    created_in_ynab: bool = False
    ynab_transaction_id: str | None = None
    is_successful: bool = True
    skip_reason: str = ""

    # Category balance (post-transaction)
    category_balance: dict[str, Any] | None = None
    overspending_warning: str = ""

    @classmethod
    def from_pipeline_state(
        cls, final_result_data: dict[str, Any]
    ) -> "TransactionEvent":
        """
        Factory: build from the ``push_to_ynab`` step's output.

        Maps ``final_result_data`` keys directly into the event model.
        Unknown keys are silently ignored (Pydantic ``model_validate``).
        """
        return cls.model_validate(final_result_data)
