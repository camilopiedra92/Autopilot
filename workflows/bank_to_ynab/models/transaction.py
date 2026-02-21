"""
Pydantic models for each stage of the multi-agent pipeline.
Each agent produces a progressively richer model.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Account Mapping (deterministic card → YNAB lookup) ───────────────
class AccountMapping(BaseModel):
    """Maps a bank card suffix to a specific YNAB budget + account."""
    card_suffix: str
    budget_id: str
    budget_name: str
    account_id: str
    account_name: str


# ── Stage 1: Email Parser output ──────────────────────────────────────
class ParsedEmail(BaseModel):
    """Structured data extracted from a bank notification email."""

    date: str = Field(description="Transaction date in YYYY-MM-DD format.")
    payee: str = Field(description="Name of the merchant or payee.")
    amount: float = Field(
        description="Transaction amount in original currency. "
        "Negative for expenses, positive for income/deposits."
    )
    card_suffix: str = Field(
        description="Last 4 digits/chars of the card used (e.g. '52e0')."
    )
    memo: str = Field(
        description="Brief description including 'terminada en XXXX' if available."
    )
    is_successful: bool = Field(
        default=True,
        description="Whether the transaction was successful. "
        "Set to False for declined, failed, reversed, or 'no exitosa' transactions.",
    )
    raw_email_snippet: str = Field(
        default="",
        description="First 200 chars of the original email for audit trail.",
    )


# ── Stage 2: Account Matcher output ──────────────────────────────────
class MatchedAccount(BaseModel):
    """YNAB account matched to the bank card from the email."""

    budget_id: str = Field(description="The YNAB Budget UUID.")
    budget_name: str = Field(default="", description="Human-readable budget name.")
    account_id: str = Field(description="The YNAB Account UUID.")
    account_name: str = Field(default="", description="Human-readable account name.")
    match_confidence: str = Field(
        default="high",
        description="Confidence level: 'high', 'medium', or 'low'.",
    )
    match_reasoning: str = Field(
        description="Why this account was selected (e.g. card suffix match in notes)."
    )


# ── Stage 3: Categorizer output ──────────────────────────────────────
class CategorizedTransaction(BaseModel):
    """Transaction with category assignment."""

    category_id: Optional[str] = Field(
        default=None,
        description="The YNAB Category UUID. None if uncertain.",
    )
    category_name: str = Field(default="", description="Human-readable category name.")
    category_reasoning: str = Field(
        description="Why this category was chosen for the merchant/payee."
    )


# ── Stage X: Web Researcher output ────────────────────────────────────
class EnrichedPayee(BaseModel):
    """Enriched entity data from the Web Researcher agent."""
    
    clean_name: str = Field(description="Normalized real name of the merchant/establishment.")
    establishment_type: str = Field(description="Type of business (e.g., 'Supermarket', 'Restaurant', 'Software Subscription').")
    website: Optional[str] = Field(default=None, description="Official website URL if found.")
    location: Optional[str] = Field(default=None, description="Physical location or neighborhood if relevant.")


# ── Post-Transaction: Category Balance snapshot ──────────────────────
class CategoryBalance(BaseModel):
    """Post-transaction category balance snapshot from YNAB."""

    category_name: str = Field(description="YNAB category name.")
    budgeted: float = Field(description="Amount budgeted this month (currency units).")
    activity: float = Field(description="Spending activity this month (currency units, negative = spending).")
    balance: float = Field(description="Remaining available balance (currency units).")
    is_overspent: bool = Field(description="True if balance is negative (overspending).")


# ── Final: Complete Transaction (all stages merged) ──────────────────
class Transaction(BaseModel):
    """Final transaction ready to be created in YNAB."""

    # From email parser
    date: str
    payee: str
    amount: float
    memo: str
    is_successful: bool = True

    # From account matcher
    budget_id: str
    account_id: str

    # From categorizer
    category_id: Optional[str] = None

    # From web researcher (optional for backward compatibility)
    enriched_name: str = ""
    establishment_type: str = ""
    website: str = ""
    location: str = ""

    # Audit trail
    match_reasoning: str = ""
    category_reasoning: str = ""
    match_confidence: str = "high"

    @property
    def amount_milliunits(self) -> int:
        """YNAB expects amounts in milliunits (amount × 1000)."""
        return int(self.amount * 1000)


