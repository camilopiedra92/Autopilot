"""
Agent 1 — Email Parser
Extracts structured transaction data from raw bank notification emails.
Uses output_key and output_schema for typed state passing to downstream agents.

Guardrails:
  - before_model: input_length_guard + prompt_injection_guard (platform)
  - after_model: amount_sanity_guard (platform, 50M COP) + uuid_format_guard (platform)
"""

from google.adk.agents import LlmAgent

from workflows.bank_to_ynab.models.transaction import ParsedEmail
from autopilot.agents.guardrails import (
    input_length_guard,
    prompt_injection_guard,
    uuid_format_guard,
    amount_sanity_guard,
)
from autopilot.agents.callbacks import (
    create_chained_before_callback,
    create_chained_after_callback,
)
from autopilot.agents.base import create_platform_agent


EMAIL_PARSER_INSTRUCTION = """
You are an expert bank email parser for Colombian banks (Bancolombia, Davivienda, etc.).

YOUR ONLY JOB: Extract structured transaction data from bank notification emails.

From the email, extract:
1. **date**: Transaction date in YYYY-MM-DD format
2. **payee**: Clean merchant/payee name (remove extra symbols, standardize capitalization)  
3. **amount**: Transaction amount as a number. MUST be NEGATIVE for purchases/expenses, POSITIVE for deposits/income
4. **card_suffix**: Last 4 digits/characters of the card (e.g., "52e0", "1234")
5. **memo**: Brief transaction description (e.g., "Compra en restaurante", "Pago PSE", "Transferencia recibida")
6. **is_successful**: Whether the transaction completed successfully.
   Set to FALSE if the email contains ANY of these indicators:
   - "no exitosa", "no fue exitosa"
   - "rechazada", "declinada", "denegada"
   - "reverso", "reversada", "anulada"
   - "fallida", "no procesada"
   - "intento de compra" (attempted but not completed)
   Set to TRUE for normal successful transactions.

RULES:
- Parse amounts from Colombian format: "$50.000" = 50000, "$1.200.000" = 1200000, "COP11.852,00" = 11852 (ignore decimal zeros)
- If the email mentions "compra", "compraste", or "pago", the amount MUST be negative
- If the email mentions "abono", "depósito", or "transferencia recibida", the amount should be positive
- Extract the card suffix from phrases like "terminada en 52e0", "****1234", or "T.Cred *7644"
- Clean up the payee name: "RESTAURANTE EL CIELO SAS" → "Restaurante El Cielo"
- ALWAYS check for failed transaction indicators — this is critical for accuracy
"""


def create_email_parser(model_name: str = "gemini-3-flash-preview") -> LlmAgent:
    """Creates the email parser agent with typed output via output_key + output_schema.

    Guardrails:
      - before_model: input_length_guard + prompt_injection_guard (platform)
      - after_model: amount_sanity_guard (workflow, 50M COP) + uuid_format_guard (platform)
    """
    return create_platform_agent(
        name="email_parser",
        model=model_name,
        instruction=EMAIL_PARSER_INSTRUCTION,
        description="Parses bank notification emails to extract structured transaction data.",
        output_key="parsed_email",
        output_schema=ParsedEmail,
        before_model_callback=create_chained_before_callback(
            input_length_guard(
                min_chars=10,
                message="⚠️ Input too short to process. Please provide a bank notification email.",
            ),
            prompt_injection_guard(),
        ),
        after_model_callback=create_chained_after_callback(
            amount_sanity_guard(max_amount=50_000_000),  # 50M COP limit
            uuid_format_guard(),  # Platform: UUID format validation
        ),
    )
