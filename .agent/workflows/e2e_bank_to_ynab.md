---
description: Run a full E2E test of the bank_to_ynab pipeline with detailed tracing
---

// turbo-all

## Run E2E Bank to YNAB Test

This workflow executes the full `bank_to_ynab` pipeline end-to-end with a real bank transaction email. It traces every stage (parsing, matching, research, categorization, YNAB push, Telegram notification) and prints a detailed colorized report.

### Prerequisites

- `.env` file must be present in the project root with valid secrets (`GOOGLE_API_KEY`, `YNAB_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`)

### Steps

1. **Ask for input**: Ask the user for the bank email text to process. If not provided, use the default embedded in the script.

2. **Run with auto_create ON** (creates transaction in YNAB + sends Telegram):

   ```bash
   cd /Users/camilopiedra/Development/N8N && python scripts/run_e2e_bank_to_ynab.py "<EMAIL_TEXT>"
   ```

3. **Run with auto_create OFF** (dry run — no YNAB creation, still sends Telegram):

   ```bash
   cd /Users/camilopiedra/Development/N8N && python scripts/run_e2e_bank_to_ynab.py "<EMAIL_TEXT>" --no-create
   ```

4. **Run with default email** (uses built-in VET AGRO test case):

   ```bash
   cd /Users/camilopiedra/Development/N8N && python scripts/run_e2e_bank_to_ynab.py
   ```

5. **Report the results**: Show the user the structured output from the script. Key things to report:
   - Overall status (success/failure) and total time
   - Each stage's input/output
   - YNAB transaction ID (if created)
   - Category balance (budgeted, spent, available)
   - Telegram message delivery status

### Notes

- The script uses colorized ANSI output for terminal readability
- All structlog events from the pipeline are also printed (step_started, tool_call_completed, etc.)
- Typical execution time is ~30-40 seconds (network-bound: Gemini API, YNAB API, Telegram API, web search)
