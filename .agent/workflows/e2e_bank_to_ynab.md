---
description: Run a full E2E test of the bank_to_ynab pipeline with detailed tracing
---

// turbo-all

## Run E2E Bank to YNAB Test

This workflow executes the full `bank_to_ynab` pipeline end-to-end **via the AgentBus**, simulating a real production Gmail event. It publishes an `email.received` event, the workflow self-matches by sender, runs the pipeline (parsing → research → categorization → YNAB push), and fires the `transaction.created` event which triggers the Telegram notification.

### Prerequisites

- `.env` file must be present in the project root with valid secrets (`GOOGLE_API_KEY`, `YNAB_ACCESS_TOKEN`, `TELEGRAM_BOT_TOKEN`)
- `EVENTBUS_BACKEND` is **not** set in `.env` (defaults to InMemory). The commands below override it inline for Pub/Sub mode.
- `SESSION_BACKEND` is **not** set in `.env` (defaults to InMemory). Add `SESSION_BACKEND=firestore` inline to test with durable sessions.

### Steps

1. **Ask for input**: Ask the user for the bank email text to process. If not provided, use the default embedded in the script.

2. **Ask for session backend**: Ask the user if they want to run with Firestore sessions (`SESSION_BACKEND=firestore`) or default in-memory. Default is in-memory.

3. **Run full E2E via AgentBus** (default — simulates real production event, creates transaction in YNAB + sends Telegram):

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && EVENTBUS_BACKEND=pubsub python scripts/run_e2e_bank_to_ynab.py
   ```

   With Firestore sessions:

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && EVENTBUS_BACKEND=pubsub SESSION_BACKEND=firestore python scripts/run_e2e_bank_to_ynab.py
   ```

4. **Run with custom email text**:

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && EVENTBUS_BACKEND=pubsub python scripts/run_e2e_bank_to_ynab.py "<EMAIL_TEXT>"
   ```

5. **Run direct pipeline** (bypasses bus, useful for debugging pipeline-only issues):

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && python scripts/run_e2e_bank_to_ynab.py --direct
   ```

6. **Run dry-run** (no YNAB creation):

   ```bash
   cd /Users/camilopiedra/Development/Autopilot && python scripts/run_e2e_bank_to_ynab.py --no-create
   ```

7. **Report the results**: Show the user the structured output from the script. Key things to report:
   - Overall status (success/failure) and total time
   - Each stage's input/output
   - YNAB transaction ID (if created)
   - Category balance (budgeted, spent, available)
   - Telegram message delivery status

### Notes

- Default mode (`--via-bus`) tests the full event-driven architecture: `email.received` → workflow self-match → pipeline → `transaction.created` → Telegram
- `auto_create` defaults to `true` from `manifest.yaml` settings. Pass `--no-create` to override.
- `SESSION_BACKEND=firestore` can be added to any command to test with durable Firestore sessions (requires GCP auth)
- The script uses colorized ANSI output for terminal readability
- Typical execution time is ~30-40 seconds (network-bound: Gemini API, YNAB API, Telegram API, web search)
