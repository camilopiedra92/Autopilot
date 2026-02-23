#!/usr/bin/env python3
"""
E2E Runner — Execute the bank_to_ynab pipeline with detailed step tracing.

Runs the full pipeline end-to-end, capturing every step's input/output,
agent calls, tool invocations, and final results.

Usage:
    python scripts/run_e2e_bank_to_ynab.py "Bancolombia: Compraste COP59.000,00 en VET AGRO..."
    python scripts/run_e2e_bank_to_ynab.py  # Uses a default test email
"""

import asyncio
import json
import sys
import time
import os
import structlog

# Ensure the project root is on sys.path for `workflows` imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

# ── Structured logging config ────────────────────────────────────────
# Configure structlog for human-readable console output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("e2e_runner")

# ANSI color codes for pretty output
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CHECK = "✅"
CROSS = "❌"
ARROW = "→"


DEFAULT_EMAIL = (
    "Bancolombia: Compraste COP59.000,00 en VET AGRO con tu T.Cred *7644, "
    "el 21/02/2026 a las 17:31. Si tienes dudas, encuentranos aqui: "
    "6045109095 o 018000931987. Estamos cerca."
)


def print_header(title: str):
    width = 70
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


def print_section(title: str):
    print(f"\n{BOLD}{MAGENTA}── {title} {'─' * (55 - len(title))}{RESET}\n")


def print_kv(key: str, value, indent: int = 2):
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}{DIM}{key}:{RESET}")
        for k, v in value.items():
            print_kv(k, v, indent + 4)
    elif isinstance(value, str) and len(value) > 100:
        print(f"{prefix}{DIM}{key}:{RESET} {value[:100]}...")
    else:
        print(f"{prefix}{DIM}{key}:{RESET} {value}")


def print_step_result(
    step_name: str, step_num: int, total: int, duration_ms: float, output_keys: list
):
    status = f"{CHECK}" if output_keys else f"{YELLOW}⏭️{RESET}"
    print(
        f"  {status} {BOLD}Step {step_num}/{total}{RESET}: {CYAN}{step_name}{RESET} {DIM}({duration_ms:.0f}ms){RESET}"
    )
    if output_keys:
        print(f"      {DIM}{ARROW} Output keys: {', '.join(output_keys)}{RESET}")


async def run_e2e(email_text: str, auto_create: bool = True, via_bus: bool = False):
    """Execute the full pipeline with detailed tracing."""
    from workflows.bank_to_ynab.workflow import BankToYnabWorkflow

    mode = "Event-Driven (via AgentBus)" if via_bus else "Direct Pipeline"
    print_header(f"Bank → YNAB — Full E2E [{mode}]")

    # ── Input ────────────────────────────────────────────────────────
    print_section("Input")
    print(f"  {DIM}Email:{RESET} {email_text[:120]}...")
    print(f"  {DIM}Auto-create:{RESET} {auto_create}")
    print(f"  {DIM}Mode:{RESET} {mode}")

    # ── Execute ──────────────────────────────────────────────────────
    print_section("Pipeline Execution")

    wf = BankToYnabWorkflow()
    await wf.setup()  # Register event subscribers (email.received, transaction.created)
    start = time.monotonic()

    if via_bus:
        # Event-driven: simulate what the Gmail webhook does
        from autopilot.core.bus import get_event_bus

        bus = get_event_bus()

        event_payload = {
            "email_id": "e2e-test-001",
            "sender": "alertasynotificaciones@bancolombia.com.co",
            "subject": "Compra realizada",
            "body": email_text,
            "label_ids": ["INBOX", "Bancos/Bancolombia"],
            "source": "e2e_test",
            "email": {
                "id": "e2e-test-001",
                "from": "alertasynotificaciones@bancolombia.com.co",
                "subject": "Compra realizada",
                "body": email_text,
                "labelIds": ["INBOX", "Bancos/Bancolombia"],
            },
        }

        print(f"  {CYAN}Publishing email.received to AgentBus...{RESET}")
        await bus.publish("email.received", event_payload, sender="e2e_test")
        elapsed = (time.monotonic() - start) * 1000

        # The bus awaited the subscriber, which called wf.run().
        # Get the result from the last run.
        if wf.last_run:
            result = wf.last_run
            # Adapt WorkflowRun to look like WorkflowResult for reporting
            from autopilot.models import WorkflowResult

            result = WorkflowResult(
                workflow_id=result.workflow_id,
                status=result.status,
                data=result.result,
                error=result.error,
                duration_ms=result.duration_ms,
            )
        else:
            print(f"  {RED}No workflow run was triggered by the event!{RESET}")
            return
    else:
        result = await wf.execute({"body": email_text, "auto_create": auto_create})
        elapsed = (time.monotonic() - start) * 1000

    # ── Status ───────────────────────────────────────────────────────
    status_color = GREEN if result.status.value == "success" else RED
    status_icon = CHECK if result.status.value == "success" else CROSS
    print(
        f"\n  {status_icon} {BOLD}{status_color}Status: {result.status.value.upper()}{RESET} {DIM}({elapsed:.0f}ms total){RESET}"
    )

    if result.error:
        print(f"\n  {RED}Error: {result.error}{RESET}")
        return

    data = result.data

    # ── Steps Completed ──────────────────────────────────────────────
    steps = data.get("__steps_completed__", [])
    if steps:
        print_section(f"Steps Completed ({len(steps)})")
        for i, step in enumerate(steps, 1):
            print(f"  {DIM}{i:2d}.{RESET} {step}")

    # ── Email Parsing ────────────────────────────────────────────────
    parsed = data.get("parsed_email")
    if parsed:
        print_section("Stage 1 — Email Parser")
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        print_kv("Date", parsed.get("date"))
        print_kv("Payee", parsed.get("payee"))
        print_kv("Amount", f"COP {parsed.get('amount', 0):,.0f}")
        print_kv("Card", f"*{parsed.get('card_suffix')}")
        print_kv("Successful", parsed.get("is_successful"))

    # ── Account Matching ─────────────────────────────────────────────
    matched = data.get("matched_account")
    if matched:
        print_section("Stage 2 — Account Matcher")
        if isinstance(matched, str):
            matched = json.loads(matched)
        print_kv(
            "Budget",
            f"{matched.get('budget_name')} ({matched.get('budget_id', '')[:8]}...)",
        )
        print_kv(
            "Account",
            f"{matched.get('account_name')} ({matched.get('account_id', '')[:8]}...)",
        )
        print_kv("Confidence", matched.get("match_confidence"))
        print_kv("Reasoning", matched.get("match_reasoning"))

    # ── Web Research ─────────────────────────────────────────────────
    enriched = data.get("enriched_payee")
    if enriched:
        print_section("Stage 3 — Web Researcher")
        if isinstance(enriched, str):
            enriched = json.loads(enriched)
        print_kv("Clean Name", enriched.get("clean_name"))
        print_kv("Type", enriched.get("establishment_type"))
        print_kv("Website", enriched.get("website"))
        print_kv("Location", enriched.get("location"))

    # ── Categorization ───────────────────────────────────────────────
    categorized = data.get("categorized_tx")
    if categorized:
        print_section("Stage 4 — Categorizer")
        if isinstance(categorized, str):
            categorized = json.loads(categorized)
        print_kv("Category", categorized.get("category_name"))
        print_kv("Category ID", categorized.get("category_id"))
        print_kv("Reasoning", categorized.get("category_reasoning"))

    # ── Transaction Synthesis ────────────────────────────────────────
    tx = data.get("transaction")
    if tx:
        print_section("Stage 5 — Synthesized Transaction")
        if isinstance(tx, str):
            tx = json.loads(tx)
        print_kv("Payee", tx.get("payee"))
        print_kv("Amount", f"COP {tx.get('amount', 0):,.0f}")
        print_kv("Date", tx.get("date"))
        print_kv("Budget ID", tx.get("budget_id"))
        print_kv("Account ID", tx.get("account_id"))
        print_kv("Category ID", tx.get("category_id"))

    # ── YNAB Result ──────────────────────────────────────────────────
    final = data.get("final_result_data")
    if final and isinstance(final, dict):
        print_section("Stage 6 — YNAB Push")
        print_kv("Created in YNAB", final.get("created_in_ynab"))
        print_kv("Transaction ID", final.get("ynab_transaction_id"))
        if final.get("skip_reason"):
            print_kv("Skip Reason", final.get("skip_reason"))

        # Category Balance
        balance = final.get("category_balance")
        if balance:
            print_section("Category Balance (Post-Transaction)")
            print_kv("Category", balance.get("category_name"))
            print_kv("Budgeted", f"COP {balance.get('budgeted', 0):,.0f}")
            print_kv("Activity", f"COP {balance.get('activity', 0):,.0f}")
            print_kv("Balance", f"COP {balance.get('balance', 0):,.0f}")
            is_overspent = balance.get("is_overspent", False)
            overspent_str = (
                f"{RED}YES ⚠️{RESET}" if is_overspent else f"{GREEN}No{RESET}"
            )
            print(f"  {DIM}Overspent:{RESET} {overspent_str}")

        if final.get("overspending_warning"):
            print(f"\n  {YELLOW}{final['overspending_warning']}{RESET}")

    # ── Telegram Notification ────────────────────────────────────────
    telegram_output = data.get("output")
    if telegram_output:
        print_section("Stage 7 — Telegram Notification")
        print(f"  {DIM}{telegram_output}{RESET}")

    # ── Summary ──────────────────────────────────────────────────────
    print_header("E2E Complete")
    print(f"  Total time: {BOLD}{elapsed:,.0f}ms{RESET}")
    print(f"  Steps:      {BOLD}{len(steps)}{RESET}")
    print(
        f"  Status:     {status_icon} {BOLD}{status_color}{result.status.value.upper()}{RESET}"
    )
    print()


if __name__ == "__main__":
    email = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else DEFAULT_EMAIL
    )
    auto = "--no-create" not in sys.argv
    via_bus = "--direct" not in sys.argv  # Default: event-driven via AgentBus
    asyncio.run(run_e2e(email, auto_create=auto, via_bus=via_bus))
