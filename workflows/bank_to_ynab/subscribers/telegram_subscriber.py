"""
Telegram Subscriber — Reactive event handler for transaction notifications.

Subscribes to ``transaction.created`` on the AgentBus and sends a
natural-language Telegram notification via the existing LLM notifier agent.

This subscriber runs concurrently with (and independently of) the main
pipeline. Errors are isolated via the bus's dead-letter pattern — a failing
Telegram send never affects pipeline success.

Design:
  - Receives ``AgentMessage`` from the bus
  - Extracts the ``TransactionEvent`` payload
  - Formats pipeline state for the LLM agent (same context the old
    ``format_notifier_input`` step produced)
  - Runs the existing ``create_telegram_notifier()`` LLM agent
"""

import structlog

from autopilot.core.bus import AgentMessage

logger = structlog.get_logger(__name__)


def _format_notifier_context(payload: dict) -> dict:
    """
    Build the state dict that the Telegram notifier LLM agent expects.

    Produces the same ``final_result_data`` and ``category_balance``
    strings that the old ``format_notifier_input`` pipeline step created,
    preserving full backward compatibility with the agent's instruction
    template.
    """
    cat_balance = payload.get("category_balance", {})

    # Build a clean summary of the transaction (excluding nested balance)
    tx_lines = []
    for key, value in payload.items():
        if key == "category_balance":
            continue  # handled separately
        tx_lines.append(f"  {key}: {value}")

    final_result_str = "\n".join(tx_lines)

    # Build an explicit, labeled category balance string
    if cat_balance:
        cat_str = (
            f"Categoría: {cat_balance.get('category_name', 'N/A')}\n"
            f"Presupuesto del mes: ${cat_balance.get('budgeted', 0):,.0f}\n"
            f"Gastado en el mes: ${abs(cat_balance.get('activity', 0)):,.0f}\n"
            f"Disponible real (incluye rollover): ${cat_balance.get('balance', 0):,.0f}\n"
            f"Sobrepasado: {'Sí' if cat_balance.get('is_overspent') else 'No'}"
        )
    else:
        cat_str = "No disponible"

    return {
        "final_result_data": final_result_str,
        "category_balance": cat_str,
        "message": f"Envía la notificación de esta transacción:\n{final_result_str}",
    }


async def on_transaction_created(msg: AgentMessage) -> None:
    """
    Reactive subscriber: send Telegram notification via LLM agent.

    Triggered by ``transaction.created`` events on the AgentBus.
    Runs the existing ``create_telegram_notifier()`` LLM agent to format
    a natural-language message and send it via the Telegram connector.
    """
    from autopilot.core.context import AgentContext
    from autopilot.core.agent import ADKAgent
    from workflows.bank_to_ynab.agents.telegram_notifier import (
        create_telegram_notifier,
    )

    payload = msg.payload
    if not payload:
        logger.info("telegram_subscriber_skipped", reason="empty payload")
        return

    logger.info(
        "telegram_subscriber_triggered",
        topic=msg.topic,
        payee=payload.get("payee", ""),
        amount=payload.get("amount", 0),
        correlation_id=msg.correlation_id,
    )

    # Build the context the LLM agent expects
    notifier_state = _format_notifier_context(payload)

    # Run the existing LLM notifier agent
    adk_agent = create_telegram_notifier()
    agent = ADKAgent(adk_agent)
    ctx = AgentContext(
        pipeline_name="telegram_subscriber",
        metadata={"source": "agentbus", "correlation_id": msg.correlation_id},
    )
    ctx.update_state(notifier_state)

    await agent.invoke(ctx, notifier_state)

    logger.info(
        "telegram_subscriber_completed",
        payee=payload.get("payee", ""),
        correlation_id=msg.correlation_id,
    )
