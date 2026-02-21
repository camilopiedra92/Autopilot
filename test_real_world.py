import asyncio
from dotenv import load_dotenv
import json

# Ensure we load env vars so we can use Google GenAI
load_dotenv()

from autopilot.registry import WorkflowRegistry
from autopilot.models import TriggerType


async def main():
    print("Loading registry...")
    registry = WorkflowRegistry()

    # We load workflows using discover (we assume it's load_all or discover)
    # Check what method it uses
    registry.discover()
    workflow = registry.get_or_raise("bank_to_ynab")

    print(f"Executing workflow: {workflow.manifest.name}")

    payload = {
        "body": """
        <html><body>
        <p>Bancolombia le informa que se ha realizado una Compra por $ 25.000,00 en UBER TRIP el 20/02/2026 a las 18:30 desde la cuenta *1234. Si tiene dudas comuníquese con la sucursal.</p>
        </body></html>
        """,
        "auto_create": False,  # so we don't actually modify their real YNAB account unless requested
    }

    # run workflow.run(...)
    result = await workflow.run(TriggerType.MANUAL, payload)

    print("--- WORKFLOW RESULT ---")
    print(f"Status: {result.status}")
    print(f"Error: {result.error}")
    print(f"Duration: {result.duration_ms} ms")
    if result.result:
        print(json.dumps(result.result, indent=2, ensure_ascii=False))
    else:
        print("Empty Result Data")


if __name__ == "__main__":
    asyncio.run(main())
