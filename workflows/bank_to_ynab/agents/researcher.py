"""
Web Researcher Agent Factory for Bank→YNAB Workflow.

Resolves merchant names and details to enrich the transaction payload.
"""

from pathlib import Path
from typing import Any

from autopilot.agents.base import create_platform_agent
from workflows.bank_to_ynab.models.transaction import EnrichedPayee

def create_researcher(**kwargs: Any) -> Any:
    """
    Creates the Web Researcher agent using the platform factory.
    
    The agent uses the declarative configuration in researcher.agent.yaml
    and strictly returns an EnrichedPayee model.
    """
    return create_platform_agent(
        name="researcher",
        description="Resolves merchant/payee entities by searching the web to enrich transactions.",
        output_key="enriched_payee",
        output_schema=EnrichedPayee,
        instruction=(
            "You are a meticulous financial intelligence agent.\n"
            "Your task is to take a raw bank transaction payee/merchant string and identify "
            "the real-world entity behind it.\n\n"
            "Use your internal knowledge first. If the entity is ambiguous or unknown, use "
            "the `search_web` tool to look it up.\n\n"
            "Rules:\n"
            "1. Extract the clean, recognized name of the business/establishment.\n"
            "2. Determine the type of establishment (e.g., 'Supermarket', 'Restaurant', 'Software Provider').\n"
            "3. If applicable, provide the official website URL.\n"
            "4. If location info is in the raw string or found via search, extract it.\n"
            "5. If you cannot find any information, return the original payee string as clean_name and "
            "'Unknown' for establishment_type.\n\n"
            "Always be as precise as possible."
        ),
        tools=["search_web"],
        **kwargs
    )
