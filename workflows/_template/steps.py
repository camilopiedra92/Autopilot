"""
DSL Helpers — Pure functions used by the example DSL workflow.

These are simple, self-contained functions designed to demonstrate
the Declarative DSL (Phase 6).  They carry zero side effects and
require no external dependencies — perfect for tests and templates.
"""


def parse_input(raw_text: str = "") -> dict:
    """Parse raw input text into structured data."""
    return {
        "parsed": raw_text.upper() if raw_text else "EMPTY",
        "char_count": len(raw_text),
    }


# Track calls for testing loop behaviour
_validate_call_count = 0


def validate_data(parsed: str = "", char_count: int = 0) -> dict:
    """Validate parsed data — succeeds when char_count > 0."""
    global _validate_call_count
    _validate_call_count += 1
    valid = char_count > 0 and len(parsed) > 0
    return {"valid": valid, "validation_attempt": _validate_call_count}


def reset_validate_counter() -> None:
    """Reset the validation call counter (for tests)."""
    global _validate_call_count
    _validate_call_count = 0


def fetch_source_a() -> dict:
    """Simulate fetching data from source A."""
    return {"source_a": {"provider": "Alpha", "score": 0.95}}


def fetch_source_b() -> dict:
    """Simulate fetching data from source B."""
    return {"source_b": {"provider": "Beta", "score": 0.88}}


def merge_results(
    parsed: str = "",
    source_a: dict | None = None,
    source_b: dict | None = None,
) -> dict:
    """Merge all enriched data into a final output."""
    return {
        "final_output": {
            "input": parsed,
            "enrichments": [
                source_a or {},
                source_b or {},
            ],
            "status": "complete",
        }
    }


# ---------------------------------------------------------------------------
# Architecture Examples: Tool Ecosystem & Custom Logic
# ---------------------------------------------------------------------------
#
# Registering reusable custom logical tools for LLM agents:
#
# NOTE: ONLY use @tool for workflow-specific custom logic.
# NEVER wrap Platform Connectors (like YNAB, Gmail) with @tool;
# the platform lazily auto-resolves those when agents reference them
#
# from autopilot.core.tools import tool
#
# @tool(tags=["template"])
# async def my_custom_logic(param: str) -> dict:
#     """Description of what this tool does."""
#     return {"result": param}
#
# For tools that need session state or auth:
# from google.adk.tools import ToolContext
# @tool(tags=["template"])
# async def auth_tool(param: str, tool_context: ToolContext) -> dict:
#     api_key = tool_context.state.get("API_KEY")
#     ...
#
# Remember to pass tools as strings (e.g., tools=["my_tool", "ynab.get_accounts"])
# Also consider attaching platform/domain guardrails directly to the agent callbacks
# and using ctx.publish("my_topic", data) for decoupled AgentBus communication.
