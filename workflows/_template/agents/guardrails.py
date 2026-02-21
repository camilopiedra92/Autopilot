"""
Workflow-Specific Guardrails.

This file demonstrates where to place domain-specific guardrails.
Platform-level guardrails (like input limits or injection checks) 
live in `autopilot.agents.guardrails` and should be reused.

Domain-specific logic (e.g., checking specific business rules mapped
to your YAML or database) belongs here.

Example:
    from autopilot.agents.callbacks import create_chained_after_callback
    from workflows.my_workflow.agents.guardrails import my_domain_guard

    agent = create_platform_agent(
        ...
        after_model_callback=create_chained_after_callback(
            my_domain_guard(),
            # ... other platform guards
        )
    )
"""

# from google.adk.agents.callback_context import CallbackContext
# from google.adk.models import LlmResponse
# from typing import Optional, Callable
# 
# def my_domain_guard() -> Callable:
#     """Example custom after-model guardrail."""
#     def _guard(context: CallbackContext, response: LlmResponse) -> Optional[LlmResponse]:
#         # Implement domain logic here (e.g. check output against a business rule)
#         return None
#     return _guard
