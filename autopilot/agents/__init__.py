"""
Platform Agents — ADK agent orchestration as a platform capability.

Provides:
  - PipelineRunner: High-level pipeline execution engine
  - Observability callbacks: before/after model logging + Prometheus + SSE
  - Tool callbacks: before/after tool logging + Prometheus + SSE
  - Reusable guardrails: Input/output validation factories
  - Callback composition: Chaining multiple callbacks
  - JSON utilities: Robust extraction from LLM output
"""

from autopilot.agents.pipeline_runner import PipelineRunner, get_pipeline_runner
from autopilot.agents.callbacks import (
    before_model_logger,
    after_model_logger,
    pipeline_session_id,
    create_chained_before_callback,
    create_chained_after_callback,
)
from autopilot.agents.tool_callbacks import (
    before_tool_logger,
    after_tool_logger,
)
from autopilot.agents.guardrails import (
    input_length_guard,
    prompt_injection_guard,
    uuid_format_guard,
)
from autopilot.agents.json_utils import extract_json
from autopilot.agents.agent_cards import load_agent_card, discover_agent_cards

__all__ = [
    # Pipeline execution
    "PipelineRunner",
    "get_pipeline_runner",
    # Model-level observability callbacks
    "before_model_logger",
    "after_model_logger",
    "pipeline_session_id",
    # Tool-level observability callbacks
    "before_tool_logger",
    "after_tool_logger",
    # Callback composition
    "create_chained_before_callback",
    "create_chained_after_callback",
    # Guardrail factories
    "input_length_guard",
    "prompt_injection_guard",
    "uuid_format_guard",
    # JSON utilities
    "extract_json",
    # Agent cards
    "load_agent_card",
    "discover_agent_cards",
]
