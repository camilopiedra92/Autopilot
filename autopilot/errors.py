"""
Structured Error Taxonomy — Typed exceptions for the AutoPilot platform.

Design principles:
  - Every error carries `retryable` + `error_code` for automated decisions
  - Hierarchy mirrors the platform layers: Pipeline → Agent → Connector → Guardrail
  - HTTP-safe: each class maps to a recommended status code
  - Structured logging friendly: all errors serialize cleanly to JSON
"""

from __future__ import annotations

__all__ = [
    # Base
    "AutoPilotError",
    # Pipeline layer
    "PipelineError",
    "PipelineTimeoutError",
    "PipelineEmptyResponseError",
    "MaxRetriesExceededError",
    # Agent layer
    "AgentError",
    "LLMRateLimitError",
    "LLMContentFilterError",
    "AgentOutputParseError",
    # Connector layer
    "ConnectorError",
    "ConnectorUnavailableError",
    "ConnectorAuthError",
    "ConnectorRateLimitError",
    # Guardrail layer
    "GuardrailError",
    "GuardrailBlockedError",
    "GuardrailValidationError",
    # DAG layer
    "DAGCycleError",
    "DAGDependencyError",
    # Session & Memory layer (V3 Phase 3)
    "SessionError",
    "MemoryServiceError",
    # Tool Ecosystem layer (V3 Phase 4)
    "ToolRegistryError",
    "MCPBridgeError",
    # Specifics
    "ToolExecutionError",
    # Tool Lifecycle (V3 Phase 7)
    "ToolCallbackError",
    "ToolAuthError",
    # Bus Layer (V3 Phase 5)
    "BusError",
    "BusTimeoutError",
    # DSL Layer (V3 Phase 6)
    "DSLValidationError",
    "DSLResolutionError",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Base
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AutoPilotError(Exception):
    """Root exception for the AutoPilot platform.

    Attributes:
        retryable: If True, the caller should consider retrying the operation.
        error_code: Machine-readable code for dashboards and alerting.
        http_status: Suggested HTTP status code for API responses.
    """

    retryable: bool = False
    error_code: str = "AUTOPILOT_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, detail: str | None = None):
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> dict:
        """Serialize for structured logging and API responses."""
        return {
            "error_code": self.error_code,
            "message": str(self),
            "detail": self.detail,
            "retryable": self.retryable,
            "http_status": self.http_status,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline Layer — Errors in the multi-agent pipeline execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PipelineError(AutoPilotError):
    """Base for all pipeline execution errors."""

    error_code = "PIPELINE_ERROR"


class PipelineTimeoutError(PipelineError):
    """Pipeline exceeded the maximum allowed execution time."""

    retryable = True
    error_code = "PIPELINE_TIMEOUT"
    http_status = 504


class PipelineEmptyResponseError(PipelineError):
    """Pipeline completed but returned no usable output."""

    retryable = True
    error_code = "PIPELINE_EMPTY_RESPONSE"
    http_status = 502


class MaxRetriesExceededError(PipelineError):
    """Loop agent exhausted all iterations without meeting the exit condition."""

    retryable = False
    error_code = "MAX_RETRIES_EXCEEDED"
    http_status = 422

    def __init__(self, message: str, *, iterations: int = 0, **kwargs):
        self.iterations = iterations
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["iterations"] = self.iterations
        return d


class DAGCycleError(PipelineError):
    """DAG contains a cycle and cannot be topologically sorted."""

    retryable = False
    error_code = "DAG_CYCLE"
    http_status = 422

    def __init__(self, message: str, *, nodes: list[str] | None = None, **kwargs):
        self.nodes = nodes or []
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["cycle_nodes"] = self.nodes
        return d


class DAGDependencyError(PipelineError):
    """DAG node references an unknown dependency."""

    retryable = False
    error_code = "DAG_DEPENDENCY"
    http_status = 422

    def __init__(self, message: str, *, node: str = "", dependency: str = "", **kwargs):
        self.node = node
        self.dependency = dependency
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["node"] = self.node
        d["dependency"] = self.dependency
        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agent Layer — Errors from individual LLM agents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AgentError(AutoPilotError):
    """Base for all agent-level errors."""

    error_code = "AGENT_ERROR"

    def __init__(self, message: str, *, agent_name: str | None = None, **kwargs):
        self.agent_name = agent_name or getattr(self, "agent_name", None)
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["agent_name"] = self.agent_name
        return d


class LLMRateLimitError(AgentError):
    """LLM provider returned a rate limit / quota exceeded error."""

    retryable = True
    error_code = "LLM_RATE_LIMIT"
    http_status = 429


class LLMContentFilterError(AgentError):
    """LLM provider blocked the request due to content safety filters."""

    retryable = False
    error_code = "LLM_CONTENT_FILTER"
    http_status = 422


class AgentOutputParseError(AgentError):
    """Agent output could not be parsed as the expected format (JSON, etc.)."""

    retryable = True
    error_code = "AGENT_OUTPUT_PARSE"
    http_status = 502


class ToolExecutionError(AgentError):
    """An agent's tool failed to execute."""

    retryable = True
    error_code = "TOOL_EXECUTION_ERROR"
    http_status = 502


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Connector Layer — Errors from external service integrations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConnectorError(AutoPilotError):
    """Base for all connector/integration errors."""

    error_code = "CONNECTOR_ERROR"

    def __init__(self, message: str, *, connector_name: str | None = None, **kwargs):
        self.connector_name = connector_name or getattr(self, "connector_name", None)
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["connector_name"] = self.connector_name
        return d


class ConnectorUnavailableError(ConnectorError):
    """External service is unreachable or returning errors."""

    retryable = True
    error_code = "CONNECTOR_UNAVAILABLE"
    http_status = 503


class ConnectorAuthError(ConnectorError):
    """Authentication/authorization failed for the external service."""

    retryable = False
    error_code = "CONNECTOR_AUTH"
    http_status = 401


class ConnectorRateLimitError(ConnectorError):
    """External service returned a rate limit error."""

    retryable = True
    error_code = "CONNECTOR_RATE_LIMIT"
    http_status = 429


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Guardrail Layer — Errors from safety/validation guardrails
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GuardrailError(AutoPilotError):
    """Base for all guardrail errors."""

    error_code = "GUARDRAIL_ERROR"

    def __init__(self, message: str, *, guardrail_name: str | None = None, **kwargs):
        self.guardrail_name = guardrail_name or getattr(self, "guardrail_name", None)
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["guardrail_name"] = self.guardrail_name
        return d


class GuardrailBlockedError(GuardrailError):
    """Guardrail blocked the request (e.g., prompt injection detected)."""

    retryable = False
    error_code = "GUARDRAIL_BLOCKED"
    http_status = 422


class GuardrailValidationError(GuardrailError):
    """Guardrail validation failed (e.g., invalid amount, bad UUID)."""

    retryable = True
    error_code = "GUARDRAIL_VALIDATION"
    http_status = 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Session & Memory Layer — Errors from V3 Phase 3 services
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SessionError(AutoPilotError):
    """Base for all session service errors."""

    error_code = "SESSION_ERROR"
    http_status = 500


class MemoryServiceError(AutoPilotError):
    """Base for all memory service errors."""

    error_code = "MEMORY_ERROR"
    http_status = 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tool Ecosystem Layer — Errors from V3 Phase 4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolRegistryError(AutoPilotError):
    """Error from the centralized tool registry (duplicate name, not found)."""

    retryable = False
    error_code = "TOOL_REGISTRY_ERROR"
    http_status = 422

    def __init__(self, message: str, *, tool_name: str = "", **kwargs):
        self.tool_name = tool_name
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        return d


class MCPBridgeError(AutoPilotError):
    """Error connecting to or using an external MCP server."""

    retryable = True
    error_code = "MCP_BRIDGE_ERROR"
    http_status = 502

    def __init__(self, message: str, *, server_name: str = "", **kwargs):
        self.server_name = server_name
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["server_name"] = self.server_name
        return d


class ToolCallbackError(AutoPilotError):
    """A tool lifecycle callback (before/after) failed."""

    retryable = False
    error_code = "TOOL_CALLBACK_ERROR"
    http_status = 500

    def __init__(
        self, message: str, *, callback_name: str = "", tool_name: str = "", **kwargs
    ):
        self.callback_name = callback_name
        self.tool_name = tool_name
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["callback_name"] = self.callback_name
        d["tool_name"] = self.tool_name
        return d


class ToolAuthError(AutoPilotError):
    """Authentication/credential error for a platform tool."""

    retryable = False
    error_code = "TOOL_AUTH_ERROR"
    http_status = 401

    def __init__(self, message: str, *, tool_name: str = "", **kwargs):
        self.tool_name = tool_name
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Bus Layer — Errors from V3 Phase 5 (Agent Bus A2A)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BusError(AutoPilotError):
    """Base for all Agent Bus errors."""

    error_code = "BUS_ERROR"
    http_status = 500


class BusTimeoutError(BusError):
    """Bus operation timed out (e.g. waiting for a message)."""

    retryable = True
    error_code = "BUS_TIMEOUT"
    http_status = 504


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSL Layer — Errors from V3 Phase 6 (Declarative DSL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DSLValidationError(AutoPilotError):
    """YAML workflow definition failed schema validation."""

    retryable = False
    error_code = "DSL_VALIDATION"
    http_status = 422

    def __init__(self, message: str, *, field: str = "", **kwargs):
        self.field = field
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["field"] = self.field
        return d


class DSLResolutionError(AutoPilotError):
    """Cannot resolve a dotted ref path to a callable or agent."""

    retryable = False
    error_code = "DSL_RESOLUTION"
    http_status = 422

    def __init__(self, message: str, *, ref: str = "", **kwargs):
        self.ref = ref
        super().__init__(message, **kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["ref"] = self.ref
        return d
