"""
Platform Models — Shared Pydantic models for the workflow framework.

Defines the core data structures used across all workflows:
  - WorkflowManifest: Declarative workflow metadata
  - TriggerConfig: How a workflow is triggered
  - SettingConfig: Workflow-specific settings
  - WorkflowResult: Standard result from workflow execution
  - WorkflowRun: Record of a single workflow execution
  - WorkflowInfo: Summary info for listing
  - AgentCard: Declarative agent metadata (YAML-driven)
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Trigger Configuration ────────────────────────────────────────────


class TriggerType(str, enum.Enum):
    """Supported trigger types for workflows."""

    WEBHOOK = "webhook"
    GMAIL_PUSH = "gmail_push"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FILE_UPLOAD = "file_upload"


class TriggerConfig(BaseModel):
    """Configuration for a single workflow trigger."""

    type: TriggerType
    # Webhook-specific
    path: str | None = None
    # Gmail push-specific
    filter: str | None = None
    label_ids: list[str] = Field(default_factory=lambda: ["INBOX"])
    # Scheduled-specific (cron expression)
    cron: str | None = None
    # Human-readable description
    description: str = ""


# ── Settings Configuration ───────────────────────────────────────────


class SettingType(str, enum.Enum):
    STRING = "string"
    SECRET = "secret"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"


class SettingConfig(BaseModel):
    """A single configurable setting for a workflow."""

    key: str
    type: SettingType = SettingType.STRING
    required: bool = False
    default: Any = None
    description: str = ""
    options: list[str] | None = None  # For SELECT type


# ── Workflow Manifest ────────────────────────────────────────────────


class WorkflowManifest(BaseModel):
    """
    Declarative manifest describing a workflow.

    Can be loaded from manifest.yaml or defined in code.
    Tells the platform everything it needs to know to
    register, route, and display the workflow.
    """

    name: str  # Unique ID (e.g. "bank_to_ynab")
    display_name: str  # Human-readable name
    description: str = ""
    version: str = "1.0.0"
    icon: str = "⚡"  # Emoji or icon identifier
    color: str = "#6366f1"  # Theme color (hex)

    triggers: list[TriggerConfig] = Field(default_factory=list)
    settings: list[SettingConfig] = Field(default_factory=list)

    # Agent names (for display/documentation)
    agents: list[str] = Field(default_factory=list)

    # Tags for filtering/categorization
    tags: list[str] = Field(default_factory=list)

    # Is this workflow currently enabled?
    enabled: bool = True


# ── Workflow Execution ───────────────────────────────────────────────


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowResult(BaseModel):
    """Standard result from a workflow execution."""

    workflow_id: str
    status: RunStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0
    stages_completed: list[str] = Field(default_factory=list)


class WorkflowRun(BaseModel):
    """Record of a single workflow execution for history/audit."""

    id: str
    workflow_id: str
    status: RunStatus
    trigger_type: TriggerType
    trigger_data: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float = 0


class WorkflowInfo(BaseModel):
    """Summary info for listing workflows."""

    name: str
    display_name: str
    description: str
    version: str
    icon: str
    color: str
    enabled: bool
    triggers: list[TriggerConfig]
    tags: list[str]
    last_run: WorkflowRun | None = None
    total_runs: int = 0
    success_rate: float = 0.0


# ── Pipeline Execution ───────────────────────────────────────────────


class PipelineResult(BaseModel):
    """
    Structured result from a PipelineRunner execution.

    Returned by PipelineRunner.run() after a multi-agent pipeline completes.
    Contains the raw final text, extracted JSON, final session state,
    and timing information.
    """

    session_id: str
    final_text: str
    parsed_json: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0


# ── Agent Cards (Declarative Agent Metadata) ─────────────────────────


class AgentType(str, enum.Enum):
    """Type of agent implementation."""

    LLM = "llm"  # LlmAgent — uses an LLM for reasoning
    CODE = "code"  # Pure Python, zero LLM calls
    CUSTOM = "custom"  # BaseAgent subclass (custom _run_async_impl)


class AgentCardIO(BaseModel):
    """Input or output schema reference for an agent card."""

    schema_ref: str = Field(description="Pydantic model name (e.g., 'ParsedEmail')")
    fields: list[str] = Field(
        default_factory=list,
        description="Key fields for quick reference",
    )


class GuardrailConfig(BaseModel):
    """Guardrail (callback) configuration for an agent."""

    before_model: list[str] = Field(default_factory=list)
    after_model: list[str] = Field(default_factory=list)


class ToolConfig(BaseModel):
    """Tool reference for an agent card."""

    name: str
    description: str = ""


class AgentCard(BaseModel):
    """
    Declarative agent metadata — loaded from .agent.yaml files.

    Describes WHAT an agent does (capabilities, I/O, tools, guardrails)
    while the Python .py file defines HOW it works (implementation).

    Compatible with Google A2A protocol for agent discovery.
    """

    name: str = Field(description="Unique agent identifier (e.g., 'email_parser')")
    display_name: str = Field(description="Human-readable name (e.g., 'Email Parser')")
    version: str = "1.0.0"
    description: str = ""
    type: AgentType = AgentType.LLM
    model: str | None = Field(
        default=None,
        description="AI model used. Only for LLM/CUSTOM agents.",
    )
    stage: int = Field(
        default=0,
        description="Pipeline execution order (1, 2, 3...)",
    )

    input: AgentCardIO | None = None
    output: AgentCardIO | None = None

    tools: list[ToolConfig] = Field(default_factory=list)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)

    state_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Session state keys this agent reads/writes (e.g., {'reads': 'parsed_email', 'writes': 'matched_account'})",
    )

    tags: list[str] = Field(default_factory=list)
