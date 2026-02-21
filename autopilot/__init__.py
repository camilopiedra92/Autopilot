"""
Platform — Multi-Workflow AI Automation Core.

Provides the shared infrastructure for declaring, discovering,
routing, and managing AI workflows, connectors, and agent orchestration.
"""

from autopilot.base_workflow import BaseWorkflow
from autopilot.models import WorkflowManifest, WorkflowResult
from autopilot.registry import WorkflowRegistry, get_registry
from autopilot.router import WorkflowRouter, get_router
from autopilot.connectors import ConnectorRegistry, get_connector_registry
from autopilot.agents import PipelineRunner, get_pipeline_runner
from autopilot.models import PipelineResult
from autopilot.core import (
    AgentContext,
    BaseAgent,
    FunctionalAgent,
    Pipeline,
    PipelineBuilder,
    DAGBuilder,
    DAGRunner,
    OrchestrationStrategy,
    ToolRegistry,
    get_tool_registry,
    AgentBus,
    get_agent_bus,
)

__all__ = [
    # Workflow framework
    "BaseWorkflow",
    "WorkflowManifest",
    "WorkflowResult",
    "WorkflowRegistry",
    "WorkflowRouter",
    "ConnectorRegistry",
    "get_registry",
    "get_router",
    "get_connector_registry",
    # Core Primitives
    "AgentContext",
    "BaseAgent",
    "FunctionalAgent",
    "Pipeline",
    "PipelineBuilder",
    # V3 DAG Orchestration
    "DAGBuilder",
    "DAGRunner",
    "OrchestrationStrategy",
    # V3 Tool Ecosystem
    "ToolRegistry",
    "get_tool_registry",
    # V3 Agent Bus (A2A)
    "AgentBus",
    "get_agent_bus",
    # ADK Bridge (internal)
    "PipelineRunner",
    "get_pipeline_runner",
    "PipelineResult",
]
