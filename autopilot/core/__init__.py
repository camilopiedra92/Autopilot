"""
AutoPilot Core — V3 Edge-Native Agentic Primitives.

This package provides the foundational building blocks for the platform:
  - AgentContext: Rich execution context (tracing, events, state, session, memory, tools)
  - BaseAgent[In, Out]: Strictly typed agent interface
  - Pipeline / PipelineBuilder: Declarative, typed workflow composition
  - DAGBuilder / DAGRunner: Graph-based parallel orchestration (V3 Phase 2)
  - FunctionalAgent: Zero-boilerplate wrapper for pure functions
  - V3 Adapters: SequentialAgentAdapter, LoopAgentAdapter, ParallelAgentAdapter
  - OrchestrationStrategy: Multi-strategy enum (SEQUENTIAL, DAG, REACT, ROUTER)
  - Session & Memory: Short-term state + long-term semantic memory (V3 Phase 3)
  - Tool Ecosystem: ToolRegistry, @tool decorator, MCP Bridge (V3 Phase 4)
  - Agent Bus: Typed pub/sub messaging for A2A communication (V3 Phase 5)
"""

from autopilot.core.context import AgentContext
from autopilot.core.agent import (
    BaseAgent,
    FunctionalAgent,
    SequentialAgentAdapter,
    LoopAgentAdapter,
    ParallelAgentAdapter,
)
from autopilot.core.pipeline import Pipeline, PipelineBuilder
from autopilot.core.dag import DAGBuilder, DAGRunner
from autopilot.core.orchestrator import OrchestrationStrategy
from autopilot.core.session import BaseSessionService, InMemorySessionService
from autopilot.core.memory import (
    BaseMemoryService,
    InMemoryMemoryService,
    Observation,
)
from autopilot.core.tools import (
    ToolInfo,
    ToolRegistry,
    get_tool_registry,
    tool,
    MCPBridge,
    MCPRegistry,
)
from autopilot.core.bus import (
    AgentBus,
    AgentMessage,
    Subscription,
    get_agent_bus,
    reset_agent_bus,
)
from autopilot.core.dsl_schema import (
    DSLWorkflowDef,
    DSLStepDef,
    DSLNodeDef,
    DSLStepType,
    DSLStrategy,
)
from autopilot.core.dsl_loader import load_workflow, load_workflow_from_dict

__all__ = [
    "AgentContext",
    "BaseAgent",
    "FunctionalAgent",
    "SequentialAgentAdapter",
    "LoopAgentAdapter",
    "ParallelAgentAdapter",
    "Pipeline",
    "PipelineBuilder",
    "DAGBuilder",
    "DAGRunner",
    "OrchestrationStrategy",
    "BaseSessionService",
    "InMemorySessionService",
    "BaseMemoryService",
    "InMemoryMemoryService",
    "Observation",
    # V3 Phase 4 — Tool Ecosystem
    "ToolInfo",
    "ToolRegistry",
    "get_tool_registry",
    "tool",
    "MCPBridge",
    "MCPRegistry",
    # V3 Phase 5 — Agent Bus (A2A)
    "AgentBus",
    "AgentMessage",
    "Subscription",
    "get_agent_bus",
    "reset_agent_bus",
    # V3 Phase 6 — Declarative DSL
    "DSLWorkflowDef",
    "DSLStepDef",
    "DSLNodeDef",
    "DSLStepType",
    "DSLStrategy",
    "load_workflow",
    "load_workflow_from_dict",
]


