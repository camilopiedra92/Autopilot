"""
OrchestrationStrategy — Declares how a workflow's agents are coordinated.

V3 introduces multi-strategy orchestration, moving beyond the legacy model
of strictly sequential pipelines.  Each strategy corresponds to a
different execution engine:

  - SEQUENTIAL: Linear A→B→C (existing Pipeline engine).
  - DAG: Directed Acyclic Graph with topological parallel execution.
  - REACT: (Future) Reasoning + Acting loops with dynamic tool selection.
  - ROUTER: (Future) LLM-based routing to the best-fit sub-agent.
"""

from enum import Enum


class OrchestrationStrategy(str, Enum):
    """
    Supported orchestration strategies for workflow execution.

    Inherits from ``str`` so values serialize cleanly to JSON/YAML.
    """

    SEQUENTIAL = "sequential"
    """Default. Linear pipeline: each step feeds the next."""

    DAG = "dag"
    """Directed Acyclic Graph: topological ordering + parallel layers."""

    REACT = "react"
    """(Future) ReAct loop: Reason → Act → Observe → repeat."""

    ROUTER = "router"
    """(Future) Dynamic routing: LLM picks the best sub-agent per request."""
