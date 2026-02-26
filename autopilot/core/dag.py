"""
DAG — Directed Acyclic Graph orchestration engine (V3 Phase 2).

Provides a builder and runner for executing agents as a dependency graph
instead of a linear pipeline.  Nodes without mutual dependencies run in
parallel via ``asyncio.gather``, maximising throughput.

Usage::

    from autopilot.core.dag import DAGBuilder

    dag = (
        DAGBuilder("analytics")
        .node("fetch", fetch_agent)
        .node("text", text_analyzer, dependencies=["fetch"])
        .node("image", image_analyzer, dependencies=["fetch"])
        .node("merge", merger, dependencies=["text", "image"])
        .build()
    )

    ctx = AgentContext(pipeline_name="analytics")
    result = await dag.execute(ctx, initial_input={"url": "..."})

The diamond pattern above executes as three layers:
  Layer 0: [fetch]          — sequential root
  Layer 1: [text, image]    — parallel
  Layer 2: [merge]          — waits for both
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from autopilot.core._artifact_persist import persist_node_artifact
from autopilot.core.agent import BaseAgent
from autopilot.core.context import AgentContext
from autopilot.core.pipeline import PipelineExecutionResult, _wrap_step, StepLike
from autopilot.errors import DAGCycleError, DAGDependencyError

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DAGNode — Internal representation of a graph vertex
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class DAGNode:
    """A single vertex in the execution graph."""

    name: str
    agent: BaseAgent
    dependencies: list[str] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DAGRunner — Async topological execution engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DAGRunner:
    """
    Executes a validated DAG of agents in topological order.

    Nodes are grouped into *layers*.  All nodes within the same layer
    have their dependencies satisfied and execute concurrently via
    ``asyncio.gather``.  Results are merged into a shared state dict
    that subsequent layers can read from.

    After each node completes, its output is automatically persisted as
    a versioned JSON artifact (``{node_name}.json``) in the configured
    artifact store (InMemory for dev, GCS for production).

    This class is not constructed directly — use ``DAGBuilder.build()``.
    """

    def __init__(self, name: str, nodes: dict[str, DAGNode], layers: list[list[str]]):
        self.name = name
        self._nodes = nodes
        self._layers = layers

    async def execute(
        self,
        ctx: AgentContext | None = None,
        *,
        initial_input: dict[str, Any] | None = None,
    ) -> PipelineExecutionResult:
        """
        Execute the DAG with full lifecycle management.

        Args:
            ctx: Execution context.  A new one is created if ``None``.
            initial_input: Seed state for the graph.

        Returns:
            PipelineExecutionResult with final state, steps completed,
            duration, and success flag.
        """
        # ── Context setup ─────────────────────────────────────────────
        if ctx is None:
            ctx = AgentContext(pipeline_name=self.name)
        elif not ctx.pipeline_name:
            ctx.pipeline_name = self.name

        if initial_input:
            ctx.update_state(initial_input)

        # Initialize ADK session lazily
        await ctx.ensure_session()

        result = PipelineExecutionResult(execution_id=ctx.execution_id)
        start = time.monotonic()

        ctx.logger.info(
            "dag_started",
            dag=self.name,
            layers=[[n for n in layer] for layer in self._layers],
            total_nodes=len(self._nodes),
        )
        await ctx.publish(
            "dag.started",
            {
                "dag": self.name,
                "total_nodes": len(self._nodes),
                "total_layers": len(self._layers),
            },
        )

        try:
            for layer_idx, layer in enumerate(self._layers):
                ctx.logger.info(
                    "dag_layer_started",
                    dag=self.name,
                    layer=layer_idx,
                    nodes=layer,
                )

                if len(layer) == 1:
                    # Single node — no gather overhead
                    await self._run_node(layer[0], ctx, result)
                else:
                    # Multiple nodes — run in parallel
                    await asyncio.gather(
                        *[self._run_node(name, ctx, result) for name in layer]
                    )

                ctx.logger.info(
                    "dag_layer_completed",
                    dag=self.name,
                    layer=layer_idx,
                )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            ctx.logger.error(
                "dag_failed",
                dag=self.name,
                error=str(exc),
            )
            await ctx.publish(
                "dag.failed",
                {
                    "dag": self.name,
                    "error": str(exc),
                },
            )
            raise

        finally:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            result.duration_ms = elapsed
            result.state = dict(ctx.state)

            if result.success:
                ctx.logger.info(
                    "dag_completed",
                    dag=self.name,
                    duration_ms=elapsed,
                    steps_completed=result.steps_completed,
                )
                await ctx.publish(
                    "dag.completed",
                    {
                        "dag": self.name,
                        "duration_ms": elapsed,
                        "steps_completed": result.steps_completed,
                    },
                )

        return result

    async def _run_node(
        self,
        node_name: str,
        ctx: AgentContext,
        result: PipelineExecutionResult,
    ) -> None:
        """Execute a single DAG node and merge its output into state."""
        node = self._nodes[node_name]

        ctx.logger.info("dag_node_started", node=node_name)
        await ctx.publish("dag.node_started", {"dag": self.name, "node": node_name})

        node_start = time.monotonic()

        # Each node receives the full accumulated state
        output = await node.agent.invoke(ctx, ctx.state)

        if output:
            ctx.update_state(output)

        elapsed = round((time.monotonic() - node_start) * 1000, 2)
        result.steps_completed.append(node_name)

        ctx.logger.info(
            "dag_node_completed",
            node=node_name,
            duration_ms=elapsed,
            output_keys=list(output.keys()) if output else [],
        )
        await ctx.publish(
            "dag.node_completed",
            {
                "dag": self.name,
                "node": node_name,
                "duration_ms": elapsed,
            },
        )

        # ── Persist node output as a versioned artifact ───────────────
        if output:
            await persist_node_artifact(
                ctx,
                engine_name=self.name,
                node_name=node_name,
                output=output,
                duration_ms=elapsed,
            )

    def __repr__(self) -> str:
        layers_str = " → ".join(f"[{', '.join(layer)}]" for layer in self._layers)
        return f"<DAGRunner {self.name!r} layers={layers_str}>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DAGBuilder — Fluent API for constructing DAGs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DAGBuilder:
    """
    Fluent builder for constructing validated DAG execution graphs.

    Registers nodes with optional dependencies, validates the graph
    structure (no cycles, no dangling refs), and produces a
    ``DAGRunner`` ready for execution.

    Auto-wraps plain functions and ADK agents into ``BaseAgent``,
    just like ``PipelineBuilder``.

    Usage::

        dag = (
            DAGBuilder("my_dag")
            .node("a", agent_a)
            .node("b", agent_b, dependencies=["a"])
            .node("c", agent_c, dependencies=["a"])
            .node("d", agent_d, dependencies=["b", "c"])
            .build()
        )
    """

    def __init__(self, name: str):
        self.name = name
        self._nodes: dict[str, DAGNode] = {}

    def node(
        self,
        name: str,
        agent: StepLike,
        *,
        dependencies: list[str] | None = None,
    ) -> "DAGBuilder":
        """
        Register a node in the DAG.

        Args:
            name: Unique name for this node.
            agent: BaseAgent, callable, or ADK agent (auto-wrapped).
            dependencies: List of node names that must complete before
                this node can execute.  Empty means the node is a root.

        Raises:
            ValueError: If a node with the same name already exists.
        """
        if name in self._nodes:
            raise ValueError(f"DAGBuilder '{self.name}': duplicate node name '{name}'.")
        self._nodes[name] = DAGNode(
            name=name,
            agent=_wrap_step(agent),
            dependencies=list(dependencies or []),
        )
        return self

    def build(self) -> DAGRunner:
        """
        Validate the graph and return a ``DAGRunner``.

        Validation steps:
          1. At least one node exists.
          2. All dependency references point to registered nodes.
          3. No cycles (Kahn's algorithm).

        Raises:
            ValueError: If the graph has no nodes.
            DAGDependencyError: If a dependency references an unknown node.
            DAGCycleError: If the graph contains a cycle.
        """
        if not self._nodes:
            raise ValueError(f"DAGBuilder '{self.name}' has no nodes.")

        # ── Validate dependency references ────────────────────────────
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise DAGDependencyError(
                        f"Node '{node.name}' depends on unknown node '{dep}'.",
                        node=node.name,
                        dependency=dep,
                    )

        # ── Compute topological layers (Kahn's algorithm) ────────────
        layers = self._compute_layers()

        return DAGRunner(self.name, dict(self._nodes), layers)

    def _compute_layers(self) -> list[list[str]]:
        """
        Compute execution layers using Kahn's topological sort.

        Returns a list of layers, where each layer is a list of node
        names that can execute concurrently.

        Raises:
            DAGCycleError: If the graph contains a cycle.
        """
        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        dependents: dict[str, list[str]] = defaultdict(list)

        for node in self._nodes.values():
            for dep in node.dependencies:
                in_degree[node.name] += 1
                dependents[dep].append(node.name)

        # Seed with root nodes (in_degree == 0)
        queue: deque[str] = deque(
            name for name, degree in in_degree.items() if degree == 0
        )

        layers: list[list[str]] = []
        processed = 0

        while queue:
            # All nodes currently in queue form one parallel layer
            layer = sorted(queue)  # Sort for deterministic ordering
            queue.clear()
            layers.append(layer)
            processed += len(layer)

            for name in layer:
                for child in dependents[name]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if processed != len(self._nodes):
            # Some nodes were never reached → cycle exists
            cycled = [n for n, d in in_degree.items() if d > 0]
            raise DAGCycleError(
                f"DAG '{self.name}' contains a cycle involving nodes: {cycled}",
                nodes=cycled,
            )

        return layers

    def __repr__(self) -> str:
        return f"<DAGBuilder {self.name!r} nodes={len(self._nodes)}>"
