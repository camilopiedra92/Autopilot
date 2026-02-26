"""
DSL Loader — Reads YAML workflow definitions and produces executable objects.

The bridge between the declarative YAML world (Phase 6) and the runtime
engine (Phases 1-5).  Given a YAML file or dict, the loader:

  1. Validates the schema with ``DSLWorkflowDef`` (Pydantic)
  2. Resolves every ``ref`` dotted path via ``importlib``
  3. Builds composed steps (loop, parallel, sequential) using V3 adapters
  4. Routes by strategy → produces a ``Pipeline`` or ``DAGRunner``

Public API::

    from autopilot.core.dsl_loader import load_workflow, load_workflow_from_dict

    # From a YAML file:
    pipeline = load_workflow("workflows/my_flow/workflow.yaml")
    result = await pipeline.execute(initial_input={...})

    # From a pre-parsed dict (useful in tests):
    dag = load_workflow_from_dict({
        "name": "analytics",
        "strategy": "dag",
        "nodes": [...],
    })

Design:
  - Zero external dependencies beyond stdlib + Pydantic + YAML.
  - ``condition_expr`` strings are compiled in a restricted sandbox.
  - Every resolution failure gives a clear, actionable error message.
"""

import importlib
from pathlib import Path
from typing import Any, Callable, Union

import structlog
import yaml

from autopilot.core.agent import (
    BaseAgent,
    FunctionalAgent,
    LoopAgentAdapter,
    ParallelAgentAdapter,
    SequentialAgentAdapter,
)
from autopilot.core.dag import DAGBuilder, DAGRunner
from autopilot.core.dsl_schema import (
    DSLStepDef,
    DSLStepType,
    DSLStrategy,
    DSLWorkflowDef,
)
from autopilot.core.pipeline import Pipeline, PipelineBuilder
from autopilot.errors import DSLResolutionError, DSLValidationError

logger = structlog.get_logger(__name__)

# Type alias for the two possible return types
Executable = Union[Pipeline, DAGRunner]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_workflow(path: str | Path) -> Executable:
    """
    Load a YAML workflow definition and return an executable Pipeline or DAGRunner.

    Args:
        path: Path to a ``.yaml`` file containing a ``DSLWorkflowDef``.

    Returns:
        ``Pipeline`` for sequential strategies, ``DAGRunner`` for DAG strategies.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        DSLValidationError: If the YAML fails schema validation.
        DSLResolutionError: If a ``ref`` cannot be resolved to a callable.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"DSL workflow file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise DSLValidationError(
            f"Workflow YAML must be a mapping, got {type(raw).__name__}.",
            field="root",
        )

    return load_workflow_from_dict(raw)


def load_workflow_from_dict(raw: dict[str, Any]) -> Executable:
    """
    Build an executable from a pre-parsed YAML dictionary.

    Useful for tests and dynamic workflow generation.

    Args:
        raw: Dictionary matching the ``DSLWorkflowDef`` schema.

    Returns:
        ``Pipeline`` for sequential strategies, ``DAGRunner`` for DAG strategies.

    Raises:
        DSLValidationError: If the dict fails schema validation.
        DSLResolutionError: If a ``ref`` cannot be resolved.
    """
    try:
        definition = DSLWorkflowDef(**raw)
    except DSLValidationError:
        raise
    except Exception as exc:
        raise DSLValidationError(
            f"Invalid DSL workflow definition: {exc}",
            field="root",
        ) from exc

    logger.info(
        "dsl_loading_workflow",
        name=definition.name,
        strategy=definition.strategy.value,
    )

    if definition.strategy == DSLStrategy.SEQUENTIAL:
        return _build_sequential(definition)
    elif definition.strategy == DSLStrategy.DAG:
        return _build_dag(definition)
    else:
        raise DSLValidationError(
            f"Unsupported strategy: {definition.strategy}",
            field="strategy",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Strategy Builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_sequential(definition: DSLWorkflowDef) -> Pipeline:
    """Build a Pipeline from a sequential DSL definition."""
    builder = PipelineBuilder(definition.name)

    for step_def in definition.steps:
        agent = _build_step(step_def)
        builder.step(agent)

    pipeline = builder.build()
    logger.info(
        "dsl_workflow_built",
        name=definition.name,
        strategy="sequential",
        step_count=len(definition.steps),
    )
    return pipeline


def _build_dag(definition: DSLWorkflowDef) -> DAGRunner:
    """Build a DAGRunner from a DAG DSL definition."""
    builder = DAGBuilder(definition.name)

    for node_def in definition.nodes:
        resolved = _resolve_ref(node_def.ref)
        agent = _to_agent(resolved, node_def.name)
        builder.node(node_def.name, agent, dependencies=node_def.dependencies)

    dag = builder.build()
    logger.info(
        "dsl_workflow_built",
        name=definition.name,
        strategy="dag",
        node_count=len(definition.nodes),
    )
    return dag


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step Builder — Recursive, handles composed types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_step(step_def: DSLStepDef) -> BaseAgent:
    """
    Recursively build a BaseAgent from a DSL step definition.

    Handles all step types: function, agent, loop, parallel, sequential.
    """
    if step_def.type == DSLStepType.FUNCTION:
        return _build_function_step(step_def)

    elif step_def.type == DSLStepType.AGENT:
        return _build_agent_step(step_def)

    elif step_def.type == DSLStepType.LOOP:
        return _build_loop_step(step_def)

    elif step_def.type == DSLStepType.PARALLEL:
        return _build_parallel_step(step_def)

    elif step_def.type == DSLStepType.SEQUENTIAL:
        return _build_sequential_step(step_def)

    else:
        raise DSLValidationError(
            f"Unknown step type '{step_def.type}' for step '{step_def.name}'.",
            field="type",
        )


def _build_function_step(step_def: DSLStepDef) -> FunctionalAgent:
    """Resolve a dotted ref to a function and wrap as FunctionalAgent."""
    if not step_def.ref:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=function) requires a 'ref' field.",
            field="ref",
        )
    func = _resolve_ref(step_def.ref)
    if not callable(func):
        raise DSLResolutionError(
            f"Ref '{step_def.ref}' resolved to non-callable: {type(func).__name__}.",
            ref=step_def.ref,
        )
    return FunctionalAgent(func, name=step_def.name, description=step_def.description)


def _build_agent_step(step_def: DSLStepDef) -> BaseAgent:
    """Resolve a dotted ref to a BaseAgent subclass or factory."""
    if not step_def.ref:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=agent) requires a 'ref' field.",
            field="ref",
        )
    resolved = _resolve_ref(step_def.ref)
    return _to_agent(resolved, step_def.name)


def _build_loop_step(step_def: DSLStepDef) -> LoopAgentAdapter:
    """Build a LoopAgentAdapter from a loop step definition."""
    if step_def.body is None:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=loop) requires a 'body' field.",
            field="body",
        )
    if not step_def.condition_expr:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=loop) requires a 'condition_expr' field.",
            field="condition_expr",
        )

    body_agent = _build_step(step_def.body)
    condition = _compile_condition(step_def.condition_expr, step_def.name)

    return LoopAgentAdapter(
        step_def.name,
        body=body_agent,
        condition=condition,
        max_iterations=step_def.max_iterations,
        description=step_def.description,
    )


def _build_parallel_step(step_def: DSLStepDef) -> ParallelAgentAdapter:
    """Build a ParallelAgentAdapter from child step definitions."""
    if not step_def.children:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=parallel) requires 'children'.",
            field="children",
        )

    branches = [_build_step(child) for child in step_def.children]
    return ParallelAgentAdapter(
        step_def.name,
        branches=branches,
        description=step_def.description,
    )


def _build_sequential_step(step_def: DSLStepDef) -> SequentialAgentAdapter:
    """Build a SequentialAgentAdapter from child step definitions."""
    if not step_def.children:
        raise DSLValidationError(
            f"Step '{step_def.name}' (type=sequential) requires 'children'.",
            field="children",
        )

    children = [_build_step(child) for child in step_def.children]
    return SequentialAgentAdapter(
        step_def.name,
        children=children,
        description=step_def.description,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ref Resolution — importlib bridge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _resolve_ref(dotted_path: str) -> Any:
    """
    Resolve a dotted import path to a Python object.

    Supports two patterns:
      - ``module.path.attribute``  →  ``importlib.import_module("module.path").attribute``
      - ``module.path``  →  ``importlib.import_module("module.path")`` (the module itself)

    Args:
        dotted_path: Fully qualified dotted path.

    Returns:
        The resolved Python object (function, class, etc.)

    Raises:
        DSLResolutionError: If the module or attribute cannot be found.
    """
    if not dotted_path or "." not in dotted_path:
        raise DSLResolutionError(
            f"Invalid ref '{dotted_path}': must be a dotted module path "
            f"(e.g. 'my_module.my_function').",
            ref=dotted_path,
        )

    # Try importing the full path as a module first
    try:
        return importlib.import_module(dotted_path)
    except ImportError:
        pass

    # Split into module + attribute
    parts = dotted_path.rsplit(".", 1)
    module_path, attr_name = parts[0], parts[1]

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise DSLResolutionError(
            f"Cannot import module '{module_path}' from ref '{dotted_path}': {exc}",
            ref=dotted_path,
        ) from exc

    if not hasattr(module, attr_name):
        raise DSLResolutionError(
            f"Module '{module_path}' has no attribute '{attr_name}'.",
            ref=dotted_path,
        )

    return getattr(module, attr_name)


def _to_agent(obj: Any, name: str) -> BaseAgent:
    """
    Convert a resolved object into a BaseAgent.

    Handles:
      - Already a BaseAgent instance → return directly
      - A class that is a BaseAgent subclass → instantiate with name
      - A callable (function / factory) → check if it returns a BaseAgent
        when called, otherwise wrap as FunctionalAgent

    Args:
        obj: The resolved Python object.
        name: Step name for the agent.

    Returns:
        A BaseAgent instance.

    Raises:
        DSLResolutionError: If the object cannot be converted.
    """
    # Already an agent instance
    if isinstance(obj, BaseAgent):
        return obj

    # A BaseAgent subclass → instantiate
    if isinstance(obj, type) and issubclass(obj, BaseAgent):
        return obj(name)

    # A callable — could be a factory or a plain function
    if callable(obj):
        import inspect

        sig = inspect.signature(obj)
        has_required_args = any(
            p.default == inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in sig.parameters.values()
        )

        # We only attempt to call it as a factory if it can be called with no arguments.
        if not has_required_args:
            try:
                result = obj()

                # If calling an async function without await, we get a coroutine.
                # Close it to avoid "coroutine was never awaited" warnings.
                if inspect.iscoroutine(result):
                    result.close()
                    raise TypeError("async function, not a factory")

                from autopilot.core.pipeline import _wrap_step, _is_adk_agent

                # Verify the result is actually an agent before wrapping
                if isinstance(result, BaseAgent) or _is_adk_agent(result):
                    wrapped = _wrap_step(result)
                    if getattr(wrapped, "name", None):
                        wrapped.name = name
                    return wrapped
            except Exception:
                # If calling it fails, fallback to wrapping the function itself
                pass

        # If it requires args, or the factory attempt didn't produce an agent, wrap it as a FunctionalAgent
        wrapped = FunctionalAgent(obj)
        wrapped.name = name
        return wrapped

    raise DSLResolutionError(
        f"Cannot convert {type(obj).__name__} to a BaseAgent for step '{name}'.",
        ref=name,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Condition Expression Compiler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _compile_condition(expr: str, step_name: str) -> Callable[[dict], bool]:
    """
    Compile a condition expression string into a callable ``(state) → bool``.

    The expression is sandboxed: only ``state`` is available in the
    evaluation namespace, with ``__builtins__`` stripped.

    Examples::

        "state.get('valid', False)"
        "state.get('counter', 0) >= 3"
        "'error' not in state"

    Args:
        expr: Python expression string.
        step_name: Step name for error context.

    Returns:
        A callable that evaluates the expression against a state dict.

    Raises:
        DSLValidationError: If the expression cannot be compiled.
    """
    try:
        code = compile(expr, f"<dsl:{step_name}:condition>", "eval")
    except SyntaxError as exc:
        raise DSLValidationError(
            f"Invalid condition_expr for step '{step_name}': {exc}",
            field="condition_expr",
        ) from exc

    def _condition(state: dict) -> bool:
        try:
            result = eval(code, {"__builtins__": {}}, {"state": state})
            return bool(result)
        except Exception as exc:
            logger.warning(
                "dsl_condition_eval_error",
                step=step_name,
                expr=expr,
                error=str(exc),
            )
            return False

    return _condition
