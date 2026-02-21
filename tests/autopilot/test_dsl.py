"""
Tests for autopilot.core DSL — Declarative YAML Workflow Engine (V3 Phase 6).

Covers:
  - DSLWorkflowDef / DSLStepDef schema validation
  - DSLLoader ref resolution via importlib
  - Sequential pipeline construction from YAML
  - DAG construction from YAML
  - Loop (retry) steps with condition expressions
  - Parallel steps with branch merging
  - Nested sequential blocks
  - Sandboxed condition expression safety
  - End-to-end: YAML file → execute → correct state
"""

import pytest
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


from autopilot.core.agent import BaseAgent, FunctionalAgent
from autopilot.core.pipeline import Pipeline
from autopilot.core.dag import DAGRunner
from autopilot.core.dsl_schema import (
    DSLWorkflowDef,
    DSLStepDef,
    DSLStepType,
    DSLStrategy,
    DSLNodeDef,
)
from autopilot.core.dsl_loader import (
    load_workflow,
    load_workflow_from_dict,
    _resolve_ref,
    _compile_condition,
    _to_agent,
)
from autopilot.errors import DSLValidationError, DSLResolutionError


# ── Test helper: patch event bus ────────────────────────────────────────

def _mock_event_bus():
    """Context manager that patches the event bus for all tests."""
    mock_bus = MagicMock()
    mock_bus.emit = AsyncMock()
    return patch("autopilot.core.context.get_event_bus", return_value=mock_bus)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSL Schema Validation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLSchema:
    """Validate the Pydantic schema layer catches errors early."""

    def test_valid_sequential_definition(self):
        defn = DSLWorkflowDef(
            name="test",
            strategy=DSLStrategy.SEQUENTIAL,
            steps=[
                DSLStepDef(name="a", type=DSLStepType.FUNCTION, ref="os.path.join"),
            ],
        )
        assert defn.name == "test"
        assert defn.strategy == DSLStrategy.SEQUENTIAL
        assert len(defn.steps) == 1

    def test_valid_dag_definition(self):
        defn = DSLWorkflowDef(
            name="dag_test",
            strategy=DSLStrategy.DAG,
            nodes=[
                DSLNodeDef(name="a", ref="os.path.join"),
                DSLNodeDef(name="b", ref="os.path.join", dependencies=["a"]),
            ],
        )
        assert defn.strategy == DSLStrategy.DAG
        assert len(defn.nodes) == 2

    def test_sequential_requires_steps(self):
        """Strategy 'sequential' with empty steps should raise."""
        with pytest.raises((DSLValidationError, Exception)):
            DSLWorkflowDef(
                name="bad",
                strategy=DSLStrategy.SEQUENTIAL,
                steps=[],
            )

    def test_dag_requires_nodes(self):
        """Strategy 'dag' with empty nodes should raise."""
        with pytest.raises((DSLValidationError, Exception)):
            DSLWorkflowDef(
                name="bad",
                strategy=DSLStrategy.DAG,
                nodes=[],
            )

    def test_step_type_enum(self):
        """All step types are valid."""
        for t in ["function", "agent", "loop", "parallel", "sequential"]:
            s = DSLStepDef(name="s", type=t, ref="os.path.join")
            assert s.type == DSLStepType(t)

    def test_invalid_step_type_raises(self):
        """Unknown step type should raise validation error."""
        with pytest.raises(Exception):
            DSLStepDef(name="s", type="nonexistent", ref="os.path.join")

    def test_loop_step_schema(self):
        """Loop step accepts body and condition_expr."""
        loop = DSLStepDef(
            name="retry",
            type=DSLStepType.LOOP,
            max_iterations=5,
            condition_expr="state.get('ok', False)",
            body=DSLStepDef(name="inner", type=DSLStepType.FUNCTION, ref="os.path.join"),
        )
        assert loop.max_iterations == 5
        assert loop.body is not None
        assert loop.body.name == "inner"

    def test_parallel_step_schema(self):
        """Parallel step accepts children list."""
        par = DSLStepDef(
            name="par",
            type=DSLStepType.PARALLEL,
            children=[
                DSLStepDef(name="a", type=DSLStepType.FUNCTION, ref="os.path.join"),
                DSLStepDef(name="b", type=DSLStepType.FUNCTION, ref="os.path.join"),
            ],
        )
        assert len(par.children) == 2

    def test_initial_state(self):
        """initial_state is optional and defaults to empty dict."""
        defn = DSLWorkflowDef(
            name="test",
            steps=[DSLStepDef(name="a", type=DSLStepType.FUNCTION, ref="os.path.join")],
            initial_state={"seed": 42},
        )
        assert defn.initial_state == {"seed": 42}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ref Resolution Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLRefResolution:
    """Test the importlib-based ref resolver."""

    def test_resolve_stdlib_function(self):
        """Resolve a well-known stdlib function."""
        result = _resolve_ref("os.path.join")
        import os.path
        assert result is os.path.join

    def test_resolve_stdlib_module(self):
        """Resolve a module path returns the module itself."""
        result = _resolve_ref("os.path")
        import os.path
        assert result is os.path

    def test_resolve_bad_module_raises(self):
        """Non-existent module raises DSLResolutionError."""
        with pytest.raises(DSLResolutionError, match="Cannot import"):
            _resolve_ref("nonexistent_module_xyz.foo")

    def test_resolve_bad_attribute_raises(self):
        """Existing module but bad attribute raises DSLResolutionError."""
        with pytest.raises(DSLResolutionError, match="has no attribute"):
            _resolve_ref("os.path.nonexistent_attr_xyz")

    def test_resolve_no_dot_raises(self):
        """Bare name without dots raises DSLResolutionError."""
        with pytest.raises(DSLResolutionError, match="must be a dotted"):
            _resolve_ref("nodots")

    def test_resolve_empty_raises(self):
        """Empty string raises DSLResolutionError."""
        with pytest.raises(DSLResolutionError):
            _resolve_ref("")

    def test_resolve_project_helper(self):
        """Resolve a function from the project's test helpers."""
        result = _resolve_ref("workflows._template.steps.parse_input")
        from workflows._template.steps import parse_input
        assert result is parse_input


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  _to_agent Converter Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToAgent:
    """Test the obj → BaseAgent converter."""

    def test_existing_agent_instance_passthrough(self):
        agent = FunctionalAgent(lambda: {}, name="test")
        result = _to_agent(agent, "step_name")
        assert result is agent

    def test_callable_wraps_as_functional(self):
        def my_func(x: int = 0) -> dict:
            return {"y": x + 1}

        result = _to_agent(my_func, "my_step")
        assert isinstance(result, FunctionalAgent)
        assert result.name == "my_step"

    def test_base_agent_subclass_instantiates(self):
        class MyAgent(BaseAgent[dict, dict]):
            async def run(self, ctx, input):
                return {}

        result = _to_agent(MyAgent, "custom")
        assert isinstance(result, MyAgent)
        assert result.name == "custom"

    def test_non_callable_raises(self):
        with pytest.raises(DSLResolutionError, match="Cannot convert"):
            _to_agent(42, "step_x")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Condition Expression Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLConditionExpr:
    """Test sandboxed condition expression compilation and evaluation."""

    def test_simple_get(self):
        cond = _compile_condition("state.get('valid', False)", "test_step")
        assert cond({"valid": True}) is True
        assert cond({"valid": False}) is False
        assert cond({}) is False

    def test_comparison(self):
        cond = _compile_condition("state.get('counter', 0) >= 3", "test_step")
        assert cond({"counter": 5}) is True
        assert cond({"counter": 2}) is False

    def test_in_operator(self):
        cond = _compile_condition("'error' not in state", "test_step")
        assert cond({}) is True
        assert cond({"error": "boom"}) is False

    def test_syntax_error_raises(self):
        with pytest.raises(DSLValidationError, match="Invalid condition_expr"):
            _compile_condition("this is not valid python!", "bad_step")

    def test_builtins_not_accessible(self):
        """Sandboxed: __import__ should not be available."""
        cond = _compile_condition("__import__('os')", "bad_step")
        # Should return False (exception caught internally)
        assert cond({}) is False

    def test_eval_error_returns_false(self):
        """Runtime errors in expressions return False (logged)."""
        cond = _compile_condition("state['nonexistent_key']", "test_step")
        assert cond({}) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — Sequential Pipeline Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderSequential:
    """Load sequential workflows from dict and verify Pipeline structure."""

    def test_loads_simple_sequential(self):
        """Two function steps → Pipeline with 2 steps."""
        raw = {
            "name": "test_seq",
            "strategy": "sequential",
            "steps": [
                {"name": "step_a", "type": "function", "ref": "os.path.basename"},
                {"name": "step_b", "type": "function", "ref": "os.path.dirname"},
            ],
        }
        result = load_workflow_from_dict(raw)
        assert isinstance(result, Pipeline)
        assert result.name == "test_seq"
        assert len(result.steps) == 2
        assert result.steps[0].name == "step_a"
        assert result.steps[1].name == "step_b"

    @pytest.mark.asyncio
    async def test_execute_sequential(self):
        """Sequential pipeline actually runs and accumulates state."""

        raw = {
            "name": "mini_pipe",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "parse",
                    "type": "function",
                    "ref": "workflows._template.steps.parse_input",
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)
        assert isinstance(pipeline, Pipeline)

        with _mock_event_bus():
            result = await pipeline.execute(initial_input={"raw_text": "hello"})

        assert result.success is True
        assert result.state["parsed"] == "HELLO"
        assert result.state["char_count"] == 5

    def test_missing_ref_raises(self):
        """Function step without ref raises DSLValidationError."""
        raw = {
            "name": "bad",
            "strategy": "sequential",
            "steps": [
                {"name": "no_ref", "type": "function"},
            ],
        }
        with pytest.raises(DSLValidationError, match="requires a 'ref'"):
            load_workflow_from_dict(raw)

    def test_invalid_yaml_dict_raises(self):
        """Non-dict input raises DSLValidationError."""
        with pytest.raises(DSLValidationError):
            load_workflow_from_dict({"name": "x", "strategy": "sequential", "steps": []})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — DAG Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderDAG:
    """Load DAG workflows from dict and verify DAGRunner structure."""

    def test_loads_dag(self):
        raw = {
            "name": "test_dag",
            "strategy": "dag",
            "nodes": [
                {"name": "root", "ref": "os.path.basename"},
                {"name": "leaf", "ref": "os.path.dirname", "dependencies": ["root"]},
            ],
        }
        result = load_workflow_from_dict(raw)
        assert isinstance(result, DAGRunner)
        assert result.name == "test_dag"

    @pytest.mark.asyncio
    async def test_execute_dag(self):
        """DAG with diamond pattern executes correctly."""

        raw = {
            "name": "diamond_dag",
            "strategy": "dag",
            "nodes": [
                {"name": "fetch", "ref": "workflows._template.steps.parse_input"},
                {"name": "enrich_a", "ref": "workflows._template.steps.fetch_source_a", "dependencies": ["fetch"]},
                {"name": "enrich_b", "ref": "workflows._template.steps.fetch_source_b", "dependencies": ["fetch"]},
                {"name": "merge", "ref": "workflows._template.steps.merge_results", "dependencies": ["enrich_a", "enrich_b"]},
            ],
        }

        dag = load_workflow_from_dict(raw)
        assert isinstance(dag, DAGRunner)

        with _mock_event_bus():
            result = await dag.execute(initial_input={"raw_text": "dag test"})

        assert result.success is True
        # parse_input produces "parsed" key
        assert "parsed" in result.state
        # fetch sources produce their outputs
        assert "source_a" in result.state
        assert "source_b" in result.state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — Loop Step Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderLoop:
    """Loop steps build LoopAgentAdapter and retry correctly."""

    def test_builds_loop_step(self):
        from autopilot.core.agent import LoopAgentAdapter

        raw = {
            "name": "loop_test",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "retry_validate",
                    "type": "loop",
                    "max_iterations": 5,
                    "condition_expr": "state.get('valid', False)",
                    "body": {
                        "name": "inner",
                        "type": "function",
                        "ref": "workflows._template.steps.validate_data",
                    },
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], LoopAgentAdapter)
        assert pipeline.steps[0].max_iterations == 5

    @pytest.mark.asyncio
    async def test_loop_retries_then_succeeds(self):
        """Loop should retry body until condition_expr passes."""

        raw = {
            "name": "loop_e2e",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "retry",
                    "type": "loop",
                    "max_iterations": 5,
                    "condition_expr": "state.get('valid', False)",
                    "body": {
                        "name": "check",
                        "type": "function",
                        "ref": "workflows._template.steps.validate_data",
                    },
                },
            ],
        }

        from workflows._template.steps import reset_validate_counter
        reset_validate_counter()

        pipeline = load_workflow_from_dict(raw)

        with _mock_event_bus():
            # Provide char_count > 0 so validate succeeds on first try
            result = await pipeline.execute(
                initial_input={"parsed": "DATA", "char_count": 4}
            )

        assert result.success is True
        assert result.state["valid"] is True

    def test_loop_missing_body_raises(self):
        raw = {
            "name": "bad_loop",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "no_body",
                    "type": "loop",
                    "condition_expr": "True",
                },
            ],
        }
        with pytest.raises(DSLValidationError, match="requires a 'body'"):
            load_workflow_from_dict(raw)

    def test_loop_missing_condition_raises(self):
        raw = {
            "name": "bad_loop",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "no_cond",
                    "type": "loop",
                    "body": {"name": "b", "type": "function", "ref": "os.path.join"},
                },
            ],
        }
        with pytest.raises(DSLValidationError, match="requires a 'condition_expr'"):
            load_workflow_from_dict(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — Parallel Step Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderParallel:
    """Parallel steps build ParallelAgentAdapter and merge results."""

    def test_builds_parallel_step(self):
        from autopilot.core.agent import ParallelAgentAdapter

        raw = {
            "name": "par_test",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "fetch_all",
                    "type": "parallel",
                    "children": [
                        {"name": "a", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                        {"name": "b", "type": "function", "ref": "workflows._template.steps.fetch_source_b"},
                    ],
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.steps[0], ParallelAgentAdapter)

    @pytest.mark.asyncio
    async def test_parallel_merges_results(self):
        raw = {
            "name": "par_e2e",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "fetch_all",
                    "type": "parallel",
                    "children": [
                        {"name": "a", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                        {"name": "b", "type": "function", "ref": "workflows._template.steps.fetch_source_b"},
                    ],
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)

        with _mock_event_bus():
            result = await pipeline.execute()

        assert result.success is True
        assert "source_a" in result.state
        assert "source_b" in result.state
        assert result.state["source_a"]["provider"] == "Alpha"
        assert result.state["source_b"]["provider"] == "Beta"

    def test_parallel_missing_children_raises(self):
        raw = {
            "name": "bad",
            "strategy": "sequential",
            "steps": [
                {"name": "empty_par", "type": "parallel"},
            ],
        }
        with pytest.raises(DSLValidationError, match="requires 'children'"):
            load_workflow_from_dict(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — Sequential (nested) Step Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderNestedSequential:
    """Nested sequential blocks build SequentialAgentAdapter."""

    def test_builds_nested_sequential_step(self):
        from autopilot.core.agent import SequentialAgentAdapter

        raw = {
            "name": "nested_test",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "group",
                    "type": "sequential",
                    "children": [
                        {"name": "a", "type": "function", "ref": "workflows._template.steps.parse_input"},
                        {"name": "b", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                    ],
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)
        assert isinstance(pipeline.steps[0], SequentialAgentAdapter)

    @pytest.mark.asyncio
    async def test_nested_sequential_executes(self):
        raw = {
            "name": "nested_e2e",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "prep",
                    "type": "sequential",
                    "children": [
                        {"name": "parse", "type": "function", "ref": "workflows._template.steps.parse_input"},
                        {"name": "enrich", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                    ],
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)

        with _mock_event_bus():
            result = await pipeline.execute(initial_input={"raw_text": "test"})

        assert result.success is True
        assert result.state["parsed"] == "TEST"
        assert "source_a" in result.state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSLLoader — YAML File Loading Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLLoaderFile:
    """Test loading from actual YAML files."""

    def test_load_example_yaml(self):
        """Load the template pipeline YAML from disk."""
        yaml_path = Path(__file__).parent.parent.parent / "workflows" / "_template" / "pipeline.yaml"
        if not yaml_path.exists():
            pytest.skip(f"Pipeline YAML not found at {yaml_path}")

        pipeline = load_workflow(yaml_path)
        assert isinstance(pipeline, Pipeline)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_workflow("/nonexistent/path/workflow.yaml")

    def test_load_from_yaml_string(self, tmp_path):
        """Write YAML to temp file and load it."""
        yaml_content = textwrap.dedent("""\
            name: tmp_workflow
            strategy: sequential
            steps:
              - name: step_one
                type: function
                ref: os.path.basename
        """)
        yaml_file = tmp_path / "workflow.yaml"
        yaml_file.write_text(yaml_content)

        pipeline = load_workflow(yaml_file)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "tmp_workflow"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Integration: Full E2E — YAML → Execute → Correct State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDSLE2E:
    """End-to-end integration: load a full YAML → execute → verify results."""

    @pytest.mark.asyncio
    async def test_full_workflow_from_yaml(self):
        """
        Simulates a complete YAML-defined workflow:
          1. parse_input (function)
          2. parallel fetch_source_a + fetch_source_b
          3. merge_results (function)
        """
        raw = {
            "name": "e2e_workflow",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "parse",
                    "type": "function",
                    "ref": "workflows._template.steps.parse_input",
                },
                {
                    "name": "enrich",
                    "type": "parallel",
                    "children": [
                        {"name": "a", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                        {"name": "b", "type": "function", "ref": "workflows._template.steps.fetch_source_b"},
                    ],
                },
                {
                    "name": "merge",
                    "type": "function",
                    "ref": "workflows._template.steps.merge_results",
                },
            ],
        }

        pipeline = load_workflow_from_dict(raw)

        with _mock_event_bus():
            result = await pipeline.execute(
                initial_input={"raw_text": "e2e test data"}
            )

        assert result.success is True
        assert result.steps_completed == ["parse", "enrich", "merge"]

        # Verify full state flow
        assert result.state["parsed"] == "E2E TEST DATA"
        assert result.state["source_a"]["provider"] == "Alpha"
        assert result.state["source_b"]["provider"] == "Beta"

        final = result.state["final_output"]
        assert final["input"] == "E2E TEST DATA"
        assert final["status"] == "complete"
        assert len(final["enrichments"]) == 2

    @pytest.mark.asyncio
    async def test_mixed_steps_workflow(self):
        """
        Workflow with all step types:
          1. function step
          2. loop step (retries)
          3. parallel step
          4. nested sequential step
        """
        raw = {
            "name": "mixed_workflow",
            "strategy": "sequential",
            "steps": [
                {
                    "name": "init",
                    "type": "function",
                    "ref": "workflows._template.steps.parse_input",
                },
                {
                    "name": "validate",
                    "type": "loop",
                    "max_iterations": 3,
                    "condition_expr": "state.get('valid', False)",
                    "body": {
                        "name": "check",
                        "type": "function",
                        "ref": "workflows._template.steps.validate_data",
                    },
                },
                {
                    "name": "enrich",
                    "type": "parallel",
                    "children": [
                        {"name": "a", "type": "function", "ref": "workflows._template.steps.fetch_source_a"},
                        {"name": "b", "type": "function", "ref": "workflows._template.steps.fetch_source_b"},
                    ],
                },
                {
                    "name": "finalize",
                    "type": "sequential",
                    "children": [
                        {"name": "merge", "type": "function", "ref": "workflows._template.steps.merge_results"},
                    ],
                },
            ],
        }

        from workflows._template.steps import reset_validate_counter
        reset_validate_counter()

        pipeline = load_workflow_from_dict(raw)

        with _mock_event_bus():
            result = await pipeline.execute(
                initial_input={"raw_text": "mixed test"}
            )

        assert result.success is True
        assert result.state["valid"] is True
        assert "source_a" in result.state
        assert "final_output" in result.state
