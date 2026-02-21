import pytest
from unittest.mock import AsyncMock

from autopilot.core.context import AgentContext
from autopilot.core.pipeline import PipelineExecutionResult
from autopilot.core.react import ReactRunner
from autopilot.core.router import RouterRunner
from autopilot.errors import MaxRetriesExceededError


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    agent.name = "mock_agent"
    agent.invoke = AsyncMock()
    return agent


@pytest.fixture
def mock_pipeline():
    pipeline = AsyncMock()
    pipeline.execute = AsyncMock()
    return pipeline


@pytest.mark.asyncio
async def test_react_runner_success(mock_agent):
    # Agent returns 'react_finished': True on the second iteration
    mock_agent.invoke.side_effect = [
        {"intermediate": "data"},
        {"react_finished": True, "final": "result"},
    ]

    runner = ReactRunner("test_react", mock_agent, max_iterations=3)
    ctx = AgentContext(pipeline_name="test_react")

    result = await runner.execute(ctx)

    assert result.success is True
    assert result.state["intermediate"] == "data"
    assert result.state["final"] == "result"
    assert result.steps_completed == ["react_iter_1", "react_iter_2"]
    assert mock_agent.invoke.call_count == 2


@pytest.mark.asyncio
async def test_react_runner_exhausted_retries(mock_agent):
    # Agent never returns 'react_finished': True
    mock_agent.invoke.return_value = {"still": "going"}

    runner = ReactRunner("test_react", mock_agent, max_iterations=2)
    ctx = AgentContext(pipeline_name="test_react")

    with pytest.raises(MaxRetriesExceededError):
        await runner.execute(ctx)

    # The runner catches the exception internally and returns a failed Result
    # Wait, the code re-raises the exception.
    assert mock_agent.invoke.call_count == 2


@pytest.mark.asyncio
async def test_router_runner_success(mock_agent, mock_pipeline):
    # Router agent decides to go to "billing" route
    mock_agent.invoke.return_value = {"route": "billing"}

    mock_pipeline_result = PipelineExecutionResult(
        success=True, state={"billing_done": True}, steps_completed=["bill_step"]
    )
    mock_pipeline.execute.return_value = mock_pipeline_result

    routes = {"billing": mock_pipeline, "support": AsyncMock()}

    runner = RouterRunner("test_router", mock_agent, routes=routes)
    ctx = AgentContext(pipeline_name="test_router")

    result = await runner.execute(ctx)

    assert result.success is True
    assert result.state["route"] == "billing"
    assert result.state["billing_done"] is True
    assert result.steps_completed == ["routed_to_billing", "bill_step"]

    mock_agent.invoke.assert_called_once()
    mock_pipeline.execute.assert_called_once()
    assert not routes["support"].execute.called


@pytest.mark.asyncio
async def test_router_runner_invalid_route(mock_agent, mock_pipeline):
    # Router agent returns a route that doesn't exist
    mock_agent.invoke.return_value = {"route": "unknown"}

    routes = {"billing": mock_pipeline}
    runner = RouterRunner("test_router", mock_agent, routes=routes)
    ctx = AgentContext(pipeline_name="test_router")

    with pytest.raises(ValueError, match="returned invalid route"):
        await runner.execute(ctx)

    assert mock_agent.invoke.call_count == 1
    assert not mock_pipeline.execute.called
