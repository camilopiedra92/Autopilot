import pytest

from autopilot.core.context import AgentContext
from autopilot.core.pipeline import Pipeline
from autopilot.core.agent import BaseAgent

class MockStep(BaseAgent[dict, dict]):
    def __init__(self, name: str, output: dict):
        super().__init__(name)
        self.output = output

    async def run(self, ctx: AgentContext, input: dict) -> dict:
        return self.output

@pytest.mark.asyncio
async def test_hitl_pause_and_resume():
    # Step 1 succeeds natively
    step1 = MockStep("step1", {"data": "step1_done"})
    
    # Step 2 requests HITL approval
    step2 = MockStep("step2", {"hitl_requested": True, "pending_approval": "invoice_123"})
    
    # Step 3 should only run after resume
    step3 = MockStep("step3", {"final_data": "step3_done"})
    
    pipeline = Pipeline("test_hitl", [step1, step2, step3])
    
    # Initial run
    ctx1 = AgentContext(pipeline_name="test_hitl")
    result1 = await pipeline.execute(ctx1)
    
    assert result1.success is True
    assert result1.paused is True
    assert result1.steps_completed == ["step1", "step2"]
    assert ctx1.state["data"] == "step1_done"
    assert ctx1.state["pending_approval"] == "invoice_123"
    assert ctx1.state["hitl_requested"] is False # Should be cleared
    assert "__steps_completed__" in ctx1.state
    assert ctx1.state["__steps_completed__"] == ["step1", "step2"]
    
    # Now simulate a webhook resuming the pipeline with the paused state
    resumed_state = dict(ctx1.state)
    resumed_state["hitl_approved"] = True
    
    ctx2 = AgentContext(pipeline_name="test_hitl")
    result2 = await pipeline.execute(ctx2, initial_input=resumed_state)
    
    assert result2.success is True
    assert result2.paused is False
    # step1 and step2 are skipped and added to completed list, step3 runs
    assert result2.steps_completed == ["step1", "step2", "step3"] 
    assert ctx2.state["final_data"] == "step3_done"
    assert ctx2.state["hitl_approved"] is True
