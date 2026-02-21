"""Template workflow — pure edge architecture implementation."""

from autopilot.base_workflow import BaseWorkflow


class TemplateWorkflow(BaseWorkflow):
    """
    Template workflow — use as a starting point for new workflows.

    This class acts as the platform registry anchor.
    - Identity & Triggers: Loaded natively from manifest.yaml
    - Execution Logic: Handled automatically by the DSL via pipeline.yaml
    
    You only need to override methods like setup(), teardown(), or 
    execute() if your workflow requires complex non-declarative logic.
    """
    pass
