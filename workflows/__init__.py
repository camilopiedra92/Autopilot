"""
Workflows Package — All automation workflows live here.

Each subdirectory is a self-contained workflow that implements BaseWorkflow.
The WorkflowRegistry auto-discovers workflows by scanning this directory.

Convention:
  workflows/
    my_workflow/
      __init__.py      # Must export: workflow = MyWorkflow()
      manifest.yaml    # Optional declarative config
      agents/          # ADK agents
      services/        # External integrations
      models/          # Pydantic models
"""
