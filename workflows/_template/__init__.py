"""
Template workflow package.

NOTE: __init__.py with a `workflow` export is no longer required.
The WorkflowRegistry auto-discovers workflows via:
  1. Classic: __init__.py with `workflow` export (this file — backward-compatible)
  2. Auto-class: workflow.py with BaseWorkflow subclass
  3. Pure YAML: manifest.yaml only

To create a new workflow, use the CLI:
  python3 -m autopilot.cli create-workflow my_workflow --display-name "My Workflow"
"""
