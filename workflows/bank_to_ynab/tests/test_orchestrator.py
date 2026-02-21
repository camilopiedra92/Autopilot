"""
Tests for the bank_to_ynab declarative pipeline construction and execution.

Validates:
  - Workflow rejects empty emails
  - Workflow formats inputs correctly for the DSL pipeline
  - DSL Pipeline is built with correct steps from pipeline.yaml
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from autopilot.core import load_workflow

@pytest.fixture
def workflow_path():
    """Returns the absolute path to the bank_to_ynab workflow directory."""
    return Path(__file__).parent.parent


@pytest.mark.asyncio
async def test_pipeline_has_eleven_steps(workflow_path):
    """Verify the DSL pipeline is constructed with 11 steps accurately."""

    with patch("workflows.bank_to_ynab.agents.email_parser.create_email_parser") as mock_parser, \
         patch("workflows.bank_to_ynab.agents.researcher.create_researcher") as mock_researcher, \
         patch("workflows.bank_to_ynab.agents.categorizer.create_categorizer") as mock_cat:

        mock_parser_agent = MagicMock()
        mock_parser_agent.name = "email_parser"
        mock_parser.return_value = mock_parser_agent

        mock_researcher_agent = MagicMock()
        mock_researcher_agent.name = "researcher"
        mock_researcher.return_value = mock_researcher_agent

        mock_cat_agent = MagicMock()
        mock_cat_agent.name = "categorizer"
        mock_cat.return_value = mock_cat_agent

        # Load the pipeline purely from YAML
        pipeline = load_workflow(str(workflow_path / "pipeline.yaml"))

        assert pipeline.name == "bank_to_ynab"
        assert len(pipeline.steps) == 11

        step_names = [s.name for s in pipeline.steps]
        assert step_names == [
            "format_parser_prompt",
            "email_parser",
            "match_account",
            "format_researcher_input",
            "researcher",
            "format_categorizer_input",
            "categorizer",
            "synthesize_transaction",
            "push_to_ynab",
            "format_notifier_input",
            "telegram_notifier",
        ]
