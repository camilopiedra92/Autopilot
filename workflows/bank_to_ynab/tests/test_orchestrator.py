"""
Tests for the bank_to_ynab declarative DAG pipeline construction.

Validates:
  - DSL DAG is built with correct nodes from pipeline.yaml
  - Node names match expected set
  - Layer structure reflects parallel execution topology
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from autopilot.core import load_workflow
from autopilot.core.dag import DAGRunner


@pytest.fixture
def workflow_path():
    """Returns the absolute path to the bank_to_ynab workflow directory."""
    return Path(__file__).parent.parent


EXPECTED_NODE_NAMES = {
    "format_parser_prompt",
    "email_parser",
    "match_account",
    "format_researcher_input",
    "researcher",
    "format_categorizer_input",
    "categorizer",
    "synthesize_transaction",
    "push_to_ynab",
    "publish_transaction_event",
}


@pytest.mark.asyncio
async def test_dag_has_ten_nodes(workflow_path):
    """Verify the DSL DAG is constructed with 10 nodes accurately."""

    with (
        patch(
            "workflows.bank_to_ynab.agents.email_parser.create_email_parser"
        ) as mock_parser,
        patch(
            "workflows.bank_to_ynab.agents.researcher.create_researcher"
        ) as mock_researcher,
        patch(
            "workflows.bank_to_ynab.agents.categorizer.create_categorizer"
        ) as mock_cat,
    ):
        mock_parser_agent = MagicMock()
        mock_parser_agent.name = "email_parser"
        mock_parser.return_value = mock_parser_agent

        mock_researcher_agent = MagicMock()
        mock_researcher_agent.name = "researcher"
        mock_researcher.return_value = mock_researcher_agent

        mock_cat_agent = MagicMock()
        mock_cat_agent.name = "categorizer"
        mock_cat.return_value = mock_cat_agent

        # Load the DAG pipeline from YAML
        dag = load_workflow(str(workflow_path / "pipeline.yaml"))

        assert isinstance(dag, DAGRunner)
        assert dag.name == "bank_to_ynab"
        assert len(dag._nodes) == 10
        assert set(dag._nodes.keys()) == EXPECTED_NODE_NAMES

        # Verify parallel layer structure:
        # Layer 0: [format_parser_prompt] — root
        # Layer 1: [email_parser]
        # Layer 2: [format_researcher_input, match_account] — PARALLEL ⚡
        # Layer 3: [researcher]
        # Layer 4: [format_categorizer_input]
        # Layer 5: [categorizer]
        # Layer 6: [synthesize_transaction]
        # Layer 7: [push_to_ynab]
        # Layer 8: [publish_transaction_event]
        assert len(dag._layers) >= 2  # At least some parallelism exists

        # The parallel layer must contain both match_account and format_researcher_input
        parallel_layer = [
            layer
            for layer in dag._layers
            if "match_account" in layer and "format_researcher_input" in layer
        ]
        assert len(parallel_layer) == 1, (
            "match_account and format_researcher_input must be in the same parallel layer"
        )
