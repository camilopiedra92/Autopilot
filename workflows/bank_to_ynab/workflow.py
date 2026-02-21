"""
BankToYnabWorkflow — Parse bank notification emails into YNAB transactions.

manifest.yaml, pipeline.yaml, and agent cards are auto-loaded by BaseWorkflow.
This file (workflow.py) provides the custom execute() override to map inputs
and outputs to the declarative pipeline.
"""

import structlog

from autopilot.base_workflow import BaseWorkflow

logger = structlog.get_logger(__name__)


class BankToYnabWorkflow(BaseWorkflow):
    """
    Bank→YNAB: Parse bank notification emails and create YNAB transactions.

    This class loads its identity from manifest.yaml and its execution logic
    from pipeline.yaml. It relies entirely on the platform's DSL loader.
    """

    pass
