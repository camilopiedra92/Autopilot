"""
Example Agent Factory.

This file demonstrates how to instantiate an ADK agent using the
platform's `create_platform_agent` factory, which automatically
wires up observability and standard callbacks as per the Edge Architecture.

DO NOT use `LlmAgent(...)` directly.
"""

from google.adk.agents import BaseAgent
from autopilot.agents.base import create_platform_agent


def create_example_agent() -> BaseAgent:
    """Creates the example agent with platform defaults."""
    return create_platform_agent(
        name="example_agent",
        instruction="You are a helpful assistant.",
        # model="gemini-3-flash-preview",  # Platform default is used automatically
        # tools=[my_custom_tool],        # Optional: Attach tools
        # before_model_callback=...,     # Optional: Attach guardrails
        # after_model_callback=...,
    )
