"""
JSON Utilities — Robust JSON extraction from LLM output.

Provides multi-strategy JSON extraction for parsing structured data
from LLM responses that may contain markdown, explanatory text, or
other non-JSON content wrapping the actual JSON payload.
"""

import json
import re


def extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output with multiple fallback strategies.

    Tries:
      1. Markdown ```json ... ``` code block
      2. Markdown ``` ... ``` code block
      3. Raw JSON object (outermost { ... })

    Raises ValueError if no valid JSON is found.
    """
    # Strategy 1: Markdown ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass  # Fall through to next strategy

    # Strategy 2: Markdown ``` ... ``` block (any language)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Raw JSON object — match outermost braces
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response: {text[:200]}")
