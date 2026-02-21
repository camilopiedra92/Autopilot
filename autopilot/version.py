"""
Single source of truth for the platform version.

Reads from pyproject.toml at import time and caches.
All other files import VERSION from here instead of hardcoding.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["VERSION", "APP_NAME"]

APP_NAME = "AutoPilot"


def _read_version() -> str:
    """Read version directly from pyproject.toml (avoids stale pip cache)."""
    try:
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if toml_path.exists():
            for line in toml_path.read_text().splitlines():
                if line.strip().startswith("version"):
                    # Parse: version = "5.0.0"
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "5.0.0"  # Hardcoded fallback


VERSION = _read_version()
