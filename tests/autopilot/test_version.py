"""Tests for autopilot.version — auto-versioning via setuptools-scm."""

import re
from importlib.metadata import version as pkg_version

from autopilot.version import APP_NAME, VERSION


def test_version_is_string():
    """VERSION must be a non-empty string."""
    assert isinstance(VERSION, str)
    assert len(VERSION) > 0


def test_version_semver_pattern():
    """VERSION must match X.Y.Z or X.Y.Z.devN+gHASH pattern."""
    # Matches: 5.0.0, 5.0.1.dev3+gabcdef1, 5.0.1.dev0+gabcdef1.d20260223
    pattern = r"^\d+\.\d+\.\d+(\.\w+(\+[\w.]+)?)?$"
    assert re.match(pattern, VERSION), (
        f"VERSION '{VERSION}' does not match semver pattern"
    )


def test_app_name():
    """APP_NAME must be 'AutoPilot'."""
    assert APP_NAME == "AutoPilot"


def test_version_matches_metadata():
    """VERSION must match importlib.metadata for the installed package."""
    assert VERSION == pkg_version("autopilot")
