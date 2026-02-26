"""Single source of truth for the platform version.

Version is auto-derived from git tags by setuptools-scm.
Never edit manually — tag a release with ``git tag vX.Y.Z`` instead.
"""

__all__ = ["VERSION", "APP_NAME"]

APP_NAME = "AutoPilot"

try:
    from importlib.metadata import version as _pkg_version

    VERSION = _pkg_version("autopilot")
except Exception:
    from autopilot._version import __version__ as VERSION
