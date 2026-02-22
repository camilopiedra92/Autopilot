"""
Platform Configuration — Extends per-workflow settings with platform-wide config.

The platform config manages:
  - Shared Google API / GCP credentials
  - Platform-level settings (port, environment, log level)
  - Workflow discovery paths
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    """Platform-wide settings. Per-workflow settings are in each workflow's manifest."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── Platform ─────────────────────────────────────────────────────
    platform_name: str = "AutoPilot"
    environment: Literal["development", "staging", "production"] = "development"
    port: int = 8080
    log_level: str = "INFO"

    # ── GCP ───────────────────────────────────────────────────────────
    google_cloud_project: str = ""
    google_api_key: str = ""

    # ── AI Model ──────────────────────────────────────────────────────
    model_name: str = "gemini-3-flash-preview"

    # ── Workflows ─────────────────────────────────────────────────────
    workflows_dir: str = "workflows"
    auto_discover_workflows: bool = True

    def bridge_adk_key(self) -> None:
        """ADK expects GOOGLE_GENAI_API_KEY."""
        if not os.getenv("GOOGLE_GENAI_API_KEY") and self.google_api_key:
            os.environ["GOOGLE_GENAI_API_KEY"] = self.google_api_key


@lru_cache
def get_platform_settings() -> PlatformSettings:
    """Singleton accessor — parsed once, cached forever."""
    settings = PlatformSettings()
    settings.bridge_adk_key()
    return settings
