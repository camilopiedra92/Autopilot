"""
Tool Authentication Framework — Credential management for platform tools.

Mirrors Google ADK's ``ToolContext.request_credential()`` /
``ToolContext.get_auth_response()`` pattern at the platform level.

This module provides:
  - ``ToolAuthConfig``: Declarative credential configuration per tool
  - ``ToolAuthManager``: Centralized credential store and auth flow management
  - Integration with ADK's native authentication when using ``LlmAgent``

ADK Alignment:
  - ADK's ToolContext provides ``request_credential(auth_config)`` and
    ``get_auth_response(auth_config)`` for runtime credential flows.
  - This module wraps that pattern into a platform-level manager that
    works for ALL tools (not just ADK agent tools), and provides a
    credential store for pre-configured service accounts and API keys.

Usage::

    from autopilot.core.tools.auth import ToolAuthConfig, get_auth_manager

    # Configure auth for a tool
    config = ToolAuthConfig(
        tool_name="ynab.create_transaction",
        auth_type="api_key",
        credential_key="YNAB_API_TOKEN",
    )

    manager = get_auth_manager()
    manager.register(config)

    # At runtime — tools can request credentials
    cred = manager.get_credential("ynab.create_transaction")
    if cred is None:
        manager.request_credential("ynab.create_transaction")
"""

from __future__ import annotations

import os
import structlog
from dataclasses import dataclass, field
from typing import Any, Literal

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolAuthConfig — Credential configuration per tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class ToolAuthConfig:
    """
    Declarative credential configuration for a platform tool.

    Attributes:
        tool_name: The tool this config applies to (e.g. "ynab.create_transaction").
        auth_type: Type of authentication ("api_key", "oauth2", "service_account").
        credential_key: Environment variable or state key where the credential lives.
        scopes: OAuth2 scopes required (if auth_type is "oauth2").
        description: Human-readable description of why this auth is needed.
    """

    tool_name: str
    auth_type: Literal["api_key", "oauth2", "service_account"] = "api_key"
    credential_key: str = ""
    scopes: tuple[str, ...] = ()
    description: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AuthCredential — Resolved credential value
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class AuthCredential:
    """
    A resolved credential ready for use by a tool.

    Attributes:
        auth_type: The type of credential.
        token: The credential value (API key, access token, etc.).
        scopes: Active scopes for this credential.
        metadata: Additional context (e.g., expiry time, refresh token).
    """

    auth_type: str
    token: str
    scopes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the credential has a non-empty token."""
        return bool(self.token)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ToolAuthManager — Credential store and auth flow management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToolAuthManager:
    """
    Centralized credential manager for platform tools.

    Manages:
      - Registration of auth configs per tool
      - Resolution of credentials from env vars, state, or session
      - Pending auth requests (for human-in-the-loop flows)
      - Credential caching within a session

    Thread-safety: safe for single-process asyncio (like Cloud Run).
    """

    def __init__(self) -> None:
        self._configs: dict[str, ToolAuthConfig] = {}
        self._credentials: dict[str, AuthCredential] = {}
        self._pending_requests: set[str] = set()

    # ── Registration ─────────────────────────────────────────────────

    def register(self, config: ToolAuthConfig) -> None:
        """Register an auth configuration for a tool."""
        self._configs[config.tool_name] = config
        logger.debug(
            "tool_auth_config_registered",
            tool=config.tool_name,
            auth_type=config.auth_type,
        )

    def get_config(self, tool_name: str) -> ToolAuthConfig | None:
        """Get the auth configuration for a tool, if registered."""
        return self._configs.get(tool_name)

    # ── Credential Resolution ────────────────────────────────────────

    def get_credential(
        self,
        tool_name: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> AuthCredential | None:
        """
        Resolve a credential for the given tool.

        Resolution order:
          1. In-memory cache (already resolved in this session)
          2. State dict (provided by AgentContext/ToolContext)
          3. Environment variable (from ToolAuthConfig.credential_key)

        Args:
            tool_name: The tool requesting credentials.
            state: Optional session/pipeline state dict.

        Returns:
            AuthCredential if found, None if not available.
        """
        # 1. Check cache
        if tool_name in self._credentials:
            cached = self._credentials[tool_name]
            if cached.is_valid:
                return cached

        config = self._configs.get(tool_name)
        if config is None:
            return None

        # 2. Check state
        if state and config.credential_key:
            token = state.get(config.credential_key)
            if token:
                cred = AuthCredential(
                    auth_type=config.auth_type,
                    token=str(token),
                    scopes=config.scopes,
                )
                self._credentials[tool_name] = cred
                return cred

        # 3. Check environment
        if config.credential_key:
            env_value = os.environ.get(config.credential_key)
            if env_value:
                cred = AuthCredential(
                    auth_type=config.auth_type,
                    token=env_value,
                    scopes=config.scopes,
                    metadata={"source": "env"},
                )
                self._credentials[tool_name] = cred
                return cred

        return None

    # ── Auth Flow Management ─────────────────────────────────────────

    def request_credential(self, tool_name: str) -> dict[str, Any]:
        """
        Signal that a tool needs credentials that are not yet available.

        This is the platform equivalent of ADK's
        ``tool_context.request_credential(auth_config)``.

        Returns:
            A dict describing the auth request, suitable for returning
            to the LLM or the user.
        """
        config = self._configs.get(tool_name)
        if config is None:
            logger.warning("auth_request_no_config", tool=tool_name)
            return {
                "status": "error",
                "error": f"No auth config registered for '{tool_name}'",
            }

        self._pending_requests.add(tool_name)
        logger.info(
            "credential_requested",
            tool=tool_name,
            auth_type=config.auth_type,
            credential_key=config.credential_key,
        )

        return {
            "status": "auth_required",
            "tool": tool_name,
            "auth_type": config.auth_type,
            "credential_key": config.credential_key,
            "scopes": list(config.scopes),
            "description": config.description or f"Credentials needed for {tool_name}",
        }

    def provide_credential(
        self,
        tool_name: str,
        token: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AuthCredential:
        """
        Provide a credential in response to a previous request.

        This is used when the user or system provides credentials after
        a ``request_credential()`` call.

        Args:
            tool_name: The tool the credential is for.
            token: The credential value.
            metadata: Optional additional context.

        Returns:
            The stored AuthCredential.
        """
        config = self._configs.get(tool_name)
        cred = AuthCredential(
            auth_type=config.auth_type if config else "api_key",
            token=token,
            scopes=config.scopes if config else (),
            metadata=metadata or {},
        )
        self._credentials[tool_name] = cred
        self._pending_requests.discard(tool_name)

        logger.info("credential_provided", tool=tool_name)
        return cred

    # ── Status ───────────────────────────────────────────────────────

    @property
    def pending_requests(self) -> list[str]:
        """Tools with pending credential requests."""
        return list(self._pending_requests)

    def has_credential(self, tool_name: str) -> bool:
        """Check if a credential is available (cached or resolvable)."""
        return tool_name in self._credentials and self._credentials[tool_name].is_valid

    def list_configs(self) -> list[ToolAuthConfig]:
        """List all registered auth configurations."""
        return list(self._configs.values())

    def clear(self) -> None:
        """Clear all configs, credentials, and pending requests."""
        self._configs.clear()
        self._credentials.clear()
        self._pending_requests.clear()

    def __repr__(self) -> str:
        return (
            f"<ToolAuthManager configs={len(self._configs)} "
            f"cached={len(self._credentials)} pending={len(self._pending_requests)}>"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_manager: ToolAuthManager | None = None


def get_auth_manager() -> ToolAuthManager:
    """Process-global singleton accessor for the ToolAuthManager."""
    global _manager
    if _manager is None:
        _manager = ToolAuthManager()
    return _manager


def reset_auth_manager() -> None:
    """Reset the global auth manager. For testing only."""
    global _manager
    _manager = None
