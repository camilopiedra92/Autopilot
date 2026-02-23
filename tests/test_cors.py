"""
CORS Hardening Tests — A6

Validates that the headless API correctly:
- Disables CORS by default (no Access-Control-Allow-Origin header)
- Enables CORS only when API_CORS_ORIGINS is explicitly configured
- Rejects browser requests from unconfigured origins
"""

import importlib
import os


def _make_app(cors_value: str | None = None):
    """Create a fresh FastAPI app with a specific API_CORS_ORIGINS value.

    CORSMiddleware is evaluated at import time in app.py, so we must
    reload the module to pick up a different env value.
    """
    env_key = "API_CORS_ORIGINS"
    original = os.environ.get(env_key)

    try:
        if cors_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = cors_value

        # Reload app module so it re-evaluates os.getenv at module scope
        import app as app_module

        importlib.reload(app_module)
        return app_module.app
    finally:
        # Restore original env state
        if original is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = original


def _preflight(client, origin: str, path: str = "/health"):
    """Send a CORS preflight (OPTIONS) request with the given Origin."""
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


class TestCORSDisabledByDefault:
    """When API_CORS_ORIGINS is unset, no CORS headers should be returned."""

    def test_no_cors_headers_on_preflight(self):
        from fastapi.testclient import TestClient

        app = _make_app(cors_value=None)
        client = TestClient(app)
        response = _preflight(client, origin="https://evil.com")

        assert "access-control-allow-origin" not in response.headers

    def test_no_cors_headers_with_empty_string(self):
        from fastapi.testclient import TestClient

        app = _make_app(cors_value="")
        client = TestClient(app)
        response = _preflight(client, origin="https://evil.com")

        assert "access-control-allow-origin" not in response.headers


class TestCORSEnabledExplicitly:
    """When API_CORS_ORIGINS is set, only those origins get CORS headers."""

    def test_allowed_origin_gets_cors_headers(self):
        from fastapi.testclient import TestClient

        app = _make_app(cors_value="https://admin.example.com")
        client = TestClient(app)
        response = _preflight(client, origin="https://admin.example.com")

        assert (
            response.headers.get("access-control-allow-origin")
            == "https://admin.example.com"
        )

    def test_unknown_origin_rejected(self):
        from fastapi.testclient import TestClient

        app = _make_app(cors_value="https://admin.example.com")
        client = TestClient(app)
        response = _preflight(client, origin="https://evil.com")

        assert "access-control-allow-origin" not in response.headers

    def test_multiple_origins(self):
        from fastapi.testclient import TestClient

        app = _make_app(cors_value="https://admin.example.com,http://localhost:3000")
        client = TestClient(app)

        # First origin allowed
        r1 = _preflight(client, origin="https://admin.example.com")
        assert (
            r1.headers.get("access-control-allow-origin") == "https://admin.example.com"
        )

        # Second origin allowed
        r2 = _preflight(client, origin="http://localhost:3000")
        assert r2.headers.get("access-control-allow-origin") == "http://localhost:3000"

        # Unknown origin still rejected
        r3 = _preflight(client, origin="https://evil.com")
        assert "access-control-allow-origin" not in r3.headers
