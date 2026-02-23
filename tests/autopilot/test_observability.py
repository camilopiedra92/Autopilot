"""Tests for autopilot.observability — setup_tracing exporter resolution."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Reset the global tracer and provider between tests."""
    import autopilot.observability as obs

    obs._tracer = None
    # Reset to a fresh provider so set_tracer_provider succeeds
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    yield
    obs._tracer = None


# ── Service Name Resolution ──────────────────────────────────────────


def test_default_service_name_is_autopilot():
    """Default service name should be APP_NAME.lower(), not a workflow name."""
    from autopilot.observability import setup_tracing

    with patch.dict("os.environ", {}, clear=True):
        tracer = setup_tracing()
        assert tracer is not None

    provider = trace.get_tracer_provider()
    resource = provider.resource  # type: ignore[attr-defined]
    assert resource.attributes["service.name"] == "autopilot"


def test_k_service_env_overrides_default():
    """K_SERVICE env var (Cloud Run) should override the default."""
    from autopilot.observability import setup_tracing

    mock_exporter = MagicMock()
    mock_class = MagicMock(return_value=mock_exporter)

    with patch.dict("os.environ", {"K_SERVICE": "my-cloud-run-svc"}, clear=True):
        with patch("autopilot.observability.BatchSpanProcessor"):
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.cloud_trace": MagicMock(
                        CloudTraceSpanExporter=mock_class
                    )
                },
            ):
                tracer = setup_tracing()
                assert tracer is not None

    provider = trace.get_tracer_provider()
    resource = provider.resource  # type: ignore[attr-defined]
    assert resource.attributes["service.name"] == "my-cloud-run-svc"


def test_explicit_service_name_overrides_all():
    """Explicit service_name arg should override both env and default."""
    from autopilot.observability import setup_tracing

    mock_exporter = MagicMock()
    mock_class = MagicMock(return_value=mock_exporter)

    with patch.dict("os.environ", {"K_SERVICE": "ignored"}, clear=True):
        with patch("autopilot.observability.BatchSpanProcessor"):
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.cloud_trace": MagicMock(
                        CloudTraceSpanExporter=mock_class
                    )
                },
            ):
                tracer = setup_tracing(service_name="custom-name")
                assert tracer is not None

    provider = trace.get_tracer_provider()
    resource = provider.resource  # type: ignore[attr-defined]
    assert resource.attributes["service.name"] == "custom-name"


# ── Exporter Selection ───────────────────────────────────────────────


def test_console_exporter_when_no_env():
    """Without K_SERVICE or otlp_endpoint, should use ConsoleSpanExporter."""
    from autopilot.observability import setup_tracing

    with patch.dict("os.environ", {}, clear=True):
        with patch("autopilot.observability.BatchSpanProcessor") as mock_bsp:
            setup_tracing()
            assert mock_bsp.called
            exporter_arg = mock_bsp.call_args[0][0]
            assert isinstance(exporter_arg, ConsoleSpanExporter)


def test_cloud_trace_on_cloud_run():
    """With K_SERVICE set, should attempt CloudTraceSpanExporter."""
    from autopilot.observability import setup_tracing

    mock_exporter = MagicMock()
    mock_class = MagicMock(return_value=mock_exporter)

    with patch.dict("os.environ", {"K_SERVICE": "prod-svc"}, clear=True):
        with patch("autopilot.observability.BatchSpanProcessor") as mock_bsp:
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.cloud_trace": MagicMock(
                        CloudTraceSpanExporter=mock_class
                    )
                },
            ):
                setup_tracing()

            assert mock_bsp.called
            exporter_arg = mock_bsp.call_args[0][0]
            assert exporter_arg is mock_exporter


def test_otlp_endpoint_takes_priority_over_cloud_trace():
    """otlp_endpoint should take priority even when K_SERVICE is set."""
    from autopilot.observability import setup_tracing

    mock_otlp_exporter = MagicMock()
    mock_otlp_class = MagicMock(return_value=mock_otlp_exporter)

    with patch.dict("os.environ", {"K_SERVICE": "prod-svc"}, clear=True):
        with patch("autopilot.observability.BatchSpanProcessor") as mock_bsp:
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": MagicMock(
                        OTLPSpanExporter=mock_otlp_class
                    )
                },
            ):
                setup_tracing(otlp_endpoint="http://collector:4317")

            assert mock_bsp.called
            exporter_arg = mock_bsp.call_args[0][0]
            assert exporter_arg is mock_otlp_exporter
