import pytest
from opentelemetry import trace

@pytest.fixture(autouse=True)
def disable_tracing():
    """Disable OpenTelemetry tracer console exports to prevent Pytest stdout closed exceptions."""
    # Set a dummy provider so background spans do not log to pytest stdout on exit
    from opentelemetry.sdk.trace import TracerProvider
    trace.set_tracer_provider(TracerProvider())
    yield
