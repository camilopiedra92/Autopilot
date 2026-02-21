"""
Centralized structured logging configuration.
Import and call `setup_logging()` once at application startup.

Replaces the duplicate structlog.configure() calls previously
scattered across app.py, main.py, and test_agent.py.
"""

import structlog


def setup_logging(level: int = 20, json_output: bool = False) -> None:
    """
    Configure structlog for the entire application.

    Args:
        level: Minimum log level (10=DEBUG, 20=INFO, 30=WARNING).
        json_output: If True, emit machine-readable JSON logs (for production).
                     If False, emit human-readable colored console logs.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
