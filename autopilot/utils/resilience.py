"""
Resilience Utils — Exponential Backoff & Jitter
Provides decorators and utilities for handling transient failures in LLM and API calls.
"""

import asyncio
import random
import structlog
from functools import wraps
from typing import Callable, Any, Type

logger = structlog.get_logger(__name__)


def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries an async function with exponential backoff and jitter.

    Args:
        retries: Max number of retries (default 3).
        initial_delay: Initial wait time in seconds (default 1.0).
        max_delay: Max cap on wait time (default 10.0).
        backoff_factor: Multiplier for exponential growth (default 2.0).
        retryable_exceptions: Tuple of exceptions to catch and retry (default catch-all).
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, retries + 2):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt > retries:
                        break  # Max retries exceeded

                    # Calculate delay with jitter
                    wait_time = min(delay, max_delay)
                    # Add +/- 10% jitter to prevent "thundering herd"
                    jitter = wait_time * random.uniform(-0.1, 0.1)
                    actual_wait = max(0, wait_time + jitter)

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        retries_left=retries - attempt + 1,
                        wait_seconds=round(actual_wait, 2),
                        error=str(e),
                    )

                    await asyncio.sleep(actual_wait)
                    delay *= backoff_factor

            logger.error(
                "retry_exhausted",
                function=func.__name__,
                attempts=retries + 1,
                error=str(last_exception),
            )
            raise last_exception  # Re-raise the last error so caller handles it

        return wrapper

    return decorator
