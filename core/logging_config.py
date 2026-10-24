"""
Structured logging configuration for the RAG framework.

Provides consistent logging with correlation IDs, performance timing,
and JSON formatting for production environments.
"""

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Optional, Callable, Any

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        record.correlation_id = correlation_id.get() or "-"
        return True


class ColoredFormatter(logging.Formatter):
    """Colorized formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_correlation: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format for logs (for production)
        include_correlation: Include correlation IDs in logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if json_format:
        # JSON format for production
        format_str = (
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","correlation_id":"%(correlation_id)s",'
            '"message":"%(message)s"}'
        )
        formatter = logging.Formatter(format_str)
    else:
        # Colored format for development
        format_str = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "[%(correlation_id)s] %(message)s"
        )
        formatter = ColoredFormatter(format_str, datefmt="%H:%M:%S")

    console_handler.setFormatter(formatter)

    if include_correlation:
        console_handler.addFilter(CorrelationIdFilter())

    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current context.

    Args:
        cid: Correlation ID to set, or None to generate a new one

    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id.get() or "-"


def log_timing(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use, or None to create one from function module
        level: Log level for timing messages
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                _logger.log(level, f"{func.__name__} completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                _logger.log(
                    logging.ERROR, f"{func.__name__} failed after {elapsed:.2f}ms: {e}"
                )
                raise

        return wrapper

    return decorator


def log_async_timing(
    logger: Optional[logging.Logger] = None, level: int = logging.DEBUG
):
    """
    Decorator to log async function execution time.

    Args:
        logger: Logger to use, or None to create one from function module
        level: Log level for timing messages
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start_time) * 1000
                _logger.log(level, f"{func.__name__} completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start_time) * 1000
                _logger.log(
                    logging.ERROR, f"{func.__name__} failed after {elapsed:.2f}ms: {e}"
                )
                raise

        return wrapper

    return decorator
