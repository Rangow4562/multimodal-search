"""
Custom exception hierarchy for the RAG framework.

All exceptions inherit from RAGException to allow catching all framework
errors with a single except clause while still allowing granular handling.
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """Base exception for all RAG framework errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        result = self.message
        if self.details:
            result += f" | Details: {self.details}"
        if self.cause:
            result += f" | Caused by: {self.cause}"
        return result


class AzureServiceError(RAGException):
    """Error communicating with Azure services (OpenAI, Search, Blob Storage)."""

    def __init__(
        self,
        message: str,
        service_name: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        details["service_name"] = service_name
        if operation:
            details["operation"] = operation
        super().__init__(message, details, cause)
        self.service_name = service_name
        self.operation = operation


class ImageProcessingError(RAGException):
    """Error processing or fetching images."""

    def __init__(
        self,
        message: str,
        image_source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        if image_source:
            details["image_source"] = image_source
        super().__init__(message, details, cause)
        self.image_source = image_source


class ValidationError(RAGException):
    """Error validating input data."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)[:100]  # Truncate for safety
        super().__init__(message, details)
        self.field_name = field_name
        self.invalid_value = invalid_value


class ConfigurationError(RAGException):
    """Error with configuration or environment variables."""

    def __init__(
        self,
        message: str,
        missing_vars: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if missing_vars:
            details["missing_vars"] = missing_vars
        super().__init__(message, details)
        self.missing_vars = missing_vars or []


class RetryExhaustedError(AzureServiceError):
    """All retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        service_name: str,
        max_retries: int,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        details = details or {}
        details["max_retries"] = max_retries
        super().__init__(message, service_name, "retry", details, cause)
        self.max_retries = max_retries


class CircuitBreakerOpenError(AzureServiceError):
    """Circuit breaker is open, service calls are blocked."""

    def __init__(
        self,
        message: str,
        service_name: str,
        reset_time: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if reset_time:
            details["reset_time"] = reset_time
        super().__init__(message, service_name, "circuit_breaker", details)
        self.reset_time = reset_time
