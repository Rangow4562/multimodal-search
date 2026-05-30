"""
Input validation utilities for the RAG framework.

Provides validation functions and decorators for sanitizing and validating
user inputs, API requests, and file paths.
"""

import re
import os
from typing import Optional, Any, List
from urllib.parse import urlparse

from .exceptions import ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


# Maximum lengths for common inputs
MAX_QUERY_LENGTH = 10000
MAX_PATH_LENGTH = 500
MAX_URL_LENGTH = 2000


def validate_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Validate and sanitize a user query.

    Args:
        query: The query string to validate
        max_length: Maximum allowed length

    Returns:
        Sanitized query string

    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty", field_name="query")

    if not isinstance(query, str):
        raise ValidationError(
            f"Query must be a string, got {type(query).__name__}",
            field_name="query",
            invalid_value=type(query).__name__,
        )

    # Strip whitespace
    query = query.strip()

    if not query:
        raise ValidationError(
            "Query cannot be empty or whitespace only", field_name="query"
        )

    if len(query) > max_length:
        raise ValidationError(
            f"Query exceeds maximum length of {max_length} characters",
            field_name="query",
            invalid_value=f"length={len(query)}",
        )

    return query


def validate_top_k(top_k: Any, min_val: int = 1, max_val: int = 100) -> int:
    """
    Validate top_k parameter.

    Args:
        top_k: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated integer

    Raises:
        ValidationError: If value is invalid
    """
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValidationError(
            "top_k must be an integer", field_name="top_k", invalid_value=top_k
        )

    if top_k < min_val or top_k > max_val:
        raise ValidationError(
            f"top_k must be between {min_val} and {max_val}",
            field_name="top_k",
            invalid_value=top_k,
        )

    return top_k


def validate_file_path(path: str, must_exist: bool = False) -> str:
    """
    Validate and sanitize a file path.

    Args:
        path: File path to validate
        must_exist: If True, verify file exists

    Returns:
        Normalized path

    Raises:
        ValidationError: If path is invalid
    """
    if not path:
        raise ValidationError("File path cannot be empty", field_name="path")

    if not isinstance(path, str):
        raise ValidationError(
            f"Path must be a string, got {type(path).__name__}", field_name="path"
        )

    if len(path) > MAX_PATH_LENGTH:
        raise ValidationError(
            f"Path exceeds maximum length of {MAX_PATH_LENGTH}", field_name="path"
        )

    # Normalize the path
    normalized = os.path.normpath(path)

    # Check for path traversal attempts
    if ".." in normalized:
        raise ValidationError(
            "Path traversal not allowed", field_name="path", invalid_value=path
        )

    if must_exist and not os.path.exists(normalized):
        raise ValidationError(
            f"File does not exist: {normalized}", field_name="path", invalid_value=path
        )

    return normalized


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """
    Validate and sanitize a URL.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes (default: http, https)

    Returns:
        Validated URL

    Raises:
        ValidationError: If URL is invalid
    """
    if not url:
        raise ValidationError("URL cannot be empty", field_name="url")

    if len(url) > MAX_URL_LENGTH:
        raise ValidationError(
            f"URL exceeds maximum length of {MAX_URL_LENGTH}", field_name="url"
        )

    allowed = allowed_schemes or ["http", "https"]

    try:
        parsed = urlparse(url)

        if parsed.scheme not in allowed:
            raise ValidationError(
                f"URL scheme must be one of: {', '.join(allowed)}",
                field_name="url",
                invalid_value=parsed.scheme,
            )

        if not parsed.netloc:
            raise ValidationError(
                "URL must have a valid host", field_name="url", invalid_value=url
            )

        return url

    except Exception as e:
        raise ValidationError(
            f"Invalid URL format: {e}", field_name="url", invalid_value=url
        )


def validate_blob_name(blob_name: str) -> str:
    """
    Validate an Azure blob name.

    Args:
        blob_name: Blob name to validate

    Returns:
        Validated blob name

    Raises:
        ValidationError: If blob name is invalid
    """
    if not blob_name:
        raise ValidationError("Blob name cannot be empty", field_name="blob_name")

    # Azure blob naming rules
    if len(blob_name) > 1024:
        raise ValidationError(
            "Blob name exceeds maximum length of 1024 characters",
            field_name="blob_name",
        )

    # Check for problematic characters
    invalid_chars = ["\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in invalid_chars:
        if char in blob_name:
            raise ValidationError(
                f"Blob name contains invalid character: {char}",
                field_name="blob_name",
                invalid_value=blob_name,
            )

    return blob_name


def sanitize_html(text: str) -> str:
    """
    Remove or escape potentially dangerous HTML content.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove script tags and their content
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL
    )

    # Remove event handlers
    text = re.sub(r"\s+on\w+\s*=\s*['\"][^'\"]*['\"]", "", text, flags=re.IGNORECASE)

    # Escape remaining HTML entities
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#x27;")

    return text


def validate_json_string(json_str: str, field_name: str = "json") -> str:
    """
    Validate that a string is valid JSON.

    Args:
        json_str: JSON string to validate
        field_name: Field name for error messages

    Returns:
        The original JSON string if valid

    Raises:
        ValidationError: If JSON is invalid
    """
    import json

    if not json_str:
        return json_str

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON: {e.msg}",
            field_name=field_name,
            invalid_value=json_str[:100] + "..." if len(json_str) > 100 else json_str,
        )
