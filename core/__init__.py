# Core Module for Azure Multimodal RAG Framework
# This package provides shared clients, utilities, and interfaces

from .config import get_config, Config
from .exceptions import (
    RAGException,
    AzureServiceError,
    ImageProcessingError,
    ValidationError,
    ConfigurationError,
)
from .logging_config import get_logger, configure_logging

__version__ = "1.0.0"
__all__ = [
    "get_config",
    "Config",
    "RAGException",
    "AzureServiceError",
    "ImageProcessingError",
    "ValidationError",
    "ConfigurationError",
    "get_logger",
    "configure_logging",
]
