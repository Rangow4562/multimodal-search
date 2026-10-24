"""
Configuration management for the RAG framework.

Provides centralized configuration with validation, environment variable
loading, and type-safe access patterns using Pydantic.
"""

import os
from typing import Optional
from functools import lru_cache
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI service configuration."""

    endpoint: str
    api_key: str
    api_version: str = "2024-06-01"
    embedding_deployment: Optional[str] = None
    completion_deployment: Optional[str] = None
    chat_deployment: Optional[str] = None

    def validate(self) -> list:
        """Return list of missing required fields."""
        missing = []
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        return missing


@dataclass
class AzureSearchConfig:
    """Azure Cognitive Search configuration."""

    endpoint: str
    admin_key: str
    index_name: str
    embedding_dim: int = 3072

    def validate(self) -> list:
        """Return list of missing required fields."""
        missing = []
        if not self.endpoint:
            missing.append("AZURE_SEARCH_ENDPOINT")
        if not self.admin_key:
            missing.append("AZURE_SEARCH_ADMIN_KEY")
        if not self.index_name:
            missing.append("AZURE_SEARCH_INDEX_NAME")
        return missing


@dataclass
class BlobStorageConfig:
    """Azure Blob Storage configuration."""

    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    default_container: str = "images"
    sas_expiry_hours: int = 24

    def validate(self) -> list:
        """Return list of missing required fields."""
        # Need either connection string OR account name + key
        if self.connection_string:
            return []
        if self.account_name and self.account_key:
            return []
        return [
            "AZURE_STORAGE_CONNECTION_STRING or (BLOB_STORAGE_ACCOUNT_NAME and BLOB_STORAGE_ACCOUNT_KEY)"
        ]

    def is_configured(self) -> bool:
        """Check if blob storage is properly configured."""
        return bool(self.connection_string or (self.account_name and self.account_key))


@dataclass
class CacheConfig:
    """Caching configuration."""

    embedding_cache_size: int = 1000
    embedding_cache_ttl: int = 3600  # seconds
    sas_token_cache_ttl: int = 3000  # seconds (less than actual expiry)
    enable_caching: bool = True


@dataclass
class RetryConfig:
    """Retry and resilience configuration."""

    max_retries: int = 3
    base_backoff: float = 1.0
    max_backoff: float = 60.0
    backoff_multiplier: float = 2.0

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class Config:
    """Main configuration class for the RAG framework."""

    openai: AzureOpenAIConfig
    search: AzureSearchConfig
    blob_storage: BlobStorageConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Application settings
    log_level: str = "INFO"
    sessions_dir: str = "./sessions"
    parsed_chunks_dir: str = "./parsed_chunks"
    max_images_per_response: int = 3
    max_conversation_history: int = 50

    def validate(self) -> list:
        """Return list of all missing required configuration."""
        missing = []
        missing.extend(self.openai.validate())
        missing.extend(self.search.validate())
        # Blob storage is optional but warn if not configured
        return missing


def _parse_connection_string(conn_str: str) -> tuple:
    """Parse Azure connection string to extract account name and key."""
    if not conn_str:
        return None, None
    try:
        parts = dict(part.split("=", 1) for part in conn_str.split(";") if "=" in part)
        return parts.get("AccountName"), parts.get("AccountKey")
    except Exception:
        return None, None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the application configuration.

    Configuration is loaded from environment variables and cached.
    Call this function to get a singleton Config instance.
    """
    # Parse connection string for account details
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    parsed_account, parsed_key = _parse_connection_string(conn_str)

    return Config(
        openai=AzureOpenAIConfig(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            completion_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT"),
            chat_deployment=os.getenv(
                "AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME"
            ),
        ),
        search=AzureSearchConfig(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "")
            or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""),
            admin_key=os.getenv("AZURE_SEARCH_ADMIN_KEY", ""),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", ""),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        ),
        blob_storage=BlobStorageConfig(
            connection_string=conn_str or None,
            account_name=os.getenv("BLOB_STORAGE_ACCOUNT_NAME") or parsed_account,
            account_key=os.getenv("BLOB_STORAGE_ACCOUNT_KEY") or parsed_key,
            default_container=os.getenv("BLOB_CONTAINER_NAME", "images"),
            sas_expiry_hours=int(os.getenv("SAS_EXPIRY_HOURS", "24")),
        ),
        cache=CacheConfig(
            embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "1000")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
        ),
        retry=RetryConfig(
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            base_backoff=float(os.getenv("BASE_BACKOFF", "1.0")),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        sessions_dir=os.getenv("SESSIONS_DIR", "./sessions"),
        parsed_chunks_dir=os.getenv("PARSED_CHUNKS_DIR", "./parsed_chunks"),
    )


def validate_config() -> tuple:
    """
    Validate configuration and return status.

    Returns:
        Tuple of (is_valid: bool, missing_vars: list)
    """
    config = get_config()
    missing = config.validate()
    return len(missing) == 0, missing
