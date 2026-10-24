"""
Unified Azure OpenAI client with retry, caching, and structured logging.

Consolidates implementations from rag_query.py, parse_document.py, and api.py.
"""

import time
import hashlib
from typing import Optional, List, Dict, Any
from functools import lru_cache

from openai import AzureOpenAI

from ..config import get_config, AzureOpenAIConfig
from ..exceptions import AzureServiceError, RetryExhaustedError
from ..logging_config import get_logger, log_timing

logger = get_logger(__name__)


class AzureOpenAIClient:
    """
    Unified Azure OpenAI client with retry logic and caching.

    Features:
        - Automatic retry with exponential backoff
        - LRU caching for embeddings
        - Structured logging
        - Support for both embeddings and completions
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        completion_deployment: Optional[str] = None,
    ):
        """
        Initialize the Azure OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            embedding_deployment: Deployment name for embeddings
            completion_deployment: Deployment name for completions
        """
        config = get_config()

        self.endpoint = endpoint or config.openai.endpoint
        self.api_key = api_key or config.openai.api_key
        self.api_version = api_version or config.openai.api_version
        self.embedding_deployment = (
            embedding_deployment or config.openai.embedding_deployment
        )
        self.completion_deployment = (
            completion_deployment or config.openai.completion_deployment
        )

        # Retry configuration
        self.max_retries = config.retry.max_retries
        self.base_backoff = config.retry.base_backoff
        self.max_backoff = config.retry.max_backoff
        self.backoff_multiplier = config.retry.backoff_multiplier

        # Caching configuration
        self.cache_enabled = config.cache.enable_caching
        self.cache_size = config.cache.embedding_cache_size

        # Initialize client
        if not self.endpoint or not self.api_key:
            raise AzureServiceError(
                "Azure OpenAI credentials not configured",
                service_name="azure_openai",
                operation="init",
            )

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        # Initialize embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

        logger.info(
            f"Azure OpenAI client initialized (endpoint: {self.endpoint[:30]}...)"
        )

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _retry_with_backoff(self, operation: str, func, *args, **kwargs):
        """Execute function with retry and exponential backoff."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == self.max_retries - 1:
                    break

                backoff = min(
                    self.base_backoff * (self.backoff_multiplier**attempt),
                    self.max_backoff,
                )

                logger.warning(
                    f"{operation} failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)

        raise RetryExhaustedError(
            f"{operation} failed after {self.max_retries} attempts",
            service_name="azure_openai",
            max_retries=self.max_retries,
            cause=last_exception,
        )

    @log_timing()
    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Get embedding vector for the given text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            List of floats representing the embedding vector
        """
        if not self.embedding_deployment:
            raise AzureServiceError(
                "Embedding deployment not configured",
                service_name="azure_openai",
                operation="get_embedding",
            )

        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._cache_key(text)
            if cache_key in self._embedding_cache:
                logger.debug("Embedding cache hit")
                return self._embedding_cache[cache_key]

        def _create_embedding():
            response = self.client.embeddings.create(
                model=self.embedding_deployment, input=text
            )
            return response.data[0].embedding

        try:
            embedding = self._retry_with_backoff(
                "Embedding creation", _create_embedding
            )

            # Cache the result
            if use_cache and self.cache_enabled:
                cache_key = self._cache_key(text)
                # Limit cache size with simple eviction
                if len(self._embedding_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest]
                self._embedding_cache[cache_key] = embedding

            return embedding

        except RetryExhaustedError:
            raise
        except Exception as e:
            raise AzureServiceError(
                f"Failed to create embedding: {e}",
                service_name="azure_openai",
                operation="get_embedding",
                cause=e,
            )

    @log_timing()
    def get_completion(
        self,
        content: Any,
        conversation_history: Optional[List[Dict]] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Get chat completion from Azure OpenAI.

        Args:
            content: Message content (string or multimodal content list)
            conversation_history: Optional list of previous messages
            system_message: Optional system message to prepend

        Returns:
            The completion response text
        """
        if not self.completion_deployment:
            raise AzureServiceError(
                "Completion deployment not configured",
                service_name="azure_openai",
                operation="get_completion",
            )

        messages = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[:-1] if conversation_history else [])

        # Add current message
        messages.append({"role": "user", "content": content})

        def _create_completion():
            response = self.client.chat.completions.create(
                model=self.completion_deployment,
                messages=messages,
            )
            return response.choices[0].message.content

        try:
            return self._retry_with_backoff("Chat completion", _create_completion)
        except RetryExhaustedError:
            raise
        except Exception as e:
            raise AzureServiceError(
                f"Failed to get completion: {e}",
                service_name="azure_openai",
                operation="get_completion",
                cause=e,
            )

    def get_completion_with_images(
        self,
        prompt: str,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Get completion with optional image inputs.

        Args:
            prompt: Text prompt
            images: List of image dicts with 'b64' and 'mime' keys

        Returns:
            The completion response text
        """
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                if isinstance(img, dict) and img.get("b64") and img.get("mime"):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img['mime']};base64,{img['b64']}"
                            },
                        }
                    )
            return self.get_completion(content)
        else:
            return self.get_completion(prompt)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._embedding_cache),
            "max_size": self.cache_size,
        }
