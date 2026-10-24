"""
Unit tests for the Azure OpenAI client.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.clients.azure_openai import AzureOpenAIClient
from core.exceptions import AzureServiceError, RetryExhaustedError


class TestAzureOpenAIClient:
    """Tests for AzureOpenAIClient class."""

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_client_initialization(self, mock_azure_openai):
        """Test client initializes correctly with config."""
        client = AzureOpenAIClient()

        assert client.endpoint is not None
        assert client.api_key is not None
        mock_azure_openai.assert_called_once()

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_client_initialization_with_explicit_params(self, mock_azure_openai):
        """Test client accepts explicit parameters."""
        client = AzureOpenAIClient(
            endpoint="https://custom.openai.azure.com",
            api_key="custom-key",
            api_version="2024-01-01",
            embedding_deployment="custom-embedding",
            completion_deployment="custom-completion",
        )

        assert client.endpoint == "https://custom.openai.azure.com"
        assert client.api_key == "custom-key"
        assert client.api_version == "2024-01-01"

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_embedding(self, mock_azure_openai, mock_openai_client):
        """Test embedding generation."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()
        embedding = client.get_embedding("Test text")

        assert len(embedding) == 3072
        mock_openai_client.embeddings.create.assert_called_once()

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_embedding_caching(self, mock_azure_openai, mock_openai_client):
        """Test embedding caching works."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()

        # First call
        embedding1 = client.get_embedding("Test text")
        # Second call with same text
        embedding2 = client.get_embedding("Test text")

        # Should only call API once due to caching
        assert mock_openai_client.embeddings.create.call_count == 1
        assert embedding1 == embedding2

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_embedding_no_cache(self, mock_azure_openai, mock_openai_client):
        """Test embedding without caching."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()

        # Two calls without cache
        client.get_embedding("Test text", use_cache=False)
        client.get_embedding("Test text", use_cache=False)

        # Should call API twice
        assert mock_openai_client.embeddings.create.call_count == 2

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_completion(self, mock_azure_openai, mock_openai_client):
        """Test chat completion."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()
        response = client.get_completion("What is AI?")

        assert response == "This is a test response."
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_completion_with_history(self, mock_azure_openai, mock_openai_client):
        """Test chat completion with conversation history."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        response = client.get_completion("What is AI?", conversation_history=history)

        assert response is not None
        # Check that messages were built correctly
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) >= 2

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_get_completion_with_images(self, mock_azure_openai, mock_openai_client):
        """Test chat completion with images."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()
        images = [{"b64": "abc123", "mime": "image/png"}]
        response = client.get_completion_with_images(
            "Describe this image", images=images
        )

        assert response is not None

    @patch("core.clients.azure_openai.AzureOpenAI")
    @patch("core.clients.azure_openai.time.sleep")
    def test_retry_on_failure(self, mock_sleep, mock_azure_openai, mock_openai_client):
        """Test retry behavior on API failure."""
        mock_azure_openai.return_value = mock_openai_client

        # Fail twice, then succeed
        mock_openai_client.embeddings.create.side_effect = [
            Exception("Rate limited"),
            Exception("Rate limited"),
            MagicMock(data=[MagicMock(embedding=[0.1] * 3072)]),
        ]

        client = AzureOpenAIClient()
        embedding = client.get_embedding("Test text", use_cache=False)

        assert len(embedding) == 3072
        assert mock_openai_client.embeddings.create.call_count == 3

    @patch("core.clients.azure_openai.AzureOpenAI")
    @patch("core.clients.azure_openai.time.sleep")
    def test_retry_exhausted(self, mock_sleep, mock_azure_openai, mock_openai_client):
        """Test RetryExhaustedError when all retries fail."""
        mock_azure_openai.return_value = mock_openai_client
        mock_openai_client.embeddings.create.side_effect = Exception("Always fails")

        client = AzureOpenAIClient()

        with pytest.raises(RetryExhaustedError) as exc_info:
            client.get_embedding("Test text", use_cache=False)

        assert "failed after" in str(exc_info.value)

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_clear_cache(self, mock_azure_openai, mock_openai_client):
        """Test cache clearing."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()
        client.get_embedding("Test text")

        assert client.cache_stats["size"] == 1

        client.clear_cache()

        assert client.cache_stats["size"] == 0

    @patch("core.clients.azure_openai.AzureOpenAI")
    def test_cache_stats(self, mock_azure_openai, mock_openai_client):
        """Test cache statistics."""
        mock_azure_openai.return_value = mock_openai_client

        client = AzureOpenAIClient()

        # Initially empty
        stats = client.cache_stats
        assert stats["size"] == 0

        # Add some embeddings
        client.get_embedding("Text 1")
        client.get_embedding("Text 2")
        client.get_embedding("Text 3")

        stats = client.cache_stats
        assert stats["size"] == 3
