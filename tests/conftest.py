"""
Pytest fixtures and configuration for the RAG framework tests.
"""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List


# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    test_env = {
        "AZURE_OPENAI_ENDPOINT": "https://test-openai.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
        "AZURE_OPENAI_COMPLETION_DEPLOYMENT": "gpt-4",
        "AZURE_SEARCH_ENDPOINT": "https://test-search.search.windows.net",
        "AZURE_SEARCH_ADMIN_KEY": "test-search-key",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=teststorage;AccountKey=testkey==;EndpointSuffix=core.windows.net",
        "BLOB_STORAGE_ACCOUNT_NAME": "teststorage",
        "BLOB_STORAGE_ACCOUNT_KEY": "testkey==",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    # Clear the cached config
    from core.config import get_config

    get_config.cache_clear()

    yield

    # Clear cache after test
    get_config.cache_clear()


# Sample test data
@pytest.fixture
def sample_embedding():
    """Return a sample embedding vector."""
    return [0.1] * 3072  # Standard embedding dimension


@pytest.fixture
def sample_documents():
    """Return sample search documents."""
    return [
        {
            "id": "doc1_page1",
            "content": "This is a sample document about machine learning.",
            "page": 1,
            "bbox_annotation": '{"document_type": "technical", "summary": "ML basics"}',
            "image_blob_urls": ["image1.png", "image2.png"],
        },
        {
            "id": "doc2_page1",
            "content": "Another document about data science.",
            "page": 1,
            "bbox_annotation": None,
            "image_blob_urls": [],
        },
    ]


@pytest.fixture
def sample_image_bytes():
    """Return sample PNG image bytes (1x1 red pixel)."""
    # Minimal valid PNG
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_jpeg_bytes():
    """Return sample JPEG image bytes."""
    # Minimal JPEG header
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 100


# Mock Azure OpenAI client
@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()

    # Mock embeddings response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1] * 3072)]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # Mock chat completions response
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(message=MagicMock(content="This is a test response."))
    ]
    mock_client.chat.completions.create.return_value = mock_completion_response

    return mock_client


@pytest.fixture
def mock_blob_service():
    """Create a mock blob service client."""
    mock_service = MagicMock()

    # Mock blob client
    mock_blob_client = MagicMock()
    mock_blob_data = MagicMock()
    mock_blob_data.readall.return_value = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    mock_blob_client.download_blob.return_value = mock_blob_data
    mock_service.get_blob_client.return_value = mock_blob_client

    return mock_service


@pytest.fixture
def mock_search_client():
    """Create a mock search client."""
    mock_client = MagicMock()

    # Mock search results
    mock_results = [
        {
            "id": "doc1_page1",
            "content": "Test content",
            "page": 1,
            "@search.score": 0.95,
        }
    ]
    mock_client.search.return_value = iter(mock_results)

    # Mock upload results
    mock_upload_result = [MagicMock(succeeded=True)]
    mock_client.upload_documents.return_value = mock_upload_result

    return mock_client


# Helper functions for tests
@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def sample_json_file(temp_test_dir):
    """Create a sample JSON file for testing."""
    import json

    file_path = temp_test_dir / "sample.json"
    data = {"test": "data", "items": [1, 2, 3]}
    file_path.write_text(json.dumps(data))
    return file_path
