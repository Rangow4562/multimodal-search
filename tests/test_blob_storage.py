"""
Unit tests for the Blob Storage client.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.clients.blob_storage import (
    BlobStorageClient,
    FetchedImage,
    fetch_images_from_blob_storage,
    generate_image_sas_url,
)


class TestBlobStorageClient:
    """Tests for BlobStorageClient class."""

    def test_client_initialization(self):
        """Test client initializes correctly with config."""
        client = BlobStorageClient()
        assert client.account_name is not None
        assert client.account_key is not None

    def test_is_configured(self):
        """Test is_configured property."""
        client = BlobStorageClient()
        assert client.is_configured is True

    def test_detect_mime_type_png(self, sample_image_bytes):
        """Test PNG MIME type detection."""
        client = BlobStorageClient()
        mime = client.detect_mime_type(sample_image_bytes)
        assert mime == "image/png"

    def test_detect_mime_type_jpeg(self, sample_jpeg_bytes):
        """Test JPEG MIME type detection."""
        client = BlobStorageClient()
        mime = client.detect_mime_type(sample_jpeg_bytes)
        assert mime == "image/jpeg"

    def test_detect_mime_type_from_extension(self):
        """Test MIME type detection from filename."""
        client = BlobStorageClient()

        # Unknown bytes but with extension hint
        mime = client.detect_mime_type(b"unknown data", "image.png")
        assert mime == "image/png"

        mime = client.detect_mime_type(b"unknown data", "photo.jpg")
        assert mime == "image/jpeg"

        mime = client.detect_mime_type(b"unknown data", "animation.gif")
        assert mime == "image/gif"

    def test_detect_mime_type_unknown(self):
        """Test MIME type detection for unknown format."""
        client = BlobStorageClient()
        mime = client.detect_mime_type(b"unknown binary data")
        assert mime is None

    @patch("core.clients.blob_storage.generate_blob_sas")
    def test_generate_sas_url(self, mock_generate_sas):
        """Test SAS URL generation."""
        mock_generate_sas.return_value = "sas-token-123"

        client = BlobStorageClient()
        url = client.generate_sas_url("test-image.png")

        assert url is not None
        assert "test-image.png" in url
        assert "sas-token-123" in url
        mock_generate_sas.assert_called_once()

    @patch("core.clients.blob_storage.generate_blob_sas")
    def test_generate_sas_url_caching(self, mock_generate_sas):
        """Test SAS URL caching."""
        mock_generate_sas.return_value = "sas-token-123"

        client = BlobStorageClient()

        # First call
        url1 = client.generate_sas_url("test-image.png")
        # Second call - should use cache
        url2 = client.generate_sas_url("test-image.png")

        assert url1 == url2
        assert mock_generate_sas.call_count == 1

    @patch("core.clients.blob_storage.generate_blob_sas")
    def test_generate_sas_url_no_cache(self, mock_generate_sas):
        """Test SAS URL without caching."""
        mock_generate_sas.return_value = "sas-token-123"

        client = BlobStorageClient()

        client.generate_sas_url("test-image.png", use_cache=False)
        client.generate_sas_url("test-image.png", use_cache=False)

        assert mock_generate_sas.call_count == 2

    @patch("core.clients.blob_storage.generate_blob_sas")
    def test_refresh_blob_sas(self, mock_generate_sas):
        """Test SAS token refresh for existing URL."""
        mock_generate_sas.return_value = "new-sas-token"

        client = BlobStorageClient()
        old_url = f"https://teststorage.blob.core.windows.net/images/test.png?old-sas"

        new_url = client.refresh_blob_sas(old_url)

        assert "new-sas-token" in new_url

    def test_refresh_blob_sas_non_azure_url(self):
        """Test that non-Azure URLs are returned unchanged."""
        client = BlobStorageClient()
        url = "https://example.com/image.png"

        result = client.refresh_blob_sas(url)

        assert result == url

    @patch("core.clients.blob_storage.BlobServiceClient")
    def test_fetch_image(
        self, mock_blob_service_class, mock_blob_service, sample_image_bytes
    ):
        """Test single image fetching."""
        mock_blob_service_class.from_connection_string.return_value = mock_blob_service

        client = BlobStorageClient()
        client._blob_service = mock_blob_service

        # Configure mock
        mock_blob_client = MagicMock()
        mock_blob_data = MagicMock()
        mock_blob_data.readall.return_value = sample_image_bytes
        mock_blob_client.download_blob.return_value = mock_blob_data
        mock_blob_service.get_blob_client.return_value = mock_blob_client

        image = client.fetch_image("test-image.png")

        assert image is not None
        assert isinstance(image, FetchedImage)
        assert image.mime_type == "image/png"
        assert image.blob_name == "test-image.png"
        assert image.base64_data is not None
        assert image.data_url.startswith("data:image/png;base64,")

    @patch("core.clients.blob_storage.BlobServiceClient")
    def test_fetch_image_empty_blob(self, mock_blob_service_class, mock_blob_service):
        """Test handling of empty blob."""
        mock_blob_service_class.from_connection_string.return_value = mock_blob_service

        client = BlobStorageClient()
        client._blob_service = mock_blob_service

        mock_blob_client = MagicMock()
        mock_blob_data = MagicMock()
        mock_blob_data.readall.return_value = b""
        mock_blob_client.download_blob.return_value = mock_blob_data
        mock_blob_service.get_blob_client.return_value = mock_blob_client

        image = client.fetch_image("empty.png")

        assert image is None

    @patch("core.clients.blob_storage.BlobServiceClient")
    def test_fetch_images_batch(
        self, mock_blob_service_class, mock_blob_service, sample_image_bytes
    ):
        """Test batch image fetching."""
        mock_blob_service_class.from_connection_string.return_value = mock_blob_service

        client = BlobStorageClient()
        client._blob_service = mock_blob_service

        mock_blob_client = MagicMock()
        mock_blob_data = MagicMock()
        mock_blob_data.readall.return_value = sample_image_bytes
        mock_blob_client.download_blob.return_value = mock_blob_data
        mock_blob_service.get_blob_client.return_value = mock_blob_client

        blob_names = ["image1.png", "image2.png", "image3.png"]
        images = client.fetch_images(blob_names)

        assert len(images) == 3

    def test_parse_blob_reference_string(self):
        """Test parsing simple blob name."""
        client = BlobStorageClient()
        result = client._parse_blob_reference("image.png")
        assert result == ["image.png"]

    def test_parse_blob_reference_json_array(self):
        """Test parsing JSON array of blob names."""
        client = BlobStorageClient()
        result = client._parse_blob_reference('["image1.png", "image2.png"]')
        assert result == ["image1.png", "image2.png"]

    def test_parse_blob_reference_list(self):
        """Test parsing list of blob names."""
        client = BlobStorageClient()
        result = client._parse_blob_reference(["image1.png", "image2.png"])
        assert result == ["image1.png", "image2.png"]

    def test_parse_blob_reference_url(self):
        """Test parsing URL to extract blob name."""
        client = BlobStorageClient()
        url = "https://storage.blob.core.windows.net/images/photo.png"
        result = client._parse_blob_reference(url)
        assert "photo.png" in result

    def test_clear_sas_cache(self):
        """Test SAS cache clearing."""
        client = BlobStorageClient()
        client._sas_cache["test"] = ("url", 12345)

        client.clear_sas_cache()

        assert len(client._sas_cache) == 0


class TestBackwardCompatibility:
    """Tests for backward-compatible functions."""

    @patch("core.clients.blob_storage.BlobStorageClient")
    def test_fetch_images_from_blob_storage_function(self, mock_client_class):
        """Test backward-compatible fetch_images_from_blob_storage function."""
        mock_instance = MagicMock()
        mock_instance.fetch_images.return_value = [
            FetchedImage(
                "b64data", "image/png", "test.png", "data:image/png;base64,b64data"
            )
        ]
        mock_client_class.return_value = mock_instance

        result = fetch_images_from_blob_storage(["test.png"])

        assert len(result) == 1
        assert result[0] == ("b64data", "image/png", "test.png")

    @patch("core.clients.blob_storage.BlobStorageClient")
    def test_generate_image_sas_url_function(self, mock_client_class):
        """Test backward-compatible generate_image_sas_url function."""
        mock_instance = MagicMock()
        mock_instance.generate_sas_url.return_value = (
            "https://test.blob.core.windows.net/images/test.png?sas"
        )
        mock_client_class.return_value = mock_instance

        result = generate_image_sas_url("test.png")

        assert result is not None
        assert "test.png" in result


class TestFetchedImage:
    """Tests for FetchedImage dataclass."""

    def test_to_dict(self):
        """Test FetchedImage.to_dict method."""
        image = FetchedImage(
            base64_data="abc123",
            mime_type="image/png",
            blob_name="test.png",
            data_url="data:image/png;base64,abc123",
        )

        result = image.to_dict()

        assert result["b64"] == "abc123"
        assert result["mime"] == "image/png"
        assert result["name"] == "test.png"
        assert result["data_url"] == "data:image/png;base64,abc123"
