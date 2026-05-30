"""
Unified Azure Blob Storage client with SAS token generation and image handling.

Consolidates blob storage operations from rag_query.py, api.py, and agent_ather_multimodel.py.
"""

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlparse, unquote

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

from ..config import get_config, BlobStorageConfig
from ..exceptions import AzureServiceError, ImageProcessingError
from ..logging_config import get_logger, log_timing

logger = get_logger(__name__)


@dataclass
class FetchedImage:
    """Represents a fetched image with metadata."""

    base64_data: str
    mime_type: str
    blob_name: str
    data_url: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "b64": self.base64_data,
            "mime": self.mime_type,
            "name": self.blob_name,
            "data_url": self.data_url,
        }


class BlobStorageClient:
    """
    Unified Azure Blob Storage client with SAS token generation and caching.

    Features:
        - SAS token generation with caching
        - Image fetching and MIME type detection
        - Batch image fetching
        - Connection reuse
    """

    # MIME type signatures for image detection
    MIME_SIGNATURES = {
        b"\x89PNG\r\n\x1a\n": "image/png",
        b"\xff\xd8": "image/jpeg",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
        b"RIFF": "image/webp",  # WebP starts with RIFF
    }

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        default_container: Optional[str] = None,
    ):
        """
        Initialize the Blob Storage client.

        Args:
            connection_string: Azure Storage connection string
            account_name: Storage account name (if not using connection string)
            account_key: Storage account key (if not using connection string)
            default_container: Default container name for operations
        """
        config = get_config()
        blob_config = config.blob_storage

        self.connection_string = connection_string or blob_config.connection_string
        self.account_name = account_name or blob_config.account_name
        self.account_key = account_key or blob_config.account_key
        self.default_container = default_container or blob_config.default_container
        self.sas_expiry_hours = blob_config.sas_expiry_hours

        # SAS token cache: {blob_url: (sas_url, expiry_time)}
        self._sas_cache: Dict[str, Tuple[str, float]] = {}
        self._sas_cache_buffer = 300  # Refresh tokens 5 min before expiry

        # Initialize blob service client
        self._blob_service: Optional[BlobServiceClient] = None

        if self.connection_string:
            self._blob_service = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            logger.info("Blob storage client initialized with connection string")
        elif self.account_name and self.account_key:
            logger.info(
                f"Blob storage client initialized for account: {self.account_name}"
            )
        else:
            logger.warning("Blob storage not configured - image operations will fail")

    @property
    def is_configured(self) -> bool:
        """Check if blob storage is properly configured."""
        return bool(self._blob_service or (self.account_name and self.account_key))

    def detect_mime_type(
        self, image_bytes: bytes, name_hint: Optional[str] = None
    ) -> Optional[str]:
        """
        Detect MIME type from image bytes or filename.

        Args:
            image_bytes: Raw image bytes
            name_hint: Optional filename for fallback detection

        Returns:
            MIME type string or None if undetected
        """
        # Check magic bytes
        for signature, mime_type in self.MIME_SIGNATURES.items():
            if image_bytes[: len(signature)] == signature:
                return mime_type

        # Fallback to extension-based detection
        if name_hint:
            lower = name_hint.lower()
            if lower.endswith(".png"):
                return "image/png"
            if lower.endswith((".jpg", ".jpeg")):
                return "image/jpeg"
            if lower.endswith(".gif"):
                return "image/gif"
            if lower.endswith(".webp"):
                return "image/webp"

        return None

    @log_timing()
    def generate_sas_url(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
        expiry_hours: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Generate a SAS URL for accessing a blob.

        Args:
            blob_name: Name of the blob
            container_name: Container name (defaults to default_container)
            expiry_hours: Hours until SAS token expires
            use_cache: Whether to use cached SAS tokens

        Returns:
            SAS URL string or None if generation fails
        """
        if not self.account_name or not self.account_key:
            logger.warning(
                "Cannot generate SAS URL: account credentials not configured"
            )
            return None

        container = container_name or self.default_container
        expiry = expiry_hours or self.sas_expiry_hours

        # Check cache
        cache_key = f"{container}/{blob_name}"
        if use_cache and cache_key in self._sas_cache:
            cached_url, expiry_time = self._sas_cache[cache_key]
            if time.time() < expiry_time - self._sas_cache_buffer:
                logger.debug(f"SAS cache hit for {blob_name}")
                return cached_url

        try:
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=container,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry),
            )

            sas_url = (
                f"https://{self.account_name}.blob.core.windows.net/"
                f"{container}/{blob_name}?{sas_token}"
            )

            # Cache the result
            if use_cache:
                expiry_time = time.time() + (expiry * 3600)
                self._sas_cache[cache_key] = (sas_url, expiry_time)

            return sas_url

        except Exception as e:
            logger.error(f"Failed to generate SAS URL for {blob_name}: {e}")
            return None

    def refresh_blob_sas(self, blob_url: str) -> str:
        """
        Refresh SAS token for an existing blob URL.

        Args:
            blob_url: Original blob URL (with or without SAS token)

        Returns:
            URL with fresh SAS token, or original URL on failure
        """
        if not self.account_name or not self.account_key:
            return blob_url

        if f"{self.account_name}.blob.core.windows.net" not in blob_url:
            return blob_url

        try:
            parsed = urlparse(blob_url)
            path_parts = parsed.path[1:].split("/", 1)

            if len(path_parts) < 2:
                return blob_url

            container_name = path_parts[0]
            blob_name = unquote(path_parts[1])

            refreshed = self.generate_sas_url(blob_name, container_name)
            return refreshed or blob_url

        except Exception as e:
            logger.error(f"Failed to refresh SAS for {blob_url}: {e}")
            return blob_url

    @log_timing()
    def fetch_image(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> Optional[FetchedImage]:
        """
        Fetch a single image from blob storage.

        Args:
            blob_name: Name of the blob to fetch
            container_name: Container name (defaults to default_container)

        Returns:
            FetchedImage object or None if fetch fails
        """
        if not self._blob_service:
            logger.warning("Blob service not initialized")
            return None

        container = container_name or self.default_container

        try:
            blob_client = self._blob_service.get_blob_client(
                container=container, blob=blob_name
            )
            blob_data = blob_client.download_blob()
            image_bytes = blob_data.readall()

            if not image_bytes:
                logger.warning(f"Empty blob: {blob_name}")
                return None

            mime_type = self.detect_mime_type(image_bytes, blob_name)
            if not mime_type:
                logger.warning(f"Could not detect MIME type for: {blob_name}")
                return None

            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64_data}"

            return FetchedImage(
                base64_data=b64_data,
                mime_type=mime_type,
                blob_name=blob_name,
                data_url=data_url,
            )

        except Exception as e:
            logger.error(f"Failed to fetch image {blob_name}: {e}")
            return None

    @log_timing()
    def fetch_images(
        self,
        blob_urls_or_names: List[Any],
        container_name: Optional[str] = None,
    ) -> List[FetchedImage]:
        """
        Fetch multiple images from blob storage.

        Args:
            blob_urls_or_names: List of blob names or URLs (can include JSON strings)
            container_name: Container name (defaults to default_container)

        Returns:
            List of FetchedImage objects for successfully fetched images
        """
        fetched_images = []

        for item in blob_urls_or_names:
            blob_names = self._parse_blob_reference(item)

            for blob_name in blob_names:
                if not blob_name or not blob_name.strip():
                    continue

                image = self.fetch_image(blob_name, container_name)
                if image:
                    fetched_images.append(image)

        logger.info(f"Fetched {len(fetched_images)} images")
        return fetched_images

    def _parse_blob_reference(self, reference: Any) -> List[str]:
        """Parse various blob reference formats to extract blob names."""
        if not reference:
            return []

        # Handle JSON string
        if isinstance(reference, str) and reference.strip().startswith("["):
            try:
                return json.loads(reference)
            except json.JSONDecodeError:
                return [reference]

        # Handle string (could be URL or blob name)
        if isinstance(reference, str):
            # Extract blob name from URL
            if "/" in reference and "blob.core.windows.net" in reference:
                try:
                    parsed = urlparse(reference)
                    return [parsed.path.split("/")[-1]]
                except Exception:
                    pass
            return [reference]

        # Handle list
        if isinstance(reference, list):
            result = []
            for item in reference:
                result.extend(self._parse_blob_reference(item))
            return result

        return []

    def clear_sas_cache(self):
        """Clear the SAS token cache."""
        self._sas_cache.clear()
        logger.info("SAS token cache cleared")


# Backward-compatible function for existing code
def fetch_images_from_blob_storage(
    image_blob_urls: List[str],
    container_name: str = "images",
) -> List[Tuple[str, str, str]]:
    """
    Backward-compatible function for fetching images.

    Returns list of tuples: (base64_data, mime_type, blob_name)
    """
    client = BlobStorageClient()
    fetched = client.fetch_images(image_blob_urls, container_name)
    return [(img.base64_data, img.mime_type, img.blob_name) for img in fetched]


def generate_image_sas_url(
    blob_name: str,
    container_name: str = "images",
    expiry_hours: int = 24,
) -> Optional[str]:
    """Backward-compatible function for generating SAS URLs."""
    client = BlobStorageClient()
    return client.generate_sas_url(blob_name, container_name, expiry_hours)
