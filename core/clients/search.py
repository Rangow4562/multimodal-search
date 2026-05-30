"""
Unified Azure Cognitive Search client for vector and text search.

Consolidates implementations from rag_query.py, parse_document.py, and api.py.
"""

import time
from typing import Optional, List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchableField,
    SimpleField,
)

from ..config import get_config
from ..exceptions import AzureServiceError, RetryExhaustedError
from ..logging_config import get_logger, log_timing

logger = get_logger(__name__)


class AzureSearchClient:
    """
    Unified Azure Cognitive Search client for both indexing and retrieval.

    Features:
        - Vector search with configurable parameters
        - Text search
        - Index management
        - Batch document upload with retry
    """

    DEFAULT_FIELDS = ["id", "content", "page", "bbox_annotation", "image_blob_urls"]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        admin_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize the Azure Search client.

        Args:
            endpoint: Azure Search endpoint URL
            admin_key: Azure Search admin key
            index_name: Name of the search index
            embedding_dim: Dimension of embedding vectors
        """
        config = get_config()
        search_config = config.search

        self.endpoint = endpoint or search_config.endpoint
        self.admin_key = admin_key or search_config.admin_key
        self.index_name = index_name or search_config.index_name
        self.embedding_dim = embedding_dim or search_config.embedding_dim

        # Retry configuration
        self.max_retries = config.retry.max_retries
        self.base_backoff = config.retry.base_backoff

        if not self.endpoint or not self.admin_key:
            raise AzureServiceError(
                "Azure Search credentials not configured",
                service_name="azure_search",
                operation="init",
            )

        self.credential = AzureKeyCredential(self.admin_key)

        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

        self.search_client: Optional[SearchClient] = None
        if self.index_name:
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

        logger.info(f"Azure Search client initialized (index: {self.index_name})")

    def _get_search_client(self) -> SearchClient:
        """Get or create search client."""
        if not self.search_client:
            if not self.index_name:
                raise AzureServiceError(
                    "Index name not specified",
                    service_name="azure_search",
                    operation="get_search_client",
                )
            self.search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )
        return self.search_client

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

                backoff = self.base_backoff * (2**attempt)
                logger.warning(
                    f"{operation} failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)

        raise RetryExhaustedError(
            f"{operation} failed after {self.max_retries} attempts",
            service_name="azure_search",
            max_retries=self.max_retries,
            cause=last_exception,
        )

    @log_timing()
    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            select_fields: Fields to include in results

        Returns:
            List of matching documents
        """
        search_client = self._get_search_client()
        select = select_fields or self.DEFAULT_FIELDS

        vector_query = {
            "kind": "vector",
            "vector": query_vector,
            "k": top_k,
            "includeDistance": True,
            "fields": "embedding",
        }

        def _search():
            results = search_client.search(
                search_text="*",
                vector_queries=[vector_query],
                select=select,
            )
            return list(results)

        try:
            return self._retry_with_backoff("Vector search", _search)
        except RetryExhaustedError:
            raise
        except Exception as e:
            raise AzureServiceError(
                f"Vector search failed: {e}",
                service_name="azure_search",
                operation="vector_search",
                cause=e,
            )

    @log_timing()
    def text_search(
        self,
        query: str,
        top_k: int = 5,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform text search.

        Args:
            query: Search query text
            top_k: Number of results to return
            select_fields: Fields to include in results

        Returns:
            List of matching documents
        """
        search_client = self._get_search_client()
        select = select_fields or self.DEFAULT_FIELDS

        def _search():
            results = search_client.search(
                search_text=query,
                select=select,
                top=top_k,
            )
            return list(results)

        try:
            return self._retry_with_backoff("Text search", _search)
        except RetryExhaustedError:
            raise
        except Exception as e:
            raise AzureServiceError(
                f"Text search failed: {e}",
                service_name="azure_search",
                operation="text_search",
                cause=e,
            )

    def delete_index(self) -> bool:
        """Delete the search index if it exists."""
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete index {self.index_name}: {e}")
            return False

    def create_index(self, recreate: bool = False) -> bool:
        """
        Create the search index with vector search capability.

        Args:
            recreate: If True, delete existing index first

        Returns:
            True if index was created/exists
        """
        if recreate:
            self.delete_index()

        # Check if index exists
        try:
            self.index_client.get_index(self.index_name)
            logger.info(f"Index {self.index_name} already exists")
            return True
        except Exception:
            pass  # Index doesn't exist

        # Define index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="bbox_annotation", type=SearchFieldDataType.String),
            SimpleField(
                name="image_blob_urls",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embedding_dim,
                vector_search_profile_name="embedding-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-config"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="embedding-profile",
                    algorithm_configuration_name="hnsw-config",
                ),
            ],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        try:
            self.index_client.create_index(index)
            logger.info(f"Created index: {self.index_name}")
            return True
        except Exception as e:
            raise AzureServiceError(
                f"Failed to create index: {e}",
                service_name="azure_search",
                operation="create_index",
                cause=e,
            )

    @log_timing()
    def upload_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upload documents to the search index.

        Args:
            documents: List of documents to upload
            batch_size: Number of documents per batch

        Returns:
            Number of successfully uploaded documents
        """
        search_client = self._get_search_client()
        uploaded = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            def _upload():
                result = search_client.upload_documents(documents=batch)
                return sum(1 for r in result if r.succeeded)

            try:
                batch_uploaded = self._retry_with_backoff(
                    f"Document upload (batch {i // batch_size + 1})", _upload
                )
                uploaded += batch_uploaded
            except Exception as e:
                logger.error(f"Failed to upload batch: {e}")

        logger.info(f"Uploaded {uploaded}/{len(documents)} documents")
        return uploaded

    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        try:
            search_client = self._get_search_client()
            results = search_client.search(search_text="*", include_total_count=True)
            return results.get_count() or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0


# Backward-compatible class alias
AzureCognitiveSearchRetriever = AzureSearchClient
AzureCognitiveSearchClient = AzureSearchClient
