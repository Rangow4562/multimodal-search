import os
import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.settings import Settings, CallbackManager
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
)

# Try to import RESPONSE_TYPE, fallback to Any if not available
try:
    from llama_index.core.schema import RESPONSE_TYPE
except ImportError:
    from typing import Any

    RESPONSE_TYPE = Any

try:
    from llama_index.core.query_engine import QueryBundle
except ImportError:
    # Fallback if QueryBundle is not available
    class QueryBundle:
        def __init__(self, query_str: str):
            self.query_str = query_str


from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, unquote
import base64
import mimetypes


# Create a simple callback handler for the API (non-Chainlit)
class APICallbackHandler:
    """Simple callback handler for API use without Chainlit."""

    # Required attributes for LlamaIndex compatibility
    event_starts_to_ignore = set()
    event_ends_to_ignore = set()

    def on_event(self, event_type, payload, **kwargs):
        """Handle callback events."""
        logger.info(f"Callback Event: {event_type}")

    def start_trace(self, *args, **kwargs):
        """Start tracing - flexible to handle various calling patterns."""
        if args:
            query_str = args[0]
            trace_id = args[1] if len(args) > 1 else kwargs.get("trace_id")
        elif "query_str" in kwargs:
            query_str = kwargs["query_str"]
            trace_id = kwargs.get("trace_id")
        else:
            query_str = str(args) if args else "Unknown query"
            trace_id = kwargs.get("trace_id")
        logger.info(
            f"Starting trace for query: {str(query_str)[:50]}... (ID: {trace_id})"
        )

    def end_trace(self, *args, **kwargs):
        """End tracing - flexible to handle various calling patterns."""
        if args:
            query_str = args[1] if len(args) > 1 else str(args[0])
        else:
            query_str = kwargs.get("query_str", "Unknown query")
        logger.info(f"Completed trace for query: {str(query_str)[:50]}...")

    # Additional callback methods that LlamaIndex might expect
    def on_trace_start(self, id_: str, query_bundle: "QueryBundle") -> None:
        """Handle trace start."""
        logger.info(f"Trace started: {id_}")

    def on_trace_end(self, id_: str, response: RESPONSE_TYPE, *args, **kwargs) -> None:
        """Handle trace end."""
        logger.info(f"Trace ended: {id_}")

    def on_event_start(self, event_type: str, payload: dict = None, **kwargs):
        """Handle event start."""
        logger.info(f"Event started: {event_type}")

    def on_event_end(self, event_type: str, payload: dict = None, **kwargs):
        """Handle event end."""
        logger.info(f"Event ended: {event_type}")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME"
)
AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_SERVICE_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = "aiassist-bemsys-index-v2"

# Blob Storage Credentials
BLOB_STORAGE_ACCOUNT_NAME = os.getenv("BLOB_STORAGE_ACCOUNT_NAME")
BLOB_STORAGE_ACCOUNT_KEY = os.getenv("BLOB_STORAGE_ACCOUNT_KEY")
BLOB_CONTAINER_SAS_TOKEN = os.getenv("BLOB_CONTAINER_SAS_TOKEN")

# Log blob storage configuration status
logger.info(
    f"Blob Storage Config - Account: {BLOB_STORAGE_ACCOUNT_NAME}, SAS Token Set: {bool(BLOB_CONTAINER_SAS_TOKEN)}"
)


class AgentRole(Enum):
    """Enumeration of different agent roles in the workflow."""

    QUERY_PROCESSOR = "query_processor"
    IMAGE_PROCESSOR = "image_processor"
    RESPONSE_ORCHESTRATOR = "response_orchestrator"
    CONTENT_ANALYZER = "content_analyzer"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""

    sender: str
    recipient: str
    message_type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentContext:
    """Shared context for agent workflow."""

    user_query: str
    retrieval_results: List[NodeWithScore] = field(default_factory=list)
    image_data: List[Any] = field(default_factory=list)
    processed_content: Dict[str, Any] = field(default_factory=dict)
    response_segments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents in the workflow."""

    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.message_handlers: Dict[str, Callable] = {}

    @abstractmethod
    async def process(
        self, context: AgentContext, message: Optional[AgentMessage] = None
    ) -> AgentContext:
        """Process the given context and optionally handle incoming messages."""
        pass

    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler

    async def send_message(
        self,
        recipient: str,
        message_type: str,
        content: Any,
        metadata: Dict[str, Any] = None,
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
        logger.info(f"Message sent from {self.name} to {recipient}: {message_type}")
        return message


class QueryProcessingAgent(BaseAgent):
    """Agent responsible for processing user queries and retrieving relevant documents."""

    def __init__(self, retriever: BaseRetriever):
        super().__init__("QueryProcessor", AgentRole.QUERY_PROCESSOR)
        self.retriever = retriever

    async def process(
        self, context: AgentContext, message: Optional[AgentMessage] = None
    ) -> AgentContext:
        """Process user query and retrieve relevant documents."""
        logger.info(f"Processing query: {context.user_query}")

        # Retrieve relevant nodes
        nodes = self.retriever.retrieve(context.user_query)
        context.retrieval_results = nodes

        logger.info(f"Retrieved {len(nodes)} nodes for query processing")

        # Store query metadata
        context.metadata["query_processed"] = True
        context.metadata["retrieval_count"] = len(nodes)
        context.metadata["retrieval_timestamp"] = datetime.now()

        return context


class ImageProcessingAgent(BaseAgent):
    """Agent responsible for processing and managing multimodal image content."""

    def __init__(self, multi_modal_llm: AzureOpenAIMultiModal, search_client=None):
        super().__init__("ImageProcessor", AgentRole.IMAGE_PROCESSOR)
        self.multi_modal_llm = multi_modal_llm
        self.search_client = (
            search_client  # Azure Search client for fetching image_blob_urls
        )

    def resolve_image_source(self, image_url_or_path):
        """Resolve image source with SAS token refresh."""
        if not image_url_or_path:
            return None

        if "blob.core.windows.net" in image_url_or_path:
            return self.refresh_blob_sas(image_url_or_path)

        if os.path.exists(image_url_or_path):
            return image_url_or_path

        return image_url_or_path

    def refresh_blob_sas(self, blob_url):
        """Add SAS token to blob URLs for access."""
        if not blob_url:
            return blob_url

        # Fix extension mismatch: Azure Search index has .jpeg but actual blobs use .jpg
        if blob_url.endswith(".jpeg"):
            blob_url = blob_url[:-5] + ".jpg"
            logger.info(f"Fixed extension from .jpeg to .jpg: {blob_url[:80]}...")

        # If URL already has a SAS token (contains ?), return as-is
        if "?" in blob_url:
            logger.info(f"URL already has SAS token: {blob_url[:80]}...")
            return blob_url

        # Use container SAS token if available (preferred - simpler and more reliable)
        if BLOB_CONTAINER_SAS_TOKEN:
            new_url = f"{blob_url}?{BLOB_CONTAINER_SAS_TOKEN}"
            logger.info(f"Applied container SAS token to URL: {new_url[:100]}...")
            return new_url
        else:
            logger.warning(
                f"BLOB_CONTAINER_SAS_TOKEN not set, falling back to account key method"
            )

        # Fallback: Generate individual blob SAS token using account key
        if not BLOB_STORAGE_ACCOUNT_NAME or not BLOB_STORAGE_ACCOUNT_KEY:
            return blob_url

        try:
            parsed_url = urlparse(blob_url)

            if (
                f"{BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
                not in parsed_url.netloc
            ):
                return blob_url

            path_parts = parsed_url.path[1:].split("/", 1)
            if len(path_parts) < 2:
                return blob_url

            container_name = path_parts[0]
            blob_name = unquote(path_parts[1])

            sas_token = generate_blob_sas(
                account_name=BLOB_STORAGE_ACCOUNT_NAME,
                container_name=container_name,
                blob_name=blob_name,
                account_key=BLOB_STORAGE_ACCOUNT_KEY,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            )

            new_url = f"https://{BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            return new_url

        except Exception as e:
            logger.error(f"Failed to refresh SAS token for {blob_url}: {e}")
            return blob_url

    async def process(
        self, context: AgentContext, message: Optional[AgentMessage] = None
    ) -> AgentContext:
        """Process images from retrieval results."""

        # Check if text-only mode is enabled
        if context.metadata.get("text_only", False):
            logger.info("Text-only mode enabled - skipping image processing")
            context.image_data = []
            context.metadata["images_processed"] = False
            context.metadata["image_count"] = 0
            return context

        logger.info("Processing images from retrieval results")

        image_nodes = []
        max_images = context.metadata.get("max_images", 10)

        for node in context.retrieval_results:
            # Get the document ID from the node
            doc_id = node.node.id_ if hasattr(node.node, "id_") else None
            logger.info(f"Processing node with id: {doc_id}")

            # Fetch image_blob_urls directly from Azure Search
            image_blob_urls = []
            if self.search_client and doc_id:
                try:
                    # Use Azure Search SDK to get the full document
                    document = self.search_client.get_document(key=doc_id)
                    image_blob_urls = document.get("image_blob_urls", [])
                    logger.info(
                        f"Fetched image_blob_urls from Azure Search: {image_blob_urls}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch document from Azure Search: {str(e)}"
                    )

            # Process image_blob_urls
            if image_blob_urls:
                # Ensure it's a list
                if isinstance(image_blob_urls, str):
                    try:
                        image_blob_urls = json.loads(image_blob_urls)
                    except json.JSONDecodeError:
                        image_blob_urls = [
                            url.strip()
                            for url in image_blob_urls.split(",")
                            if url.strip()
                        ]

                for img_url in image_blob_urls:
                    if img_url and len(image_nodes) < max_images:
                        try:
                            resolved_url = self.resolve_image_source(img_url)
                            if resolved_url:
                                is_local = os.path.exists(resolved_url)

                                # Validate remote URL exists before adding
                                if not is_local:
                                    import requests

                                    try:
                                        head_response = requests.head(
                                            resolved_url, timeout=5
                                        )
                                        if head_response.status_code != 200:
                                            logger.warning(
                                                f"Image URL not accessible (HTTP {head_response.status_code}): {resolved_url[:80]}..."
                                            )
                                            continue
                                    except requests.RequestException as req_err:
                                        logger.warning(
                                            f"Failed to validate image URL: {resolved_url[:80]}... - {req_err}"
                                        )
                                        continue

                                image_node = (
                                    ImageNode(image_path=resolved_url)
                                    if is_local
                                    else ImageNode(image_url=resolved_url)
                                )
                                image_nodes.append(NodeWithScore(node=image_node))
                                logger.info(
                                    f"Added validated image: {resolved_url[:80]}..."
                                )
                        except Exception as e:
                            logger.error(f"Failed to process image_blob_url: {str(e)}")

        context.image_data = image_nodes
        context.metadata["images_processed"] = True
        context.metadata["image_count"] = len(image_nodes)

        logger.info(f"Processed {len(image_nodes)} images from image_blob_urls")

        return context


class ContentAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing and structuring content."""

    def __init__(self, multi_modal_llm: AzureOpenAIMultiModal):
        super().__init__("ContentAnalyzer", AgentRole.CONTENT_ANALYZER)
        self.multi_modal_llm = multi_modal_llm

        # System context for all prompts
        self.system_context = """You are PAECETrak ROW AI Assist, an advanced AI assistant specialized in Department of Transportation (DOT) systems, Right-of-Way (ROW) management, and permit processing workflows. You are built to help DOT staff and users navigate complex regulatory processes, understand technical documentation, and accomplish their tasks efficiently."""

        # ═══════════════════════════════════════════════════════════════════════════
        # MULTIMODAL PROMPT TEMPLATE (with images)
        # ═══════════════════════════════════════════════════════════════════════════
        self.qa_prompt_template = """## ROLE & CONTEXT
You are PAECETrak ROW AI Assist, an advanced internal AI assistant for Department of Transportation (DOT) systems. You specialize in:
- Right-of-Way (ROW) management and permit processing
- E-Permitting system workflows and user interfaces
- Technical documentation and procedural guidance
- Regulatory compliance and process optimization

## INPUT DATA

### Visual Resources ({image_count} images available)
{images_section}

### Document Context
The following text has been extracted from official DOT documentation, user manuals, and system guides:

```
{context_str}
```

## INSTRUCTIONS

### Response Requirements
1. **Accuracy First**: Only provide information that can be verified from the provided context and images. If information is not available, clearly state this.

2. **Visual Analysis**: 
   - Carefully analyze all provided images for relevant UI elements, workflows, buttons, forms, and diagrams
   - When referencing screenshots, describe what the user should see and look for
   - Identify key visual elements that help answer the query

3. **Image References** (CRITICAL):
   - When referencing images, ALWAYS use the complete URL provided above
   - Use markdown image syntax: `![Descriptive Caption](FULL_URL_HERE)`
   - For inline references: `See [Image N - Description](FULL_URL_HERE)`
   - NEVER use just filenames - always use the complete accessible URL
   - Example: `![Send Email Dialog](https://sa1docstoragegenai.blob.core.windows.net/...?sas_token)`

4. **Step-by-Step Guidance**:
   - Break down complex procedures into numbered steps
   - Include specific UI element names (buttons, menus, fields)
   - Reference relevant screenshots at each step when available
   - Highlight important warnings or prerequisites

5. **Response Format**:
   - Use clear headers and bullet points for readability
   - Bold important terms, button names, and UI elements
   - Include relevant tips or best practices when applicable
   - End with a summary or next steps if appropriate

### Quality Standards
- Be concise but comprehensive
- Use professional, clear language appropriate for DOT staff
- If multiple approaches exist, present the recommended method first
- Acknowledge any limitations in the available information

## USER QUERY
{query_str}

## RESPONSE
"""

        # ═══════════════════════════════════════════════════════════════════════════
        # TEXT-ONLY PROMPT TEMPLATE (without images)
        # ═══════════════════════════════════════════════════════════════════════════
        self.text_only_prompt = PromptTemplate(
            """\
## ROLE & CONTEXT
You are PAECETrak ROW AI Assist, an advanced internal AI assistant for Department of Transportation (DOT) systems. You specialize in Right-of-Way (ROW) management, E-Permitting workflows, and technical documentation.

## DOCUMENT CONTEXT
The following text has been extracted from official DOT documentation, user manuals, and system guides:

```
{context_str}
```

## INSTRUCTIONS

### Response Requirements
1. **Source-Based Answers**: Only provide information that can be verified from the provided context. If information is not available, clearly state this.

2. **Step-by-Step Guidance**:
   - Break down complex procedures into numbered steps
   - Include specific UI element names (buttons, menus, fields) when referenced in documentation
   - Highlight important warnings or prerequisites

3. **Response Format**:
   - Use clear headers and bullet points for readability
   - Bold important terms, button names, and UI elements
   - Include relevant tips or best practices when applicable
   - Be concise but comprehensive

4. **Professional Standards**:
   - Use clear language appropriate for DOT staff
   - If multiple approaches exist, present the recommended method first
   - Acknowledge any limitations in the available information

## USER QUERY
{query_str}

## RESPONSE
"""
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # EXTERNAL SEARCH PROMPT (context + general knowledge)
        # ═══════════════════════════════════════════════════════════════════════════
        self.external_search_prompt = PromptTemplate(
            """\
## ROLE & CONTEXT
You are PAECETrak ROW AI Assist, an advanced internal AI assistant for Department of Transportation (DOT) systems. For this query, you have permission to supplement knowledge base information with your general knowledge when helpful.

## KNOWLEDGE BASE CONTEXT
The following text has been extracted from official DOT documentation:

```
{context_str}
```

## INSTRUCTIONS

### Response Strategy
1. **Primary Source**: First, attempt to answer using the provided knowledge base context
2. **Supplementary Information**: If the context is insufficient, you may use your general knowledge to provide helpful information
3. **Clear Attribution**: Always clearly distinguish between:
   - 📚 **From Knowledge Base**: Information directly from provided context
   - 🌐 **General Knowledge**: Information from your training data

### Response Requirements
1. **Accuracy & Attribution**:
   - Clearly mark which parts of your answer come from the knowledge base vs. general knowledge
   - Use visual indicators or headers to distinguish sources
   - Be honest about confidence levels

2. **Comprehensive Coverage**:
   - Provide complete, helpful answers even if context is limited
   - Offer relevant general knowledge that may help the user
   - Suggest where to find authoritative information if needed

3. **Response Format**:
   - Use clear headers and bullet points
   - Bold important terms and concepts
   - Include practical tips and best practices
   - End with actionable next steps or recommendations

## USER QUERY
{query_str}

## RESPONSE
"""
        )

        # Legacy prompt for backward compatibility
        self.qa_prompt = PromptTemplate(
            f"""\
{self.system_context}

Below we give parsed text from slides in parsed markdown format, as well as the image.

---------------------
{{context_str}}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.

Query: {{query_str}}
Answer: """
        )

    async def process(
        self, context: AgentContext, message: Optional[AgentMessage] = None
    ) -> AgentContext:
        """Analyze content and generate response."""

        # Check processing modes
        is_text_only = context.metadata.get("text_only", False)
        is_external_search = context.metadata.get("external_search", False)
        max_images = context.metadata.get("max_images", 10)

        # Log the processing mode
        if is_external_search:
            mode_desc = "external search mode"
            logger.info(
                f"Analyzing content in {mode_desc} with text-only={is_text_only}"
            )
        elif is_text_only:
            mode_desc = "text-only mode"
            logger.info(f"Analyzing content in {mode_desc}")
        else:
            mode_desc = "multimodal mode"
            logger.info(f"Analyzing content in {mode_desc}")

        # Create context string from text nodes
        context_str = "\n\n".join(
            [
                node.get_content(metadata_mode=MetadataMode.LLM)
                for node in context.retrieval_results
            ]
        )

        # Build images section with URLs for multimodal mode
        images_section = ""
        if not is_text_only and context.image_data:
            image_lines = ["Available Images with URLs:"]
            for idx, img_node_with_score in enumerate(context.image_data, 1):
                img_node = img_node_with_score.node
                img_url = getattr(img_node, "image_url", None) or getattr(
                    img_node, "image_path", None
                )
                if img_url:
                    image_lines.append(f"{idx}. Image {idx}: {img_url}")
            images_section = "\n".join(image_lines)
            logger.info(
                f"Built images section with {len(context.image_data)} image URLs"
            )

        # Choose appropriate prompt based on mode
        if is_external_search:
            # For external search, we can use either text-only or multimodal external prompt
            if is_text_only:
                # For now, use text-only prompt but indicate external search capability
                fmt_prompt = self.text_only_prompt.format(
                    context_str=context_str, query_str=context.user_query
                )
                # Add a note about external search capability
                fmt_prompt += "\n\nNote: You may also use general knowledge beyond the provided context if it would be helpful to answer the user's question."
            else:
                fmt_prompt = self.external_search_prompt.format(
                    context_str=context_str, query_str=context.user_query
                )
        elif is_text_only:
            fmt_prompt = self.text_only_prompt.format(
                context_str=context_str, query_str=context.user_query
            )
        else:
            # Use the new template with images section
            image_count = len(context.image_data) if context.image_data else 0
            fmt_prompt = self.qa_prompt_template.format(
                image_count=image_count,
                images_section=(
                    images_section if images_section else "(No images available)"
                ),
                context_str=context_str,
                query_str=context.user_query,
            )

        # Get response from multimodal LLM
        try:
            # Convert image nodes for processing only in multimodal mode
            image_documents = []
            if not is_text_only:
                for img_node_with_score in context.image_data:
                    img_node = img_node_with_score.node

                    # Debug: log what attributes the node has
                    has_path = hasattr(img_node, "image_path") and img_node.image_path
                    has_url = hasattr(img_node, "image_url") and img_node.image_url
                    logger.info(
                        f"Processing image node - has_path: {has_path}, has_url: {has_url}"
                    )

                    if has_url:
                        logger.info(
                            f"Image URL being sent to LLM: {img_node.image_url[:120]}..."
                        )

                    # Handle local file conversion to base64
                    if hasattr(img_node, "image_path") and img_node.image_path:
                        if os.path.exists(img_node.image_path):
                            with open(img_node.image_path, "rb") as image_file:
                                encoded_string = base64.b64encode(
                                    image_file.read()
                                ).decode("utf-8")
                                mime_type, _ = mimetypes.guess_type(img_node.image_path)
                                if not mime_type:
                                    mime_type = "image/jpeg"
                                data_url = f"data:{mime_type};base64,{encoded_string}"
                                new_node = ImageNode(image_url=data_url)
                                new_node.metadata = img_node.metadata
                                image_documents.append(new_node)
                        else:
                            # If it's a URL, create new node with URL
                            new_node = ImageNode(image_url=img_node.image_path)
                            new_node.metadata = img_node.metadata
                            image_documents.append(new_node)
                    else:
                        image_documents.append(img_node)

            llm_response = self.multi_modal_llm.complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
            )

            context.processed_content["response_text"] = str(llm_response)
            context.processed_content["source_nodes"] = context.retrieval_results
            context.processed_content["image_nodes"] = context.image_data
            context.metadata["content_analyzed"] = True
            context.metadata["analysis_timestamp"] = datetime.now()
            context.metadata["external_search_used"] = is_external_search

        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            context.processed_content["response_text"] = (
                "I apologize, but I encountered an error while processing your query. Please ensure the Azure services are properly configured and try again."
            )

        return context


class ResponseOrchestrationAgent(BaseAgent):
    """Agent responsible for orchestrating the final response with proper formatting."""

    def __init__(self):
        super().__init__("ResponseOrchestrator", AgentRole.RESPONSE_ORCHESTRATOR)

    async def process(
        self, context: AgentContext, message: Optional[AgentMessage] = None
    ) -> AgentContext:
        """Orchestrate final response with smart segmentation and image matching."""
        logger.info("Orchestrating final response")

        response_text = context.processed_content.get("response_text", "")

        if not response_text:
            context.response_segments = [
                {"text": "No response generated.", "images": []}
            ]
            return context

        # Prepare image data for display
        display_images = []
        unique_image_urls = set()

        # Process image nodes for display
        if context.processed_content.get("image_nodes"):
            for img_node in context.processed_content["image_nodes"]:
                image_source = None
                is_local = False

                if hasattr(img_node.node, "image_path") and img_node.node.image_path:
                    image_source = img_node.node.image_path
                    is_local = os.path.exists(image_source)
                elif hasattr(img_node.node, "image_url") and img_node.node.image_url:
                    image_source = img_node.node.image_url
                    is_local = False

                if image_source and image_source not in unique_image_urls:
                    unique_image_urls.add(image_source)

                    try:
                        if not is_local:
                            # Refresh SAS token for remote URLs
                            refreshed_url = self.resolve_image_source(image_source)
                            if refreshed_url:
                                image_source = refreshed_url

                        # Create API-compatible image data structure with actual image content
                        img_element = {
                            "type": "image",
                            "name": f"Image {len(display_images)+1}",
                            "source": image_source,
                            "is_local": is_local,
                            "display": "inline",
                        }

                        # Add accessible image URL or base64 data
                        if is_local:
                            # For local images, convert to base64 for API response
                            try:
                                import base64

                                with open(image_source, "rb") as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode(
                                        "utf-8"
                                    )
                                    # Get MIME type
                                    import mimetypes

                                    mime_type = (
                                        mimetypes.guess_type(image_source)[0]
                                        or "image/png"
                                    )
                                    img_element["data_url"] = (
                                        f"data:{mime_type};base64,{img_data}"
                                    )
                                    img_element["url"] = img_element["data_url"]
                            except Exception as img_error:
                                logger.warning(
                                    f"Failed to convert image to base64: {img_error}"
                                )
                                img_element["url"] = image_source
                        else:
                            # For remote images, use the (refreshed) URL directly
                            img_element["url"] = image_source

                        # Extract image filename and keywords for matching
                        img_filename = (
                            os.path.basename(image_source).split(".")[0].lower()
                        )
                        keywords = [img_filename] + [
                            w for w in img_filename.split("_") if len(w) > 2
                        ]

                        display_images.append(
                            (
                                image_source,
                                img_element,
                                {"filename": img_filename, "keywords": keywords},
                            )
                        )

                    except Exception as e:
                        logger.error(f"Failed to create image element: {str(e)}")

        # Create paired text-image segments
        paired_segments = self.create_paired_segments(response_text, display_images)

        context.response_segments = paired_segments
        context.display_images = display_images  # Store display_images in context
        context.metadata["response_orchestrated"] = True
        context.metadata["segment_count"] = len(paired_segments)
        context.metadata["image_count"] = len(display_images)

        return context

    def create_paired_segments(self, text, image_data):
        """Create text segments with associated images for inline display."""
        if not image_data:
            return [{"text": text, "images": []}]

        # First, segment the text
        text_segments = self.smart_text_segmentation(text, image_data)

        # Then, match images to each segment
        paired_segments = []
        used_images = set()

        for i, segment in enumerate(text_segments):
            segment_images = self.match_images_to_segment(
                segment, image_data, used_images, i
            )

            # Mark images as used
            for img_tuple in segment_images:
                used_images.add(img_tuple[0])  # Add image source to used set

            paired_segments.append(
                {
                    "text": segment,
                    "images": [
                        img_tuple[1] for img_tuple in segment_images
                    ],  # Extract image elements
                }
            )

        # Add any remaining images at the end
        remaining_images = [
            img_tuple[1] for img_tuple in image_data if img_tuple[0] not in used_images
        ]

        if remaining_images:
            if paired_segments:
                # Add to last segment if it exists
                paired_segments[-1]["images"].extend(remaining_images)
            else:
                # Create a new segment if none exist
                paired_segments.append({"text": "", "images": remaining_images})

        return paired_segments

    def match_images_to_segment(
        self, segment, image_data, used_images, segment_position
    ):
        """Match relevant images to a text segment."""
        matched_images = []
        segment_lower = segment.lower()

        # Email step keywords for better matching
        email_step_keywords = {
            "access": [
                "send email",
                "button",
                "compose",
                "email functionality",
                "interface",
            ],
            "enter": ["to field", "subject field", "message", "details", "cc", "bcc"],
            "attach": ["attach", "document", "files", "upload", "checkbox"],
            "save": ["save", "cancel", "send email", "button"],
            "complete": ["record", "update", "log", "sent"],
        }

        # Determine step type based on content
        step_type = None
        if segment_position == 0 and any(
            k in segment_lower for k in email_step_keywords["access"]
        ):
            step_type = "access"
        elif any(k in segment_lower for k in email_step_keywords["enter"]):
            step_type = "enter"
        elif any(k in segment_lower for k in email_step_keywords["attach"]):
            step_type = "attach"
        elif any(k in segment_lower for k in email_step_keywords["save"]):
            step_type = "save"
        elif segment_position >= len(email_step_keywords) - 1 and any(
            k in segment_lower for k in email_step_keywords["complete"]
        ):
            step_type = "complete"

        # Score each unused image for this segment
        image_scores = []
        for img_source, img_element, metadata in image_data:
            if img_source in used_images:
                continue

            score = 0
            filename = metadata.get("filename", "")
            keywords = metadata.get("keywords", [])

            # Filename match (strongest signal)
            if filename and filename in segment_lower:
                score += 10

            # Keywords match
            for keyword in keywords:
                if keyword and keyword in segment_lower:
                    score += 5

            # Step type matching
            if step_type:
                if step_type == "access" and any(
                    k in filename.lower() for k in ["email", "interface", "button"]
                ):
                    score += 15
                elif step_type == "enter" and any(
                    k in filename.lower() for k in ["field", "form", "detail"]
                ):
                    score += 15
                elif step_type == "attach" and any(
                    k in filename.lower() for k in ["attach", "document", "file"]
                ):
                    score += 15
                elif step_type == "save" and any(
                    k in filename.lower() for k in ["save", "button", "cancel"]
                ):
                    score += 15
                elif step_type == "complete" and any(
                    k in filename.lower() for k in ["record", "log", "complete"]
                ):
                    score += 15

            # Common image reference patterns
            if re.search(r"(?i)figure|image|diagram|screenshot|illustration", segment):
                score += 2
            if re.search(
                r"(?i)shown|displayed|pictured|illustrated|depicted|interface|button|click",
                segment,
            ):
                score += 1
            if re.search(
                r"(?i)step|instruction|process|how to|click|select|choose|enter|fill",
                segment,
            ):
                score += 1

            # Position-based scoring (images often align with step order)
            available_images = [img for img in image_data if img[0] not in used_images]
            if len(available_images) > 3:
                img_position = available_images.index(
                    (img_source, img_element, metadata)
                )
                position_diff = abs(segment_position - img_position)
                if position_diff <= 1:
                    score += 3
                elif position_diff <= 2:
                    score += 1

            if score >= 1:
                image_scores.append((score, img_source, img_element, metadata))

        # Sort by score and return best matches (up to 2 per segment)
        image_scores.sort(reverse=True, key=lambda x: x[0])

        for score, img_source, img_element, metadata in image_scores[:2]:
            matched_images.append((img_source, img_element, metadata))

        return matched_images

    def resolve_image_source(self, image_url_or_path):
        """Resolve image source with SAS token refresh."""
        if not image_url_or_path:
            return None

        if "blob.core.windows.net" in image_url_or_path and BLOB_STORAGE_ACCOUNT_NAME:
            try:
                parsed_url = urlparse(image_url_or_path)

                if (
                    f"{BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
                    not in parsed_url.netloc
                ):
                    return image_url_or_path

                path_parts = parsed_url.path[1:].split("/", 1)
                if len(path_parts) < 2:
                    return image_url_or_path

                container_name = path_parts[0]
                blob_name = unquote(path_parts[1])

                sas_token = generate_blob_sas(
                    account_name=BLOB_STORAGE_ACCOUNT_NAME,
                    container_name=container_name,
                    blob_name=blob_name,
                    account_key=BLOB_STORAGE_ACCOUNT_KEY,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1),
                )

                new_url = f"https://{BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
                return new_url

            except Exception as e:
                logger.error(f"Failed to refresh SAS token: {e}")
                return image_url_or_path

        return image_url_or_path

    def smart_text_segmentation(self, text, image_data):
        """Segment text with image reference detection."""
        patterns = [
            r"(figure|fig\.?|image|img|diagram|screenshot|illustration)s?\s*(\d+|\w+)?",
            r"shown\s+(below|above|in the (figure|image|diagram))",
            r"following\s+(figure|image|diagram)",
            r"(see|refer to)\s+(figure|image|diagram)",
            r"(as\s+(shown|illustrated|depicted|seen)(\s+in|\s+by)?)",
            r"image \d+",
            r"(shown|illustrated|depicted) (below|above)",
            r"visual (representation|depiction)",
            r"screen\s*shot",
            r"interface",
            r"button",
            r"section \d+\.\d+",
            r"steps? \d+",
            r"instructions?",
            r"(send|compose) (an? )?email",
            r"email (functionality|interface)",
            r"(to|subject|message|cc|bcc) field",
            r"attach (button|documents?)",
            r"documents? and log",
            r"save button",
            r"cancel button",
        ]

        step_headline_pattern = (
            r"^(Access|Enter|Attach|Save|Cancel|Email|Click|Select|Fill|Choose)"
        )
        step_pattern = r"^\s*(\d+\.\s+|\*\s+|•\s+|-\s+)"

        # Try step-based segmentation first
        text_lines = text.split("\n")
        steps_or_sections = []

        for i, line in enumerate(text_lines):
            if not line.strip():
                continue

            if (
                re.search(step_headline_pattern, line.strip(), re.IGNORECASE)
                and len(line.strip()) < 100
            ):
                steps_or_sections.append(i)

            if re.search(step_pattern, line.strip()):
                steps_or_sections.append(i)

        if len(steps_or_sections) >= 2:
            segments = []
            for i in range(len(steps_or_sections)):
                start_idx = steps_or_sections[i]
                end_idx = (
                    steps_or_sections[i + 1]
                    if i < len(steps_or_sections) - 1
                    else len(text_lines)
                )

                segment_lines = text_lines[start_idx:end_idx]
                segments.append("\n".join(segment_lines))
            return segments

        # Fall back to paragraph-based segmentation
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) < len(image_data) * 0.7:
            new_paragraphs = []
            for p in paragraphs:
                lines = p.split("\n")
                current_para = []

                for line in lines:
                    if re.search(step_pattern, line) or re.search(
                        step_headline_pattern, line, re.IGNORECASE
                    ):
                        if current_para:
                            new_paragraphs.append("\n".join(current_para))
                            current_para = []
                    current_para.append(line)

                if current_para:
                    new_paragraphs.append("\n".join(current_para))

            paragraphs = new_paragraphs

        segments = []
        current_segment = []

        # Create keyword set for faster lookups
        all_keywords = set()
        for _, _, metadata in image_data:
            if "keywords" in metadata:
                for keyword in metadata["keywords"]:
                    if keyword and len(keyword) > 3:
                        all_keywords.add(keyword)

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            image_reference = any(
                re.search(pattern, paragraph, re.IGNORECASE) for pattern in patterns
            )

            keyword_match = any(
                keyword in paragraph.lower() for keyword in all_keywords
            )

            is_step = re.search(step_pattern, paragraph, re.MULTILINE)
            is_heading = (
                re.search(step_headline_pattern, paragraph.strip(), re.IGNORECASE)
                and len(paragraph.strip()) < 100
            )

            if (
                image_reference or is_step or keyword_match or is_heading
            ) and current_segment:
                segments.append("\n\n".join(current_segment))
                current_segment = [paragraph]
            else:
                current_segment.append(paragraph)

        if current_segment:
            segments.append("\n\n".join(current_segment))

        if len(segments) < len(image_data) * 0.6:
            return paragraphs

        return segments


class AgentWorkflowManager:
    """Manager class for coordinating agent workflows."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_steps: List[tuple] = []

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the workflow manager."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def add_workflow_step(self, agent_name: str, condition: Optional[Callable] = None):
        """Add a step to the workflow."""
        self.workflow_steps.append((agent_name, condition))

    async def execute_workflow(self, context: AgentContext) -> AgentContext:
        """Execute the complete workflow."""
        logger.info("Starting agent workflow execution")

        for agent_name, condition in self.workflow_steps:
            if agent_name not in self.agents:
                logger.error(f"Agent {agent_name} not found in registered agents")
                continue

            agent = self.agents[agent_name]

            # Check condition if provided
            if condition and not condition(context):
                logger.info(f"Skipping agent {agent_name} due to condition")
                continue

            try:
                logger.info(f"Executing agent: {agent_name}")
                context = await agent.process(context)
            except Exception as e:
                logger.error(f"Error in agent {agent_name}: {str(e)}")
                # Continue with other agents even if one fails

        logger.info("Agent workflow execution completed")
        return context


# Custom Azure OpenAI MultiModal class for image handling
class CustomAzureOpenAIMultiModal(AzureOpenAIMultiModal):
    def complete(self, prompt, image_documents, **kwargs):
        converted_images = []
        for img_node in image_documents:
            if (
                hasattr(img_node, "image_path")
                and img_node.image_path
                and not img_node.image_url
            ):
                try:
                    if os.path.exists(img_node.image_path):
                        with open(img_node.image_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode(
                                "utf-8"
                            )
                            mime_type, _ = mimetypes.guess_type(img_node.image_path)
                            if not mime_type:
                                mime_type = "image/jpeg"
                            data_url = f"data:{mime_type};base64,{encoded_string}"

                            new_node = ImageNode(image_url=data_url)
                            new_node.metadata = img_node.metadata
                            converted_images.append(new_node)
                    else:
                        new_node = ImageNode(image_url=img_node.image_path)
                        new_node.metadata = img_node.metadata
                        converted_images.append(new_node)
                except Exception as e:
                    logger.error(f"Failed to convert image node: {e}")
                    converted_images.append(img_node)
            else:
                converted_images.append(img_node)

        return super().complete(
            prompt=prompt, image_documents=converted_images, **kwargs
        )


# Initialize services
def initialize_services():
    """Initialize all required services."""
    # Initialize LLMs
    llm = AzureOpenAI(
        model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
        deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        streaming=True,
    )

    embed_model = AzureOpenAIEmbedding(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Set up callback manager (API-compatible)
    callback_manager = CallbackManager([APICallbackHandler()])

    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager

    # Initialize search clients
    credential = AzureKeyCredential(SEARCH_SERVICE_API_KEY)
    index_client = SearchIndexClient(
        endpoint=SEARCH_SERVICE_ENDPOINT, credential=credential
    )

    # Initialize vector store
    # Index schema: id, content, page, embedding, bbox_annotation, image_blob_urls
    # Configure additional metadata fields to be retrieved
    metadata_fields = {
        "image_blob_urls": "image_blob_urls",  # Collection(Edm.String) - blob URLs for images
        "page": "page",  # Edm.Int32 - page number
    }

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=index_client,
        index_name=INDEX_NAME,
        index_management=IndexManagement.VALIDATE_INDEX,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="embedding",
        embedding_dimensionality=3072,
        metadata_string_field_key="bbox_annotation",  # Using bbox_annotation as metadata field
        doc_id_field_key="id",  # Using id as doc_id since no separate doc_id field exists
        filterable_metadata_field_keys=metadata_fields,  # Additional fields to retrieve
        language_analyzer="en.lucene",
        vector_algorithm_type="hnsw",
    )

    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    # Initialize custom multimodal LLM
    azure_openai_mm_llm = CustomAzureOpenAIMultiModal(
        deployment_name=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        model=AZURE_OPENAI_CHAT_COMPLETION_DEPLOYED_MODEL_NAME,
        max_new_tokens=4096,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    # Create a SearchClient for direct document fetching (for image_blob_urls)
    search_client = SearchClient(
        endpoint=SEARCH_SERVICE_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    return index, azure_openai_mm_llm, search_client


# Setup agent workflow
def setup_agent_workflow(index, multi_modal_llm, search_client=None):
    """Set up the agent workflow with all agents."""
    workflow_manager = AgentWorkflowManager()

    # Create agents
    query_agent = QueryProcessingAgent(index.as_retriever(similarity_top_k=3))
    image_agent = ImageProcessingAgent(multi_modal_llm, search_client=search_client)
    content_agent = ContentAnalysisAgent(multi_modal_llm)
    response_agent = ResponseOrchestrationAgent()

    # Register agents
    workflow_manager.register_agent(query_agent)
    workflow_manager.register_agent(image_agent)
    workflow_manager.register_agent(content_agent)
    workflow_manager.register_agent(response_agent)

    # Define workflow steps
    workflow_manager.add_workflow_step("QueryProcessor")
    workflow_manager.add_workflow_step("ImageProcessor")
    workflow_manager.add_workflow_step("ContentAnalyzer")
    workflow_manager.add_workflow_step("ResponseOrchestrator")

    return workflow_manager


# Legacy Chainlit functions removed - agent workflow now accessed via REST API
# The agent system is initialized in agent_api.py and used through HTTP endpoints

# Note: This file now contains only the agent framework classes
# The actual API server functionality is in agent_api.py
