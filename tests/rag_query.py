import os
import json
import base64
import sys
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from io import BytesIO
import argparse
import uuid

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from PIL import Image

load_dotenv()


# -------------------------------
# ANSI Color Codes for Terminal Output
# -------------------------------
class Colors:
    """ANSI color codes for terminal formatting."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"


def colored(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


# -------------------------------
# Conversation Message Structure
# -------------------------------
@dataclass
class ConversationMessage:
    """Structure for storing individual conversation messages."""

    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    images: List[str] = field(default_factory=list)  # Image blob names
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        return cls(**data)


# -------------------------------
# Conversation Session Manager
# -------------------------------
@dataclass
class ConversationSession:
    """Manages conversation history and session state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    history: List[ConversationMessage] = field(default_factory=list)
    max_history: int = 50
    sessions_dir: str = "./sessions"

    def add_message(
        self,
        role: str,
        content: str,
        images: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Add a message to the conversation history."""
        msg = ConversationMessage(
            role=role, content=content, images=images or [], metadata=metadata or {}
        )
        self.history.append(msg)
        self.updated_at = datetime.now().isoformat()

        # Trim history if exceeds max
        if len(self.history) > self.max_history:
            # Keep system messages and recent messages
            system_msgs = [m for m in self.history if m.role == "system"]
            other_msgs = [m for m in self.history if m.role != "system"]
            keep_count = self.max_history - len(system_msgs)
            self.history = system_msgs + other_msgs[-keep_count:]

    def get_context_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages formatted for OpenAI API."""
        messages = []
        recent = (
            self.history[-max_messages:]
            if len(self.history) > max_messages
            else self.history
        )

        for msg in recent:
            if msg.role in ["user", "assistant", "system"]:
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.updated_at = datetime.now().isoformat()

    def save(self, name: str = None):
        """Save session to JSON file."""
        os.makedirs(self.sessions_dir, exist_ok=True)
        filename = name or self.session_id
        filepath = os.path.join(self.sessions_dir, f"{filename}.json")

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [m.to_dict() for m in self.history],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    @classmethod
    def load(cls, name: str, sessions_dir: str = "./sessions") -> "ConversationSession":
        """Load session from JSON file."""
        filepath = os.path.join(sessions_dir, f"{name}.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Session '{name}' not found")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())[:8]),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            sessions_dir=sessions_dir,
        )
        session.history = [
            ConversationMessage.from_dict(m) for m in data.get("history", [])
        ]

        return session

    def display_history(self, last_n: int = 10):
        """Display recent conversation history."""
        messages = (
            self.history[-last_n:] if len(self.history) > last_n else self.history
        )

        if not messages:
            print(colored("  (No conversation history)", Colors.GRAY))
            return

        for msg in messages:
            role_color = Colors.CYAN if msg.role == "user" else Colors.GREEN
            role_label = "You" if msg.role == "user" else "AI"

            print(colored(f"\n  [{role_label}]", role_color + Colors.BOLD))
            # Truncate long messages
            content = (
                msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            )
            print(f"  {content}")

            if msg.images:
                print(colored(f"  📷 Images: {len(msg.images)}", Colors.GRAY))


# -------------------------------
# Azure OpenAI Client (Enhanced)
# -------------------------------
class AzureOpenAIClient:
    def __init__(
        self,
        endpoint,
        embedding_deployment,
        completion_deployment,
        api_key,
        api_version,
    ):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.embedding_deployment = embedding_deployment
        self.completion_deployment = completion_deployment

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model=self.embedding_deployment, input=text
        )
        return response.data[0].embedding

    def get_completion(self, content, conversation_history: List[Dict] = None):
        """
        Get completion with optional conversation history.
        Content is a list of dicts [{"type": "text","text":...}, {"type":"image_url","image_url":{"url":...}}, ...]
        """
        messages = []

        # Add conversation history for context
        if conversation_history:
            messages.extend(conversation_history[:-1])  # All but the current message

        # Add current message with multimodal content
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.completion_deployment,
            messages=messages,
        )
        return response.choices[0].message.content


# -------------------------------
# Azure Cognitive Search Retriever
# -------------------------------
class AzureCognitiveSearchRetriever:
    def __init__(self, endpoint, admin_key, index_name):
        credential = AzureKeyCredential(admin_key)
        self.search_client = SearchClient(
            endpoint=endpoint, index_name=index_name, credential=credential
        )

    def vector_search(self, vector, top_k=5):
        vector_query = {
            "kind": "vector",
            "vector": vector,
            "k": top_k,
            "includeDistance": True,
            "fields": "embedding",
        }
        results = self.search_client.search(
            search_text="*",
            vector_queries=[vector_query],
            select=["content", "page", "bbox_annotation", "image_blob_urls", "id"],
        )
        return list(results)

    def text_search(self, query, top_k=5):
        results = self.search_client.search(
            search_text=query,
            select=["content", "page", "bbox_annotation", "image_blob_urls", "id"],
            top=top_k,
        )
        return list(results)


# -------------------------------
# Fetch images from Azure Blob Storage
# -------------------------------
def fetch_images_from_blob_storage(
    image_blob_urls: List[str], container_name: str = None
) -> List[Tuple[str, str, str]]:
    # Get container name from environment if not provided
    if container_name is None:
        container_name = (
            os.getenv("IMAGE_CONTAINER_NAME")
            or os.getenv("AZURE_SEARCH_INDEX_NAME")
            or os.getenv("CONTAINER_NAME")
            or "images"
        )
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        return []

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    fetched_images = []

    for image_blob_url in image_blob_urls:
        # Parse blob names
        if isinstance(image_blob_url, str) and image_blob_url.strip().startswith("["):
            try:
                blob_names = json.loads(image_blob_url)
            except Exception:
                blob_names = [image_blob_url]
        elif isinstance(image_blob_url, str):
            blob_names = [image_blob_url]
        elif isinstance(image_blob_url, list):
            blob_names = image_blob_url
        else:
            blob_names = []

        for blob_name in blob_names:
            if not blob_name or not blob_name.strip():
                continue
            try:
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob_name
                )
                blob_data = blob_client.download_blob()
                img_bytes = blob_data.readall()

                if not img_bytes:
                    continue

                mime_type = detect_image_mime_type(img_bytes, blob_name)
                if not mime_type:
                    continue

                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                fetched_images.append((img_b64, mime_type, blob_name))

            except Exception:
                continue

    return fetched_images


def detect_image_mime_type(image_bytes, blob_name):
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    elif image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    elif image_bytes[:6] in [b"GIF87a", b"GIF89a"]:
        return "image/gif"
    return None


# -------------------------------
# Generate SAS URL for Blob Images
# -------------------------------
def get_image_container_name() -> str:
    """Get the container name for images from environment variables."""
    # Try specific image container env var first, then fall back to search index name
    return (
        os.getenv("IMAGE_CONTAINER_NAME")
        or os.getenv("AZURE_SEARCH_INDEX_NAME")
        or os.getenv("CONTAINER_NAME")
        or "images"
    )


def generate_image_sas_url(
    blob_name: str, container_name: str = None, expiry_hours: int = 24
) -> Optional[str]:
    """Generate a SAS URL for an image blob that can be accessed directly."""
    # Use provided container or get from environment
    if container_name is None:
        container_name = get_image_container_name()

    account_name = os.getenv("BLOB_STORAGE_ACCOUNT_NAME") or os.getenv(
        "AZURE_STORAGE_ACCOUNT_NAME"
    )
    account_key = os.getenv("BLOB_STORAGE_ACCOUNT_KEY") or os.getenv(
        "AZURE_STORAGE_ACCOUNT_KEY"
    )

    if not account_name or not account_key:
        # Try to extract from connection string
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            parts = dict(
                part.split("=", 1)
                for part in connection_string.split(";")
                if "=" in part
            )
            account_name = parts.get("AccountName")
            account_key = parts.get("AccountKey")

    if not account_name or not account_key:
        return None

    try:
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )
        return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    except Exception as e:
        return None


# -------------------------------
# Retrieve markdown content from parsed_chunks
# -------------------------------
def get_markdown_content(chunk_ids, parsed_chunks_dir="./parsed_chunks"):
    markdown_content = {}
    if not os.path.exists(parsed_chunks_dir):
        return markdown_content

    for filename in os.listdir(parsed_chunks_dir):
        if not filename.endswith("_parsed_chunks.json"):
            continue
        filepath = os.path.join(parsed_chunks_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for chunk in chunks:
                cid = chunk.get("id")
                if cid in chunk_ids:
                    markdown_content[cid] = chunk.get("content", "")
        except Exception:
            pass
    return markdown_content


# -------------------------------
# Build GPT content with text + images
# -------------------------------
def build_prompt_with_images(
    query, chunks, max_images=3, include_markdown=True, conversation_context: str = ""
):
    content = []

    # Retrieve markdown for chunks
    chunk_ids = [c.get("id") for c in chunks]
    markdown_map = get_markdown_content(chunk_ids) if include_markdown else {}

    # First, select up to max_images images and record their blob names in order
    selected_images = []  # list of tuples (img_b64, mime, blob_name)
    image_names = []
    for chunk in chunks:
        for img_tuple in chunk.get("image_data", []):
            if len(selected_images) >= max_images:
                break
            img_b64, mime, blob_name = img_tuple
            selected_images.append((img_b64, mime, blob_name))
            image_names.append(blob_name)
        if len(selected_images) >= max_images:
            break

    # Build instruction header with enumerated Images list including SAS URLs
    images_lines = []
    image_url_map = {}  # Store mapping of image name to SAS URL
    for idx, name in enumerate(image_names, 1):
        sas_url = generate_image_sas_url(name)
        if sas_url:
            images_lines.append(f"{idx}. {name} - URL: {sas_url}")
            image_url_map[name] = sas_url
        else:
            images_lines.append(f"{idx}. {name}")
    images_block = "\n".join(images_lines) if images_lines else "(none)"

    # Include conversation context if available
    context_section = ""
    if conversation_context:
        context_section = (
            f"\nPrevious conversation context:\n{conversation_context}\n\n"
        )

    header_text = (
        "You are an internal AI assistant for  PAECETrak ROW AI Assist. You are given the following:\n\n"
        "1. Text content from documents.\n"
        "2. Up to 3 images, each provided with their accessible URLs.\n"
        f"{context_section}"
        "Instructions:\n\n"
        "- Analyze both the text and the attached images.\n"
        "- If a question refers to a diagram, technical drawing, or visual information, check the images for relevant content.\n"
        '- When referencing images in your answer, use the FULL URL provided above in markdown format, e.g., "![Image Description](full_url_here)".\n'
        '- You can also reference images like: "See [Image 1](full_url_here) for the technical drawing."\n'
        "- Include relevant text from the documents, annotations, or markdown content in your answer.\n"
        "- Do NOT use just the image filename as the link - always use the complete URL provided.\n"
        "- If this is a follow-up question, use context from the conversation to provide a coherent response.\n\n"
        "Available Images with URLs:\n"
        f"{images_block}\n\n"
    )
    content.append({"type": "text", "text": header_text})

    # Add document text blocks (content + annotation)
    for chunk in chunks:
        text_block = chunk.get("content", "")
        if include_markdown and markdown_map.get(chunk.get("id")):
            text_block += "\n" + markdown_map[chunk.get("id")]

        annotation_block = ""
        bbox_annotation = chunk.get("bbox_annotation")
        if bbox_annotation:
            try:
                ann = (
                    json.loads(bbox_annotation)
                    if isinstance(bbox_annotation, str)
                    else bbox_annotation
                )
                lines = []
                if ann.get("document_type"):
                    lines.append(f"Type: {ann['document_type']}")
                if ann.get("short_description"):
                    lines.append(f"Description: {ann['short_description']}")
                if ann.get("summary"):
                    lines.append(f"Summary: {ann['summary']}")
                annotation_block = "\n".join(lines)
            except Exception:
                annotation_block = ""

        if text_block or annotation_block:
            content.append(
                {
                    "type": "text",
                    "text": text_block
                    + ("\n" + annotation_block if annotation_block else ""),
                }
            )

    # Add selected images in the same order as enumerated
    for img_b64, mime, _ in selected_images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
        )

    # Append the question
    content.append({"type": "text", "text": f"Question: {query}"})

    return content, image_names, selected_images, image_url_map


# -------------------------------
# Post-process AI response to fix image links
# -------------------------------
import re


def post_process_response(
    answer: str, image_url_map: Dict[str, str], all_blob_urls: List[str] = None
) -> str:
    """
    Post-process the AI response to replace image filename references with actual SAS URLs.
    Handles patterns like:
    - ![img-1.jpeg](img-1.jpeg)
    - [Image 1](img-1.jpeg)
    - Generic image references (img-1, img-2, etc.)
    """
    if not image_url_map and not all_blob_urls:
        return answer

    # Build complete URL map including all blob URLs
    complete_url_map = dict(image_url_map) if image_url_map else {}
    if all_blob_urls:
        for blob_name in all_blob_urls:
            if blob_name not in complete_url_map:
                sas_url = generate_image_sas_url(blob_name)
                if sas_url:
                    complete_url_map[blob_name] = sas_url

    if not complete_url_map:
        return answer

    processed_answer = answer

    # Create a list of all available SAS URLs in order
    url_list = list(complete_url_map.values())
    blob_names_list = list(complete_url_map.keys())

    # First, replace exact blob name matches
    for img_name, sas_url in complete_url_map.items():
        escaped_name = re.escape(img_name)

        # Pattern 1: ![anything](exact_blob_name) -> ![anything](sas_url)
        pattern1 = rf"(!\[[^\]]*\])\({escaped_name}\)"
        processed_answer = re.sub(pattern1, rf"\1({sas_url})", processed_answer)

        # Pattern 2: [anything](exact_blob_name) -> [anything](sas_url)
        pattern2 = rf"(\[[^\]]+\])\({escaped_name}\)"
        processed_answer = re.sub(pattern2, rf"\1({sas_url})", processed_answer)

    # Now handle generic image references like img-1.jpeg, img-2.jpeg, etc.
    # These need to be mapped to actual URLs based on index
    def replace_generic_image(match):
        """Replace generic image filename with actual SAS URL based on index."""
        prefix = match.group(1)  # The ![...] or [...] part
        img_ref = match.group(2)  # The filename like img-1.jpeg

        # Extract index from patterns like img-1, img-2, Image 1, etc.
        index_match = re.search(r"(\d+)", img_ref)
        if index_match:
            idx = int(index_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= idx < len(url_list):
                return f"{prefix}({url_list[idx]})"

        # If no index found or out of range, use first available URL
        if url_list:
            return f"{prefix}({url_list[0]})"
        return match.group(0)  # Return original if no URLs available

    # Pattern for generic image references: ![...](img-N.ext) or [...](img-N.ext)
    generic_pattern = r"(!\[[^\]]*\]|\[[^\]]+\])\((img-?\d+\.(?:jpeg|jpg|png|gif))\)"
    processed_answer = re.sub(
        generic_pattern, replace_generic_image, processed_answer, flags=re.IGNORECASE
    )

    # Also handle patterns like (img-N.jpeg) without markdown link prefix - make them clickable
    def make_image_clickable(match):
        full_match = match.group(0)
        img_ref = match.group(1)

        # Extract index
        index_match = re.search(r"(\d+)", img_ref)
        if index_match:
            idx = int(index_match.group(1)) - 1
            if 0 <= idx < len(url_list):
                return f"[{img_ref}]({url_list[idx]})"

        if url_list:
            return f"[{img_ref}]({url_list[0]})"
        return full_match

    # Find standalone image references that aren't already in links
    # Match img-N.ext that's not preceded by ]( or followed by )
    standalone_pattern = r"(?<!\]\()(?<!\()(\bimg-?\d+\.(?:jpeg|jpg|png|gif)\b)(?!\))"
    processed_answer = re.sub(
        standalone_pattern, make_image_clickable, processed_answer, flags=re.IGNORECASE
    )

    return processed_answer


# -------------------------------
# Chat Interface
# -------------------------------
class ChatInterface:
    """Handles the continuous chat interaction."""

    COMMANDS = {
        "/exit": "Exit the chat session",
        "/quit": "Exit the chat session (alias)",
        "/clear": "Clear conversation history",
        "/history": "Show recent conversation history",
        "/save": "Save session (/save [name])",
        "/load": "Load session (/load <name>)",
        "/images": "Show images from last response",
        "/help": "Show this help message",
    }

    def __init__(self, retriever, openai_client, session: ConversationSession = None):
        self.retriever = retriever
        self.openai_client = openai_client
        self.session = session or ConversationSession()
        self.last_images = []  # Store images from last response
        self.running = True

    def print_header(self):
        """Print welcome header."""
        print()
        print(
            colored(
                "╔════════════════════════════════════════════════════════╗",
                Colors.CYAN,
            )
        )
        print(
            colored(
                "║        PAECETrak ROW AI Assist - Chat Mode            ║",
                Colors.CYAN + Colors.BOLD,
            )
        )
        print(
            colored(
                "╚════════════════════════════════════════════════════════╝",
                Colors.CYAN,
            )
        )
        print()
        print(colored("  Type your question or use /help for commands", Colors.GRAY))
        print(colored(f"  Session ID: {self.session.session_id}", Colors.GRAY))
        print()

    def print_help(self):
        """Display help information."""
        print()
        print(colored("  ╭─────────────────────────────────────╮", Colors.BLUE))
        print(
            colored(
                "  │         Available Commands          │", Colors.BLUE + Colors.BOLD
            )
        )
        print(colored("  ├─────────────────────────────────────┤", Colors.BLUE))
        for cmd, desc in self.COMMANDS.items():
            print(
                colored(f"  │ {cmd:12} ", Colors.CYAN)
                + colored(f"{desc:23} │", Colors.GRAY)
            )
        print(colored("  ╰─────────────────────────────────────╯", Colors.BLUE))
        print()

    def handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns True if should continue, False to exit."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd in ["/exit", "/quit"]:
            print(colored("\n  👋 Goodbye! Session saved.", Colors.YELLOW))
            self.session.save()
            return False

        elif cmd == "/clear":
            self.session.clear()
            self.last_images = []
            print(colored("\n  ✓ Conversation history cleared.", Colors.GREEN))

        elif cmd == "/history":
            print(colored("\n  📜 Recent Conversation:", Colors.YELLOW + Colors.BOLD))
            self.session.display_history()

        elif cmd == "/save":
            name = arg or self.session.session_id
            filepath = self.session.save(name)
            print(colored(f"\n  ✓ Session saved to: {filepath}", Colors.GREEN))

        elif cmd == "/load":
            if not arg:
                print(colored("\n  ❌ Usage: /load <session_name>", Colors.RED))
            else:
                try:
                    self.session = ConversationSession.load(arg)
                    print(
                        colored(
                            f"\n  ✓ Session '{arg}' loaded ({len(self.session.history)} messages)",
                            Colors.GREEN,
                        )
                    )
                except FileNotFoundError:
                    print(colored(f"\n  ❌ Session '{arg}' not found", Colors.RED))

        elif cmd == "/images":
            if not self.last_images:
                print(colored("\n  (No images from last response)", Colors.GRAY))
            else:
                print(
                    colored(
                        "\n  📷 Images from last response:", Colors.YELLOW + Colors.BOLD
                    )
                )
                print(
                    colored("  ─────────────────────────────────────────", Colors.GRAY)
                )
                for idx, name in enumerate(self.last_images, 1):
                    print(colored(f"    Image {idx}: ", Colors.CYAN) + f"{name}")
                    sas_url = generate_image_sas_url(name)
                    if sas_url:
                        print(colored(f"    🔗 URL: ", Colors.GREEN) + f"{sas_url}")
                    else:
                        print(colored(f"    ⚠️  URL not available", Colors.YELLOW))
                    print()

        elif cmd == "/help":
            self.print_help()

        else:
            print(
                colored(
                    f"\n  ❌ Unknown command: {cmd}. Type /help for available commands.",
                    Colors.RED,
                )
            )

        return True

    def show_typing_indicator(self):
        """Show typing indicator."""
        print(colored("\n  ⏳ AI is thinking...", Colors.GRAY), end="\r")

    def process_query(self, query: str):
        """Process a user query and generate response."""
        self.show_typing_indicator()

        try:
            # Get embedding and search
            query_vector = self.openai_client.get_embedding(query)
            results = self.retriever.vector_search(query_vector, top_k=5)

            if not results:
                results = self.retriever.text_search(query, top_k=5)

            if not results:
                answer = "I couldn't find any relevant information in the documents."
                self.session.add_message("assistant", answer)
                print(" " * 50)  # Clear typing indicator
                print(colored("\n  🤖 AI:", Colors.GREEN + Colors.BOLD))
                print(f"  {answer}")
                return

            # Filter valid chunks
            chunks = [r for r in results if r.get("content")]
            if not chunks:
                answer = "I found results but they don't contain readable content."
                self.session.add_message("assistant", answer)
                print(" " * 50)
                print(colored("\n  🤖 AI:", Colors.GREEN + Colors.BOLD))
                print(f"  {answer}")
                return

            # Flatten image URLs
            all_blob_urls = []
            for chunk in chunks:
                urls = chunk.get("image_blob_urls") or []
                if isinstance(urls, str) and urls.strip().startswith("["):
                    try:
                        urls = json.loads(urls)
                    except Exception:
                        urls = [urls]
                elif isinstance(urls, str):
                    urls = [urls]
                elif not isinstance(urls, list):
                    urls = []

                urls = [u for u in urls if u and u.strip()]
                chunk["image_blob_urls"] = urls
                all_blob_urls.extend(urls)

            # Fetch images
            fetched_images = fetch_images_from_blob_storage(all_blob_urls)

            # Assign images back to chunks
            img_idx = 0
            for chunk in chunks:
                chunk_images = []
                for _ in chunk["image_blob_urls"]:
                    if img_idx < len(fetched_images):
                        chunk_images.append(fetched_images[img_idx])
                        img_idx += 1
                chunk["image_data"] = chunk_images

            # Build conversation context from history
            conversation_context = ""
            recent_msgs = self.session.history[-4:]  # Last 2 exchanges
            if recent_msgs:
                context_parts = []
                for msg in recent_msgs:
                    role = "User" if msg.role == "user" else "Assistant"
                    context_parts.append(f"{role}: {msg.content[:300]}")
                conversation_context = "\n".join(context_parts)

            # Build GPT content
            gpt_content, referenced_image_names, selected_images, image_url_map = (
                build_prompt_with_images(
                    query,
                    chunks,
                    max_images=3,
                    include_markdown=True,
                    conversation_context=conversation_context,
                )
            )

            # Get conversation history for API
            history_messages = self.session.get_context_messages(max_messages=6)

            # Get completion
            try:
                answer = self.openai_client.get_completion(
                    gpt_content, history_messages
                )
            except Exception as e:
                # Fallback: text only
                text_only_content = [c for c in gpt_content if c["type"] == "text"]
                answer = self.openai_client.get_completion(
                    text_only_content, history_messages
                )

            # Post-process the answer to replace image filename references with SAS URLs
            # Debug output to diagnose image URL issues
            if all_blob_urls:
                print(
                    colored(
                        f"\n  [DEBUG] Found {len(all_blob_urls)} blob URLs",
                        Colors.YELLOW,
                    )
                )
                for i, url in enumerate(all_blob_urls[:3]):
                    print(colored(f"    - {url[:60]}...", Colors.GRAY))
            else:
                print(
                    colored(
                        "\n  [DEBUG] No blob URLs found in search results",
                        Colors.YELLOW,
                    )
                )

            if image_url_map:
                print(
                    colored(
                        f"  [DEBUG] Generated {len(image_url_map)} SAS URLs",
                        Colors.YELLOW,
                    )
                )
            else:
                print(colored("  [DEBUG] No SAS URLs generated", Colors.YELLOW))

            answer = post_process_response(answer, image_url_map, all_blob_urls)

            # Store in history
            self.session.add_message("user", query)
            self.session.add_message("assistant", answer, images=referenced_image_names)
            self.last_images = (
                referenced_image_names if referenced_image_names else all_blob_urls
            )

            # Display response
            print(" " * 50)  # Clear typing indicator
            print(colored("\n  🤖 AI:", Colors.GREEN + Colors.BOLD))
            print()

            # Format and display answer
            for line in answer.split("\n"):
                print(f"  {line}")

            # Show referenced images with SAS URLs
            # Use referenced_image_names if available, otherwise fall back to all_blob_urls from search
            display_images = (
                referenced_image_names if referenced_image_names else all_blob_urls
            )
            if display_images:
                print()
                print(colored("  📷 Referenced Images:", Colors.CYAN + Colors.BOLD))
                print(
                    colored("  ─────────────────────────────────────────", Colors.GRAY)
                )
                for idx, name in enumerate(display_images, 1):
                    print(colored(f"    Image {idx}: ", Colors.CYAN) + f"{name}")
                    sas_url = generate_image_sas_url(name)
                    if sas_url:
                        print(colored(f"    🔗 URL: ", Colors.GREEN) + f"{sas_url}")
                    else:
                        print(
                            colored(
                                f"    ⚠️  URL not available (check blob storage credentials)",
                                Colors.YELLOW,
                            )
                        )
                    print()
            elif not display_images and all_blob_urls:
                # Images were in search but couldn't be processed
                print()
                print(
                    colored(
                        "  📷 Images found in documents:", Colors.CYAN + Colors.BOLD
                    )
                )
                print(
                    colored("  ─────────────────────────────────────────", Colors.GRAY)
                )
                for idx, name in enumerate(all_blob_urls[:5], 1):  # Limit to first 5
                    print(colored(f"    Image {idx}: ", Colors.CYAN) + f"{name}")
                    sas_url = generate_image_sas_url(name)
                    if sas_url:
                        print(colored(f"    🔗 URL: ", Colors.GREEN) + f"{sas_url}")
                    print()

            print()

        except KeyboardInterrupt:
            print(colored("\n  ⚠️  Query cancelled", Colors.YELLOW))
        except Exception as e:
            print(colored(f"\n  ❌ Error: {str(e)}", Colors.RED))

    def run(self):
        """Run the continuous chat loop."""
        self.print_header()

        while self.running:
            try:
                # Get user input
                user_input = input(
                    colored("  You: ", Colors.CYAN + Colors.BOLD)
                ).strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("/"):
                    self.running = self.handle_command(user_input)
                    continue

                # Process query
                self.process_query(user_input)

            except KeyboardInterrupt:
                print(
                    colored(
                        "\n\n  👋 Session interrupted. Type /exit to quit or continue chatting.",
                        Colors.YELLOW,
                    )
                )
            except EOFError:
                print(colored("\n\n  👋 Goodbye!", Colors.YELLOW))
                self.session.save()
                break


# -------------------------------
# Single Query Mode (Backward Compatibility)
# -------------------------------
def run_single_query():
    """Run in single-query mode (original behavior)."""
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    completion_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    retriever = AzureCognitiveSearchRetriever(
        search_endpoint, search_admin_key, search_index_name
    )
    openai_client = AzureOpenAIClient(
        openai_endpoint,
        embedding_deployment,
        completion_deployment,
        openai_api_key,
        openai_api_version,
    )

    user_query = input("Enter your question about product summary reports:\n> ")
    query_vector = openai_client.get_embedding(user_query)
    results = retriever.vector_search(query_vector, top_k=5)

    if not results:
        results = retriever.text_search(user_query, top_k=5)
        if not results:
            print("❌ No results found")
            return

    chunks = [r for r in results if r.get("content")]
    if not chunks:
        print("❌ No valid chunks with content")
        return

    all_blob_urls = []
    for chunk in chunks:
        urls = chunk.get("image_blob_urls") or []
        if isinstance(urls, str) and urls.strip().startswith("["):
            try:
                urls = json.loads(urls)
            except Exception:
                urls = [urls]
        elif isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            urls = []

        urls = [u for u in urls if u and u.strip()]
        chunk["image_blob_urls"] = urls
        all_blob_urls.extend(urls)

    fetched_images = fetch_images_from_blob_storage(all_blob_urls)

    img_idx = 0
    for chunk in chunks:
        chunk_images = []
        for _ in chunk["image_blob_urls"]:
            if img_idx < len(fetched_images):
                chunk_images.append(fetched_images[img_idx])
                img_idx += 1
        chunk["image_data"] = chunk_images

    gpt_content, referenced_image_names, selected_images, image_url_map = (
        build_prompt_with_images(
            user_query, chunks, max_images=3, include_markdown=True
        )
    )

    try:
        answer = openai_client.get_completion(gpt_content)
    except Exception:
        text_only_content = [c for c in gpt_content if c["type"] == "text"]
        answer = openai_client.get_completion(text_only_content)

    # Post-process the answer to replace image filename references with SAS URLs
    answer = post_process_response(answer, image_url_map, all_blob_urls)

    print("\n--- AI Answer ---\n")
    print(answer)

    if referenced_image_names:
        print("\n--- Referenced Images ---")
        for idx, name in enumerate(referenced_image_names, 1):
            sas_url = generate_image_sas_url(name)
            print(f"Image {idx}: {name}")
            if sas_url:
                print(f"  URL: {sas_url}")


# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG Query + GPT-4o Vision - Continuous Chat Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_query.py              # Start continuous chat
  python rag_query.py --single     # Single query mode (original)
  python rag_query.py --session my_session  # Load existing session
        """,
    )
    parser.add_argument(
        "--single",
        "-s",
        action="store_true",
        help="Run in single-query mode (original behavior)",
    )
    parser.add_argument("--session", type=str, help="Load an existing session by name")
    parser.add_argument(
        "--max-history",
        type=int,
        default=50,
        help="Maximum conversation history length (default: 50)",
    )

    args = parser.parse_args()

    # Single query mode
    if args.single:
        run_single_query()
        return

    # Initialize clients
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    completion_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    retriever = AzureCognitiveSearchRetriever(
        search_endpoint, search_admin_key, search_index_name
    )
    openai_client = AzureOpenAIClient(
        openai_endpoint,
        embedding_deployment,
        completion_deployment,
        openai_api_key,
        openai_api_version,
    )

    # Load or create session
    session = None
    if args.session:
        try:
            session = ConversationSession.load(args.session)
            print(f"Loaded session: {args.session}")
        except FileNotFoundError:
            print(f"Session '{args.session}' not found. Creating new session.")

    if session is None:
        session = ConversationSession(max_history=args.max_history)

    # Run chat interface
    chat = ChatInterface(retriever, openai_client, session)
    chat.run()


if __name__ == "__main__":
    main()
