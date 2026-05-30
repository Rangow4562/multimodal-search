import os
import json
import base64
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# FastAPI app initialization
# ----------------------------
app = FastAPI(
    title="RAG Query API",
    description="API for querying documents using RAG with vision support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------
# Pydantic models
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    include_images: Optional[bool] = True

class SourceInfo(BaseModel):
    id: Optional[str]
    page: Optional[int]
    content_preview: Optional[str]
    has_images: bool
    has_annotations: bool

class ImageData(BaseModel):
    data_url: str

class QueryResponse(BaseModel):
    answer: str
    answer_html: str
    sources: List[SourceInfo]
    images_used: List[ImageData] = []

class HealthResponse(BaseModel):
    status: str
    message: str

# ----------------------------
# Global clients
# ----------------------------
openai_client = None
search_client = None
blob_service_client = None

# ----------------------------
# Azure OpenAI Client
# ----------------------------
class AzureOpenAIClient:
    def __init__(self, endpoint, embedding_deployment, completion_deployment, api_key, api_version):
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.embedding_deployment = embedding_deployment
        self.completion_deployment = completion_deployment

    def get_embedding(self, text):
        response = self.client.embeddings.create(model=self.embedding_deployment, input=text)
        return response.data[0].embedding

    def get_completion(self, prompt, images: Optional[List[dict]] = None):
        # images: list of {"b64": str, "mime": str}
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                try:
                    b64 = img.get("b64") if isinstance(img, dict) else None
                    mime = img.get("mime") if isinstance(img, dict) else None
                    if b64 and mime:
                        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                except Exception:
                    continue
            response = self.client.chat.completions.create(
                model=self.completion_deployment,
                messages=[{"role": "user", "content": content}]
            )
        else:
            response = self.client.chat.completions.create(
                model=self.completion_deployment,
                messages=[{"role": "user", "content": prompt}]
            )
        return response.choices[0].message.content

# ----------------------------
# Azure Cognitive Search Retriever
# ----------------------------
class AzureCognitiveSearchRetriever:
    def __init__(self, endpoint, admin_key, index_name):
        credential = AzureKeyCredential(admin_key)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    def vector_search(self, vector, top_k=5):
        vector_query = {
            "kind": "vector",
            "vector": vector,
            "fields": "embedding",
            "k": top_k,
            "includeDistance": True
        }
        results = self.search_client.search(
            search_text="*",
            vector_queries=[vector_query],
            select=["id", "content", "page", "bbox_annotation", "image_blob_urls"]
        )
        return list(results)

# ----------------------------
# Image fetching
# ----------------------------
def _detect_image_mime_type(image_bytes: bytes, name_hint: Optional[str] = None) -> Optional[str]:
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if image_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    if image_bytes[:6] in [b'GIF87a', b'GIF89a']:
        return 'image/gif'
    if name_hint:
        lower = name_hint.lower()
        if lower.endswith('.png'):
            return 'image/png'
        if lower.endswith('.jpg') or lower.endswith('.jpeg'):
            return 'image/jpeg'
        if lower.endswith('.gif'):
            return 'image/gif'
    return None

def fetch_images_from_blob_storage(image_blob_urls, container_name="images"):
    if not blob_service_client:
        return []

    fetched_images = []  # list of { 'b64': str, 'mime': str, 'name': str, 'data_url': str }
    for image_blob_url in image_blob_urls:
        try:
            if isinstance(image_blob_url, str) and image_blob_url.strip().startswith('['):
                blob_names = json.loads(image_blob_url)
            else:
                blob_names = [image_blob_url]

            for blob_url in blob_names:
                blob_name = blob_url.split('/')[-1] if isinstance(blob_url, str) and blob_url.startswith('http') else blob_url
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_data = blob_client.download_blob()
                image_bytes = blob_data.readall()
                if not image_bytes:
                    continue
                mime = _detect_image_mime_type(image_bytes, name_hint=blob_name if isinstance(blob_name, str) else None)
                if not mime:
                    continue
                b64 = base64.b64encode(image_bytes).decode('utf-8')
                fetched_images.append({
                    "b64": b64,
                    "mime": mime,
                    "name": blob_name if isinstance(blob_name, str) else str(blob_name),
                    "data_url": f"data:{mime};base64,{b64}"
                })
        except Exception as e:
            print(f"❌ Error fetching image {image_blob_url}: {e}")
    return fetched_images

# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(query, contexts, include_images=True):
    context_text = ""
    images_data = []
    image_labels = []

    for doc in contexts:
        content = doc.get("content", "")
        if not content:
            continue
        page = doc.get("page", "unknown")
        context_text += f"Page {page}:\n{content}\n"

        bbox_annotation = doc.get("bbox_annotation")
        if bbox_annotation:
            try:
                if isinstance(bbox_annotation, str):
                    bbox_data = json.loads(bbox_annotation)
                else:
                    bbox_data = bbox_annotation
                context_text += "\nDocument Annotation:\n"
                if bbox_data.get("document_type"):
                    context_text += f"- Type: {bbox_data['document_type']}\n"
                if bbox_data.get("short_description"):
                    context_text += f"- Description: {bbox_data['short_description']}\n"
                if bbox_data.get("summary"):
                    context_text += f"- Summary: {bbox_data['summary']}\n"
            except Exception:
                context_text += "\n[Contains annotation data]\n"

        # Collect images with labels
        if include_images and doc.get("image_blob_urls"):
            label = f"Image {len(image_labels)+1}"
            images_data.append({
                "image_blob_urls": doc["image_blob_urls"],
                "label": label,
                "chunk_id": doc.get("id", "unknown"),
                "page": page
            })
            image_labels.append(label)

        context_text += "\n---\n\n"

    # Instruction for AI to produce a direct, useful answer and reference images
    prompt = (
        "You are an AI assistant. Follow these rules strictly:\n"
        "- Answer the user's question directly and concisely.\n"
        "- Do NOT describe what sources you have in general; answer with the facts.\n"
        "- Do NOT use Markdown bold or italics (no **text**, *text*, or _text_). Use plain text.\n"
        "- If visual info is relevant, reference images by label (e.g., 'See Image 1').\n"
        "- When citing a location, include BOTH document name and page, e.g., 'Qualification_Summary_KTE57_D, page 5'.\n"
        "  (Hint: the document name is the part of the chunk id before '_page'.)\n"
        "- Include only the most relevant facts from the text/annotations/markdown; avoid boilerplate.\n"
        "- If the answer cannot be determined from the provided content, say so clearly.\n\n"
        f"Context:\n{context_text}\n"
        f"Question: {query}\n"
    )

    return prompt, images_data

# ----------------------------
# Startup event
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global openai_client, search_client, blob_service_client

    required_vars = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_ADMIN_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_COMPLETION_DEPLOYMENT",
        "AZURE_OPENAI_API_KEY"
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    try:
        openai_client = AzureOpenAIClient(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            completion_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        )

        search_client = AzureCognitiveSearchRetriever(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            admin_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        )

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        print("✅ Clients initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        raise RuntimeError(f"Client initialization failed: {e}")

# ----------------------------
# Health check
# ----------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="API is running")

# ----------------------------
# RAG Query Endpoint (Lovable-ready)
# ----------------------------
@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    try:
        query_vector = openai_client.get_embedding(query_request.question)
        results = search_client.vector_search(query_vector, top_k=query_request.top_k)
        if not results:
            return QueryResponse(answer="No relevant documents found.", sources=[], images_used=[])

        valid_results = [doc for doc in results if doc.get("content")]
        if not valid_results:
            return QueryResponse(answer="No valid content found.", sources=[], images_used=[])

        prompt, images_data = build_prompt(query_request.question, valid_results, include_images=query_request.include_images)

        # Fetch images as base64 with mime
        fetched_images = []
        image_labels = []
        if query_request.include_images and images_data:
            all_blob_urls = []
            for img in images_data:
                blob_urls = img.get("image_blob_urls")
                if blob_urls:
                    if isinstance(blob_urls, str) and blob_urls.strip().startswith('['):
                        all_blob_urls.extend(json.loads(blob_urls))
                    elif isinstance(blob_urls, list):
                        all_blob_urls.extend(blob_urls)
                    else:
                        all_blob_urls.append(blob_urls)
                image_labels.append(img.get("label"))

            if all_blob_urls:
                fetched_images = fetch_images_from_blob_storage(all_blob_urls)

        # Generate AI answer with image references
        answer = openai_client.get_completion(prompt, images=fetched_images if fetched_images else None)

        # Sanitize bold/italics from answer
        def _sanitize_answer(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Remove ** for bold and single * or _ for italics
            sanitized = text.replace('**', '')
            # Replace single asterisks not in bullet prefixes; simple approach: remove remaining '*'
            sanitized = sanitized.replace('*', '')
            sanitized = sanitized.replace('_', '')
            return sanitized

        answer_sanitized = _sanitize_answer(answer)

        # Build sources
        sources = []
        for doc in valid_results:
            sources.append(SourceInfo(
                id=doc.get("id"),
                page=doc.get("page"),
                content_preview=(doc.get("content", "")[:200] + "...") if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                has_images=bool(doc.get("image_blob_urls")),
                has_annotations=bool(doc.get("bbox_annotation"))
            ))

        # Map images to Lovable-ready data URLs
        images_for_lovable = [ImageData(data_url=img.get("data_url")) for img in fetched_images]

        # Build HTML answer with embedded images
        imgs_html = "".join([f"<div><img src=\"{img.get('data_url')}\" alt=\"{img.get('name')}\" /></div>" for img in fetched_images])
        answer_html = f"{answer_sanitized}\n\n{imgs_html}" if imgs_html else answer_sanitized

        return QueryResponse(answer=answer_sanitized, answer_html=answer_html, sources=sources, images_used=images_for_lovable)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
async def root():
    return {"message": "RAG Query API is running", "docs": "/docs"}
