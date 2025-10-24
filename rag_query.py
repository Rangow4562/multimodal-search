import os
import json
import base64
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from typing import List, Tuple, Optional
from io import BytesIO
import argparse
import imghdr

from PIL import Image

load_dotenv()


# -------------------------------
# Azure OpenAI Client
# -------------------------------
class AzureOpenAIClient:
    def __init__(self, endpoint, embedding_deployment, completion_deployment, api_key, api_version):
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.embedding_deployment = embedding_deployment
        self.completion_deployment = completion_deployment

    def get_embedding(self, text):
        response = self.client.embeddings.create(model=self.embedding_deployment, input=text)
        return response.data[0].embedding

    def get_completion(self, content):
        """
        Content is a list of dicts [{"type": "text","text":...}, {"type":"image_url","image_url":{"url":...}}, ...]
        """
        response = self.client.chat.completions.create(
            model=self.completion_deployment,
            messages=[{"role": "user", "content": content}]
        )
        return response.choices[0].message.content


# -------------------------------
# Azure Cognitive Search Retriever
# -------------------------------
class AzureCognitiveSearchRetriever:
    def __init__(self, endpoint, admin_key, index_name):
        credential = AzureKeyCredential(admin_key)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    def vector_search(self, vector, top_k=5):
        vector_query = {
            "kind": "vector",
            "vector": vector,
            "k": top_k,
            "includeDistance": True,
            "fields": "embedding"
        }
        results = self.search_client.search(
            search_text="*",
            vector_queries=[vector_query],
            select=["content", "page", "bbox_annotation", "image_blob_urls", "id"]
        )
        return list(results)

    def text_search(self, query, top_k=5):
        results = self.search_client.search(
            search_text=query,
            select=["content", "page", "bbox_annotation", "image_blob_urls", "id"],
            top=top_k
        )
        return list(results)


# -------------------------------
# Fetch images from Azure Blob Storage
# -------------------------------
def fetch_images_from_blob_storage(image_blob_urls: List[str], container_name="images") -> List[Tuple[str, str, str]]:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found")
        return []

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    fetched_images = []

    for image_blob_url in image_blob_urls:
        # Parse blob names
        if isinstance(image_blob_url, str) and image_blob_url.strip().startswith('['):
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
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_data = blob_client.download_blob()
                img_bytes = blob_data.readall()

                if not img_bytes:
                    continue

                mime_type = detect_image_mime_type(img_bytes, blob_name)
                if not mime_type:
                    continue

                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                fetched_images.append((img_b64, mime_type, blob_name))

            except Exception as e:
                print(f"❌ Failed to fetch blob {blob_name}: {e}")
                continue

    return fetched_images


def detect_image_mime_type(image_bytes, blob_name):
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    elif image_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    elif image_bytes[:6] in [b'GIF87a', b'GIF89a']:
        return 'image/gif'
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
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            for chunk in chunks:
                cid = chunk.get('id')
                if cid in chunk_ids:
                    markdown_content[cid] = chunk.get('content', '')
        except Exception as e:
            print(f"⚠️ Failed to read {filename}: {e}")
    return markdown_content


# -------------------------------
# Build GPT content with text + images
# -------------------------------
def build_prompt_with_images(query, chunks, max_images=3, include_markdown=True):
    content = []

    # Retrieve markdown for chunks
    chunk_ids = [c.get('id') for c in chunks]
    markdown_map = get_markdown_content(chunk_ids) if include_markdown else {}

    # First, select up to max_images images and record their blob names in order
    selected_images = []  # list of tuples (img_b64, mime, blob_name)
    image_names = []
    for chunk in chunks:
        for img_tuple in chunk.get('image_data', []):
            if len(selected_images) >= max_images:
                break
            img_b64, mime, blob_name = img_tuple
            selected_images.append((img_b64, mime, blob_name))
            image_names.append(blob_name)
        if len(selected_images) >= max_images:
            break

    # Build instruction header with enumerated Images list
    images_lines = []
    for idx, name in enumerate(image_names, 1):
        images_lines.append(f"{idx}. {name}")
    images_block = "\n".join(images_lines) if images_lines else "(none)"

    header_text = (
        "You are an internal AI assistant for Tower Cold Chain. You are given the following:\n\n"
        "1. Text content from documents.\n"
        "2. Up to 3 images, each provided as a URL or blob name.\n\n"
        "Instructions:\n\n"
        "- Analyze both the text and the attached images.\n"
        "- If a question refers to a diagram, technical drawing, or visual information, check the images for relevant content.\n"
        "- In your answer, explicitly reference the images you are using, e.g., \"See Image 1 for the technical drawing of KTM22R.\"\n"
        "- Include relevant text from the documents, annotations, or markdown content in your answer.\n"
        "- Do NOT attempt to generate new images; only refer to the existing images.\n\n"
        "Images:\n"
        f"{images_block}\n\n"
    )
    content.append({"type": "text", "text": header_text})

    # Add document text blocks (content + annotation)
    for chunk in chunks:
        text_block = chunk.get('content', '')
        if include_markdown and markdown_map.get(chunk.get('id')):
            text_block += "\n" + markdown_map[chunk.get('id')]

        annotation_block = ""
        bbox_annotation = chunk.get('bbox_annotation')
        if bbox_annotation:
            try:
                ann = json.loads(bbox_annotation) if isinstance(bbox_annotation, str) else bbox_annotation
                lines = []
                if ann.get('document_type'):
                    lines.append(f"Type: {ann['document_type']}")
                if ann.get('short_description'):
                    lines.append(f"Description: {ann['short_description']}")
                if ann.get('summary'):
                    lines.append(f"Summary: {ann['summary']}")
                annotation_block = "\n".join(lines)
            except Exception:
                annotation_block = ""

        if text_block or annotation_block:
            content.append({"type": "text", "text": text_block + ("\n" + annotation_block if annotation_block else "")})

    # Add selected images in the same order as enumerated
    for img_b64, mime, _ in selected_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})

    # Append the question
    content.append({"type": "text", "text": f"Question: {query}"})

    return content, image_names, selected_images


# -------------------------------
# Main RAG + GPT-4o Vision
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Query + GPT-4o Vision")
    args = parser.parse_known_args()[0]

    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    completion_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    retriever = AzureCognitiveSearchRetriever(search_endpoint, search_admin_key, search_index_name)
    openai_client = AzureOpenAIClient(openai_endpoint, embedding_deployment, completion_deployment, openai_api_key, openai_api_version)

    user_query = input("Enter your question about product summary reports:\n> ")
    query_vector = openai_client.get_embedding(user_query)
    results = retriever.vector_search(query_vector, top_k=5)

    if not results:
        results = retriever.text_search(user_query, top_k=5)
        if not results:
            print("❌ No results found")
            return

    # Filter valid chunks
    chunks = [r for r in results if r.get('content')]
    if not chunks:
        print("❌ No valid chunks with content")
        return

    # Flatten image URLs safely
    all_blob_urls = []
    for chunk in chunks:
        urls = chunk.get('image_blob_urls') or []
        if isinstance(urls, str) and urls.strip().startswith('['):
            try:
                urls = json.loads(urls)
            except Exception:
                urls = [urls]
        elif isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            urls = []

        urls = [u for u in urls if u and u.strip()]
        chunk['image_blob_urls'] = urls
        all_blob_urls.extend(urls)

    fetched_images = fetch_images_from_blob_storage(all_blob_urls)

    # Assign images back to chunks
    img_idx = 0
    for chunk in chunks:
        chunk_images = []
        for _ in chunk['image_blob_urls']:
            if img_idx < len(fetched_images):
                chunk_images.append(fetched_images[img_idx])
                img_idx += 1
        chunk['image_data'] = chunk_images

    # Build GPT content with max 3 images
    gpt_content, referenced_image_names, selected_images = build_prompt_with_images(user_query, chunks, max_images=3, include_markdown=True)

    # Get completion
    try:
        answer = openai_client.get_completion(gpt_content)
    except Exception as e:
        print(f"❌ GPT-4o Vision failed: {e}")
        # Fallback: text only
        text_only_content = [c for c in gpt_content if c['type'] == 'text']
        answer = openai_client.get_completion(text_only_content)

    # Sanitize bold/italics
    def _sanitize_answer(text: str) -> str:
        if not isinstance(text, str):
            return text
        s = text.replace('**', '')
        s = s.replace('*', '')
        s = s.replace('_', '')
        return s
    answer_sanitized = _sanitize_answer(answer)

    print("\n--- AI Answer ---\n")
    print(answer_sanitized)

    if referenced_image_names:
        print("\n--- Referenced Images ---")
        for idx, name in enumerate(referenced_image_names, 1):
            print(f"Image {idx}: {name}")

    # Emit a UI-friendly JSON payload with answer, answer_html, and inline data URLs for images
    try:
        ui_images = []
        for (img_b64, mime, blob_name) in selected_images:
            ui_images.append({
                "name": blob_name,
                "mime": mime,
                "data_url": f"data:{mime};base64,{img_b64}"
            })
        imgs_html = "".join([f"<div><img src=\"{img['data_url']}\" alt=\"{img['name']}\" /></div>" for img in ui_images])
        answer_html = f"{answer_sanitized}\n\n{imgs_html}" if imgs_html else answer_sanitized
        ui_payload = {
            "answer": answer_sanitized,
            "answer_html": answer_html,
            "images": ui_images
        }
        print("\n--- UI Payload ---")
        print(json.dumps(ui_payload))
    except Exception as e:
        print(f"⚠️ Failed to emit UI payload: {e}")

    print("\n--- Retrieved Chunks ---")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c.get('id','unknown')} | Page {c.get('page','unknown')}")
        print(f"Content preview: {c.get('content','')[:150]}{'...' if len(c.get('content',''))>150 else ''}")
        if c.get('image_blob_urls'):
            print(f"Images: {c['image_blob_urls']}")
            if c.get('image_data'):
                print(f"Image Status: Fetched from blob storage")
        if c.get('bbox_annotation'):
            print(f"Contains annotation data")
        print()


if __name__ == "__main__":
    main()
