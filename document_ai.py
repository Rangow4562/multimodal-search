import os
import io
import json
import time
import base64
from dotenv import load_dotenv
from typing import Optional, List
import PyPDF2

from blob_utils import list_blob_files, download_pdf_from_blob

# Load environment variables
load_dotenv()

# Configurable chunk size (pages per API request)
PAGES_PER_CHUNK = int(os.getenv("MISTRAL_PAGES_PER_CHUNK", "10"))


def split_pdf_into_chunks(pdf_bytes: bytes, pages_per_chunk: int = 10) -> List[bytes]:
    """
    Split a PDF into chunks of `pages_per_chunk` pages each.
    Returns a list of PDF byte objects, one for each chunk.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    chunks = []

    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        writer = PyPDF2.PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        chunk_buffer = io.BytesIO()
        writer.write(chunk_buffer)
        chunk_buffer.seek(0)
        chunks.append(chunk_buffer.read())

    print(
        f"📄 Split PDF into {len(chunks)} chunks of up to {pages_per_chunk} pages each (total: {total_pages} pages)"
    )
    return chunks


def send_pdf_to_mistral(
    base64_pdf,
    api_key,
    endpoint_url,
    model_name,
    request_timeout: Optional[float] = None,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
):
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model_name,
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}",
        },
        "bbox_annotation_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "properties": {
                        "document_type": {
                            "title": "Document_Type",
                            "description": "The type of the image.",
                            "type": "string",
                        },
                        "short_description": {
                            "title": "Short_Description",
                            "description": "A description in English describing the image.",
                            "type": "string",
                        },
                        "summary": {
                            "title": "Summary",
                            "description": "Summarize the image.",
                            "type": "string",
                        },
                    },
                    "required": ["document_type", "short_description", "summary"],
                    "title": "BBOXAnnotation",
                    "type": "object",
                    "additionalProperties": False,
                },
                "name": "document_annotation",
                "strict": True,
            },
        },
        "include_image_base64": True,
    }

    retryable_statuses = {408, 429, 500, 502, 503, 504}
    attempt = 0
    while True:
        attempt += 1
        start_time = time.time()
        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=request_timeout or 120,
            )
            end_time = time.time()
            response.raise_for_status()
            result = response.json()
            print(
                f"✅ Processed in {end_time - start_time:.2f} seconds (attempt {attempt})"
            )
            return result
        except requests.RequestException as req_err:
            status = getattr(req_err.response, "status_code", None)
            response_body = None
            try:
                if req_err.response is not None:
                    response_body = req_err.response.text[:500]  # First 500 chars
            except Exception:
                pass

            if attempt <= max_retries and (
                status in retryable_statuses or status is None
            ):
                sleep_seconds = backoff_seconds * attempt
                print(
                    f"⚠️ Request failed with {status}. Retrying in {sleep_seconds:.1f}s (attempt {attempt}/{max_retries})..."
                )
                if response_body:
                    print(f"   Response: {response_body}")
                time.sleep(sleep_seconds)
                continue

            # Final failure - print full error details
            print(f"❌ Final failure: status={status}")
            if response_body:
                print(f"   Response body: {response_body}")
            raise


def merge_chunk_results(chunk_results: List[dict]) -> dict:
    """
    Merge results from multiple chunks into a single combined result.
    Combines pages from all chunks into a unified response.
    """
    if not chunk_results:
        return {}

    if len(chunk_results) == 1:
        return chunk_results[0]

    # Start with the first result as base
    merged = {
        "pages": [],
        "chunk_count": len(chunk_results),
    }

    page_offset = 0
    for idx, chunk in enumerate(chunk_results):
        if "pages" in chunk:
            for page in chunk["pages"]:
                # Adjust page index to reflect position in original document
                adjusted_page = page.copy()
                if "index" in adjusted_page:
                    adjusted_page["original_index"] = adjusted_page["index"]
                    adjusted_page["index"] = page_offset + adjusted_page["index"]
                adjusted_page["chunk_index"] = idx
                merged["pages"].append(adjusted_page)
            page_offset += len(chunk["pages"])

    # Copy any other top-level keys from the first chunk
    for key in chunk_results[0]:
        if key not in merged and key != "pages":
            merged[key] = chunk_results[0][key]

    return merged


def process_all_blobs():
    # Load configs
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME")
    api_key = os.getenv("MISTRAL_API_KEY")
    endpoint_url = os.getenv("MISTRAL_ENDPOINT")
    model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-document-ai-2505")
    output_dir = os.getenv("MISTRAL_OUTPUT_DIR", "document_ai_results")
    request_timeout = float(os.getenv("MISTRAL_REQUEST_TIMEOUT_SECONDS", "120"))
    max_retries = int(os.getenv("MISTRAL_MAX_RETRIES", "3"))
    backoff_seconds = float(os.getenv("MISTRAL_BACKOFF_SECONDS", "2"))
    pages_per_chunk = int(os.getenv("MISTRAL_PAGES_PER_CHUNK", "10"))

    # Validate required configuration
    missing_vars = []
    if not connection_string:
        missing_vars.append("AZURE_STORAGE_CONNECTION_STRING")
    if not container_name:
        missing_vars.append("CONTAINER_NAME")
    if not api_key:
        missing_vars.append("MISTRAL_API_KEY")
    if not endpoint_url:
        missing_vars.append("MISTRAL_ENDPOINT")
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file and retry.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all PDFs in the container
    pdf_blobs = list_blob_files(connection_string, container_name, suffix=".pdf")

    if not pdf_blobs:
        print("❌ No PDF files found in blob storage.")
        return

    print(f"📝 Found {len(pdf_blobs)} PDFs to process.")
    print(f"⚙️ Processing with {pages_per_chunk} pages per chunk")

    for blob_name in pdf_blobs:
        print(f"\n==== Processing: {blob_name} ====")
        try:
            # Skip if already processed
            base_name = os.path.splitext(os.path.basename(blob_name))[0]
            output_file = os.path.join(
                output_dir, f"{base_name}_document_ai_result.json"
            )
            if os.path.exists(output_file):
                print(
                    f"⏭️ Skipping {blob_name} (already processed - {output_file} exists)"
                )
                continue

            print(f"🔄 Downloading {blob_name}...")
            base64_pdf = download_pdf_from_blob(
                connection_string, container_name, blob_name
            )

            # Decode to raw bytes for splitting
            pdf_bytes = base64.b64decode(base64_pdf)

            # Split PDF into chunks
            pdf_chunks = split_pdf_into_chunks(pdf_bytes, pages_per_chunk)

            # Delay between chunks to avoid rate limiting
            chunk_delay = float(os.getenv("MISTRAL_CHUNK_DELAY_SECONDS", "3"))

            # Process each chunk
            chunk_results = []
            for i, chunk_bytes in enumerate(pdf_chunks):
                chunk_output_file = os.path.join(
                    output_dir, f"{base_name}_chunk_{i + 1}.json"
                )

                # Check if this chunk was already processed (resume support)
                if os.path.exists(chunk_output_file):
                    print(
                        f"\n⏭️ Chunk {i + 1}/{len(pdf_chunks)} already processed, loading from cache..."
                    )
                    with open(chunk_output_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    chunk_results.append(result)
                    continue

                # Add delay between chunks (except for first chunk)
                if i > 0:
                    print(
                        f"\n⏳ Waiting {chunk_delay}s before next chunk to avoid rate limiting..."
                    )
                    time.sleep(chunk_delay)

                print(f"\n📤 Processing chunk {i + 1}/{len(pdf_chunks)}...")
                chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                result = send_pdf_to_mistral(
                    chunk_base64,
                    api_key,
                    endpoint_url,
                    model_name,
                    request_timeout,
                    max_retries,
                    backoff_seconds,
                )
                chunk_results.append(result)

                # Save individual chunk result for debugging/recovery
                with open(chunk_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved chunk {i + 1} to: {chunk_output_file}")

            # Merge all chunk results
            merged_result = merge_chunk_results(chunk_results)

            # Save merged result
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged_result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Saved merged result to: {output_file}")

            # Cleanup: Delete chunk files after successful merge
            print(f"🧹 Cleaning up {len(pdf_chunks)} chunk files...")
            for i in range(len(pdf_chunks)):
                chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.json")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            print(f"✅ Chunk files deleted")

        except Exception as e:
            print(f"❌ Error processing {blob_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    process_all_blobs()
