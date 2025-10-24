import os
import json
import time
from dotenv import load_dotenv
from typing import Optional

from blob_utils import list_blob_files, download_pdf_from_blob

# Load environment variables
load_dotenv()

def send_pdf_to_mistral(base64_pdf, api_key, endpoint_url, model_name, request_timeout: Optional[float] = None, max_retries: int = 3, backoff_seconds: float = 2.0):
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
                        "document_type": {"title": "Document_Type", "description": "The type of the image.", "type": "string"},
                        "short_description": {"title": "Short_Description", "description": "A description in English describing the image.", "type": "string"},
                        "summary": {"title": "Summary", "description": "Summarize the image.", "type": "string"},
                    },
                    "required": ["document_type", "short_description", "summary"],
                    "title": "BBOXAnnotation",
                    "type": "object",
                    "additionalProperties": False,
                },
                "name": "document_annotation",
                "strict": True,
            }
        },
        "include_image_base64": True,
    }

    retryable_statuses = {408, 429, 500, 502, 503, 504}
    attempt = 0
    while True:
        attempt += 1
        start_time = time.time()
        try:
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=request_timeout or 60)
            end_time = time.time()
            response.raise_for_status()
            result = response.json()
            print(f"✅ Processed in {end_time - start_time:.2f} seconds (attempt {attempt})")
            return result
        except requests.RequestException as req_err:
            status = getattr(req_err.response, "status_code", None)
            if attempt <= max_retries and (status in retryable_statuses or status is None):
                sleep_seconds = backoff_seconds * attempt
                print(f"⚠️ Request failed with {status}. Retrying in {sleep_seconds:.1f}s (attempt {attempt}/{max_retries})...")
                time.sleep(sleep_seconds)
                continue
            raise

def process_all_blobs():
    # Load configs
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME")
    api_key = os.getenv("MISTRAL_API_KEY")
    endpoint_url = os.getenv("MISTRAL_ENDPOINT")
    model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-document-ai-2505")
    output_dir = os.getenv("MISTRAL_OUTPUT_DIR", "document_ai_results")
    request_timeout = float(os.getenv("MISTRAL_REQUEST_TIMEOUT_SECONDS", "60"))
    max_retries = int(os.getenv("MISTRAL_MAX_RETRIES", "3"))
    backoff_seconds = float(os.getenv("MISTRAL_BACKOFF_SECONDS", "2"))
    
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
    for blob_name in pdf_blobs:
        print(f"\n==== Processing: {blob_name} ====")
        try:
            # Skip if already processed
            base_name = os.path.splitext(os.path.basename(blob_name))[0]
            output_file = os.path.join(output_dir, f"{base_name}_document_ai_result.json")
            if os.path.exists(output_file):
                print(f"⏭️ Skipping {blob_name} (already processed - {output_file} exists)")
                continue

            print(f"🔄 Processing {blob_name}...")
            base64_pdf = download_pdf_from_blob(connection_string, container_name, blob_name)
            result = send_pdf_to_mistral(base64_pdf, api_key, endpoint_url, model_name, request_timeout, max_retries, backoff_seconds)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {blob_name}: {e}")

if __name__ == "__main__":
    process_all_blobs()
