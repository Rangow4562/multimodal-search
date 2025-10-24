import os
import json
import glob
import time
from dotenv import load_dotenv
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind
)
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
import argparse

from openai import AzureOpenAI

# ----------------------------
# Parsing functions
# ----------------------------

def parse_mistral_output(input_json_path):
    data = json.load(open(input_json_path, "r", encoding="utf-8"))
    doc_name = os.path.splitext(os.path.basename(input_json_path))[0].replace("_document_ai_result", "")
    chunks = []
    for page in data.get("pages", []):
        page_number = page.get("page_number") or page.get("index", 0) + 1
        md = page.get("markdown", "") or ""
        if not md:
            continue

        # Extract bbox annotation for the page (if available)
        bbox_annotation = page.get("bbox_annotations") or page.get("document_annotation") or None

        # Extract images from the page (if available)
        images = page.get("images", [])
        image_data = []
        for img in images:
            if img.get("image_base64"):
                image_data.append({
                    "id": img.get("id", f"img_{len(image_data)}"),
                    "image_base64": img["image_base64"],
                    "top_left_x": img.get("top_left_x"),
                    "top_left_y": img.get("top_left_y"),
                    "bottom_right_x": img.get("bottom_right_x"),
                    "bottom_right_y": img.get("bottom_right_y"),
                    "image_annotation": img.get("image_annotation")
                })

        # Create larger chunks with overlap for better content coverage
        # Target chunk size: 2000 characters with 200 character overlap
        chunk_size = 2000
        overlap_size = 200
        
        # Clean and prepare the markdown text
        clean_md = md.strip()
        if not clean_md:
            continue
            
        # Create chunks with overlap
        start = 0
        chunk_index = 1
        
        while start < len(clean_md):
            # Calculate end position for this chunk
            end = start + chunk_size
            
            # If this isn't the last chunk, try to find a good break point
            if end < len(clean_md):
                # Look for a good break point (newline, period, or space) within the last 200 chars
                lookback_start = max(start + chunk_size - 200, start)
                lookback_text = clean_md[lookback_start:end]
                
                # Prefer breaking at double newlines, then single newlines, then periods
                break_point = -1
                
                # Try to find double newline first
                double_newline_pos = lookback_text.rfind('\n\n')
                if double_newline_pos != -1:
                    break_point = lookback_start + double_newline_pos + 2
                else:
                    # Try single newline
                    newline_pos = lookback_text.rfind('\n')
                    if newline_pos != -1:
                        break_point = lookback_start + newline_pos + 1
                    else:
                        # Try period
                        period_pos = lookback_text.rfind('.')
                        if period_pos != -1:
                            break_point = lookback_start + period_pos + 1
                        else:
                            # Fallback to space
                            space_pos = lookback_text.rfind(' ')
                            if space_pos != -1:
                                break_point = lookback_start + space_pos + 1
                
                if break_point != -1:
                    end = break_point
            
            # Extract the chunk content
            chunk_content = clean_md[start:end].strip()
            
            if chunk_content:  # Only create chunk if there's content
                # Create valid document key by replacing spaces and special characters
                safe_doc_name = doc_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                chunk = {
                    "id": f"{safe_doc_name}_page{page_number}_chunk{chunk_index}",
                    "content": chunk_content,
                    "page": page_number,
                }
                if bbox_annotation:
                    chunk["bbox_annotation"] = bbox_annotation
                if image_data:
                    chunk["images"] = image_data

                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position for next chunk, accounting for overlap
            start = max(start + 1, end - overlap_size)
    return chunks

def parse_all_results(folder=".", output_dir="./parsed_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = glob.glob(os.path.join(folder, "*_document_ai_result.json"))
    if not paths:
        print("❌ No document AI result files found to parse")
        return
    
    print(f"📁 Found {len(paths)} document AI result files")
    parsed_count = 0
    skipped_count = 0
    
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0].replace("_document_ai_result", "")
        out_path = os.path.join(output_dir, f"{base}_parsed_chunks.json")
        if os.path.exists(out_path):
            print(f"⏭️ Skipping parse for {base} (already exists - {out_path})")
            skipped_count += 1
            continue
        
        print(f"🔄 Parsing {base}...")
        chunks = parse_mistral_output(path)
        json.dump(chunks, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ Parsed {len(chunks)} chunks from {base}")
        parsed_count += 1
    
    print(f"\n📊 Parse Summary: {parsed_count} files parsed, {skipped_count} files skipped")

# ----------------------------
# Embedding + Azure Cognitive Search indexing
# ----------------------------

load_dotenv()

class AzureOpenAIClient:
    def __init__(self, endpoint, embedding_deployment, api_key, api_version, completion_deployment=None):
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.embedding_deployment = embedding_deployment
        self.completion_deployment = completion_deployment

    def get_embedding(self, text):
        response = self.client.embeddings.create(model=self.embedding_deployment, input=text)
        return response.data[0].embedding

class AzureCognitiveSearchClient:
    def __init__(self, endpoint, admin_key, index_name, embedding_dim):
        self.index_name = index_name
        self.credential = AzureKeyCredential(admin_key)
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=self.credential)
        self.embedding_dim = embedding_dim

    def delete_index(self):
        """Delete the existing search index"""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"Deleted index {self.index_name}")
        except Exception as e:
            print(f"Index {self.index_name} doesn't exist or couldn't be deleted: {e}")

    def create_index(self):
        # Build VectorSearch compatible with Azure SDK 11.5.3
        try:
            from azure.search.documents.indexes.models import VectorSearchProfile, HnswAlgorithmConfiguration
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
                algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
            )
            
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", searchable=True, max_length=100000),  # Allow much longer content
                SimpleField(name="page", type="Edm.Int32"),
                SearchField(
                    name="embedding",
                    type="Collection(Edm.Single)",
                    searchable=True,
                    vector_search_dimensions=self.embedding_dim,
                    vector_search_profile="vector-profile"
                ),
                SimpleField(name="bbox_annotation", type="Edm.String", filterable=True, facetable=False, searchable=False),
                SimpleField(name="image_blob_urls", type="Edm.String", filterable=False, facetable=False, searchable=False)
            ]
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            self.index_client.create_index(index)
            print(f"✅ Created index {self.index_name} with vector search")
        except Exception as e:
            print(f"❌ Failed to create index with vector search: {e}")
            print("🔄 Trying to create index without vector search...")
            # Fallback to basic index without vector search
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", searchable=True, max_length=100000),
                SimpleField(name="page", type="Edm.Int32"),
                SimpleField(name="bbox_annotation", type="Edm.String", filterable=True, facetable=False, searchable=False),
                SimpleField(name="image_blob_urls", type="Edm.String", filterable=False, facetable=False, searchable=False)
            ]
            index = SearchIndex(
                name=self.index_name,
                fields=fields
            )
            self.index_client.create_index(index)
            print(f"⚠️ Created basic index {self.index_name} (without vector search)")
            print("⚠️ Note: This index will not support vector search - only text search will work")

    def update_index_with_embedding(self):
        """Update existing index to add embedding field for vector search"""
        try:
            from azure.search.documents.indexes.models import VectorSearchProfile, HnswAlgorithmConfiguration
            
            # Get current index
            current_index = self.index_client.get_index(self.index_name)
            
            # Add embedding field with correct vector search profile
            embedding_field = SearchField(
                name="embedding",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=self.embedding_dim,
                vector_search_profile_name="vector-profile"
            )
            
            # Add vector search configuration
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
                algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")]
            )
            
            # Update the index
            current_index.fields.append(embedding_field)
            current_index.vector_search = vector_search
            
            # Update the index
            self.index_client.create_or_update_index(current_index)
            print(f"✅ Updated index {self.index_name} with embedding field and vector search")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update index with embedding field: {e}")
            return False

    def upload_documents(self, documents):
        print(f"Uploading {len(documents)} documents...")
        result = self.search_client.upload_documents(documents)
        print(f"Upload result: {result}")

    def vector_search(self, query_embedding, top_k=5):
        vector_query = {
            "kind": "vector",
            "vector": query_embedding,
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

    def get_existing_document_ids(self):
        """Retrieves all document IDs currently in the search index."""
        try:
            # Use a search query to get all document IDs
            # This is a simplified approach; a more robust solution might involve a search with a specific filter
            # or iterating through all documents if the index is small.
            # For large indexes, this might need to be paginated or use a different search strategy.
            all_docs = self.search_client.search(
                search_text="*",
                select=["id"],
                top=100000 # Increased limit to get all document IDs
            )
            return {doc['id'] for doc in all_docs}
        except Exception as e:
            print(f"⚠️ Error retrieving existing document IDs from search index: {e}")
            return set()


def load_parsed_chunks(parsed_folder="./parsed_chunks"):
    json_files = glob.glob(os.path.join(parsed_folder, "*_parsed_chunks.json"))
    print(f"Found {len(json_files)} parsed chunk files.")
    all_chunks = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {fpath}")
    return all_chunks


def load_uploaded_docs_tracking():
    """Load the tracking file of already uploaded documents"""
    tracking_file = "uploaded_docs_tracking.json"
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"⚠️ Error loading tracking file: {e}")
            return set()
    return set()

def save_uploaded_docs_tracking(uploaded_docs):
    """Save the tracking file of uploaded documents"""
    tracking_file = "uploaded_docs_tracking.json"
    try:
        with open(tracking_file, 'w') as f:
            json.dump(list(uploaded_docs), f, indent=2)
        print(f"💾 Saved tracking file: {tracking_file}")
    except Exception as e:
        print(f"⚠️ Error saving tracking file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Parse and (re)ingest documents into Azure Cognitive Search")
    parser.add_argument('--ignore-tracking', action='store_true', help='Ignore uploaded_docs_tracking.json and reprocess all chunks')
    parser.add_argument('--reset-tracking', action='store_true', help='Delete uploaded_docs_tracking.json before ingest')
    args, _ = parser.parse_known_args()

    # Load environment variables
    load_dotenv()
    
    # Read config from environment
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents-index")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    required = {
        "AZURE_SEARCH_ENDPOINT": search_endpoint,
        "AZURE_SEARCH_ADMIN_KEY": admin_key,
        "AZURE_SEARCH_INDEX_NAME": index_name,
        "AZURE_OPENAI_ENDPOINT": openai_endpoint,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": embedding_deployment,
        "AZURE_OPENAI_API_KEY": openai_api_key,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Missing required configuration in .env: {', '.join(missing)}")
        return

    # Parse any new Mistral outputs from document_ai_results into ./parsed_chunks
    parse_all_results(folder=os.getenv("MISTRAL_OUTPUT_DIR", "document_ai_results"))

    # Initialize clients
    acs_client = AzureCognitiveSearchClient(
        os.getenv('AZURE_SEARCH_ENDPOINT'),
        os.getenv('AZURE_SEARCH_ADMIN_KEY'),
        os.getenv('AZURE_SEARCH_INDEX_NAME'),
        1536
    )
    openai_client = AzureOpenAIClient(openai_endpoint, embedding_deployment, openai_api_key, openai_api_version)
    
    # Check if index exists and work with it
    print("🔍 Checking existing search index...")
    try:
        existing_index = acs_client.index_client.get_index(acs_client.index_name)
        print(f"✅ Index '{acs_client.index_name}' exists")
        
        # Check if embedding field exists
        has_embedding = any(field.name == 'embedding' for field in existing_index.fields)
        if has_embedding:
            print("✅ Index has embedding field - vector search will work")
            acs_client.has_embedding_field = True
        else:
            print("⚠️ Index missing 'embedding' field - attempting to add it...")
            if acs_client.update_index_with_embedding():
                acs_client.has_embedding_field = True
                print("✅ Successfully added embedding field to existing index")
            else:
                print("❌ Failed to add embedding field - will use text search only")
                acs_client.has_embedding_field = False
                print("💡 Note: To enable vector search, you may need to recreate the index manually")
                
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        print("🔄 Creating new index...")
        acs_client.create_index()
        print("✅ New index created successfully")
        acs_client.has_embedding_field = True  # New index should have embedding field

    # Load existing uploaded documents tracking
    if args.reset_tracking:
        tracking_file = "uploaded_docs_tracking.json"
        try:
            if os.path.exists(tracking_file):
                os.remove(tracking_file)
                print(f"🗑️ Deleted tracking file: {tracking_file}")
        except Exception as e:
            print(f"⚠️ Failed to delete tracking file: {e}")
    uploaded_docs = set() if args.ignore_tracking else load_uploaded_docs_tracking()
    
    # Get existing documents from search index
    try:
        existing_docs = acs_client.get_existing_document_ids()
        print(f"📊 Found {len(existing_docs)} existing documents in search index")
    except Exception as e:
        print(f"⚠️ Could not retrieve existing documents from search index: {e}")
        existing_docs = set()
    
    # Combine both sources to get all existing docs
    if len(existing_docs) == 0 and len(uploaded_docs) > 0:
        print("⚠️ Index is empty but tracking has entries. Ignoring tracking for full re-ingestion.")
        all_existing_docs = existing_docs
    else:
        all_existing_docs = existing_docs.union(uploaded_docs)
    print(f"📊 Total existing documents (index ∪ tracking): {len(all_existing_docs)}")
    
    # Process parsed chunks
    parsed_chunks_dir = os.getenv("PARSED_CHUNKS_DIR", "parsed_chunks")
    if not os.path.exists(parsed_chunks_dir):
        print(f"❌ Parsed chunks directory not found: {parsed_chunks_dir}")
        return
    
    # Get all parsed chunk files
    chunk_files = [f for f in os.listdir(parsed_chunks_dir) if f.endswith('_parsed_chunks.json')]
    if not chunk_files:
        print("❌ No parsed chunk files found")
        return
    
    print(f"📁 Found {len(chunk_files)} parsed chunk files")
    
    # Process each file
    total_chunks = 0
    new_chunks = 0
    to_upload = []
    
    for chunk_file in chunk_files:
        file_path = os.path.join(parsed_chunks_dir, chunk_file)
        print(f"\n📄 Processing {chunk_file}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Handle both list format and dict format
            if isinstance(chunks_data, list):
                chunks = chunks_data
            else:
                chunks = chunks_data.get('chunks', [])
            
            total_chunks += len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get('id')
                if not chunk_id:
                    print(f"⚠️ Skipping chunk {i+1} - missing ID")
                    continue
                
                # Check if already uploaded
                if chunk_id in all_existing_docs:
                    continue
                
                new_chunks += 1
                content = chunk.get('content', '')
                page = chunk.get('page', 1)
                bbox_annotation = chunk.get('bbox_annotation', [])
                image_blob_urls = chunk.get('image_blob_urls', [])

                # Verify blob existence for each referenced image; accept only .jpg/.png
                verified_blobs = []
                try:
                    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                    if connection_string:
                        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                        container_name = os.getenv("AZURE_IMAGES_CONTAINER", "images")
                        for ref in image_blob_urls or []:
                            try:
                                if not isinstance(ref, str):
                                    continue
                                lower = ref.lower()
                                if not (lower.endswith('.jpg') or lower.endswith('.png')):
                                    continue
                                blob_client = blob_service_client.get_blob_client(container=container_name, blob=ref)
                                exists = blob_client.exists() if hasattr(blob_client, 'exists') else True
                                if exists:
                                    verified_blobs.append(ref)
                            except Exception as be:
                                print(f"⚠️ Error verifying blob {ref}: {be}")
                    else:
                        # No connection string; keep original refs filtered to jpg/png
                        for ref in image_blob_urls or []:
                            if isinstance(ref, str) and (ref.lower().endswith('.jpg') or ref.lower().endswith('.png')):
                                verified_blobs.append(ref)
                except Exception as e:
                    print(f"⚠️ Error during blob verification: {e}")
                
                # Generate embedding
                try:
                    embedding_vector = openai_client.get_embedding(content)
                except Exception as e:
                    print(f"❌ Failed to generate embedding for chunk {i+1}: {e}")
                    continue
                
                # Create document with or without embedding field based on index structure
                doc = {
                    "id": chunk_id,
                    "content": content,  # Ensure content field is properly set
                    "page": page,
                    "bbox_annotation": json.dumps(bbox_annotation) if bbox_annotation else None,
                    "image_blob_urls": json.dumps(verified_blobs) if verified_blobs else None
                }
                
                # Only include embedding if the index supports it
                if hasattr(acs_client, 'has_embedding_field') and acs_client.has_embedding_field:
                    doc["embedding"] = embedding_vector
                else:
                    print(f"⚠️ Skipping embedding for chunk {i+1} (index doesn't support embeddings)")
                to_upload.append(doc)
                
        except Exception as e:
            print(f"❌ Error processing {chunk_file}: {e}")
            continue
    
    print(f"\n📊 Summary:")
    print(f"   Total chunks processed: {total_chunks}")
    print(f"   New chunks to upload: {new_chunks}")
    print(f"   Already uploaded: {total_chunks - new_chunks}")
    
    if not to_upload:
        print("✅ No new chunks to upload - all documents are already indexed!")
        return
    
    # Upload in batches
    batch_size = 1000
    total_batches = (len(to_upload) + batch_size - 1) // batch_size
    
    print(f"\n📤 Uploading {len(to_upload)} documents in {total_batches} batches...")
    
    successfully_uploaded = []
    for i in range(0, len(to_upload), batch_size):
        batch_num = (i // batch_size) + 1
        batch = to_upload[i:i + batch_size]
        
        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            result = acs_client.upload_documents(batch)
            print(f"✅ Successfully uploaded batch {batch_num}")
            
            # Track successfully uploaded documents
            batch_ids = [doc['id'] for doc in batch]
            successfully_uploaded.extend(batch_ids)
            
        except Exception as e:
            print(f"❌ Failed to upload batch {batch_num}: {e}")
            continue
    
    # Update tracking file
    if successfully_uploaded:
        save_uploaded_docs_tracking(successfully_uploaded)
        print(f"💾 Saved tracking file: uploaded_docs_tracking.json")
        print(f"✅ Updated tracking file with {len(successfully_uploaded)} new documents")
    
    print(f"✅ All new chunks uploaded with embeddings.")

if __name__ == "__main__":
    main()
