<<<<<<< HEAD
# ather-multimodal-rag

A robust Multimodal Retrieval-Augmented Generation (RAG) pipeline designed for processing PDFs, extracting structured text, images, and layout annotations using Mistral Document AI, indexing vector embeddings in Azure AI Search, and querying using Azure OpenAI vision models.

## New Workflow (OCR JSON → Ingest → Query)

1) Prepare OCR JSON
- Place Mistral OCR JSON files in `./ocr_json/`.

2) Ingest JSON → Blob Storage + Vector DB
- Upload images with strict naming and index chunks with embeddings:

```bash
python ingest.py --ocr-dir ./ocr_json --container images --index $AZURE_SEARCH_INDEX_NAME
```

3) Validate/Clean Images (optional)
- Remove junk filenames:

```bash
python rag_query.py --clean-blobs
```

- Validate and repair base64-stored blobs:

```bash
python rag_query.py --validate-images --fix-invalid
```

4) Query

```bash
python rag_query.py
```

Notes:
- Images are stored as `<doc_id>_page{page}_chunk{chunk}_img-{n}.(jpg|png)` with correct content-type.
- Only `.jpg`/`.png` are indexed; missing blobs are skipped gracefully during ingestion and query.


## Overview

This pipeline processes PDFs stored in Azure Blob Storage through the following stages:

1. **PDF Discovery & Download** - Lists and downloads PDFs from Azure Blob Storage
2. **Document AI Processing** - Uses Mistral Document AI to extract text, images, and annotations
3. **Image Extraction** - Extracts images to separate blob storage for efficient indexing
4. **Chunking & Embedding** - Creates text chunks, generates embeddings, and indexes in Azure Cognitive Search
5. **Query & Retrieval** - Performs vector search and generates answers using Azure OpenAI (with vision model support)

## Architecture

```
Azure Blob Storage (PDFs) 
    ↓
Document AI (Mistral) → JSON with text + images + annotations
    ↓
Parser → Chunked text + image references
    ↓
Image Extractor → Images stored in blob storage
    ↓
Indexer → Azure Cognitive Search with vector embeddings
    ↓
Query Engine → RAG with vision model support
```

## Files

- **`blob_utils.py`** - Azure Blob Storage utilities for PDF discovery and download
- **`document_ai.py`** - Orchestrates PDF processing through Mistral Document AI
- **`parse_document.py`** - Parses Mistral output, creates chunks, and indexes in Azure Cognitive Search
- **`extract_images.py`** - Extracts images from chunks and stores them in blob storage
- **`rag_query.py`** - Interactive RAG query engine with vision model support
- **`test_rag_flow.py`** - Integration tests for the entire pipeline

## Setup

### Environment Variables
Create a `.env` file with the following variables:

```bash
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
CONTAINER_NAME=your_pdf_container

# Mistral Document AI
MISTRAL_API_KEY=your_api_key
MISTRAL_ENDPOINT=your_endpoint_url
MISTRAL_MODEL_NAME=mistral-document-ai-2505

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_ADMIN_KEY=your_admin_key
AZURE_SEARCH_INDEX_NAME=documents-index
EMBEDDING_DIM=1536

# Azure OpenAI (via OpenAI SDK)
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
AZURE_OPENAI_COMPLETION_DEPLOYMENT=your_vision_completion_deployment
```

### Dependencies

```bash
pip install azure-storage-blob azure-search-documents openai azure-identity python-dotenv requests
```

## Usage

### 1. Process PDFs with Document AI

```bash
python document_ai.py
```

This will:
- List all PDFs in your blob container
- Process each PDF through Mistral Document AI
- Save results as `*_document_ai_result.json`

### 2. Parse and Create Chunks

```bash
python parse_document.py
```

**First run only**: Uncomment the `create_index()` call to create the Azure Cognitive Search index.

This will:
- Parse Mistral output into paragraph-level chunks
- Generate embeddings for each chunk
- Index chunks in Azure Cognitive Search

### 3. Extract Images to Blob Storage

```bash
python extract_images.py
```

This will:
- Extract images from parsed chunks
- Store images in a separate `images` blob container
- Update chunk files with image blob references
- Store metadata for each image

### 4. Re-index with Image References

```bash
python parse_document.py --reindex
```

This will:
- Load updated chunks with image blob references
- Re-index in Azure Cognitive Search
- Include `image_blob_url` field for image retrieval

### 5. Query with Images

```bash
python rag_query.py
```

This will:
- Accept your question
- Perform vector search to retrieve relevant chunks
- Fetch images from blob storage if available
- Use vision model to generate comprehensive answers

## Image Handling

### Storage Strategy

Instead of storing large base64 images in the search index (which can cause size limits and performance issues), this pipeline:

1. **Extracts images** from PDF chunks to a separate `images` blob container
2. **Stores references** (`image_blob_url`) in the search index
3. **Fetches images on-demand** when needed for vision model queries
4. **Maintains metadata** including bounding box annotations and source information

### Image Metadata

Each image is stored with:
- **Image file**: `page_{page}_{chunk_id}.png`
- **Metadata file**: `page_{page}_{chunk_id}.png.metadata.json`
- **Content**: Original chunk ID, page number, bbox annotations, source PDF

## Vision Model Integration

The pipeline automatically detects when images are available and:

- **Text-only queries**: Uses standard completion model
- **Image-containing queries**: Switches to vision model with image content
- **Hybrid responses**: Combines text analysis with visual understanding

## Testing

Run the integration test to verify your setup:

```bash
python test_rag_flow.py
```

This will check:
- Environment variables
- File structure
- Azure service connectivity
- Pipeline readiness

## Troubleshooting

### Common Issues

1. **Index creation fails**: Ensure you have admin access to Azure Cognitive Search
2. **Image upload fails**: Check blob storage connection string and permissions
3. **Vision model errors**: Verify your Azure OpenAI deployment supports vision
4. **Missing chunks**: Ensure you've run the parser before indexing

### Performance Tips

- **Batch processing**: Images are processed in batches for efficiency
- **Selective indexing**: Only index fields you need for search
- **Vector optimization**: Uses HNSW algorithm for fast similarity search
- **Blob references**: Avoid duplicating large image data

## Example Queries

### Text-only
```
"What are the key features of the product?"
```

### Image-aware
```
"Show me the diagram on page 3 and explain what it illustrates"
"What does the chart in the document show about performance?"
```

## Security Notes

- Uses Azure Managed Identity where possible
- Stores sensitive data in Azure Key Vault (recommended)
- Images are stored in private blob containers
- Search index access is controlled via admin keys

## Scaling Considerations

- **Large PDFs**: Process in smaller batches
- **Many images**: Consider image compression or thumbnail generation
- **High query volume**: Implement caching for frequently accessed images
- **Storage costs**: Monitor blob storage usage and implement lifecycle policies

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update this README for any changes
4. Ensure all environment variables are documented

## License

This project is provided as-is for development purposes.
=======
# multimodal-search
Document processing and multimodal RAG pipeline using Azure AI Search &amp; OpenAI Vision.
>>>>>>> multi/main
