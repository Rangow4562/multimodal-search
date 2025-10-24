# blob_utils.py
from azure.storage.blob import BlobServiceClient
import os
import base64

def list_blob_files(connection_string, container_name, suffix=".pdf"):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    return [blob.name for blob in container_client.list_blobs() if blob.name.endswith(suffix)]

def process_all_pdfs():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("CONTAINER_NAME")
    pdf_blobs = list_blob_files(conn_str, container, suffix=".pdf")
    for blob_name in pdf_blobs:
        print(f"Processing: {blob_name}")
        # Now download and process blob_name as needed
        base64_pdf = download_pdf_from_blob(conn_str, container, blob_name)
        # Call document_ai, embed, etc...


def download_pdf_from_blob(connection_string, container_name, blob_name):
    service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = service.get_blob_client(container=container_name, blob=blob_name)
    data = blob_client.download_blob().readall()
    return base64.b64encode(data).decode("utf-8")

if __name__ == "__main__":
    process_all_pdfs()


