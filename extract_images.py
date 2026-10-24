import os
import json
import base64
import glob
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from io import BytesIO

try:
    from PIL import Image
except Exception:
    Image = None


def _safe_doc_name_from_chunks_file(chunk_file_path: str) -> str:
    base = os.path.basename(chunk_file_path)
    if base.endswith("_parsed_chunks.json"):
        base = base[: -len("_parsed_chunks.json")]
    # Normalize to safe token
    safe = base.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    return safe


def _extract_chunk_number(chunk_id: str) -> int:
    # chunk_id format: <safe_doc_name>_page{page}_chunk{chunk_index}
    try:
        after = chunk_id.split("_chunk", 1)[1]
        num = ""
        for ch in after:
            if ch.isdigit():
                num += ch
            else:
                break
        return int(num) if num else 1
    except Exception:
        return 1


load_dotenv()


def extract_images_from_chunks_to_blob_storage(
    parsed_folder="./parsed_chunks", container_name="aiassist-bemsys-index-v2"
):
    """
    Extract images from parsed chunks and store them in Azure Blob Storage.
    This should be run after document_ai.py and before parse_document.py.

    Args:
        parsed_folder: Folder containing parsed chunk JSON files
        container_name: Blob container name for storing images

    Returns:
        Number of images successfully stored
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment")
        return 0

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Ensure container exists
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.get_container_properties()
        print(f"✅ Using existing container: {container_name}")
    except:
        print(f"Creating container: {container_name}")
        blob_service_client.create_container(container_name)

    # Find all parsed chunk files
    chunk_files = glob.glob(os.path.join(parsed_folder, "*_parsed_chunks.json"))
    if not chunk_files:
        print(f"❌ No parsed chunk files found in {parsed_folder}")
        return 0

    print(f"📁 Found {len(chunk_files)} parsed chunk files")

    total_images_stored = 0

    for chunk_file in chunk_files:
        print(f"\n📄 Processing: {os.path.basename(chunk_file)}")

        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            images_in_file = 0
            file_has_images = False

            safe_doc_name = _safe_doc_name_from_chunks_file(chunk_file)

            for chunk in chunks:
                images = chunk.get("images", [])
                if not images:
                    continue

                file_has_images = True
                chunk_id = chunk.get("id", "unknown")
                page = chunk.get("page", 0)
                bbox_annotation = chunk.get("bbox_annotation")
                chunk_num = _extract_chunk_number(chunk_id)

                for img_idx, img in enumerate(images):
                    image_base64 = img.get("image_base64")
                    if not image_base64:
                        continue

                    try:
                        # Required naming: <doc_name>_page{page}_chunk{chunk}_img-{n}.<ext>
                        img_seq = img_idx

                        # Decode base64 to binary and normalize to a clean image using Pillow
                        b64_str = image_base64.strip()
                        if b64_str.startswith("data:") and ";base64," in b64_str:
                            b64_str = b64_str.split(";base64,", 1)[1]

                        try:
                            raw_bytes = base64.b64decode(b64_str, validate=True)
                        except Exception:
                            # Try lenient decode if validate raises
                            raw_bytes = base64.b64decode(b64_str, validate=False)
                        if len(raw_bytes) == 0:
                            raise ValueError("decoded image is empty")

                        if Image is None:
                            raise RuntimeError(
                                "Pillow is required to process and normalize images"
                            )

                        try:
                            pil_img_ctx = Image.open(BytesIO(raw_bytes))
                        except Exception as open_err:
                            first20 = raw_bytes[:20].hex()
                            print(
                                f"  ⚠️ Cannot identify image for {chunk_id} img {img_idx}: {open_err}; first20={first20}; decoded_size={len(raw_bytes)}"
                            )
                            continue

                        with pil_img_ctx as pil_img:
                            # Convert mode where necessary
                            if pil_img.mode not in ("RGB", "L"):
                                pil_img = pil_img.convert("RGB")

                            fmt = pil_img.format.lower() if pil_img.format else "jpeg"
                            if fmt not in ("jpeg", "png"):
                                # Prefer saving as PNG for non-jpeg/png
                                fmt = "png"

                            out = BytesIO()
                            if fmt == "jpeg":
                                pil_img.save(
                                    out, format="JPEG", quality=90, optimize=True
                                )
                                content_type = "image/jpeg"
                                filename = f"{safe_doc_name}_page{page}_chunk{chunk_num}_img-{img_seq}.jpg"
                            else:
                                pil_img.save(out, format="PNG", optimize=True)
                                content_type = "image/png"
                                filename = f"{safe_doc_name}_page{page}_chunk{chunk_num}_img-{img_seq}.png"

                            image_bytes = out.getvalue()

                        # Check if image already exists in blob storage (now that filename is known)
                        blob_client = blob_service_client.get_blob_client(
                            container=container_name, blob=filename
                        )
                        try:
                            blob_client.get_blob_properties()
                            blob_url = blob_client.url
                            print(f"  ⏭️ Skipping image: {filename} (already exists)")
                            if "image_blob_urls" not in chunk:
                                chunk["image_blob_urls"] = []
                            if blob_url not in chunk["image_blob_urls"]:
                                chunk["image_blob_urls"].append(blob_url)
                            continue
                        except Exception:
                            # Not existing; proceed to upload
                            pass

                        # Upload normalized single image to blob storage with proper content type
                        blob_client = blob_service_client.get_blob_client(
                            container=container_name, blob=filename
                        )
                        blob_client.upload_blob(
                            image_bytes, overwrite=True, content_type=content_type
                        )

                        # Update the chunk to include the full blob URL
                        blob_url = blob_client.url
                        if "image_blob_urls" not in chunk:
                            chunk["image_blob_urls"] = []
                        chunk["image_blob_urls"].append(blob_url)

                        images_in_file += 1
                        print(
                            f"  ✅ Stored image: {filename} ({content_type}, {len(image_bytes)} bytes)"
                        )

                    except Exception as e:
                        print(
                            f"  ❌ Error storing image for chunk {chunk_id}, image {img_idx}: {e}"
                        )

            # Save updated chunks with blob URLs if any images were processed
            if file_has_images:
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                print(f"  💾 Updated {chunk_file} with image references")

            total_images_stored += images_in_file

        except Exception as e:
            print(f"❌ Error processing {chunk_file}: {e}")

    print(f"\n🎯 Total images stored: {total_images_stored}")
    return total_images_stored


def extract_images_to_local(
    parsed_folder="./parsed_chunks", output_folder="./extracted_images"
):
    """
    Extract images from parsed chunks and save them to a local folder.
    This is useful for local testing or when blob storage is not needed.

    Args:
        parsed_folder: Folder containing parsed chunk JSON files
        output_folder: Local folder to save extracted images

    Returns:
        Number of images successfully saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder: {output_folder}")

    # Find all parsed chunk files
    chunk_files = glob.glob(os.path.join(parsed_folder, "*_parsed_chunks.json"))
    if not chunk_files:
        print(f"❌ No parsed chunk files found in {parsed_folder}")
        return 0

    print(f"📁 Found {len(chunk_files)} parsed chunk files")

    total_images_saved = 0

    for chunk_file in chunk_files:
        print(f"\n📄 Processing: {os.path.basename(chunk_file)}")

        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            images_in_file = 0
            safe_doc_name = _safe_doc_name_from_chunks_file(chunk_file)

            # Create a subfolder for each document
            doc_folder = os.path.join(output_folder, safe_doc_name)
            os.makedirs(doc_folder, exist_ok=True)

            for chunk in chunks:
                images = chunk.get("images", [])
                if not images:
                    continue

                chunk_id = chunk.get("id", "unknown")
                page = chunk.get("page", 0)
                chunk_num = _extract_chunk_number(chunk_id)

                for img_idx, img in enumerate(images):
                    image_base64 = img.get("image_base64")
                    if not image_base64:
                        continue

                    try:
                        img_seq = img_idx

                        # Decode base64 to binary
                        b64_str = image_base64.strip()
                        if b64_str.startswith("data:") and ";base64," in b64_str:
                            b64_str = b64_str.split(";base64,", 1)[1]

                        try:
                            raw_bytes = base64.b64decode(b64_str, validate=True)
                        except Exception:
                            raw_bytes = base64.b64decode(b64_str, validate=False)

                        if len(raw_bytes) == 0:
                            raise ValueError("decoded image is empty")

                        if Image is None:
                            raise RuntimeError(
                                "Pillow is required to process and normalize images"
                            )

                        try:
                            pil_img_ctx = Image.open(BytesIO(raw_bytes))
                        except Exception as open_err:
                            first20 = raw_bytes[:20].hex()
                            print(
                                f"  ⚠️ Cannot identify image for {chunk_id} img {img_idx}: {open_err}; first20={first20}"
                            )
                            continue

                        with pil_img_ctx as pil_img:
                            # Convert mode where necessary
                            if pil_img.mode not in ("RGB", "L"):
                                pil_img = pil_img.convert("RGB")

                            fmt = pil_img.format.lower() if pil_img.format else "jpeg"
                            if fmt not in ("jpeg", "png"):
                                fmt = "png"

                            # Determine filename
                            if fmt == "jpeg":
                                filename = (
                                    f"page{page}_chunk{chunk_num}_img-{img_seq}.jpg"
                                )
                            else:
                                filename = (
                                    f"page{page}_chunk{chunk_num}_img-{img_seq}.png"
                                )

                            filepath = os.path.join(doc_folder, filename)

                            # Check if file already exists
                            if os.path.exists(filepath):
                                print(f"  ⏭️ Skipping: {filename} (already exists)")
                                continue

                            # Save to local file
                            if fmt == "jpeg":
                                pil_img.save(
                                    filepath, format="JPEG", quality=90, optimize=True
                                )
                            else:
                                pil_img.save(filepath, format="PNG", optimize=True)

                            file_size = os.path.getsize(filepath)
                            images_in_file += 1
                            print(f"  ✅ Saved: {filename} ({file_size} bytes)")

                    except Exception as e:
                        print(
                            f"  ❌ Error saving image for chunk {chunk_id}, image {img_idx}: {e}"
                        )

            total_images_saved += images_in_file

        except Exception as e:
            print(f"❌ Error processing {chunk_file}: {e}")

    print(f"\n🎯 Total images saved locally: {total_images_saved}")
    print(f"📁 Images saved to: {os.path.abspath(output_folder)}")
    return total_images_saved


def main():
    """
    Main function to extract images from parsed chunks.
    Supports both local storage and Azure Blob Storage.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract images from parsed document chunks"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Save images to local folder instead of Azure Blob Storage",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./extracted_images",
        help="Output folder for local extraction (default: ./extracted_images)",
    )
    parser.add_argument(
        "--parsed-folder",
        type=str,
        default="./parsed_chunks",
        help="Folder containing parsed chunk files (default: ./parsed_chunks)",
    )
    args = parser.parse_args()

    if args.local:
        print("🖼️ Image Extraction to Local Folder")
        print("=" * 50)
        images_saved = extract_images_to_local(
            parsed_folder=args.parsed_folder, output_folder=args.output
        )
        if images_saved > 0:
            print(
                f"\n✅ Successfully extracted and saved {images_saved} images locally"
            )
        else:
            print("\n⚠️ No images were found to extract")
            print("📝 Make sure you have run document_ai.py and the parser first")
    else:
        print("🖼️ Image Extraction to Blob Storage")
        print("=" * 50)
        images_stored = extract_images_from_chunks_to_blob_storage(
            parsed_folder=args.parsed_folder
        )
        if images_stored > 0:
            print(f"\n✅ Successfully extracted and stored {images_stored} images")
            print(
                "📝 Next step: Run parse_document.py to index the chunks with image references"
            )
        else:
            print("\n⚠️ No images were found to extract")
            print("📝 Make sure you have run document_ai.py and the parser first")


if __name__ == "__main__":
    main()
