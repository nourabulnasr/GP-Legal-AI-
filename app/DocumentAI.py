"""
Google Document AI OCR Pipeline (Production Safe)
- Splits large PDFs into <=15-page chunks (Python SDK limit)
- Uses native PDF parsing (best quality)
- Runs OCR safely
- Merges output into a single TXT file

Bytes-based API for main.py uploads:
- documentai_extract_text_from_bytes(pdf_bytes) -> str
- documentai_extract_pages_from_bytes(pdf_bytes) -> List[Dict] with "page", "text"
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader, PdfWriter
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

# =========================
# CONFIG (env overrides for main.py)
# =========================

PROJECT_ID = os.environ.get("DOCUMENT_AI_PROJECT_ID", "project-38e1b771-78f7-468c-b5a").strip()
LOCATION = os.environ.get("DOCUMENT_AI_LOCATION", "us").strip()
PROCESSOR_ID = os.environ.get("DOCUMENT_AI_PROCESSOR_ID", "62b4bb1781737a13").strip()

INPUT_PDF = "laws/raw/Labor Law for 2025 in egypt.pdf"

PDF_CHUNKS_DIR = Path("laws/tmp_pdf_chunks")
OUTPUT_DIR = Path("laws/processed")

PAGES_PER_CHUNK = 15   # 🔒 SAFE LIMIT FOR PYTHON SDK
OUTPUT_TXT = OUTPUT_DIR / "labor_law_2025_raw.txt"

# =========================
# PDF SPLITTER
# =========================

def split_pdf(input_pdf: str, output_dir: Path, pages_per_chunk: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)

    chunk_files = []
    chunk_index = 1

    for start in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        end = min(start + pages_per_chunk, total_pages)

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        chunk_path = output_dir / f"chunk_{chunk_index}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)

        print(f"📄 Created PDF chunk: {chunk_path}")
        chunk_files.append(chunk_path)
        chunk_index += 1

    return chunk_files

# =========================
# OCR (SAFE MODE)
# =========================

def ocr_pdf(pdf_path: Path, client, processor_name: str) -> str:
    with open(pdf_path, "rb") as f:
        content = f.read()

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(
            content=content,
            mime_type="application/pdf"
        ),
        process_options=documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=True
            )
        )
    )

    result = client.process_document(request=request)
    return result.document.text or ""


def _get_client_and_processor():
    """Return (client, processor_name) for Document AI. Uses env or module config."""
    if not PROJECT_ID or not PROCESSOR_ID:
        return None, None
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    return client, processor_name


def _is_configured() -> bool:
    """True if Document AI env/config is set."""
    return bool(PROJECT_ID and PROCESSOR_ID)


def documentai_extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract full text from PDF bytes (uploaded file).
    Splits into <=15-page chunks when needed. Used by main.py OCR pipeline.
    """
    if not pdf_bytes or not _is_configured():
        return ""
    try:
        client, processor_name = _get_client_and_processor()
        if not client or not processor_name:
            return ""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        if total_pages <= PAGES_PER_CHUNK:
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=pdf_bytes,
                    mime_type="application/pdf",
                ),
                process_options=documentai.ProcessOptions(
                    ocr_config=documentai.OcrConfig(enable_native_pdf_parsing=True),
                ),
            )
            result = client.process_document(request=request)
            return (result.document.text or "").strip()
        # Split and process chunks
        parts = []
        with tempfile.TemporaryDirectory(prefix="docai_") as tmpdir:
            out_dir = Path(tmpdir)
            for start in range(0, total_pages, PAGES_PER_CHUNK):
                writer = PdfWriter()
                end = min(start + PAGES_PER_CHUNK, total_pages)
                for i in range(start, end):
                    writer.add_page(reader.pages[i])
                chunk_path = out_dir / "chunk.pdf"
                with open(chunk_path, "wb") as f:
                    writer.write(f)
                text = ocr_pdf(chunk_path, client, processor_name)
                if text:
                    parts.append(text)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def documentai_extract_pages_from_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text per page from PDF bytes. Returns [{"page": 0, "text": "..."}, ...].
    Used by main.py to build ocr_chunks. Splits into <=15-page chunks when needed.
    """
    if not pdf_bytes or not _is_configured():
        return []
    full_text = documentai_extract_text_from_bytes(pdf_bytes)
    if not full_text:
        return []
    # Split by form-feed if Document AI returns page breaks; else single chunk
    parts = full_text.split("\f")
    if len(parts) > 1:
        return [{"page": i, "text": p.strip()} for i, p in enumerate(parts) if p.strip()]
    return [{"page": 0, "text": full_text}]


# =========================
# MAIN PIPELINE
# =========================

def process_document():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("✂️ Splitting PDF...")
    pdf_chunks = split_pdf(INPUT_PDF, PDF_CHUNKS_DIR, PAGES_PER_CHUNK)

    print("🔗 Initializing Document AI client...")
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(
        PROJECT_ID,
        LOCATION,
        PROCESSOR_ID
    )

    all_text_parts = []

    for i, pdf_chunk in enumerate(pdf_chunks, 1):
        print(f"🔍 OCR processing ({i}/{len(pdf_chunks)}): {pdf_chunk.name}")
        text = ocr_pdf(pdf_chunk, client, processor_name)
        all_text_parts.append(text)

    print("🧩 Merging OCR results...")
    final_text = "\n\n".join(all_text_parts)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_text)

    print("\n✅ OCR completed successfully!")
    print(f"📄 Output TXT: {OUTPUT_TXT.resolve()}")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    process_document()
