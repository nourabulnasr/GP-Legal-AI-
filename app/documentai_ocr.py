"""
Google Document AI OCR integration.

Requires:
  - DOCUMENT_AI_PROJECT_ID, DOCUMENT_AI_LOCATION, DOCUMENT_AI_PROCESSOR_ID in env
  - GOOGLE_APPLICATION_CREDENTIALS pointing to service account JSON key

Usage:
  from app.documentai_ocr import documentai_extract_text, _HAS_DOCUMENTAI
  if _HAS_DOCUMENTAI:
      text = documentai_extract_text(pdf_bytes)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_HAS_DOCUMENTAI = False
try:
    from google.cloud import documentai_v1 as documentai
    _HAS_DOCUMENTAI = True
except ImportError:
    documentai = None  # type: ignore


def _is_configured() -> bool:
    return bool(
        _HAS_DOCUMENTAI
        and os.environ.get("DOCUMENT_AI_PROJECT_ID")
        and os.environ.get("DOCUMENT_AI_PROCESSOR_ID")
    )


def documentai_extract_text(
    pdf_bytes: bytes,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    processor_id: Optional[str] = None,
) -> str:
    """
    Extract text from PDF using Google Document AI.

    Uses env vars if not passed:
      DOCUMENT_AI_PROJECT_ID, DOCUMENT_AI_LOCATION (default: us), DOCUMENT_AI_PROCESSOR_ID
    """
    if not _HAS_DOCUMENTAI or documentai is None:
        return ""

    project_id = project_id or os.environ.get("DOCUMENT_AI_PROJECT_ID", "").strip()
    location = location or os.environ.get("DOCUMENT_AI_LOCATION", "us").strip()
    processor_id = processor_id or os.environ.get("DOCUMENT_AI_PROCESSOR_ID", "").strip()

    if not project_id or not processor_id:
        return ""

    try:
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        result = client.process_document(request=request)
        doc = result.document
        return (doc.text or "").strip()
    except Exception:
        return ""


def documentai_extract_pages(
    pdf_bytes: bytes,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    processor_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF. Returns list of {page: int, text: str}.
    Uses Document AI full text; splits by form-feed or returns single chunk.
    """
    full_text = documentai_extract_text(pdf_bytes, project_id, location, processor_id)
    if not full_text:
        return []

    parts = full_text.split("\f")
    if len(parts) > 1:
        return [{"page": i, "text": p.strip()} for i, p in enumerate(parts) if p.strip()]
    return [{"page": 0, "text": full_text}]
