# -*- coding: utf-8 -*-
"""Shared PDF text extraction for contract uploads and law RAG PDF ingestion.

Order matches ``ocr_check_and_search`` PDF handling: Document AI (bytes-based)
when configured, then legacy ``documentai_ocr`` pages, then PyMuPDF with
per-page raster OCR when embedded text is too short.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from .documentai_ocr import (
        documentai_extract_pages,
        _is_configured as _documentai_legacy_configured,
    )
except Exception:
    documentai_extract_pages = lambda *a, **k: []  # type: ignore
    _documentai_legacy_configured = lambda: False  # type: ignore

try:
    from .DocumentAI import (
        documentai_extract_pages_from_bytes,
        _is_configured as _documentai_py_configured,
    )
except Exception:
    documentai_extract_pages_from_bytes = None  # type: ignore
    _documentai_py_configured = lambda: False  # type: ignore


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = img.point(lambda x: 0 if x < 160 else 255, mode="1")
        img = img.convert("L")
        return img
    except Exception:
        return img


def is_text_too_short(txt: str) -> bool:
    return len((txt or "").strip()) < 40


def ocr_image_to_text(img_bytes: bytes) -> str:
    try:
        import pytesseract

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = preprocess_for_ocr(img)
        config = "--oem 1 --psm 6"
        return pytesseract.image_to_string(img, lang="ara+eng", config=config)
    except Exception:
        return ""


def extract_pdf_pages_from_bytes(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], bool]:
    """Extract per-page text from PDF bytes.

    Returns ``(pages, used_document_ai_or_page_ocr)`` where each page dict has
    ``page`` (0-based int) and ``text`` (str). The boolean flags whether
    Document AI or Tesseract page OCR was used (for UI / logging).
    """
    if not pdf_bytes or not pdf_bytes[:5].startswith(b"%PDF"):
        return [], False

    used_heavy = False
    dai_pages: List[Dict[str, Any]] = []
    if documentai_extract_pages_from_bytes and _documentai_py_configured():
        try:
            dai_pages = documentai_extract_pages_from_bytes(pdf_bytes)
        except Exception:
            pass
    if not dai_pages and _documentai_legacy_configured() and documentai_extract_pages:
        try:
            dai_pages = documentai_extract_pages(pdf_bytes)
        except Exception:
            pass

    if dai_pages and any((p.get("text") or "").strip() for p in dai_pages):
        used_heavy = True
        out: List[Dict[str, Any]] = []
        for p in dai_pages:
            out.append(
                {
                    "page": int(p.get("page", len(out))),
                    "text": p.get("text", "") or "",
                }
            )
        return out, used_heavy

    if fitz is None:
        txt = ocr_image_to_text(pdf_bytes) or ""
        if txt.strip():
            return [{"page": 0, "text": txt}], True
        return [], False

    ocr_chunks: List[Dict[str, Any]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            page_txt = page.get_text("text") or ""
            if is_text_too_short(page_txt):
                used_heavy = True
                pix = page.get_pixmap(dpi=300)
                page_txt = ocr_image_to_text(pix.tobytes("png")) or ""
            ocr_chunks.append({"page": i, "text": page_txt})
    finally:
        doc.close()

    if not any((c.get("text") or "").strip() for c in ocr_chunks):
        txt = ocr_image_to_text(pdf_bytes) or ""
        if txt.strip():
            return [{"page": 0, "text": txt}], True
        return [], used_heavy

    return ocr_chunks, used_heavy


def extract_full_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Join all pages with blank lines (same spacing as contract normalization step)."""
    pages, _ = extract_pdf_pages_from_bytes(pdf_bytes)
    return "\n\n".join((p.get("text") or "") for p in pages).strip()
