# -*- coding: utf-8 -*-
"""Extract labor-law articles from official PDF text (PyMuPDF + Arabic article markers)."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

ARTICLE_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(?:المادة|مادة)\s*[\(\)]?\s*(\d{1,4})\s*[\)\(]?\s*[:\-–]?\s*",
    re.MULTILINE,
)


def nfkc_norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DIGIT_MAP)
    s = s.replace("\u00A0", " ")
    s = s.replace("\u200f", "").replace("\u200e", "")
    return s


def _extract_pdf_text_pymupdf_only(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    return "\n".join(parts)


def extract_pdf_text(pdf_path: Path) -> str:
    """Raw PDF text for article splitting.

    Default (``LAW_PDF_EXTRACTION_MODE=auto``): same pipeline as contract uploads
    (Document AI when configured, else PyMuPDF with per-page OCR when text is sparse).

    Set ``LAW_PDF_EXTRACTION_MODE=pymupdf_only`` for the previous fast path (embedded text only).
    """
    pdf_path = Path(pdf_path)
    mode = (os.environ.get("LAW_PDF_EXTRACTION_MODE") or "auto").strip().lower()
    if mode == "pymupdf_only":
        return _extract_pdf_text_pymupdf_only(pdf_path)
    from .pdf_text_pipeline import extract_full_text_from_pdf_bytes

    return extract_full_text_from_pdf_bytes(pdf_path.read_bytes())


def build_articles_from_text(full_text: str, *, source_path: str) -> List[Dict[str, Any]]:
    full_text = (full_text or "").replace("\r", "")
    matches = list(ARTICLE_SPLIT_RE.finditer(full_text))
    articles: List[Dict[str, Any]] = []
    if not matches:
        return articles

    for i, m in enumerate(matches):
        art_no = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) < 80:
            continue
        articles.append(
            {
                "article": str(art_no),
                "text": body,
                "title": "",
                "source": source_path,
            }
        )
    return articles


def extract_articles_from_pdf(
    pdf_path: Path,
    *,
    law_display_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Read PDF from disk and return article dicts suitable for laws/processed/labor_law_articles.json.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw = extract_pdf_text(pdf_path)
    text = nfkc_norm(raw)
    articles = build_articles_from_text(text, source_path=pdf_path.as_posix())
    law = (
        (law_display_name or "").strip()
        or os.environ.get("LAW_DEFAULT_NAME", "").strip()
        or "قانون العمل رقم 14 لسنة 2025"
    )
    for a in articles:
        a["law"] = law
        a.setdefault("title", "")
    return articles
