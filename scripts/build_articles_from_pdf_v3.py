# scripts/build_articles_from_pdf_v3.py
from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

import fitz  # PyMuPDF


RAW_PDF = Path("laws/raw/Labor Law for 2025 in egypt.pdf")
OUT_JSON = Path("laws/processed/labor14_2025_articles.from_pdf.v3.json")


# Arabic-Indic + Eastern-Arabic digits -> Western digits
_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def nfkc_norm(s: str) -> str:
    """Fix Arabic presentation forms (ﻣﺎﺩﺓ -> مادة) and normalize text."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DIGIT_MAP)
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("\u200f", "").replace("\u200e", "")  # RTL marks
    return s


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    return "\n".join(parts)


# Match:
# ( المادة 132 : )   OR   مادة(132) :   OR   مادة) 132 (
ARTICLE_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(?:المادة|مادة)\s*[\(\)]?\s*(\d{1,4})\s*[\)\(]?\s*[:\-–]?\s*",
    re.MULTILINE,
)


def build_articles_from_text(full_text: str) -> list[dict]:
    full_text = full_text.replace("\r", "")
    matches = list(ARTICLE_SPLIT_RE.finditer(full_text))
    articles: list[dict] = []

    if not matches:
        return articles

    for i, m in enumerate(matches):
        art_no = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()

        # skip tiny garbage
        if len(body) < 80:
            continue

        articles.append(
            {
                "article": str(art_no),
                "text": body,
                "source": RAW_PDF.as_posix(),
            }
        )

    return articles


def main():
    if not RAW_PDF.exists():
        raise SystemExit(f"RAW PDF not found: {RAW_PDF}")

    raw_text = extract_pdf_text(RAW_PDF)
    text = nfkc_norm(raw_text)

    # quick sanity check: do we see "مادة" now?
    has_mada = ("مادة" in text) or ("المادة" in text)

    articles = build_articles_from_text(text)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print("✅ Built articles from PDF (V3)")
    print(
        {
            "pdf_text_len": len(text),
            "contains_mada": has_mada,
            "articles_out": len(articles),
            "out_path": str(OUT_JSON),
        }
    )


if __name__ == "__main__":
    main()

