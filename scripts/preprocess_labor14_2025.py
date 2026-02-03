import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent

ARTICLES_IN = ROOT / "laws" / "processed" / "labor14_2025_articles.json"

# Outputs
ARTICLES_OUT = ROOT / "laws" / "processed" / "labor14_2025_articles.cleaned.json"
CHUNKS_OUT = ROOT / "chunks" / "labor14_2025_chunks.jsonl"
REPORT_OUT = ROOT / "laws" / "processed" / "labor14_2025_report.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
MULTI_SPACE = re.compile(r"[ \t]+")

# Arabic/Persian digits -> Latin
DIGIT_MAP = str.maketrans(
    {
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # ✅ NFKC fixes Arabic presentation forms (ﻣﺎﺩﺓ -> مادة)
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("\u200f", "").replace("\u200e", "")  # RTL marks
    s = ARABIC_DIACRITICS.sub("", s)
    s = s.replace("ـ", "")  # tatweel

    # ✅ digits normalization
    s = s.translate(DIGIT_MAP)

    s = MULTI_SPACE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clamp_chunk(text: str, max_chars: int = 1800) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"\n\n+|(?<=\.)\s+|(?<=؛)\s+|(?<=:)\s+", text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def main():
    if not ARTICLES_IN.exists():
        raise SystemExit(f"Missing articles file: {ARTICLES_IN}")

    articles = json.loads(ARTICLES_IN.read_text(encoding="utf-8"))
    if not isinstance(articles, list) or len(articles) == 0:
        raise SystemExit("articles.json must be a non-empty list")

    cleaned_articles: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    split_count = 0
    empty_dropped = 0
    total_out = 0

    for a in articles:
        if not isinstance(a, dict):
            continue

        art_no = str(a.get("article") or a.get("no") or "")
        title = a.get("title") or ""
        text = a.get("text") or a.get("body") or a.get("content") or ""

        title_n = normalize_text(title)
        text_n = normalize_text(text)

        obj = dict(a)
        obj["article"] = art_no
        obj["title"] = title_n
        obj["text"] = text_n
        obj["source"] = obj.get("source") or "laws/raw/Labor Law for 2025 in egypt.pdf"
        obj["law"] = obj.get("law") or "قانون العمل رقم 14 لسنة 2025"
        cleaned_articles.append(obj)

        # build chunks from the article text
        if len(text_n) < 30:
            empty_dropped += 1
            continue

        pieces = clamp_chunk(text_n, max_chars=1800)
        if len(pieces) > 1:
            split_count += 1

        for j, piece in enumerate(pieces):
            cid = f"labor14_2025__art_{art_no}"
            if len(pieces) > 1:
                cid = f"{cid}__{j}"

            chunks.append(
                {
                    "id": cid,
                    "text": piece,
                    "normalized_text": piece,
                    "source": "labor14_2025",
                    "law": obj["law"],
                    "article": art_no,
                    "title": title_n,
                }
            )
            total_out += 1

    # Deduplicate by normalized_text (helps RAG index quality)
    seen = set()
    deduped = []
    dup = 0
    for c in chunks:
        key = c.get("normalized_text", "")
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        deduped.append(c)

    # Write outputs
    ARTICLES_OUT.write_text(
        json.dumps(cleaned_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_OUT.open("w", encoding="utf-8") as w:
        for c in deduped:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {
        "articles_in": len(articles),
        "articles_out": len(cleaned_articles),
        "chunks_out_lines": total_out,
        "chunks_split_count": split_count,
        "chunks_empty_dropped": empty_dropped,
        "chunks_duplicates_removed": dup,
        "notes": [
            "✅ Built chunks from articles (not legacy chunks file)",
            "✅ Added NFKC normalization (fixes Arabic presentation forms مثل ﻣﺎﺩﺓ -> مادة)",
            "✅ Normalized Arabic/Persian digits (۱۳۲ / ١٣٢ -> 132)",
            "Removed diacritics/tatweel/RTL marks, collapsed spaces",
            "Clamped chunks to <= ~1800 chars, split on paragraph/sentences",
            "Added strong metadata: law/article/title/source for better RAG",
            "Deduped by normalized_text for better RAG indexing",
        ],
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Preprocess DONE (articles -> chunks)")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
