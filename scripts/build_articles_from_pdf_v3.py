# scripts/build_articles_from_pdf_v3.py — CLI wrapper for app.law_pdf_extract
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_PDF = ROOT / "laws" / "raw" / "Labor Law for 2025 in egypt.pdf"
OUT_JSON = ROOT / "laws" / "processed" / "labor14_2025_articles.from_pdf.v3.json"


def main() -> None:
    from app.law_pdf_extract import extract_articles_from_pdf

    pdf = RAW_PDF
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1]).expanduser().resolve()
    if not pdf.is_file():
        raise SystemExit(f"RAW PDF not found: {pdf}")

    articles = extract_articles_from_pdf(pdf)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

    has_mada = False
    try:
        from app.law_pdf_extract import extract_pdf_text, nfkc_norm

        t = nfkc_norm(extract_pdf_text(pdf))
        has_mada = ("مادة" in t) or ("المادة" in t)
    except Exception:
        pass

    print("OK: Built articles from PDF (V3)")
    print(
        json.dumps(
            {
                "pdf_text_len": sum(len(a.get("text", "")) for a in articles),
                "contains_mada": has_mada,
                "articles_out": len(articles),
                "out_path": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
