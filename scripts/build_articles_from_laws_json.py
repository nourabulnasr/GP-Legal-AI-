from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "laws" / "labor_14_2025.json"
OUT_PATH = ROOT / "laws" / "processed" / "labor14_2025_articles.json"

def main():
    data = json.load(open(IN_PATH, "r", encoding="utf-8"))
    arts = (data or {}).get("articles")

    out = []

    # Case A: articles is dict: {"1": {"text":...}, ...}
    if isinstance(arts, dict):
        for k, v in arts.items():
            if isinstance(v, dict):
                text = v.get("text") or v.get("body") or v.get("content") or ""
                title = v.get("title") or v.get("heading") or ""
            else:
                text = str(v or "")
                title = ""
            text = (text or "").strip()
            if not text:
                continue
            out.append({
                "article": str(k),
                "title": title.strip() if isinstance(title, str) else "",
                "text": text,
                "law": data.get("law_name") or data.get("title") or "قانون العمل رقم 14 لسنة 2025",
                "source": "laws/labor_14_2025.json",
            })

    # Case B: articles is list already
    elif isinstance(arts, list):
        for item in arts:
            if not isinstance(item, dict):
                continue
            art_no = item.get("article") or item.get("id") or item.get("number")
            text = item.get("text") or item.get("body") or item.get("content") or ""
            title = item.get("title") or item.get("heading") or ""
            text = (text or "").strip()
            if not text:
                continue
            out.append({
                "article": str(art_no) if art_no is not None else None,
                "title": title.strip() if isinstance(title, str) else "",
                "text": text,
                "law": item.get("law") or data.get("law_name") or "قانون العمل رقم 14 لسنة 2025",
                "source": "laws/labor_14_2025.json",
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("✅ Built articles list")
    print({"articles_out": len(out), "out_path": str(OUT_PATH)})

if __name__ == "__main__":
    main()

