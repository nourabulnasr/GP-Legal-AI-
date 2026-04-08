from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json


def load_chunks_as_docs(chunks_dir: Path) -> List[Dict[str, Any]]:
    """
    Reads *.jsonl under chunks_dir safely (tolerant to bad lines).
    Returns docs list with:
      - page_content: str
      - metadata: dict (law/article/title/source/id/page/chunk_id/source_file)
    """
    docs: List[Dict[str, Any]] = []
    chunks_dir = Path(chunks_dir)

    if not chunks_dir.exists():
        return docs

    for p in sorted(chunks_dir.glob("*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        # ignore malformed JSON lines
                        continue

                    text = (
                        obj.get("normalized_text")
                        or obj.get("text")
                        or obj.get("page_content")
                        or ""
                    )
                    text = str(text).strip()
                    if not text:
                        continue

                    meta = {
                        "source_file": p.name,
                        "id": obj.get("id"),
                        "chunk_id": obj.get("chunk_id"),
                        "page": obj.get("page"),
                        "law": obj.get("law"),
                        "article": obj.get("article"),
                        "title": obj.get("title"),
                        # keep original 'source' if present, else default to file stem (or full filename)
                        "source": obj.get("source") or p.stem,
                    }

                    docs.append({"page_content": text, "metadata": meta})
        except Exception:
            # if file can't be read, skip it
            continue

    return docs
