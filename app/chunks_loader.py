from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Neutral preprocess output; legacy filename kept for migration / old trees.
_CHUNK_NEUTRAL_CLEANED = "labor_law_chunks.cleaned.jsonl"
_CHUNK_LEGACY_CLEANED = "labor14_2025_chunks.cleaned.jsonl"


def list_chunk_jsonl_files(chunks_dir: Path) -> List[Path]:
    """
    Paths to load for the in-memory law retriever.

    If ``labor_law_chunks.cleaned.jsonl`` exists, only that file is used (avoids mixing
    with a leftover legacy JSONL). Otherwise ``labor14_2025_chunks.cleaned.jsonl``, then glob.
    """
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.is_dir():
        return []
    neutral = chunks_dir / _CHUNK_NEUTRAL_CLEANED
    legacy = chunks_dir / _CHUNK_LEGACY_CLEANED
    if neutral.is_file():
        return [neutral]
    if legacy.is_file():
        return [legacy]
    return sorted(chunks_dir.glob("*.jsonl"))


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

    jsonl_files = list_chunk_jsonl_files(chunks_dir)

    for p in jsonl_files:
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
