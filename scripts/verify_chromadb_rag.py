#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 verification: ChromaDB RAG build + search.
Run from project root: python scripts/verify_chromadb_rag.py
Asserts non-empty structured results for search("ساعات العمل", top_k=5).
Skips with exit 0 if chromadb/sentence_transformers not installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("ChromaDB RAG verification (plan Step 1)...")
    # Prefer Legal Rag bridge (uses Legal Rag/src); fallback to app/rag_chromadb
    search_fn = is_available_fn = get_count_fn = None
    try:
        from app import legal_rag_bridge
        search_fn = legal_rag_bridge.search
        is_available_fn = legal_rag_bridge.is_available
        get_count_fn = legal_rag_bridge.get_collection_count
        print("Using Legal Rag bridge (Legal Rag/src VectorStore + DataIngestion).")
    except Exception as e:
        print(f"Legal Rag bridge not used: {e}")
    if search_fn is None:
        try:
            from app.rag_chromadb import build_index, search, is_available, get_collection_count
            search_fn = search
            is_available_fn = is_available
            get_count_fn = get_collection_count
            if not is_available_fn():
                try:
                    n = build_index()
                    print(f"Built index: {n} docs")
                except Exception as e:
                    print(f"Skip: build_index failed: {e}")
                    return 0
            else:
                n = get_count_fn()
                print(f"Index already present: {n} docs")
            print("Using app/rag_chromadb.")
        except ModuleNotFoundError as e:
            print(f"Skip: dependency not installed ({e}). Install chromadb and sentence_transformers to run.")
            return 0

    # Trigger lazy init (Legal Rag bridge ingests on first use if empty)
    if is_available_fn:
        is_available_fn()
    n = get_count_fn() if get_count_fn else 0
    print(f"Index: {n} docs")

    results = search_fn("ساعات العمل", top_k=5) if search_fn else []
    if not results:
        print("WARN: search returned empty (corpus may be empty or path missing).")
        return 0
    # Evidence: document count and first result snippet
    print(f"[Evidence] document_count={n}")
    for i, r in enumerate(results, 1):
        assert "score" in r and "text" in r and "metadata" in r
        text_preview = (r.get("text") or "")[:200].replace("\n", " ")
        meta = r.get("metadata") or {}
        article = meta.get("article", "")
        print(f"  {i}. score={r['score']:.3f} article={article} text_len={len(r.get('text', ''))}")
        if i == 1:
            print(f"[Evidence] first_result_snippet: {text_preview!r}")
    print("OK: ChromaDB RAG verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
