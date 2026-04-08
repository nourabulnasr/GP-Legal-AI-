# -*- coding: utf-8 -*-
"""
ChromaDB-backed RAG over Egyptian Labor Law (Legal RAG corpus).
Loads from Legal Rag/data or chunks/; persists to chroma_legal.
Same API shape as rag_utils.Retriever.search for drop-in use in main.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = Path(__file__).resolve().parent.parent

# Corpus: Legal Rag data or project chunks
LEGAL_RAG_DATA = _BASE / "Legal Rag" / "data" / "labor14_2025_chunks.cleaned.jsonl"
CHUNKS_FALLBACK = _BASE / "chunks" / "labor14_2025_chunks.cleaned.jsonl"
PERSIST_DIR = os.environ.get("CHROMA_LEGAL_DIR", str(_BASE / "chroma_legal"))
COLLECTION_NAME = "legal_labor_law"

_client = None
_collection = None
_embedding_fn = None
_initialized = False


def _corpus_path() -> Path:
    path_env = os.environ.get("LEGAL_RAG_DATA_PATH", "").strip()
    if path_env:
        return Path(path_env)
    if LEGAL_RAG_DATA.exists():
        return LEGAL_RAG_DATA
    return CHUNKS_FALLBACK


def _load_docs(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = (obj.get("normalized_text") or obj.get("text") or "").strip()
            if len(text) < 20:
                continue
            doc_id = obj.get("id") or f"doc_{len(docs)}"
            meta = {
                "law": str(obj.get("law", "")),
                "article": str(obj.get("article", "")),
                "source": str(obj.get("source", "") or path.stem),
            }
            docs.append({"id": doc_id, "text": text, "metadata": meta})
    return docs


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("LEGAL_RAG_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        _embedding_fn = SentenceTransformer(model_name)
        return _embedding_fn
    except Exception as e:
        raise RuntimeError(f"ChromaDB RAG requires sentence_transformers: {e}") from e


def build_index(
    corpus_dir: Optional[Path] = None,
    persist_path: Optional[str] = None,
) -> int:
    """
    Build or rebuild ChromaDB index from corpus. Exposed per plan.
    corpus_dir: path to folder containing JSONL (or path to single .jsonl); default = Legal RAG data/chunks.
    persist_path: ChromaDB persist directory; default = CHROMA_LEGAL_DIR / chroma_legal.
    Returns: number of documents indexed.
    """
    global _client, _collection, _initialized
    import chromadb
    from chromadb.config import Settings

    path = Path(persist_path or PERSIST_DIR)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # Clear if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        coll = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        pass

    if corpus_dir is not None:
        cp = Path(corpus_dir)
        if cp.is_file() and cp.suffix.lower() == ".jsonl":
            docs = _load_docs(cp)
        elif cp.is_dir():
            jsonl_files = list(cp.glob("*.jsonl"))
            docs = []
            for fp in sorted(jsonl_files):
                docs.extend(_load_docs(fp))
        else:
            docs = []
    else:
        docs = _load_docs(_corpus_path())

    if not docs:
        _client = client
        _collection = coll
        _initialized = True
        return 0

    model = _get_embedding_fn()
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    for m in metadatas:
        for k, v in list(m.items()):
            if v is None:
                m[k] = ""
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    coll.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())
    _client = client
    _collection = coll
    _initialized = True
    return len(docs)


def _get_client():
    global _client, _collection, _initialized
    if _initialized and _client is not None and _collection is not None:
        return _client, _collection

    import chromadb
    from chromadb.config import Settings

    path = Path(PERSIST_DIR)
    path.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count = _collection.count()

    if count == 0:
        corpus_path = _corpus_path()
        docs = _load_docs(corpus_path)
        if not docs:
            _initialized = True
            return _client, _collection

        model = _get_embedding_fn()
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        for m in metadatas:
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        _collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())

    _initialized = True
    return _client, _collection


def search(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for law chunks. Returns same shape as rag_utils Retriever:
    [{"score": float, "text": str, "metadata": dict}, ...]
    """
    if not query or not query.strip():
        return []

    try:
        _, coll = _get_client()
        model = _get_embedding_fn()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        where = None
        if filters:
            where = {k: v for k, v in filters.items() if v is not None and v != ""}

        n_results = min(max(1, top_k * 2), 50)  # fetch extra then filter by min_score
        res = coll.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
            where=where if where else None,
        )

        out: List[Dict[str, Any]] = []
        if not res or not res["ids"] or not res["ids"][0]:
            return out

        # Chroma cosine distance: similarity = 1 - distance
        dists = res["distances"][0]
        docs = res["documents"][0]
        metadatas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
        ids = res["ids"][0]

        for i, doc_id in enumerate(ids):
            dist = dists[i] if i < len(dists) else 1.0
            sim = 1.0 - float(dist)
            if sim < min_score:
                continue
            text = docs[i] if i < len(docs) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            out.append({"score": sim, "text": text, "metadata": meta})
            if len(out) >= top_k:
                break

        return out

    except Exception as e:
        print(f"[ChromaDB RAG] search failed: {e!r}")
        return []


def is_available() -> bool:
    """True if ChromaDB and corpus are usable."""
    try:
        _get_client()
        return _collection is not None and _collection.count() >= 0
    except Exception:
        return False


def get_collection_count() -> int:
    """Return number of documents in the collection (0 if not built)."""
    try:
        _, coll = _get_client()
        return coll.count()
    except Exception:
        return 0
