# -*- coding: utf-8 -*-
"""
Bridge to Legal Rag: uses Legal Rag/src (VectorStore, RAGEngine, DataIngestion)
with paths resolved from project root. Same API as rag_chromadb for drop-in use in main.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = Path(__file__).resolve().parent.parent
_LEGAL_RAG_ROOT = _BASE / "Legal Rag"
_LEGAL_RAG_SRC = _LEGAL_RAG_ROOT / "src"

# Ensure Legal Rag src is importable
if str(_LEGAL_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGAL_RAG_ROOT))

_config = None
_vector_store = None
_initialized = False
_init_error: Optional[str] = None


def _get_config():
    """Build config with absolute paths so it works when app runs from project root."""
    global _config
    if _config is not None:
        return _config
    try:
        from src.config import Config

        yaml_path = _LEGAL_RAG_ROOT / "config.yaml"
        if yaml_path.exists():
            _config = Config.from_yaml(str(yaml_path))
        else:
            _config = Config()
        # Override paths to be absolute
        _config.data.labor_law_file = str(_LEGAL_RAG_ROOT / "data" / "labor14_2025_chunks.cleaned.jsonl")
        _config.data.sample_contracts_dir = str(_LEGAL_RAG_ROOT / "data" / "sample_contracts")
        _config.vector_store.persist_directory = str(_LEGAL_RAG_ROOT / "chroma_db")
        _config.vector_store.collection_name = getattr(
            _config.vector_store, "collection_name", "egyptian_labor_laws"
        )
        return _config
    except Exception as e:
        raise RuntimeError(f"Legal Rag config failed: {e}") from e


def _get_vector_store():
    """Lazy init: VectorStore + optional ingestion when collection is empty."""
    global _vector_store, _initialized, _init_error
    if _initialized:
        return _vector_store
    _initialized = True
    try:
        from src.vector_store import VectorStore
        from src.data_ingestion import DataIngestion

        config = _get_config()
        _vector_store = VectorStore(config=config)
        stats = _vector_store.get_collection_stats()
        count = stats.get("document_count", 0) or 0
        if count == 0:
            labor_file = Path(config.data.labor_law_file)
            if labor_file.exists():
                ingestion = DataIngestion(config=config)
                ok = ingestion.ingest_labor_law_data()
                if ok:
                    stats = _vector_store.get_collection_stats()
                    count = stats.get("document_count", 0)
            else:
                _init_error = f"Corpus not found: {labor_file}"
        return _vector_store
    except Exception as e:
        _init_error = str(e)
        _vector_store = None
        return None


def search(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search using Legal Rag VectorStore. Same shape as rag_chromadb.search:
    [{"score": float, "text": str, "metadata": dict}, ...]
    """
    if not query or not query.strip():
        return []
    try:
        vs = _get_vector_store()
        if vs is None:
            return []
        where = None
        if filters:
            where = {k: v for k, v in filters.items() if v is not None and v != ""}
        results = vs.query_with_scores(
            query_text=query,
            n_results=min(max(1, top_k * 2), 50),
            min_score=min_score,
            where=where,
        )
        out: List[Dict[str, Any]] = []
        for r in results:
            if r.get("score", 0) >= min_score:
                out.append({
                    "score": r.get("score", 0),
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata") or {},
                })
            if len(out) >= top_k:
                break
        return out
    except Exception as e:
        print(f"[Legal RAG bridge] search failed: {e!r}")
        return []


def is_available() -> bool:
    """True if Legal Rag is usable (config + corpus/chroma)."""
    try:
        _get_config()
        vs = _get_vector_store()
        if vs is None:
            return False
        stats = vs.get_collection_stats()
        return (stats.get("document_count") or 0) >= 0
    except Exception:
        return False


def get_collection_count() -> int:
    """Number of documents in the Legal Rag collection (0 if not built)."""
    try:
        vs = _get_vector_store()
        if vs is None:
            return 0
        stats = vs.get_collection_stats()
        return int(stats.get("document_count") or 0)
    except Exception:
        return 0


def get_device() -> Optional[str]:
    """Resolved device (cpu/cuda) from Legal Rag config, or None if not loaded."""
    try:
        cfg = _get_config()
        return cfg.get_device()
    except Exception:
        return None


def get_embedding_model_name() -> Optional[str]:
    """Embedding model name from Legal Rag config, or None if not loaded."""
    try:
        cfg = _get_config()
        return getattr(cfg.embeddings, "model_name", None)
    except Exception:
        return None


def get_collection_name() -> Optional[str]:
    """Chroma collection name from Legal Rag config, or None if not loaded."""
    try:
        cfg = _get_config()
        return getattr(getattr(cfg, "vector_store", None), "collection_name", None)
    except Exception:
        return None
