# -*- coding: utf-8 -*-
"""Year-neutral paths and metadata keys for the Egyptian labor-law RAG corpus."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent

# Canonical processed artifacts (year-neutral filenames)
ARTICLES_CANONICAL = _REPO / "laws" / "processed" / "labor_law_articles.json"
ARTICLES_CLEANED_JSON = _REPO / "laws" / "processed" / "labor_law_articles.cleaned.json"
CHUNKS_JSONL = _REPO / "chunks" / "labor_law_chunks.jsonl"
CHUNKS_CLEANED_JSONL = _REPO / "chunks" / "labor_law_chunks.cleaned.jsonl"
LEGAL_RAG_CHUNKS_CLEANED_JSONL = _REPO / "Legal Rag" / "data" / "labor_law_chunks.cleaned.jsonl"
PREPROCESS_REPORT_JSON = _REPO / "laws" / "processed" / "labor_law_preprocess_report.json"

# Chunk metadata / Chroma filter value (not a file path)
LAW_CHUNK_SOURCE = "labor_law"
CHUNK_ID_PREFIX = "labor_law"

# Legacy filenames (pre rename) — migration and read fallbacks
LEGACY_ARTICLES_JSON = _REPO / "laws" / "processed" / "labor14_2025_articles.json"
LEGACY_ARTICLES_CLEANED_JSON = _REPO / "laws" / "processed" / "labor14_2025_articles.cleaned.json"
LEGACY_CHUNKS_JSONL = _REPO / "chunks" / "labor14_2025_chunks.jsonl"
LEGACY_CHUNKS_CLEANED_JSONL = _REPO / "chunks" / "labor14_2025_chunks.cleaned.jsonl"
LEGACY_LEGAL_RAG_CHUNKS_CLEANED_JSONL = _REPO / "Legal Rag" / "data" / "labor14_2025_chunks.cleaned.jsonl"
LEGACY_PREPROCESS_REPORT_JSON = _REPO / "laws" / "processed" / "labor14_2025_report.json"


def migrate_legacy_labor_law_artifacts() -> None:
    """If neutral-named files are missing but legacy files exist, copy once (no overwrite)."""
    pairs = [
        (LEGACY_ARTICLES_JSON, ARTICLES_CANONICAL),
        (LEGACY_ARTICLES_CLEANED_JSON, ARTICLES_CLEANED_JSON),
        (LEGACY_CHUNKS_JSONL, CHUNKS_JSONL),
        (LEGACY_CHUNKS_CLEANED_JSONL, CHUNKS_CLEANED_JSONL),
        (LEGACY_LEGAL_RAG_CHUNKS_CLEANED_JSONL, LEGAL_RAG_CHUNKS_CLEANED_JSONL),
        (LEGACY_PREPROCESS_REPORT_JSON, PREPROCESS_REPORT_JSON),
    ]
    for old, new in pairs:
        if new.exists() or not old.is_file():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)


def resolved_cleaned_chunks_path() -> Optional[Path]:
    """Prefer Legal Rag data copy, then repo chunks/; neutral names before legacy."""
    for p in (
        LEGAL_RAG_CHUNKS_CLEANED_JSONL,
        CHUNKS_CLEANED_JSONL,
        LEGACY_LEGAL_RAG_CHUNKS_CLEANED_JSONL,
        LEGACY_CHUNKS_CLEANED_JSONL,
    ):
        if p.is_file():
            return p
    return None


def resolved_legal_rag_chunks_file() -> Path:
    """Path passed to Legal Rag ingest (prefer existing neutral, else legacy, else neutral target)."""
    if LEGAL_RAG_CHUNKS_CLEANED_JSONL.is_file():
        return LEGAL_RAG_CHUNKS_CLEANED_JSONL
    if LEGACY_LEGAL_RAG_CHUNKS_CLEANED_JSONL.is_file():
        return LEGACY_LEGAL_RAG_CHUNKS_CLEANED_JSONL
    return LEGAL_RAG_CHUNKS_CLEANED_JSONL
