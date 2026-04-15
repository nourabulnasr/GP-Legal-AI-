# -*- coding: utf-8 -*-
"""
Pipeline: labor law articles JSON -> chunk JSONL -> Chroma reindex + in-memory retriever reload.
Used by admin API, URL poller, and scripts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.law_paths import (
    ARTICLES_CANONICAL,
    CHUNKS_CLEANED_JSONL,
    LEGAL_RAG_CHUNKS_CLEANED_JSONL,
    resolved_cleaned_chunks_path,
)

_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = _ROOT / "chunks"
INCOMING_DIR = _ROOT / "laws" / "raw" / "incoming"
META_DIR = _ROOT / "laws" / "meta"
BACKUPS_ROOT = META_DIR / "backups"
SYNC_STATE_PATH = META_DIR / "sync_state.json"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_canonical_artifacts() -> Optional[Path]:
    """Copy current articles + chunk JSONL into laws/meta/backups/<stamp>/."""
    stamp = _utc_stamp()
    dest = BACKUPS_ROOT / stamp
    any_file = False
    dest.mkdir(parents=True, exist_ok=True)
    if ARTICLES_CANONICAL.is_file():
        shutil.copy2(ARTICLES_CANONICAL, dest / ARTICLES_CANONICAL.name)
        any_file = True
    if LEGAL_RAG_CHUNKS_CLEANED_JSONL.is_file():
        shutil.copy2(LEGAL_RAG_CHUNKS_CLEANED_JSONL, dest / LEGAL_RAG_CHUNKS_CLEANED_JSONL.name)
        any_file = True
    if CHUNKS_CLEANED_JSONL.is_file():
        shutil.copy2(CHUNKS_CLEANED_JSONL, dest / CHUNKS_CLEANED_JSONL.name)
        any_file = True
    if CHUNKS_DIR.is_dir():
        for p in sorted(CHUNKS_DIR.glob("labor_law*.jsonl")) + sorted(CHUNKS_DIR.glob("labor14_2025*.jsonl")):
            if p.is_file():
                shutil.copy2(p, dest / p.name)
                any_file = True
    return dest if any_file else None


def write_canonical_articles(articles: List[Dict[str, Any]], *, backup: bool = True) -> Path:
    if backup:
        backup_canonical_artifacts()
    ARTICLES_CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    ARTICLES_CANONICAL.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ARTICLES_CANONICAL


def run_preprocess() -> None:
    script = _ROOT / "scripts" / "preprocess_labor14_2025.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing preprocess script: {script}")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        check=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def preview_pdf_articles(pdf_path: Path, *, law_display_name: Optional[str] = None) -> Dict[str, Any]:
    from app.law_pdf_extract import extract_articles_from_pdf

    articles = extract_articles_from_pdf(pdf_path, law_display_name=law_display_name)
    preview = [
        {"article": a.get("article"), "title": (a.get("title") or "")[:120], "text_len": len(a.get("text") or "")}
        for a in articles[:15]
    ]
    return {"article_count": len(articles), "preview": preview}


def full_pipeline_from_articles_json(*, backup: bool = True) -> Dict[str, Any]:
    """Assume ARTICLES_CANONICAL already written; run preprocess + reindex."""
    if not ARTICLES_CANONICAL.is_file():
        raise FileNotFoundError(f"Missing {ARTICLES_CANONICAL}")
    if backup:
        backup_canonical_artifacts()
    run_preprocess()
    return reindex_all_rag()


def full_pipeline_from_pdf(
    pdf_path: Path,
    *,
    law_display_name: Optional[str] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    from app.law_pdf_extract import extract_articles_from_pdf

    articles = extract_articles_from_pdf(pdf_path, law_display_name=law_display_name)
    if not articles:
        raise ValueError("No articles extracted from PDF (parser may not match this gazette layout).")
    write_canonical_articles(articles, backup=backup)
    run_preprocess()
    return reindex_all_rag()


def reindex_all_rag() -> Dict[str, Any]:
    """Rebuild Chroma (bridge or fallback) and reload main in-memory retriever."""
    out: Dict[str, Any] = {"chroma": {}, "retriever": {}}
    corpus = resolved_cleaned_chunks_path()

    try:
        import app.main as mainmod

        mod = getattr(mainmod, "rag_chromadb", None)
    except Exception as e:
        out["chroma"] = {"error": repr(e)}
        mod = None

    if mod is not None and hasattr(mod, "reindex_from_corpus"):
        ok, msg = mod.reindex_from_corpus(corpus)
        cnt = int(mod.get_collection_count()) if hasattr(mod, "get_collection_count") else 0
        out["chroma"] = {
            "backend": getattr(mod, "__name__", "rag_chromadb"),
            "ok": ok,
            "message": msg,
            "count": cnt,
        }
    else:
        out["chroma"] = {"skipped": True, "reason": "no rag_chromadb module"}

    try:
        import app.main as mainmod

        if hasattr(mainmod, "reload_chunk_retriever"):
            out["retriever"] = mainmod.reload_chunk_retriever()
        else:
            out["retriever"] = {"ok": False, "reason": "reload_chunk_retriever missing"}
    except Exception as e:
        out["retriever"] = {"ok": False, "error": repr(e)}

    return out


def _rag_memory_only_backend() -> bool:
    return (os.environ.get("LEGAL_RAG_QUERY_BACKEND") or "").strip().lower() == "memory_only"


def reindex_is_successful(result: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, reason).

    Default: Chroma must succeed and retriever reload must report ok.
    With LEGAL_RAG_QUERY_BACKEND=memory_only: only retriever reload must succeed; Chroma is optional.
    """
    chroma = (result or {}).get("chroma") or {}
    retr = (result or {}).get("retriever") or {}

    if retr.get("ok") is False:
        return False, str(retr.get("error") or retr.get("reason") or "retriever reload failed")

    if _rag_memory_only_backend():
        return True, "ok"

    if chroma.get("skipped"):
        return False, "chroma backend unavailable"
    if chroma.get("ok") is False:
        return False, str(chroma.get("message") or "chroma reindex failed")
    if chroma.get("error"):
        return False, str(chroma.get("error"))
    return True, "ok"


def save_uploaded_pdf(upload_bytes: bytes, filename: str) -> Path:
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in filename if c.isalnum() or c in "._- ")[:180] or "law.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    dest = INCOMING_DIR / f"{_utc_stamp()}_{safe}"
    dest.write_bytes(upload_bytes)
    return dest


def _load_sync_state() -> Dict[str, Any]:
    if not SYNC_STATE_PATH.is_file():
        return {}
    try:
        return json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sync_state(obj: Dict[str, Any]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def poll_official_url_once() -> Dict[str, Any]:
    """
    Conditional GET for LAW_OFFICIAL_PDF_URL. If changed, download and run full_pipeline_from_pdf.
    """
    url = (os.environ.get("LAW_OFFICIAL_PDF_URL") or "").strip()
    if not url:
        return {"skipped": True, "reason": "LAW_OFFICIAL_PDF_URL not set"}

    import httpx

    state = _load_sync_state()
    headers: Dict[str, str] = {}
    etag = state.get("etag")
    lm = state.get("last_modified")
    if etag:
        headers["If-None-Match"] = str(etag)
    if lm:
        headers["If-Modified-Since"] = str(lm)

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(url, headers=headers)
        if r.status_code == 304:
            return {"changed": False, "status": 304}
        r.raise_for_status()
        body = r.content
        if len(body) < 100 or not body[:5].startswith(b"%PDF"):
            return {"changed": False, "error": "response is not a PDF", "len": len(body)}

        new_etag = r.headers.get("etag")
        new_lm = r.headers.get("last-modified")
        tmp = META_DIR / "incoming_poll.pdf"
        META_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(body)

        law_name = (os.environ.get("LAW_DEFAULT_NAME") or "").strip() or None
        result = full_pipeline_from_pdf(tmp, law_display_name=law_name, backup=True)
        state.update(
            {
                "etag": new_etag or state.get("etag"),
                "last_modified": new_lm or state.get("last_modified"),
                "last_ok_at": _utc_stamp(),
                "last_url": url,
            }
        )
        _save_sync_state(state)
        result["changed"] = True
        result["bytes"] = len(body)
        return result


_poll_stop = threading.Event()
_poll_thread: Optional[threading.Thread] = None


def start_background_poller() -> None:
    """Start daemon thread if LAW_POLL_ENABLED=1 and LAW_OFFICIAL_PDF_URL set."""
    if (os.environ.get("LAW_POLL_ENABLED", "").strip() != "1"):
        return
    if not (os.environ.get("LAW_OFFICIAL_PDF_URL") or "").strip():
        print("[law poll] LAW_POLL_ENABLED=1 but LAW_OFFICIAL_PDF_URL empty; skipping poller.")
        return
    global _poll_thread
    if _poll_thread and _poll_thread.is_alive():
        return

    try:
        interval = int(os.environ.get("LAW_POLL_INTERVAL_SECONDS", "86400"))
    except ValueError:
        interval = 86400
    interval = max(60, interval)

    def loop() -> None:
        while not _poll_stop.is_set():
            try:
                out = poll_official_url_once()
                if out.get("changed") or out.get("error") or out.get("skipped"):
                    print("[law poll]", out)
            except Exception as e:
                print("[law poll] error:", repr(e))
            if _poll_stop.wait(timeout=interval):
                break

    _poll_stop.clear()
    _poll_thread = threading.Thread(target=loop, daemon=True, name="law-official-poll")
    _poll_thread.start()
    print(f"[law poll] background poller started (interval={interval}s).")


def stop_background_poller() -> None:
    _poll_stop.set()
