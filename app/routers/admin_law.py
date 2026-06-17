# -*- coding: utf-8 -*-
"""Admin-only endpoints to update labor law corpus and rebuild RAG indexes."""
from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.deps import require_admin
from app.db.models import User

router = APIRouter(prefix="/admin/law", tags=["admin-law"])

MAX_PDF_BYTES = int(os.environ.get("LAW_UPLOAD_MAX_BYTES", str(35 * 1024 * 1024)))

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        base = dict(_jobs.get(job_id, {}))
        base.update(kwargs)
        _jobs[job_id] = base


def _run_pdf_job(job_id: str, pdf_path: str, law_name: Optional[str]) -> None:
    from pathlib import Path

    from app.law_update_service import full_pipeline_from_pdf, reindex_is_successful

    _set_job(job_id, status="running")
    try:
        out = full_pipeline_from_pdf(Path(pdf_path), law_display_name=law_name, backup=True)
        ok, reason = reindex_is_successful(out)
        if ok:
            _set_job(job_id, status="done", result=out)
        else:
            _set_job(job_id, status="error", result=out, error=reason)
    except Exception as e:
        _set_job(job_id, status="error", error=repr(e))


def _run_reindex_job(job_id: str) -> None:
    from app.law_update_service import full_pipeline_from_articles_json, reindex_is_successful

    _set_job(job_id, status="running")
    try:
        out = full_pipeline_from_articles_json(backup=True)
        ok, reason = reindex_is_successful(out)
        if ok:
            _set_job(job_id, status="done", result=out)
        else:
            _set_job(job_id, status="error", result=out, error=reason)
    except Exception as e:
        _set_job(job_id, status="error", error=repr(e))


@router.post("/preview-pdf")
def preview_law_pdf(
    file: UploadFile = File(...),
    law_display_name: Optional[str] = Form(None),
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    from pathlib import Path
    import tempfile

    from app.law_update_service import preview_pdf_articles

    raw = file.file.read(MAX_PDF_BYTES + 1)
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF too large")
    if not raw[:5].startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        path = Path(tmp.name)
    try:
        return preview_pdf_articles(path, law_display_name=law_display_name)
    finally:
        try:
            path.unlink()
        except Exception:
            pass


@router.post("/upload-pdf")
def upload_law_pdf(
    file: UploadFile = File(...),
    law_display_name: Optional[str] = Form(None),
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    from pathlib import Path

    from app.law_update_service import save_uploaded_pdf

    raw = file.file.read(MAX_PDF_BYTES + 1)
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF too large")
    if not raw[:5].startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")

    saved = save_uploaded_pdf(raw, file.filename or "law.pdf")
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", pdf=str(saved))
    t = threading.Thread(
        target=_run_pdf_job,
        args=(job_id, str(saved), law_display_name),
        daemon=True,
        name=f"law-pdf-{job_id[:8]}",
    )
    t.start()
    return {"job_id": job_id, "status": "queued", "saved_pdf": str(saved)}


@router.post("/reindex")
def reindex_law_rag(_: User = Depends(require_admin)) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", mode="reindex")
    t = threading.Thread(target=_run_reindex_job, args=(job_id,), daemon=True, name=f"law-reindex-{job_id[:8]}")
    t.start()
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_law_job(job_id: str, _: User = Depends(require_admin)) -> Dict[str, Any]:
    with _jobs_lock:
        j = dict(_jobs.get(job_id) or {})
    if not j:
        raise HTTPException(404, "Unknown job_id")
    return {"job_id": job_id, **j}


@router.get("/download-chunks-cleaned")
def download_chunks_cleaned(_: User = Depends(require_admin)) -> FileResponse:
    """Serve the cleaned labor-law chunk JSONL produced by preprocess (same file used for RAG ingest)."""
    from app.law_update_service import resolved_cleaned_chunks_path

    path = resolved_cleaned_chunks_path()
    if path is None:
        raise HTTPException(
            404,
            detail="No cleaned chunks JSONL found. Run Upload and rebuild or Reindex only first.",
        )
    return FileResponse(
        path=str(path.resolve()),
        filename=path.name,
        media_type="application/x-ndjson; charset=utf-8",
    )


@router.post("/sync-from-url")
def sync_law_from_url(
    _: User = Depends(require_admin),
) -> Dict[str, Any]:
    """One-shot poll of LAW_OFFICIAL_PDF_URL (admin only)."""
    from app.law_update_service import poll_official_url_once

    return poll_official_url_once()
