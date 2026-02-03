# -*- coding: utf-8 -*-
"""
Optional jobs router. Placeholder for background tasks (e.g. reindex, ingest).
Mount is optional; if this module is missing, main.py skips it cleanly.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/health")
def jobs_health():
    """Health check for jobs service (placeholder)."""
    return {"status": "ok", "service": "jobs"}
