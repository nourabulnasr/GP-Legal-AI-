# -*- coding: utf-8 -*-
"""
Opt-in HTTP tests for POST /ocr_check_and_search (heavy: imports full app).

From project root (uses in-process TestClient; no separate server required):
  set RUN_OCR_INTEGRATION=1
  python -m pytest app/tests/test_ocr_check_and_search_integration.py -v

When Google credentials are configured, expects translation_provider google_cloud_translate_v2
when the client is available; otherwise local_lfm_translate or skip depending on runtime setup.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_FIXTURE_PDF = Path(__file__).resolve().parent.parent.parent / "test_fixtures" / "sample_contract_de.pdf"


@pytest.mark.integration
def test_ocr_check_and_search_language_detection_and_translation_payload():
    if os.getenv("RUN_OCR_INTEGRATION", "").strip().lower() not in ("1", "true", "yes"):
        pytest.skip("Set RUN_OCR_INTEGRATION=1 to run (imports app.main; may be slow).")

    from fastapi.testclient import TestClient

    from app.main import app

    assert _FIXTURE_PDF.is_file(), f"missing fixture: {_FIXTURE_PDF}"

    client = TestClient(app)
    with open(_FIXTURE_PDF, "rb") as f:
        data = f.read()

    files = {"file": ("sample_contract_de.pdf", data, "application/pdf")}
    form = {
        "save": "false",
        "translation_only": "true",
        "use_rag": "false",
        "use_ml": "false",
        "use_llm": "false",
        "translate_to_ar": "true",
        "translate_per_chunk_mt": "false",
    }

    r = client.post("/ocr_check_and_search", files=files, data=form)
    assert r.status_code == 200, r.text[:2000]
    body = r.json()

    ld = body.get("language_detection")
    assert isinstance(ld, dict), body.keys()
    assert "language_code" in ld
    assert "confidence" in ld
    assert "is_mixed" in ld

    tx = body.get("translation")
    assert isinstance(tx, dict)
    assert "translation_provider" in tx
    assert "translation_status" in tx

    def _google_translate_v2_importable() -> bool:
        try:
            from google.cloud import translate_v2 as _tv2  # noqa: F401

            return True
        except Exception:
            return False

    creds_configured = bool(
        (os.getenv("GOOGLE_TRANSLATION_API_KEY") or "").strip()
        or (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    )
    mt_allowed = os.getenv("EXTERNAL_MT_DISABLED", "").strip().lower() not in ("1", "true", "yes")
    google_not_disabled = os.getenv("DISABLE_GOOGLE_MT", "").strip().lower() not in ("1", "true", "yes")
    expect_google = creds_configured and mt_allowed and google_not_disabled and _google_translate_v2_importable()
    if expect_google:
        assert tx.get("translation_provider") in (
            "google_cloud_translate_v2",
            "local_lfm_translate",
            "argos_translate",
        ), tx
    elif os.getenv("ENABLE_LOCAL_LLM_TRANSLATION", "1").strip().lower() not in ("0", "false", "no"):
        assert tx.get("translation_provider") in ("local_lfm_translate", None, "none"), tx

    chunks = body.get("ocr_chunks") or []
    assert isinstance(chunks, list) and len(chunks) >= 1
    if tx.get("translation_provider") and tx.get("translation_status") not in ("disabled", "skipped"):
        assert any(
            isinstance(c, dict) and str(c.get("translated_ar_text") or "").strip()
            for c in chunks
        ), "expected translated_ar_text on at least one chunk when MT ran"
