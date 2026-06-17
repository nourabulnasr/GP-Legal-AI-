# -*- coding: utf-8 -*-
"""Shared logic for /legato/* routes: RAG + LFM explain, risk, summaries.

Imports from app.main are deferred to request time to avoid circular import at startup.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.models import Analysis, User
from app.utils_text import detect_language, llm_locale_from_detection, norm_ar

# Small LRU for identical explain requests (demo path perf; no DB snapshot in key).
_EXPLAIN_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_EXPLAIN_CACHE_MAX = 32


def _normalize_for_match(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _pick_rule_hit(
    rule_hits: List[Dict[str, Any]],
    *,
    rule_id: Optional[str],
    clause_text: str,
) -> Optional[Dict[str, Any]]:
    """Resolve the best rule hit for explain-clause (avoid always picking the first error)."""
    hits = list(rule_hits or [])
    rid = (rule_id or "").strip()
    if rid:
        for h in hits:
            if (h.get("rule_id") or h.get("id")) == rid:
                return h

    needle = _normalize_for_match(clause_text)
    if needle:
        for h in hits:
            matched = _normalize_for_match(h.get("matched_text") or "")
            if matched and (needle in matched or matched in needle):
                return h
        best: Optional[Dict[str, Any]] = None
        best_score = 0
        needle_words = {w for w in needle.split() if len(w) > 2}
        if needle_words:
            for h in hits:
                matched = _normalize_for_match(h.get("matched_text") or "")
                if not matched:
                    continue
                score = sum(1 for w in needle_words if w in matched)
                if score > best_score:
                    best_score = score
                    best = h
            if best is not None and best_score >= 2:
                return best

    if hits:
        return next(
            (h for h in hits if (h.get("severity") in ("error", "high"))),
            hits[0],
        )
    return None


def _hits_to_law_articles(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for x in hits or []:
        out.append({"text": x.get("text", ""), "metadata": x.get("metadata") or {}})
    return out


def _live_rag_for_violation(hit: Dict[str, Any], clause_text: str, contract_tags: List[str]) -> List[Dict[str, Any]]:
    from app.main import (
        build_rag_query,
        _rag_search_labor_corpus,
        _rag_min_score,
        _filter_to_labor_only,
        _rag_backend_ready,
        _dedupe_rag_hits_by_metadata,
        _rag_prioritize_hits_for_rule_hit,
    )

    if not _rag_backend_ready():
        return []

    rid = (hit.get("rule_id") or "").upper()
    desc = hit.get("description") or ""
    matched_short = ((hit.get("matched_text") or "") or "")[:220]
    boosts: List[str] = []
    if "WORKING_HOURS" in rid:
        boosts = ["ساعات العمل", "الحد الأقصى", "8 ساعات", "قانون العمل 14 لسنة 2025"]
    elif "ANNUAL_LEAVE" in rid:
        boosts = ["الإجازة السنوية", "الحد الأدنى", "15 يوم", "قانون العمل 14 لسنة 2025"]
    elif "PROBATION" in rid:
        boosts = ["فترة الاختبار", "3 أشهر", "قانون العمل 14 لسنة 2025"]
    elif "SALARY" in rid:
        boosts = ["الأجر", "طريقة الصرف", "موعد صرف الأجر", "قانون العمل 14 لسنة 2025"]
    else:
        boosts = ["قانون العمل", "عقد العمل"]

    q = " ".join([desc] + boosts + ([matched_short] if matched_short else []))
    qv = norm_ar(q)
    qv_boosted = build_rag_query(qv, clause_text, contract_tags)
    ms = _rag_min_score()
    hits_v = _rag_search_labor_corpus(qv_boosted, top_k=14, min_score=ms)
    hits_v = _filter_to_labor_only(_dedupe_rag_hits_by_metadata(hits_v))
    hits_v = _rag_prioritize_hits_for_rule_hit(hits_v, hit)
    return hits_v[:4]


def run_explain_clause(
    *,
    clause_text: str,
    language: Optional[str],
    analysis_id: Optional[int],
    rule_id: Optional[str],
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    """Returns dict with explanation, language, rule_id, sources (optional)."""
    text = (clause_text or "").strip()
    if not text:
        raise ValueError("clause_text is required")

    lang = (language or "").strip().lower() if language else ""
    if lang not in ("ar", "en"):
        lang = llm_locale_from_detection(detect_language(text))

    if analysis_id is None and len(text) <= 6000:
        ck = hashlib.sha256(
            f"{text}|{lang}|{rule_id or ''}|{current_user.id}".encode("utf-8")
        ).hexdigest()
        if ck in _EXPLAIN_CACHE:
            _EXPLAIN_CACHE.move_to_end(ck)
            return dict(_EXPLAIN_CACHE[ck])

    try:
        from app import local_llm as llm
    except Exception as e:
        raise RuntimeError(f"local_llm import failed: {e}") from e

    if not getattr(llm, "is_available", lambda: False)():
        raise RuntimeError("Local LFM not available (set LOCAL_LLM_PATH or add LFM2.5-1.2B-Instruct).")

    explain_fn = getattr(llm, "explain_violation", None)
    if not explain_fn:
        raise RuntimeError("explain_violation missing on local_llm")

    rule_id_eff = rule_id or ""
    description = "User requested explanation for this clause."
    matched = text[:2000]
    law_articles: List[Dict[str, Any]] = []
    contract_tags: List[str] = []
    data: Optional[Dict[str, Any]] = None

    if analysis_id is not None:
        row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not row:
            raise ValueError("analysis not found")
        is_admin = getattr(current_user, "role", "user") == "admin"
        if row.user_id != current_user.id and not is_admin:
            raise PermissionError("forbidden")
        try:
            data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
        except Exception:
            data = {}

        rule_hits = data.get("rule_hits") or []
        contract_tags = list(data.get("contract_tags") or [])
        hit = _pick_rule_hit(rule_hits, rule_id=rule_id_eff or None, clause_text=text)
        if hit:
            rule_id_eff = str(hit.get("rule_id") or hit.get("id") or "CLAUSE")
            description = (hit.get("description") or description)[:2000]
            matched = (hit.get("matched_text") or text)[:2000]
            rag_by = data.get("rag_by_violation") or []
            block = next((b for b in rag_by if b.get("rule_id") == rule_id_eff), None)
            if block and (block.get("hits") or []):
                law_articles = _hits_to_law_articles(block.get("hits") or [])
            else:
                law_articles = _hits_to_law_articles(_live_rag_for_violation(hit, text, contract_tags))

    if not law_articles:
        from app.main import (
            _rag_search_labor_corpus,
            _rag_min_score,
            _filter_to_labor_only,
            _rag_backend_ready,
            build_rag_query,
            _dedupe_rag_hits_by_metadata,
        )

        if _rag_backend_ready():
            boosted = build_rag_query(norm_ar(text[:400]), text, contract_tags)
            ms = _rag_min_score()
            hits = _rag_search_labor_corpus(boosted, top_k=10, min_score=ms)
            hits = _filter_to_labor_only(_dedupe_rag_hits_by_metadata(hits))[:4]
            law_articles = _hits_to_law_articles(hits)

    if not rule_id_eff:
        rule_id_eff = "CLAUSE_REVIEW"

    expl = explain_fn(
        rule_id=rule_id_eff,
        description=description,
        matched_text=matched,
        law_articles=law_articles,
        max_new_tokens=400,
        language=lang,
    )
    out = {
        "explanation": expl,
        "language": lang,
        "rule_id": rule_id_eff,
        "sources": [{"article": (a.get("metadata") or {}).get("article"), "law": (a.get("metadata") or {}).get("law")} for a in law_articles[:6]],
    }
    if analysis_id is None and len(text) <= 6000:
        ck = hashlib.sha256(
            f"{text}|{lang}|{rule_id or ''}|{current_user.id}".encode("utf-8")
        ).hexdigest()
        _EXPLAIN_CACHE[ck] = dict(out)
        _EXPLAIN_CACHE.move_to_end(ck)
        while len(_EXPLAIN_CACHE) > _EXPLAIN_CACHE_MAX:
            _EXPLAIN_CACHE.popitem(last=False)
    return out


def risk_payload_from_analysis(analysis_id: int, db: Session, current_user: User) -> Dict[str, Any]:
    row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not row:
        raise ValueError("analysis not found")
    is_admin = getattr(current_user, "role", "user") == "admin"
    if row.user_id != current_user.id and not is_admin:
        raise PermissionError("forbidden")
    try:
        data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
    except Exception:
        data = {}
    rule_hits = data.get("rule_hits") or []
    sev_counts: Dict[str, int] = {}
    for h in rule_hits:
        s = str(h.get("severity") or "unknown")
        sev_counts[s] = sev_counts.get(s, 0) + 1
    error_count = sev_counts.get("error", 0) + sev_counts.get("high", 0)
    warning_count = sev_counts.get("warning", 0)
    info_count = sev_counts.get("info", 0)
    return {
        "analysis_id": analysis_id,
        # Flutter-expected keys
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "total_hits": len(rule_hits),
        "unified_ml_risk": data.get("full_text_unified_risk"),
        "rule_hits": rule_hits,
        "needs_review": bool(row.needs_review),
        # backward-compat
        "rule_hits_count": len(rule_hits),
        "severity_counts": sev_counts,
        "full_text_unified_risk": data.get("full_text_unified_risk"),
        "clause_level_unified_risks": data.get("clause_level_unified_risks"),
        "unified_ml_above_threshold": data.get("unified_ml_above_threshold"),
        "pipeline_steps": data.get("pipeline_steps"),
    }


def summarize_clauses_llm(
    clauses: List[str],
    language: str,
) -> List[Dict[str, Any]]:
    from app import local_llm as llm

    if not getattr(llm, "is_available", lambda: False)():
        raise RuntimeError("Local LFM not available")
    gen = getattr(llm, "generate", None)
    if not gen:
        raise RuntimeError("local_llm.generate missing")

    out: List[Dict[str, Any]] = []
    lang = (
        language
        if language in ("ar", "en")
        else llm_locale_from_detection(detect_language((clauses[0] or "")[:4000] if clauses else ""))
    )
    for i, c in enumerate(clauses):
        c = (c or "").strip()
        if not c:
            continue
        if lang == "ar":
            prompt = f"لخص البند التالي في جملة أو جملتين كحد أقصى، بدون إضافة قوانين مخترعة:\n\n{c[:4000]}\n\nالملخص:"
        else:
            prompt = f"Summarize the following contract clause in at most two sentences. Do not invent law citations:\n\n{c[:4000]}\n\nSummary:"
        summary = gen(prompt, max_new_tokens=200, do_sample=False)
        out.append({"clause_index": i, "summary": summary.strip()})
    return out


def contract_text_from_result(data: Dict[str, Any], *, max_chars: int = 12000) -> str:
    """Best-effort contract body from a saved analysis result_json."""
    full_text = str(data.get("full_text") or "").strip()
    chunks = data.get("ocr_chunks") or []

    def _chunk_text(chunk: Any) -> str:
        if not isinstance(chunk, dict):
            return ""
        return str(
            chunk.get("translated_ar_text")
            or chunk.get("translated_text")
            or chunk.get("normalized_text")
            or chunk.get("text")
            or ""
        ).strip()

    from_chunks = "\n\n".join(t for c in chunks if (t := _chunk_text(c)))
    body = from_chunks if len(from_chunks) >= len(full_text) else full_text
    if not body and from_chunks:
        body = from_chunks
    return body[:max_chars] if body else ""


def _comparison_is_usable(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 40:
        return False
    if t.startswith("[LLM") or t.startswith("[Gemini"):
        return False
    alpha = sum(1 for c in t if c.isalpha())
    return alpha >= 20


def _compare_with_gemini(text_a: str, text_b: str, language: str) -> str:
    from app.routers.chat import _get_gemini_client, _gemini_generate

    client = _get_gemini_client()
    if not client:
        return ""
    ta = (text_a or "")[:6000]
    tb = (text_b or "")[:6000]
    if language == "ar":
        prompt = (
            "قارن بين عقدي العمل التاليين. اذكر أوجه التشابه والاختلاف الرئيسية في نقاط مرقمة "
            "(الراتب، ساعات العمل، الإجازات، فترة التجربة، إنهاء العقد، إلخ). "
            "لا تخترع بنوداً غير موجودة في النص.\n\n"
            f"العقد أ:\n{ta}\n\nالعقد ب:\n{tb}"
        )
    else:
        prompt = (
            "Compare the following two employment contracts. List key similarities and differences "
            "in bullet points (salary, hours, leave, probation, termination, etc.). "
            "Do not invent clauses not present in the text.\n\n"
            f"Contract A:\n{ta}\n\nContract B:\n{tb}"
        )
    try:
        return _gemini_generate(client, prompt).strip()
    except Exception:
        return ""


def compare_contracts_llm(text_a: str, text_b: str, language: str) -> str:
    from app import local_llm as llm

    lang = (
        language
        if language in ("ar", "en")
        else llm_locale_from_detection(detect_language((text_a or "") + "\n\n" + (text_b or ""))[:8000])
    )
    max_each = int(os.getenv("COMPARE_MAX_CHARS_EACH", "1800"))
    ta = (text_a or "")[:max_each]
    tb = (text_b or "")[:max_each]

    comparison = ""
    if getattr(llm, "is_available", lambda: False)():
        gen = getattr(llm, "generate", None)
        if gen:
            if lang == "ar":
                prompt = (
                    "قارن بين النصين التاليين من عقود العمل. "
                    "اذكر أوجه التشابه والاختلاف الرئيسية في نقاط مختصرة:\n\n"
                    f"النص أ:\n{ta}\n\nالنص ب:\n{tb}\n\nالمقارنة:"
                )
            else:
                prompt = (
                    "Compare the following two employment contract excerpts. "
                    "List key similarities and differences briefly:\n\n"
                    f"Text A:\n{ta}\n\nText B:\n{tb}\n\nComparison:"
                )
            comparison = gen(prompt, max_new_tokens=512, do_sample=False).strip()
            comparison = re.sub(r"^\.\.\.\s*\)\.?\s*$", "", comparison).strip()

    if _comparison_is_usable(comparison):
        return comparison

    gemini = _compare_with_gemini(text_a, text_b, lang)
    if _comparison_is_usable(gemini):
        return gemini

    if comparison:
        return comparison
    if gemini:
        return gemini
    raise RuntimeError("Could not generate a contract comparison. Try again later.")


def new_share_token() -> str:
    return secrets.token_urlsafe(24)


def default_expires_at(days: int = 30) -> datetime:
    return datetime.utcnow() + timedelta(days=days)
