"""
Mobile / Phase-5 feature API: explain clause, compare, negotiation chat, shares,
deal threads, timeline, legal profiles, signatures, risk summary.

LLM-heavy endpoints use **local LFM** only (``app.lfm_runtime`` / ``llm.generate``) — ready for adapter/LoRA fine-tuning.
Gemini is not used here.
"""
from __future__ import annotations

import json
import secrets
import difflib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin
from app.db.session import get_db
from app.db.models import (
    User,
    Analysis,
    AnalysisShare,
    DealThread,
    DealMessage,
    TimelineEvent,
    LegalProfile,
    SignatureRecord,
)
from app.lfm_runtime import lfm_failed_for_http, lfm_full_prompt


router = APIRouter(prefix="/legato", tags=["legato-mobile"])


def _load_analysis_json(db: Session, analysis_id: int, user: User) -> tuple[Analysis, Dict[str, Any]]:
    row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    is_admin = getattr(user, "role", "user") == "admin"
    if row.user_id != user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid analysis JSON")
    return row, data if isinstance(data, dict) else {}


# --- Explain clause ---


class ExplainClauseRequest(BaseModel):
    clause_text: str = Field(..., min_length=1)
    analysis_id: Optional[int] = None
    rule_id: Optional[str] = None


class ExplainClauseResponse(BaseModel):
    explanation: str
    disclaimer: str = "Informational only — not legal advice."


@router.post("/explain-clause", response_model=ExplainClauseResponse)
def explain_clause(
    payload: ExplainClauseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    context = ""
    if payload.analysis_id is not None:
        _, result = _load_analysis_json(db, payload.analysis_id, current_user)
        from app.routers.chat import _build_context

        context = _build_context(result)
    rule_hint = f"Focus on rule id: {payload.rule_id}\n" if payload.rule_id else ""
    prompt = f"""You are an Egyptian labor-law tutor (Law 14/2025 where relevant). Educational tone only — not legal advice.

Write the answer in Arabic first (main content). You may add one short English paragraph after if helpful.

Strict rules:
- Answer directly. Do NOT ask the user any questions. Do NOT output numbered lists of "questions to ask a lawyer."
- Use these sections with headings: (1) المعنى باختصار (2) نقاط الانتباه والمخاطر المحتملة (3) ملخص إنجليزي قصير (optional)

{rule_hint}
## Analysis context (may be empty)
{context[:8000]}

## Clause to explain
{payload.clause_text[:6000]}"""
    text = lfm_full_prompt(prompt, max_new_tokens=640)
    if lfm_failed_for_http(text):
        raise HTTPException(status_code=503, detail=text)
    return ExplainClauseResponse(explanation=text)


# --- Summarize clauses ---


class SummarizeClausesRequest(BaseModel):
    clauses: List[str] = Field(..., min_length=1)
    analysis_id: Optional[int] = None


class SummarizeClausesResponse(BaseModel):
    summaries: List[Dict[str, str]]


@router.post("/summarize-clauses", response_model=SummarizeClausesResponse)
def summarize_clauses(
    payload: SummarizeClausesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    extra = ""
    if payload.analysis_id is not None:
        _, result = _load_analysis_json(db, payload.analysis_id, current_user)
        from app.routers.chat import _build_context

        extra = _build_context(result)[:4000]
    out: List[Dict[str, str]] = []
    for i, c in enumerate(payload.clauses[:20]):
        prompt = f"""Summarize this employment-contract clause in 2-3 short bullet points (Arabic preferred, English ok).
Context (truncated): {extra}
Clause {i + 1}: {c[:4000]}"""
        s = lfm_full_prompt(prompt, max_new_tokens=256)
        if lfm_failed_for_http(s):
            raise HTTPException(status_code=503, detail=s)
        out.append({"index": str(i), "summary": s})
    return SummarizeClausesResponse(summaries=out)


# --- Compare ---


class CompareRequest(BaseModel):
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    analysis_id_a: Optional[int] = None
    analysis_id_b: Optional[int] = None


class CompareResponse(BaseModel):
    unified_diff: str
    lines_added: int
    lines_removed: int
    ai_summary: Optional[str] = None


def _extract_contract_text(result: Dict[str, Any]) -> str:
    chunks = result.get("ocr_chunks") or []
    parts: List[str] = []
    for c in chunks:
        t = (c.get("normalized_text") or c.get("text") or "") if isinstance(c, dict) else ""
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


@router.post("/compare", response_model=CompareResponse)
def compare_contracts(
    payload: CompareRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    a = (payload.text_a or "").strip()
    b = (payload.text_b or "").strip()
    if payload.analysis_id_a is not None:
        _, ra = _load_analysis_json(db, payload.analysis_id_a, current_user)
        a = _extract_contract_text(ra) or a
    if payload.analysis_id_b is not None:
        _, rb = _load_analysis_json(db, payload.analysis_id_b, current_user)
        b = _extract_contract_text(rb) or b
    if not a or not b:
        raise HTTPException(status_code=400, detail="Provide text_a & text_b or two analysis IDs with extractable text.")

    la = a.splitlines()
    lb = b.splitlines()
    diff = list(difflib.unified_diff(la, lb, lineterm="", n=2))
    added = sum(1 for x in diff if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in diff if x.startswith("-") and not x.startswith("---"))
    uni = "\n".join(diff[:5000])
    # When both versions are one long line (or line diff is empty) but texts differ, show word-level diff.
    if not uni.strip() and a != b:
        wa = a.split()
        wb = b.split()
        diff_w = list(difflib.unified_diff(wa, wb, lineterm="", n=1))
        uni = "\n".join(diff_w[:8000])
        added = sum(1 for x in diff_w if x.startswith("+") and not x.startswith("+++"))
        removed = sum(1 for x in diff_w if x.startswith("-") and not x.startswith("---"))

    summary_prompt = f"""Summarize the main differences between contract Version A and Version B for a lawyer.
Output 5–8 bullet points in Arabic; you may add one English line at the end if needed.
Do not invent facts — only what differs between the excerpts below.

Version A (first 3500 chars):
{a[:3500]}

Version B (first 3500 chars):
{b[:3500]}
"""
    ai_raw = lfm_full_prompt(summary_prompt, max_new_tokens=512)
    ai = ai_raw if not ai_raw.strip().startswith("[LFM unavailable]") else None

    return CompareResponse(
        unified_diff=uni[:100000],
        lines_added=added,
        lines_removed=removed,
        ai_summary=ai,
    )


# --- Negotiation chat ---


class NegotiationChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    analysis_id: Optional[int] = None
    history: Optional[List[Dict[str, str]]] = None


class NegotiationChatResponse(BaseModel):
    content: str


@router.post("/negotiation-chat", response_model=NegotiationChatResponse)
def negotiation_chat(
    payload: NegotiationChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ctx = ""
    if payload.analysis_id is not None:
        _, result = _load_analysis_json(db, payload.analysis_id, current_user)
        from app.routers.chat import _build_context

        ctx = _build_context(result)[:12000]
    hist = ""
    if payload.history:
        for m in payload.history[-8:]:
            role = m.get("role") or "user"
            hist += f"{role}: {m.get('content', '')}\n"
    prompt = f"""You are a negotiation coach for employment contracts (Egypt labor law context when relevant).
Suggest neutral, professional redlines and talking points — not adversarial attacks.
Always remind that final review belongs to qualified counsel.
If context is missing, give general negotiation guidance.

## Contract context
{ctx}

## History
{hist}

User: {payload.message}

Assistant:"""
    text = lfm_full_prompt(prompt, max_new_tokens=512)
    if lfm_failed_for_http(text):
        raise HTTPException(status_code=503, detail=text)
    return NegotiationChatResponse(content=text)


# --- Risk summary ---


@router.get("/risk/{analysis_id}")
def risk_summary(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _, data = _load_analysis_json(db, analysis_id, current_user)
    hits = data.get("rule_hits") or []
    risk = data.get("full_text_unified_risk")
    labor = data.get("labor_summary") or {}
    n = len(hits) if isinstance(hits, list) else 0
    severity = "high" if n > 5 else ("medium" if n > 0 else "low")
    return {
        "analysis_id": analysis_id,
        "hit_count": n,
        "full_text_unified_risk": risk,
        "severity_band": severity,
        "labor_summary_status": labor.get("status") if isinstance(labor, dict) else None,
    }


# --- Shares ---


class CreateShareRequest(BaseModel):
    analysis_id: int
    expires_days: int = Field(30, ge=1, le=365)


class CreateShareResponse(BaseModel):
    token: str
    share_path: str


@router.post("/shares", response_model=CreateShareResponse)
def create_share(
    payload: CreateShareRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == payload.analysis_id).first()
    if not row or row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    tok = secrets.token_urlsafe(24)
    exp = datetime.utcnow() + timedelta(days=payload.expires_days)
    sh = AnalysisShare(
        token=tok,
        analysis_id=row.id,
        owner_user_id=current_user.id,
        expires_at=exp,
    )
    db.add(sh)
    db.commit()
    return CreateShareResponse(token=tok, share_path=f"/legato/shares/public/{tok}")


@router.get("/shares/public/{token}")
def get_public_share(token: str, db: Session = Depends(get_db)):
    sh = db.query(AnalysisShare).filter(AnalysisShare.token == token).first()
    if not sh:
        raise HTTPException(status_code=404, detail="Not found")
    if sh.expires_at is not None and sh.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Link expired")
    row = db.query(Analysis).filter(Analysis.id == sh.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis missing")
    try:
        data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
    except Exception:
        data = {}
    return {
        "filename": row.filename,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "result": data,
        "disclaimer": "Shared read-only view — informational only.",
    }


# --- Deal threads ---


class CreateThreadRequest(BaseModel):
    analysis_id: int
    title: Optional[str] = None


class PostMessageRequest(BaseModel):
    body: str = Field(..., min_length=1)


@router.post("/deal-threads")
def create_thread(
    payload: CreateThreadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _, _ = _load_analysis_json(db, payload.analysis_id, current_user)
    t = DealThread(
        analysis_id=payload.analysis_id,
        title=payload.title or "Discussion",
        created_by_user_id=current_user.id,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {"id": t.id, "analysis_id": t.analysis_id, "title": t.title}


@router.get("/deal-threads")
def list_threads(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _, _ = _load_analysis_json(db, analysis_id, current_user)
    rows = db.query(DealThread).filter(DealThread.analysis_id == analysis_id).order_by(DealThread.created_at.desc()).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]


@router.get("/deal-threads/{thread_id}/messages")
def list_messages(
    thread_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    th = db.query(DealThread).filter(DealThread.id == thread_id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Thread not found")
    _, _ = _load_analysis_json(db, th.analysis_id, current_user)
    rows = (
        db.query(DealMessage)
        .filter(DealMessage.thread_id == thread_id)
        .order_by(DealMessage.created_at.asc())
        .all()
    )
    out = []
    for m in rows:
        u = db.query(User).filter(User.id == m.user_id).first()
        out.append(
            {
                "id": m.id,
                "user_id": m.user_id,
                "email": u.email if u else "",
                "body": m.body,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
        )
    return out


@router.post("/deal-threads/{thread_id}/messages")
def post_message(
    thread_id: int,
    payload: PostMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    th = db.query(DealThread).filter(DealThread.id == thread_id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Thread not found")
    _, _ = _load_analysis_json(db, th.analysis_id, current_user)
    m = DealMessage(thread_id=thread_id, user_id=current_user.id, body=payload.body[:20000])
    db.add(m)
    db.commit()
    db.refresh(m)
    return {"id": m.id, "created_at": m.created_at.isoformat() if m.created_at else None}


# --- Timeline ---


class TimelineEventCreate(BaseModel):
    analysis_id: int
    label: str = Field(..., min_length=1)
    event_date: str = Field(..., description="ISO date YYYY-MM-DD")
    source: str = "manual"


@router.post("/timeline/events")
def create_timeline_event(
    payload: TimelineEventCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _, _ = _load_analysis_json(db, payload.analysis_id, current_user)
    ev = TimelineEvent(
        analysis_id=payload.analysis_id,
        user_id=current_user.id,
        label=payload.label[:512],
        event_date=payload.event_date[:32],
        source=payload.source[:32] if payload.source in ("manual", "extracted") else "manual",
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return {"id": ev.id}


@router.get("/timeline/all")
def timeline_all(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    rows = db.query(TimelineEvent).order_by(TimelineEvent.event_date.desc()).limit(500).all()
    out = []
    for ev in rows:
        a = db.query(Analysis).filter(Analysis.id == ev.analysis_id).first()
        out.append(
            {
                "id": ev.id,
                "analysis_id": ev.analysis_id,
                "filename": a.filename if a else None,
                "label": ev.label,
                "event_date": ev.event_date,
                "source": ev.source,
                "created_at": ev.created_at.isoformat() if ev.created_at else None,
            }
        )
    return out


# --- Legal profile (network MVP) ---


class ProfilePayload(BaseModel):
    display_name: Optional[str] = None
    headline: Optional[str] = None
    organization: Optional[str] = None
    bio: Optional[str] = None


@router.get("/profile/me")
def get_profile_me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    p = db.query(LegalProfile).filter(LegalProfile.user_id == current_user.id).first()
    if not p:
        return {
            "user_id": current_user.id,
            "email": current_user.email,
            "display_name": None,
            "headline": None,
            "organization": None,
            "bio": None,
        }
    return {
        "user_id": p.user_id,
        "email": current_user.email,
        "display_name": p.display_name,
        "headline": p.headline,
        "organization": p.organization,
        "bio": p.bio,
    }


@router.put("/profile/me")
def put_profile_me(
    payload: ProfilePayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    p = db.query(LegalProfile).filter(LegalProfile.user_id == current_user.id).first()
    if not p:
        p = LegalProfile(user_id=current_user.id)
        db.add(p)
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        if hasattr(p, k) and k != "user_id":
            setattr(p, k, v)
    p.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(p)
    return {"status": "ok"}


@router.get("/network/profiles")
def list_network_profiles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    rows = db.query(LegalProfile).order_by(LegalProfile.updated_at.desc()).limit(200).all()
    out = []
    for p in rows:
        u = db.query(User).filter(User.id == p.user_id).first()
        out.append(
            {
                "user_id": p.user_id,
                "email": u.email if u else "",
                "display_name": p.display_name,
                "headline": p.headline,
                "organization": p.organization,
            }
        )
    return out


# --- E-sign (in-app record, not DocuSign) ---


class SignatureRequest(BaseModel):
    analysis_id: int
    signer_name: str = Field(..., min_length=1)
    consent_acknowledged: bool = False
    signature_png_base64: Optional[str] = None


@router.post("/signatures")
def record_signature(
    payload: SignatureRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not payload.consent_acknowledged:
        raise HTTPException(status_code=400, detail="consent_acknowledged must be true")
    _, _ = _load_analysis_json(db, payload.analysis_id, current_user)
    meta = {
        "signer_name": payload.signer_name,
        "has_image": bool(payload.signature_png_base64),
        "image_len": len(payload.signature_png_base64 or ""),
    }
    rec = SignatureRecord(
        analysis_id=payload.analysis_id,
        user_id=current_user.id,
        signer_name=payload.signer_name[:255],
        consent_version="legato-v1",
        payload_json=json.dumps(meta, ensure_ascii=False),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {
        "id": rec.id,
        "message": "Signature record stored in-app. For legally binding e-sign, integrate a certified provider separately.",
    }
