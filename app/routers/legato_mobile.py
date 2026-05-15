# -*- coding: utf-8 -*-
"""Mobile Legato API: /legato/* — explain, risk, shares, deal threads, profile, etc."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin
from app.db.models import (
    Analysis,
    User,
    LegatoShare,
    LegatoDealThread,
    LegatoDealMessage,
    LegatoTimelineEvent,
    LegatoProfile,
    LegatoSignature,
)
from app.db.session import get_db
from app import legato_service
from app.utils_text import detect_language, llm_locale_from_detection

router = APIRouter(prefix="/legato", tags=["legato-mobile"])


# ---------- explain clause ----------
class ExplainClauseBody(BaseModel):
    clause_text: str = Field(..., min_length=1)
    analysis_id: Optional[int] = None
    rule_id: Optional[str] = None
    language: Optional[str] = Field(default=None, description="ar or en; omit to auto-detect")


@router.post("/explain-clause")
def explain_clause(
    body: ExplainClauseBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return legato_service.run_explain_clause(
            clause_text=body.clause_text,
            language=body.language,
            analysis_id=body.analysis_id,
            rule_id=body.rule_id,
            db=db,
            current_user=current_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- risk ----------
@router.get("/risk/{analysis_id}")
def risk_summary(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return legato_service.risk_payload_from_analysis(analysis_id, db, current_user)
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------- summarize ----------
class SummarizeBody(BaseModel):
    clauses: List[str] = Field(..., min_length=1)
    analysis_id: Optional[int] = None
    language: Optional[str] = None


@router.post("/summarize-clauses")
def summarize_clauses(
    body: SummarizeBody,
    current_user: User = Depends(get_current_user),
):
    lang = (body.language or "ar").lower()
    if lang not in ("ar", "en"):
        sample = "\n\n".join((c or "").strip() for c in (body.clauses or [])[:5])[:8000]
        lang = llm_locale_from_detection(detect_language(sample))
    try:
        summaries = legato_service.summarize_clauses_llm(body.clauses, lang)
        return {"summaries": summaries, "language": lang}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- compare ----------
class CompareBody(BaseModel):
    text_a: Optional[str] = None
    text_b: Optional[str] = None
    analysis_id_a: Optional[int] = None
    analysis_id_b: Optional[int] = None
    language: Optional[str] = None


@router.post("/compare")
def compare_contracts(
    body: CompareBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    def _load_text(aid: Optional[int]) -> str:
        if aid is None:
            return ""
        row = db.query(Analysis).filter(Analysis.id == aid).first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Analysis {aid} not found")
        adm = getattr(current_user, "role", "user") == "admin"
        if row.user_id != current_user.id and not adm:
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            data = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
        except Exception:
            data = {}
        parts = data.get("ocr_chunks") or []
        return "\n\n".join(str(p.get("normalized_text") or p.get("text") or "") for p in parts)[:12000]

    ta = (body.text_a or "").strip() or _load_text(body.analysis_id_a)
    tb = (body.text_b or "").strip() or _load_text(body.analysis_id_b)
    if not ta or not tb:
        raise HTTPException(status_code=400, detail="Provide text_a/text_b or both analysis ids")
    lang = (body.language or "").strip().lower()
    if not lang:
        lang = llm_locale_from_detection(detect_language((ta or "") + "\n\n" + (tb or ""))[:8000])
    elif lang not in ("ar", "en"):
        lang = llm_locale_from_detection(detect_language((ta or "") + "\n\n" + (tb or ""))[:8000])
    try:
        comparison = legato_service.compare_contracts_llm(ta, tb, lang)
        return {"comparison": comparison, "language": lang}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- negotiation (Gemini; same family as /chat) ----------
class NegotiationBody(BaseModel):
    message: str = Field(..., min_length=1)
    analysis_id: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None


def _negotiation_gemini(context: str, message: str, history: Optional[List[Dict[str, Any]]]) -> str:
    from app.routers.chat import _get_gemini_client, _gemini_generate

    client_or_legacy = _get_gemini_client()
    if not client_or_legacy:
        return ""
    sys = (
        "You are a negotiation coach for employment contracts. Help the user strategize professionally. "
        "If contract context is provided, ground suggestions in it. Do not invent law article numbers.\n\n"
        f"Context:\n{context[:20000]}"
    )
    hist = ""
    if history:
        for m in history[-8:]:
            role = "User" if str(m.get("role", "")).lower() == "user" else "Assistant"
            hist += f"{role}: {m.get('content', '')}\n"
    full_prompt = f"{sys}\n\n{hist}User: {message}\n\nAssistant:"
    try:
        return _gemini_generate(client_or_legacy, full_prompt)
    except Exception as e:
        return f"[Negotiation chat error: {e!r}]"


@router.post("/negotiation-chat")
def negotiation_chat(
    body: NegotiationBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.routers.chat import _build_context

    context = ""
    if body.analysis_id is not None:
        row = db.query(Analysis).filter(Analysis.id == body.analysis_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        adm = getattr(current_user, "role", "user") == "admin"
        if row.user_id != current_user.id and not adm:
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            result = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid analysis data")
        context = _build_context(result)

    content = _negotiation_gemini(context, body.message, body.history)
    if not content or content.startswith("[Negotiation chat error:"):
        # Fallback: short LFM reply
        try:
            from app import local_llm as llm

            if getattr(llm, "is_available", lambda: False)():
                prompt = f"Negotiation advice (no Gemini API). Be brief.\n\n{context[:4000]}\n\nUser: {body.message}\n\nAdvice:"
                content = llm.generate(prompt, max_new_tokens=256, do_sample=False)
            else:
                content = "Set GEMINI_API_KEY for negotiation chat, or configure local LFM."
        except Exception as e:
            content = f"[Negotiation unavailable: {e!r}]"
    return {"content": content}


# ---------- shares ----------
class ShareCreateBody(BaseModel):
    analysis_id: int = Field(..., ge=1)
    expires_days: int = Field(default=30, ge=1, le=365)


@router.post("/shares")
def create_share(
    body: ShareCreateBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == body.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if row.user_id != current_user.id and getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    token = legato_service.new_share_token()
    exp = legato_service.default_expires_at(body.expires_days)
    sh = LegatoShare(
        token=token,
        analysis_id=body.analysis_id,
        user_id=current_user.id,
        expires_at=exp,
    )
    db.add(sh)
    db.commit()
    db.refresh(sh)
    return {"token": token, "expires_at": exp.isoformat() + "Z", "share_id": sh.id}


@router.get("/shares/public/{token}")
def public_share(token: str, db: Session = Depends(get_db)):
    sh = db.query(LegatoShare).filter(LegatoShare.token == token).first()
    if not sh:
        raise HTTPException(status_code=404, detail="Not found")
    if sh.expires_at and datetime.utcnow() > sh.expires_at:
        raise HTTPException(status_code=410, detail="Share link expired")
    row = db.query(Analysis).filter(Analysis.id == sh.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis missing")
    try:
        payload = json.loads(row.result_json) if isinstance(row.result_json, str) else row.result_json
    except Exception:
        payload = {}
    return {
        "analysis_id": row.id,
        "filename": row.filename,
        "result": payload,
        "shared_at": sh.created_at.isoformat() + "Z" if sh.created_at else None,
    }


# ---------- deal threads ----------
class DealThreadCreate(BaseModel):
    analysis_id: int
    title: Optional[str] = None


@router.post("/deal-threads")
def deal_thread_create(
    body: DealThreadCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == body.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    # Shared threads: allow any authenticated user to start a discussion on an analysis id.
    th = LegatoDealThread(analysis_id=body.analysis_id, user_id=current_user.id, title=body.title or "Discussion")
    db.add(th)
    db.commit()
    db.refresh(th)
    return {"id": th.id, "analysis_id": th.analysis_id, "title": th.title, "created_at": th.created_at.isoformat() + "Z"}


@router.get("/deal-threads")
def deal_thread_list(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    # Shared threads: any authenticated user may list threads for a given analysis id.
    threads = (
        db.query(LegatoDealThread)
        .filter(LegatoDealThread.analysis_id == analysis_id)
        .order_by(LegatoDealThread.created_at.desc())
        .all()
    )
    return [
        {
            "id": t.id,
            "analysis_id": t.analysis_id,
            "title": t.title,
            "created_at": t.created_at.isoformat() + "Z",
            "created_by": t.user_id,
        }
        for t in threads
    ]


class DealMessageBody(BaseModel):
    body: str = Field(..., min_length=1)


@router.get("/deal-threads/{thread_id}/messages")
def deal_messages_list(
    thread_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    th = db.query(LegatoDealThread).filter(LegatoDealThread.id == thread_id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Thread not found")
    # Shared threads: any authenticated user may read messages.
    msgs = (
        db.query(LegatoDealMessage)
        .filter(LegatoDealMessage.thread_id == thread_id)
        .order_by(LegatoDealMessage.created_at.asc())
        .all()
    )
    out = []
    for m in msgs:
        email = ""
        if getattr(m, "author_id", None):
            u = db.query(User).filter(User.id == m.author_id).first()
            email = u.email if u else ""
        out.append(
            {
                "id": m.id,
                "author_id": getattr(m, "author_id", None),
                "email": email,
                "body": m.body,
                "created_at": m.created_at.isoformat() + "Z",
            }
        )
    return out


@router.post("/deal-threads/{thread_id}/messages")
def deal_message_post(
    thread_id: int,
    body: DealMessageBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    th = db.query(LegatoDealThread).filter(LegatoDealThread.id == thread_id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Thread not found")
    # Shared threads: any authenticated user may post.
    m = LegatoDealMessage(thread_id=thread_id, body=body.body, author_id=current_user.id)
    db.add(m)
    db.commit()
    db.refresh(m)
    return {"id": m.id, "author_id": current_user.id, "email": current_user.email, "body": m.body, "created_at": m.created_at.isoformat() + "Z"}


# ---------- timeline ----------
class TimelineEventBody(BaseModel):
    analysis_id: int
    label: str = Field(..., min_length=1)
    event_date: str = Field(..., description="ISO date string")
    source: str = "manual"


@router.post("/timeline/events")
def timeline_create(
    body: TimelineEventBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == body.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if row.user_id != current_user.id and getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    ev = LegatoTimelineEvent(
        analysis_id=body.analysis_id,
        user_id=current_user.id,
        label=body.label,
        event_date=body.event_date,
        source=(body.source or "manual")[:64],
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return {"id": ev.id, "label": ev.label, "event_date": ev.event_date, "created_at": ev.created_at.isoformat() + "Z"}


@router.get("/timeline/all")
def timeline_admin_all(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    rows = db.query(LegatoTimelineEvent).order_by(LegatoTimelineEvent.created_at.desc()).limit(500).all()
    return [
        {
            "id": r.id,
            "analysis_id": r.analysis_id,
            "user_id": r.user_id,
            "label": r.label,
            "event_date": r.event_date,
            "source": r.source,
            "created_at": r.created_at.isoformat() + "Z",
        }
        for r in rows
    ]


@router.get("/timeline/me")
def timeline_me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(LegatoTimelineEvent)
        .filter(LegatoTimelineEvent.user_id == current_user.id)
        .order_by(LegatoTimelineEvent.event_date.desc(), LegatoTimelineEvent.created_at.desc())
        .limit(200)
        .all()
    )
    return [
        {
            "id": r.id,
            "analysis_id": r.analysis_id,
            "label": r.label,
            "event_date": r.event_date,
            "source": r.source,
            "created_at": r.created_at.isoformat() + "Z",
        }
        for r in rows
    ]


# ---------- profile ----------
@router.get("/profile/me")
def profile_get(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    if not row:
        return {"user_id": current_user.id, "profile": {}}
    try:
        return {"user_id": current_user.id, "profile": json.loads(row.payload_json or "{}")}
    except Exception:
        return {"user_id": current_user.id, "profile": {}}


@router.put("/profile/me")
def profile_put(
    fields: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    if not row:
        row = LegatoProfile(user_id=current_user.id, payload_json="{}")
        db.add(row)
    row.payload_json = json.dumps(fields, ensure_ascii=False)
    db.add(row)
    db.commit()
    return {"ok": True}


@router.get("/network/profiles")
def network_profiles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = db.query(LegatoProfile).filter(LegatoProfile.user_id != current_user.id).limit(100).all()
    out = []
    for r in rows:
        try:
            prof = json.loads(r.payload_json or "{}")
        except Exception:
            prof = {}
        out.append({"user_id": r.user_id, "profile": prof})
    return out


# ---------- signatures ----------
class SignatureBody(BaseModel):
    analysis_id: int
    signer_name: str = Field(..., min_length=1)
    consent_acknowledged: bool = False
    signature_png_base64: Optional[str] = None


@router.post("/signatures")
def signatures_create(
    body: SignatureBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == body.analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if row.user_id != current_user.id and getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    sig = LegatoSignature(
        analysis_id=body.analysis_id,
        user_id=current_user.id,
        signer_name=body.signer_name,
        consent_acknowledged=bool(body.consent_acknowledged),
        signature_png_base64=body.signature_png_base64,
    )
    db.add(sig)
    db.commit()
    db.refresh(sig)
    return {"id": sig.id, "created_at": sig.created_at.isoformat() + "Z"}
