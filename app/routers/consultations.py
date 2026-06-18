"""Consultation requests and verified lawyer directory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import ConsultationRequest, LawyerApplication, User, UserNotification
from app.db.session import get_db
from app.routers.social import (
    _avatar_url_for_user,
    _display_name,
    _parse_profile_row,
    _public_base_url,
    _title_company,
)
from app.services.consultation_helpers import (
    approved_lawyer_rate,
    create_or_get_direct_for_consultation,
    ensure_consultation_session_valid,
)

router = APIRouter(prefix="/api", tags=["consultations"])

MIN_CONSULTATION_MINUTES = 15


class ConsultationRequestBody(BaseModel):
    lawyer_id: int
    duration_minutes: int = Field(..., ge=MIN_CONSULTATION_MINUTES)
    notes: Optional[str] = Field(None, max_length=2000)


class ConsultationRespondBody(BaseModel):
    action: str = Field(..., pattern="^(accept|reject)$")


def _serialize_consultation(db: Session, row: ConsultationRequest) -> Dict[str, Any]:
    from app.db.models import LegatoProfile

    requester = db.query(User).filter(User.id == row.requester_id).first()
    lawyer = db.query(User).filter(User.id == row.lawyer_id).first()
    req_row = db.query(LegatoProfile).filter(LegatoProfile.user_id == row.requester_id).first()
    law_row = db.query(LegatoProfile).filter(LegatoProfile.user_id == row.lawyer_id).first()
    req_prof = _parse_profile_row(req_row)
    law_prof = _parse_profile_row(law_row)
    base = _public_base_url()
    total = None
    if row.hourly_rate is not None:
        total = round(row.hourly_rate * (row.duration_minutes / 60.0), 2)
    return {
        "id": row.id,
        "requester_id": row.requester_id,
        "requester_name": _display_name(requester, req_prof) if requester else "?",
        "requester_email": requester.email if requester else "",
        "lawyer_id": row.lawyer_id,
        "lawyer_name": _display_name(lawyer, law_prof) if lawyer else "?",
        "duration_minutes": row.duration_minutes,
        "hourly_rate": row.hourly_rate,
        "estimated_total": total,
        "notes": row.notes or "",
        "status": row.status,
        "conversation_id": row.conversation_id,
        "session_started_at": row.session_started_at.isoformat() + "Z" if row.session_started_at else None,
        "session_ends_at": row.session_ends_at.isoformat() + "Z" if row.session_ends_at else None,
        "created_at": row.created_at.isoformat() + "Z",
        "responded_at": row.responded_at.isoformat() + "Z" if row.responded_at else None,
        "lawyer_avatar_url": _avatar_url_for_user(row.lawyer_id, law_prof, base, profile_row=law_row),
        "requester_avatar_url": _avatar_url_for_user(row.requester_id, req_prof, base, profile_row=req_row),
    }


@router.get("/network/lawyers")
def list_verified_lawyers(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Directory of approved lawyers with hourly rate for the Network tab."""
    from app.db.models import LegatoProfile

    rows = (
        db.query(LawyerApplication, User)
        .join(User, LawyerApplication.user_id == User.id)
        .filter(LawyerApplication.status == "approved")
        .order_by(User.id.asc())
        .limit(min(limit, 100))
        .all()
    )
    base = _public_base_url()
    out: List[Dict[str, Any]] = []
    for app, u in rows:
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == u.id).first()
        prof = _parse_profile_row(prow)
        rate = getattr(app, "negotiated_hourly_rate", None) or getattr(app, "hourly_rate", None)
        out.append(
            {
                "user_id": u.id,
                "name": _display_name(u, prof),
                "subtitle": _title_company(prof),
                "location": prof.get("location") or "",
                "avatar_url": _avatar_url_for_user(u.id, prof, base, profile_row=prow),
                "is_verified_lawyer": True,
                "hourly_rate": rate,
                "years_of_experience": getattr(app, "years_of_experience", None),
            }
        )
    return {"items": out}


@router.post("/consultations")
def create_consultation_request(
    body: ConsultationRequestBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.lawyer_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot request consultation with yourself")
    if body.duration_minutes < MIN_CONSULTATION_MINUTES:
        raise HTTPException(status_code=400, detail=f"Minimum duration is {MIN_CONSULTATION_MINUTES} minutes")

    lawyer = db.query(User).filter(User.id == body.lawyer_id).first()
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found")
    rate = approved_lawyer_rate(db, body.lawyer_id)
    if rate is None:
        raise HTTPException(status_code=400, detail="This user is not a verified lawyer")

    pending = (
        db.query(ConsultationRequest)
        .filter(
            ConsultationRequest.requester_id == current_user.id,
            ConsultationRequest.lawyer_id == body.lawyer_id,
            ConsultationRequest.status == "pending",
        )
        .first()
    )
    if pending:
        raise HTTPException(status_code=400, detail="You already have a pending request with this lawyer")

    row = ConsultationRequest(
        requester_id=current_user.id,
        lawyer_id=body.lawyer_id,
        duration_minutes=body.duration_minutes,
        hourly_rate=rate,
        notes=(body.notes or "").strip() or None,
        status="pending",
    )
    db.add(row)
    db.flush()
    db.add(
        UserNotification(
            recipient_id=body.lawyer_id,
            actor_id=current_user.id,
            type="consultation_request",
            reference_id=row.id,
            message=(
                f"New consultation request: {body.duration_minutes} min"
                f"{f' — {body.notes.strip()[:80]}' if body.notes and body.notes.strip() else ''}"
            ),
            read=False,
        )
    )
    db.commit()
    db.refresh(row)
    return _serialize_consultation(db, row)


@router.get("/consultations/{consultation_id}")
def get_consultation(
    consultation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(ConsultationRequest).filter(ConsultationRequest.id == consultation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Consultation not found")
    if current_user.id not in (row.requester_id, row.lawyer_id) and getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return _serialize_consultation(db, row)


@router.post("/consultations/{consultation_id}/respond")
def respond_consultation(
    consultation_id: int,
    body: ConsultationRespondBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(ConsultationRequest).filter(ConsultationRequest.id == consultation_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Consultation not found")
    if row.lawyer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the lawyer can respond")
    if row.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {row.status}")

    now = datetime.now(timezone.utc)
    row.responded_at = now

    if body.action == "reject":
        row.status = "rejected"
        db.add(row)
        db.add(
            UserNotification(
                recipient_id=row.requester_id,
                actor_id=current_user.id,
                type="consultation_response",
                reference_id=row.id,
                message="Your consultation request was declined.",
                read=False,
            )
        )
        db.commit()
        return _serialize_consultation(db, row)

    conv = create_or_get_direct_for_consultation(db, row.requester_id, row.lawyer_id)
    row.conversation_id = conv.id
    row.status = "active"
    row.session_started_at = now
    row.session_ends_at = now + timedelta(minutes=row.duration_minutes)
    db.add(row)
    db.add(
        UserNotification(
            recipient_id=row.requester_id,
            actor_id=current_user.id,
            type="consultation_response",
            reference_id=row.id,
            message=f"Consultation accepted — {row.duration_minutes} min session started.",
            read=False,
        )
    )
    db.commit()
    db.refresh(row)
    payload = _serialize_consultation(db, row)
    payload["conversation_id"] = conv.id
    return payload


@router.get("/consultations/session/conversation/{conversation_id}")
def consultation_session_for_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.routers.messaging import _require_member

    _require_member(db, conversation_id, current_user.id)
    row, err = ensure_consultation_session_valid(db, conversation_id)
    if row is None:
        return {"active": False}
    if err:
        return {
            "active": False,
            "status": row.status,
            "message": err,
            "consultation_id": row.id,
        }
    remaining_sec = None
    if row.session_ends_at:
        ends = row.session_ends_at
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=timezone.utc)
        remaining_sec = max(0, int((ends - datetime.now(timezone.utc)).total_seconds()))
    return {
        "active": True,
        "consultation_id": row.id,
        "status": row.status,
        "duration_minutes": row.duration_minutes,
        "session_ends_at": row.session_ends_at.isoformat() + "Z" if row.session_ends_at else None,
        "remaining_seconds": remaining_sec,
    }
