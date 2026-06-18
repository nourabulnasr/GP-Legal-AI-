"""Shared helpers for consultation requests and timed chat sessions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.db.models import ConsultationRequest, NetworkInvite, UserConversation, UserConversationMember

# Egypt is UTC+2 year-round (DST ended in 2014). Fixed offset matches the Flutter client;
# ZoneInfo("Africa/Cairo") can still apply +3 in summer on hosts with stale tzdata.
EGYPT_TZ = timezone(timedelta(hours=2))
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite returns naive datetimes; we persist UTC wall time."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_naive(dt: datetime) -> datetime:
    """Store UTC wall time in naive DATETIME columns."""
    return as_utc(dt).replace(tzinfo=None)  # type: ignore[union-attr]


def expire_consultation_if_due(db: Session, row: ConsultationRequest) -> ConsultationRequest:
    """Mark an active consultation expired once its session window has passed."""
    if row.status != "active":
        return row
    ends = as_utc(row.session_ends_at)
    if ends is not None and ends <= datetime.now(timezone.utc):
        row.status = "expired"
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def consultation_payload_for_list(db: Session, row: ConsultationRequest) -> Dict[str, Any]:
    """Serialize consultation state for conversation list/detail payloads."""
    row = expire_consultation_if_due(db, row)
    payload: Dict[str, Any] = {
        "id": row.id,
        "status": row.status,
        "duration_minutes": row.duration_minutes,
    }
    if row.scheduled_at:
        sched = as_utc(row.scheduled_at)
        payload["scheduled_at"] = sched.isoformat().replace("+00:00", "Z") if sched else None
    if row.status == "active" and row.session_ends_at:
        ends = as_utc(row.session_ends_at)
        if ends:
            remaining = max(0, int((ends - datetime.now(timezone.utc)).total_seconds()))
            if remaining <= 0:
                row.status = "expired"
                db.add(row)
                db.commit()
                db.refresh(row)
                payload["status"] = "expired"
            else:
                payload["remaining_seconds"] = remaining
                payload["session_ends_at"] = ends.isoformat().replace("+00:00", "Z")
    return payload


def format_scheduled_egypt(dt: Optional[datetime]) -> str:
    """Human-readable Egypt local time, e.g. '18 Jun · 7:00 AM'."""
    dt = as_utc(dt)
    if dt is None:
        return "scheduled time"
    local = dt.astimezone(EGYPT_TZ)
    hour12 = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    return f"{local.day} {_MONTHS[local.month - 1]} · {hour12}:{local.minute:02d} {ampm}"


def maybe_activate_scheduled_consultation(db: Session, row: ConsultationRequest) -> bool:
    """Start the timed session when payment is confirmed and scheduled_at has arrived."""
    if row.status != "confirmed":
        return row.status == "active"
    sched = as_utc(row.scheduled_at)
    if sched is None:
        return False
    now = datetime.now(timezone.utc)
    if sched > now:
        return False
    ends = sched + timedelta(minutes=row.duration_minutes)
    row.status = "active"
    row.session_started_at = utc_naive(sched)
    row.session_ends_at = utc_naive(ends)
    db.add(row)
    db.commit()
    db.refresh(row)
    return True


def users_connected(db: Session, a_id: int, b_id: int) -> bool:
    if a_id == b_id:
        return False
    row = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(
                and_(NetworkInvite.requester_id == a_id, NetworkInvite.addressee_id == b_id),
                and_(NetworkInvite.requester_id == b_id, NetworkInvite.addressee_id == a_id),
            ),
        )
        .first()
    )
    return row is not None


def active_consultation_between(db: Session, a_id: int, b_id: int) -> Optional[ConsultationRequest]:
    now = datetime.now(timezone.utc)
    row = (
        db.query(ConsultationRequest)
        .filter(
            ConsultationRequest.status == "active",
            or_(
                and_(ConsultationRequest.requester_id == a_id, ConsultationRequest.lawyer_id == b_id),
                and_(ConsultationRequest.requester_id == b_id, ConsultationRequest.lawyer_id == a_id),
            ),
        )
        .order_by(ConsultationRequest.id.desc())
        .first()
    )
    if not row:
        return None
    if row.session_ends_at and row.session_ends_at.replace(tzinfo=timezone.utc) <= now:
        row.status = "expired"
        db.add(row)
        db.commit()
        return None
    return row


def can_message_users(db: Session, a_id: int, b_id: int) -> bool:
    return users_connected(db, a_id, b_id) or active_consultation_between(db, a_id, b_id) is not None


def consultation_for_conversation(db: Session, conversation_id: int) -> Optional[ConsultationRequest]:
    return (
        db.query(ConsultationRequest)
        .filter(ConsultationRequest.conversation_id == conversation_id)
        .order_by(ConsultationRequest.id.desc())
        .first()
    )


def confirmed_consultation_waiting_between(db: Session, a_id: int, b_id: int) -> Optional[ConsultationRequest]:
    """Paid booking whose scheduled start is still in the future."""
    now = datetime.now(timezone.utc)
    row = (
        db.query(ConsultationRequest)
        .filter(
            ConsultationRequest.status == "confirmed",
            ConsultationRequest.payment_status == "paid",
            or_(
                and_(ConsultationRequest.requester_id == a_id, ConsultationRequest.lawyer_id == b_id),
                and_(ConsultationRequest.requester_id == b_id, ConsultationRequest.lawyer_id == a_id),
            ),
        )
        .order_by(ConsultationRequest.id.desc())
        .first()
    )
    if not row:
        return None
    sched = as_utc(row.scheduled_at)
    if sched is None or sched <= now:
        return None
    return row


def ensure_consultation_session_valid(db: Session, conversation_id: int) -> Tuple[Optional[ConsultationRequest], Optional[str]]:
    """Return (consultation, error_message). None error means OK to send."""
    row = consultation_for_conversation(db, conversation_id)
    if not row:
        return None, None
    if row.status == "confirmed":
        if not maybe_activate_scheduled_consultation(db, row):
            label = format_scheduled_egypt(row.scheduled_at)
            return row, f"Consultation starts at {label}. Messaging opens then."
    if row.status == "expired":
        return row, "Consultation done. This chat has ended."
    if row.status != "active":
        return row, "This consultation session is not active."
    now = datetime.now(timezone.utc)
    ends = row.session_ends_at
    if ends is not None:
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=timezone.utc)
        if ends <= now:
            row.status = "expired"
            db.add(row)
            db.commit()
            return row, "Consultation session has ended."
    return row, None


def approved_lawyer_rate(db: Session, lawyer_user_id: int) -> Optional[float]:
    from app.db.models import LawyerApplication

    app = (
        db.query(LawyerApplication)
        .filter(LawyerApplication.user_id == lawyer_user_id, LawyerApplication.status == "approved")
        .first()
    )
    if not app:
        return None
    rate = getattr(app, "negotiated_hourly_rate", None) or getattr(app, "hourly_rate", None)
    return float(rate) if rate is not None else None


def create_or_get_direct_for_consultation(db: Session, user_a: int, user_b: int) -> UserConversation:
    from app.routers.messaging import _find_direct_conversation

    existing = _find_direct_conversation(db, user_a, user_b)
    if existing:
        return existing
    conv = UserConversation(kind="direct", title=None, created_by=user_a)
    db.add(conv)
    db.flush()
    for uid in (user_a, user_b):
        db.add(UserConversationMember(conversation_id=conv.id, user_id=uid))
    db.flush()
    return conv
