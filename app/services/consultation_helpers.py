"""Shared helpers for consultation requests and timed chat sessions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.db.models import ConsultationRequest, NetworkInvite, UserConversation, UserConversationMember


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


def ensure_consultation_session_valid(db: Session, conversation_id: int) -> Tuple[Optional[ConsultationRequest], Optional[str]]:
    """Return (consultation, error_message). None error means OK to send."""
    row = consultation_for_conversation(db, conversation_id)
    if not row:
        return None, None
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
