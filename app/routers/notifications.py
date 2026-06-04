# -*- coding: utf-8 -*-
"""Feed activity alerts: likes, comments, connection posts."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import User, UserNotification, LegatoProfile
from app.db.session import get_db
from app.services.social_notifications import actor_display_name
from app.routers.social import _parse_profile_row, _avatar_url_for_user, _public_base_url

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _actor_avatar_url(db: Session, actor_id: int) -> str:
    u = db.query(User).filter(User.id == actor_id).first()
    if not u:
        return ""
    prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == actor_id).first()
    prof = _parse_profile_row(prow)
    return _avatar_url_for_user(actor_id, prof, _public_base_url(), profile_row=prow) or ""


def _serialize(n: UserNotification, db: Session) -> Dict[str, Any]:
    return {
        "id": n.id,
        "type": n.type,
        "message": n.message,
        "post_id": n.post_id,
        "actor_id": n.actor_id,
        "actor_name": actor_display_name(db, n.actor_id),
        "actor_avatar_url": _actor_avatar_url(db, n.actor_id),
        "read": bool(n.read),
        "created_at": n.created_at.isoformat() + "Z",
    }


@router.get("")
def list_notifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(40, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(UserNotification).filter(UserNotification.recipient_id == current_user.id)
    total = q.count()
    unread = (
        db.query(UserNotification)
        .filter(UserNotification.recipient_id == current_user.id, UserNotification.read.is_(False))
        .count()
    )
    rows = (
        q.order_by(UserNotification.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "items": [_serialize(n, db) for n in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "unread_count": unread,
    }


@router.get("/unread-count")
def unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    count = (
        db.query(UserNotification)
        .filter(UserNotification.recipient_id == current_user.id, UserNotification.read.is_(False))
        .count()
    )
    return {"unread_count": count}


@router.post("/{notification_id}/read")
def mark_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(UserNotification)
        .filter(UserNotification.id == notification_id, UserNotification.recipient_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Notification not found")
    row.read = True
    db.add(row)
    db.commit()
    return {"ok": True}


@router.post("/read-all")
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    (
        db.query(UserNotification)
        .filter(UserNotification.recipient_id == current_user.id, UserNotification.read.is_(False))
        .update({"read": True}, synchronize_session=False)
    )
    db.commit()
    return {"ok": True}


@router.delete("/{notification_id}")
def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(UserNotification)
        .filter(UserNotification.id == notification_id, UserNotification.recipient_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Notification not found")
    db.delete(row)
    db.commit()
    return {"ok": True}
