from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import LegatoProfile, NetworkInvite, User, UserNotification


def _parse_profile_row(row: Optional[LegatoProfile]) -> dict:
    import json

    if not row:
        return {}
    try:
        data = json.loads(row.payload_json or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def actor_display_name(db: Session, user_id: int) -> str:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return f"User {user_id}"
    prof = _parse_profile_row(
        db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).first()
    )
    name = (prof.get("displayName") or prof.get("name") or "").strip()
    if name:
        return name
    email = user.email or ""
    if "@" in email:
        return email.split("@")[0].replace(".", " ").title()
    return f"User {user_id}"


def connection_user_ids(db: Session, user_id: int) -> list[int]:
    rows = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(NetworkInvite.requester_id == user_id, NetworkInvite.addressee_id == user_id),
        )
        .all()
    )
    out: list[int] = []
    for inv in rows:
        other = inv.addressee_id if inv.requester_id == user_id else inv.requester_id
        if other != user_id:
            out.append(other)
    return out


def notify_user(
    db: Session,
    *,
    recipient_id: int,
    actor_id: int,
    ntype: str,
    message: str,
    post_id: Optional[int] = None,
) -> None:
    if recipient_id == actor_id:
        return
    msg = (message or "")[:512]
    if not msg:
        return
    db.add(
        UserNotification(
            recipient_id=recipient_id,
            actor_id=actor_id,
            type=ntype,
            post_id=post_id,
            message=msg,
            read=False,
        )
    )


def notify_post_liked(db: Session, *, actor_id: int, post_author_id: int, post_id: int) -> None:
    name = actor_display_name(db, actor_id)
    notify_user(
        db,
        recipient_id=post_author_id,
        actor_id=actor_id,
        ntype="like",
        post_id=post_id,
        message=f"{name} liked your post",
    )


def notify_post_commented(
    db: Session,
    *,
    actor_id: int,
    post_author_id: int,
    post_id: int,
    comment_preview: str,
) -> None:
    name = actor_display_name(db, actor_id)
    preview = (comment_preview or "").strip()
    if len(preview) > 80:
        preview = preview[:77] + "..."
    suffix = f': "{preview}"' if preview else ""
    notify_user(
        db,
        recipient_id=post_author_id,
        actor_id=actor_id,
        ntype="comment",
        post_id=post_id,
        message=f"{name} commented on your post{suffix}",
    )


def notify_connections_new_post(
    db: Session,
    *,
    author_id: int,
    post_id: int,
    recipient_ids: Optional[Iterable[int]] = None,
) -> None:
    name = actor_display_name(db, author_id)
    targets = list(recipient_ids) if recipient_ids is not None else connection_user_ids(db, author_id)
    for rid in targets:
        notify_user(
            db,
            recipient_id=rid,
            actor_id=author_id,
            ntype="connection_post",
            post_id=post_id,
            message=f"{name} posted an update",
        )
