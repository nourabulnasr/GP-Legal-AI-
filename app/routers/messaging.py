"""Private and group messaging between connected users."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import (
    LegatoProfile,
    NetworkInvite,
    User,
    UserConversation,
    UserConversationMember,
    UserMessage,
)
from app.db.session import get_db

router = APIRouter(prefix="/api/messages", tags=["messages"])


def _display_name(u: User, db: Session) -> str:
    from app.routers.social import _parse_profile_row, _display_name as social_display

    prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == u.id).first()
    prof = _parse_profile_row(prow)
    return social_display(u, prof)


def _user_public(db: Session, user_id: int) -> Dict[str, Any]:
    from app.routers.social import _parse_profile_row, _avatar_url_for_user, _public_base_url

    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        return {"user_id": user_id, "name": "?", "email": "", "avatar_url": ""}
    prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).first()
    prof = _parse_profile_row(prow)
    base = _public_base_url()
    return {
        "user_id": user_id,
        "name": _display_name(u, db),
        "email": u.email,
        "avatar_url": _avatar_url_for_user(user_id, prof, base, profile_row=prow) or "",
    }


def _users_connected(db: Session, a_id: int, b_id: int) -> bool:
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


def _connected_peer_ids(db: Session, user_id: int) -> set[int]:
    rows = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(NetworkInvite.requester_id == user_id, NetworkInvite.addressee_id == user_id),
        )
        .all()
    )
    out: set[int] = set()
    for inv in rows:
        peer = inv.addressee_id if inv.requester_id == user_id else inv.requester_id
        out.add(peer)
    return out


def _conversation_payload(
    db: Session,
    conv: UserConversation,
    viewer_id: int,
) -> Dict[str, Any]:
    members = (
        db.query(UserConversationMember)
        .filter(UserConversationMember.conversation_id == conv.id)
        .all()
    )
    member_ids = [m.user_id for m in members]
    peers: List[Dict[str, Any]] = []
    member_list: List[Dict[str, Any]] = []
    for uid in member_ids:
        info = _user_public(db, uid)
        member_list.append(info)
        if uid == viewer_id:
            continue
        peers.append(info)
    last = (
        db.query(UserMessage)
        .filter(UserMessage.conversation_id == conv.id)
        .order_by(UserMessage.id.desc())
        .first()
    )
    title = conv.title
    if conv.kind == "direct" and peers:
        title = peers[0].get("name") or peers[0].get("email") or "Chat"
    return {
        "id": conv.id,
        "kind": conv.kind,
        "title": title or ("Group chat" if conv.kind == "group" else "Chat"),
        "member_ids": member_ids,
        "members": member_list,
        "peers": peers,
        "created_by": conv.created_by,
        "created_at": conv.created_at.isoformat() + "Z",
        "last_message": last.body if last else None,
        "last_message_at": last.created_at.isoformat() + "Z" if last else None,
    }


def _find_direct_conversation(db: Session, user_id: int, peer_id: int) -> Optional[UserConversation]:
    my_ids = {
        r[0]
        for r in db.query(UserConversationMember.conversation_id)
        .filter(UserConversationMember.user_id == user_id)
        .all()
    }
    peer_ids = {
        r[0]
        for r in db.query(UserConversationMember.conversation_id)
        .filter(UserConversationMember.user_id == peer_id)
        .all()
    }
    shared = my_ids & peer_ids
    if not shared:
        return None
    return (
        db.query(UserConversation)
        .filter(UserConversation.kind == "direct", UserConversation.id.in_(shared))
        .order_by(UserConversation.id.asc())
        .first()
    )


class DirectConversationBody(BaseModel):
    peer_user_id: int


class GroupConversationBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    member_ids: List[int] = Field(default_factory=list, max_length=20)


class MessageBody(BaseModel):
    body: str = Field(..., min_length=1, max_length=8000)


class ConversationUpdateBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)


class MarkReadBody(BaseModel):
    message_id: Optional[int] = None


class AddMembersBody(BaseModel):
    member_ids: List[int] = Field(..., min_length=1, max_length=20)


@router.get("/conversations")
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    member_rows = (
        db.query(UserConversationMember.conversation_id)
        .filter(UserConversationMember.user_id == current_user.id)
        .all()
    )
    ids = [r[0] for r in member_rows]
    if not ids:
        return {"items": []}
    convs = (
        db.query(UserConversation)
        .filter(UserConversation.id.in_(ids))
        .order_by(UserConversation.id.desc())
        .all()
    )
    items = [_conversation_payload(db, c, current_user.id) for c in convs]
    items.sort(key=lambda x: x.get("last_message_at") or x.get("created_at") or "", reverse=True)
    return {"items": items}


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = _require_member(db, conversation_id, current_user.id)
    return _conversation_payload(db, conv, current_user.id)


@router.post("/conversations/direct")
def create_or_get_direct(
    body: DirectConversationBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    peer_id = body.peer_user_id
    if peer_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot chat with yourself")
    peer = db.query(User).filter(User.id == peer_id).first()
    if not peer:
        raise HTTPException(status_code=404, detail="User not found")
    if not _users_connected(db, current_user.id, peer_id):
        raise HTTPException(status_code=403, detail="You must be connected to message this user")
    existing = _find_direct_conversation(db, current_user.id, peer_id)
    if existing:
        return _conversation_payload(db, existing, current_user.id)
    conv = UserConversation(kind="direct", title=None, created_by=current_user.id)
    db.add(conv)
    db.flush()
    for uid in (current_user.id, peer_id):
        db.add(UserConversationMember(conversation_id=conv.id, user_id=uid))
    db.commit()
    db.refresh(conv)
    return _conversation_payload(db, conv, current_user.id)


@router.post("/conversations/group")
def create_group(
    body: GroupConversationBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    member_ids = sorted(set(body.member_ids + [current_user.id]))
    if len(member_ids) < 2:
        raise HTTPException(status_code=400, detail="A group needs at least 2 members")
    connected = _connected_peer_ids(db, current_user.id)
    for mid in member_ids:
        if mid == current_user.id:
            continue
        if mid not in connected:
            raise HTTPException(status_code=403, detail=f"User {mid} is not in your connections")
        if not db.query(User).filter(User.id == mid).first():
            raise HTTPException(status_code=404, detail=f"User {mid} not found")
    conv = UserConversation(
        kind="group",
        title=body.title.strip(),
        created_by=current_user.id,
    )
    db.add(conv)
    db.flush()
    for uid in member_ids:
        db.add(UserConversationMember(conversation_id=conv.id, user_id=uid))
    db.commit()
    db.refresh(conv)
    return _conversation_payload(db, conv, current_user.id)


@router.post("/conversations/{conversation_id}/members")
def add_group_members(
    conversation_id: int,
    body: AddMembersBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = _require_member(db, conversation_id, current_user.id)
    if conv.kind != "group":
        raise HTTPException(status_code=400, detail="Members can only be added to group chats")
    if conv.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the group creator can add members")
    connected = _connected_peer_ids(db, current_user.id)
    existing = {
        r[0]
        for r in db.query(UserConversationMember.user_id)
        .filter(UserConversationMember.conversation_id == conversation_id)
        .all()
    }
    for mid in body.member_ids:
        if mid == current_user.id or mid in existing:
            continue
        if mid not in connected:
            raise HTTPException(status_code=403, detail=f"User {mid} is not in your connections")
        if not db.query(User).filter(User.id == mid).first():
            raise HTTPException(status_code=404, detail=f"User {mid} not found")
        db.add(UserConversationMember(conversation_id=conversation_id, user_id=mid))
        existing.add(mid)
    db.commit()
    db.refresh(conv)
    return _conversation_payload(db, conv, current_user.id)


def _require_member(db: Session, conversation_id: int, user_id: int) -> UserConversation:
    conv = db.query(UserConversation).filter(UserConversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    mem = (
        db.query(UserConversationMember)
        .filter(
            UserConversationMember.conversation_id == conversation_id,
            UserConversationMember.user_id == user_id,
        )
        .first()
    )
    if not mem:
        raise HTTPException(status_code=403, detail="Not a member of this conversation")
    return conv


def _message_status(
    db: Session,
    conversation_id: int,
    viewer_id: int,
    message_id: int,
) -> str:
    others = (
        db.query(UserConversationMember)
        .filter(
            UserConversationMember.conversation_id == conversation_id,
            UserConversationMember.user_id != viewer_id,
        )
        .all()
    )
    if not others:
        return "sent"
    for mem in others:
        if (mem.last_read_message_id or 0) < message_id:
            return "sent"
    return "seen"


def _mark_conversation_read(
    db: Session,
    conversation_id: int,
    user_id: int,
    message_id: Optional[int] = None,
) -> None:
    mem = (
        db.query(UserConversationMember)
        .filter(
            UserConversationMember.conversation_id == conversation_id,
            UserConversationMember.user_id == user_id,
        )
        .first()
    )
    if not mem:
        return
    if message_id is None:
        last = (
            db.query(UserMessage)
            .filter(UserMessage.conversation_id == conversation_id)
            .order_by(UserMessage.id.desc())
            .first()
        )
        message_id = last.id if last else None
    if message_id is None:
        return
    if (mem.last_read_message_id or 0) < message_id:
        mem.last_read_message_id = message_id
        db.add(mem)
        db.commit()


@router.patch("/conversations/{conversation_id}")
def update_conversation(
    conversation_id: int,
    body: ConversationUpdateBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = _require_member(db, conversation_id, current_user.id)
    if conv.kind != "group":
        raise HTTPException(status_code=400, detail="Only group chats can be renamed")
    if conv.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the group creator can rename this group")
    conv.title = body.title.strip()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _conversation_payload(db, conv, current_user.id)


@router.post("/conversations/{conversation_id}/read")
def mark_conversation_read(
    conversation_id: int,
    body: MarkReadBody = MarkReadBody(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_member(db, conversation_id, current_user.id)
    _mark_conversation_read(db, conversation_id, current_user.id, body.message_id)
    return {"ok": True}


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_member(db, conversation_id, current_user.id)
    msgs = (
        db.query(UserMessage)
        .filter(UserMessage.conversation_id == conversation_id)
        .order_by(UserMessage.created_at.asc())
        .all()
    )
    if msgs:
        _mark_conversation_read(db, conversation_id, current_user.id, msgs[-1].id)
    out = []
    for m in msgs:
        u = db.query(User).filter(User.id == m.author_id).first()
        item = {
            "id": m.id,
            "author_id": m.author_id,
            "author_name": _display_name(u, db) if u else "?",
            "email": u.email if u else "",
            "body": m.body,
            "created_at": m.created_at.isoformat() + "Z",
            "is_mine": m.author_id == current_user.id,
        }
        if m.author_id == current_user.id:
            item["status"] = _message_status(db, conversation_id, current_user.id, m.id)
        out.append(item)
    return {"items": out}


@router.post("/conversations/{conversation_id}/messages")
def post_message(
    conversation_id: int,
    body: MessageBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_member(db, conversation_id, current_user.id)
    msg = UserMessage(
        conversation_id=conversation_id,
        author_id=current_user.id,
        body=body.body.strip(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {
        "id": msg.id,
        "author_id": current_user.id,
        "author_name": _display_name(current_user, db),
        "email": current_user.email,
        "body": msg.body,
        "created_at": msg.created_at.isoformat() + "Z",
        "is_mine": True,
        "status": "sent",
    }
