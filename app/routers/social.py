# -*- coding: utf-8 -*-
"""Professional networking API: /api/posts, /api/profile/*, /api/network/* — SQLite + SQLAlchemy."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import (
    User,
    LawyerApplication,
    LegatoProfile,
    SocialPost,
    SocialPostLike,
    SocialPostComment,
    SocialPostShare,
    NetworkInvite,
    SkillEndorsement,
    ProfileRecommendation,
    ProfileUserDocument,
    UserNotification,
)
from app.db.session import get_db
from app.db.init_db import (
    _ext_from_mime,
    _guess_image_mime,
    _normalize_uploaded_image,
    _write_avatar_image_file,
    _write_post_image_file,
)
from app.services.social_notifications import (
    notify_connections_new_post,
    notify_post_commented,
    notify_post_liked,
)

router = APIRouter(prefix="/api", tags=["social"])

_ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".jfif", ".heic", ".heif"}


def _public_base_url(request: Optional[Request] = None) -> str:
    """Public URL prefix for /static/post_images (must match client API base)."""
    for key in ("IMAGE_BASE_URL", "PUBLIC_API_URL", "API_PUBLIC_URL"):
        val = (os.getenv(key) or "").strip().rstrip("/")
        if val:
            return val
    if request is not None:
        return str(request.base_url).rstrip("/")
    return "http://localhost:8000"


def _normalize_image_url(url: Optional[str], base: str) -> Optional[str]:
    if not url:
        return None
    base = base.rstrip("/")
    if url.startswith("/"):
        return f"{base}{url}"
    if "localhost" in url or "127.0.0.1" in url:
        marker = "/static/"
        if marker in url:
            return f"{base}{url[url.index(marker):]}"
    return url


def _parse_profile_row(row: Optional[LegatoProfile]) -> Dict[str, Any]:
    if not row:
        return {}
    try:
        return json.loads(row.payload_json or "{}")
    except Exception:
        return {}


def _display_name(user: User, prof: Dict[str, Any]) -> str:
    name = (prof.get("displayName") or prof.get("name") or "").strip()
    if name:
        return name
    email = user.email or ""
    if "@" in email:
        return email.split("@")[0].replace(".", " ").title()
    return f"User {user.id}"


def _title_company(prof: Dict[str, Any]) -> str:
    t = (prof.get("title") or "").strip()
    c = (prof.get("company") or "").strip()
    if t and c:
        return f"{t} · {c}"
    return t or c or "Legal professional"


def _avatar_url_for_user(
    user_id: int,
    prof: Dict[str, Any],
    base: str,
    *,
    profile_row: Optional[LegatoProfile] = None,
) -> str:
    """Resolve a public avatar URL from DB bytes or profile JSON."""
    if profile_row is not None and getattr(profile_row, "avatar_bytes", None):
        ext = _ext_from_mime(getattr(profile_row, "avatar_mime_type", None))
        return f"{base.rstrip('/')}/static/profile_avatars/user_{user_id}{ext}"
    return _normalize_image_url(
        prof.get("avatarUrl") or prof.get("avatar_url") or "",
        base,
    ) or ""


def _batch_post_stats(
    db: Session,
    post_ids: List[int],
    viewer_id: int,
) -> tuple:
    """Return (likes_counts, comments_counts, shares_counts, viewer_liked_set) for a list of post IDs.
    4 queries total regardless of feed size."""
    if not post_ids:
        return {}, {}, {}, set()
    likes_counts = {
        row.post_id: row.cnt
        for row in db.query(SocialPostLike.post_id, func.count(SocialPostLike.id).label("cnt"))
        .filter(SocialPostLike.post_id.in_(post_ids))
        .group_by(SocialPostLike.post_id)
        .all()
    }
    comments_counts = {
        row.post_id: row.cnt
        for row in db.query(SocialPostComment.post_id, func.count(SocialPostComment.id).label("cnt"))
        .filter(SocialPostComment.post_id.in_(post_ids))
        .group_by(SocialPostComment.post_id)
        .all()
    }
    shares_counts = {
        row.post_id: row.cnt
        for row in db.query(SocialPostShare.post_id, func.count(SocialPostShare.id).label("cnt"))
        .filter(SocialPostShare.post_id.in_(post_ids))
        .group_by(SocialPostShare.post_id)
        .all()
    }
    viewer_liked = {
        row.post_id
        for row in db.query(SocialPostLike.post_id)
        .filter(SocialPostLike.post_id.in_(post_ids), SocialPostLike.user_id == viewer_id)
        .all()
    }
    return likes_counts, comments_counts, shares_counts, viewer_liked


def _batch_connection_status(
    db: Session,
    viewer_id: int,
    user_ids: List[int],
) -> Dict[int, str]:
    """Map author user_id -> none | pending | connected for feed Connect buttons."""
    peers = [uid for uid in user_ids if uid != viewer_id]
    if not peers:
        return {}
    out: Dict[int, str] = {uid: "none" for uid in peers}
    rows = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status.in_(("accepted", "pending")),
            or_(
                and_(NetworkInvite.requester_id == viewer_id, NetworkInvite.addressee_id.in_(peers)),
                and_(NetworkInvite.addressee_id == viewer_id, NetworkInvite.requester_id.in_(peers)),
            ),
        )
        .order_by(NetworkInvite.id.desc())
        .all()
    )
    for inv in rows:
        peer = inv.addressee_id if inv.requester_id == viewer_id else inv.requester_id
        if peer not in out:
            continue
        if inv.status == "accepted":
            out[peer] = "connected"
        elif out[peer] != "connected":
            out[peer] = "pending"
    return out


def _verified_lawyer_ids_batch(db: Session, user_ids: List[int]) -> set:
    """One query: returns the set of user_ids whose LawyerApplication is approved."""
    if not user_ids:
        return set()
    rows = (
        db.query(LawyerApplication.user_id)
        .filter(
            LawyerApplication.user_id.in_(user_ids),
            LawyerApplication.status == "approved",
        )
        .all()
    )
    return {r.user_id for r in rows}


def _serialize_post(
    post: SocialPost,
    db: Session,
    viewer_id: int,
    *,
    authors_map: Optional[Dict[int, User]] = None,
    profiles_map: Optional[Dict[int, Dict[str, Any]]] = None,
    profile_rows_map: Optional[Dict[int, LegatoProfile]] = None,
    likes_counts: Optional[Dict[int, int]] = None,
    comments_counts: Optional[Dict[int, int]] = None,
    shares_counts: Optional[Dict[int, int]] = None,
    viewer_liked: Optional[set] = None,
    image_base: Optional[str] = None,
    connection_status: Optional[str] = None,
    verified_lawyer_ids: Optional[set] = None,
) -> Dict[str, Any]:
    # Use pre-fetched data when available (batch path), else fall back to single queries.
    if authors_map is not None:
        author = authors_map.get(post.author_id)
    else:
        author = db.query(User).filter(User.id == post.author_id).first()
    if not author:
        raise HTTPException(status_code=500, detail="Post author missing")

    prow: Optional[LegatoProfile] = None
    if profiles_map is not None:
        prof = profiles_map.get(post.author_id, {})
        if profile_rows_map is not None:
            prow = profile_rows_map.get(post.author_id)
    else:
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == post.author_id).first()
        prof = _parse_profile_row(prow)

    try:
        tags = json.loads(post.tags_json or "[]")
        if not isinstance(tags, list):
            tags = []
    except Exception:
        tags = []

    lc = likes_counts.get(post.id, 0) if likes_counts is not None else db.query(SocialPostLike).filter(SocialPostLike.post_id == post.id).count()
    cc = comments_counts.get(post.id, 0) if comments_counts is not None else db.query(SocialPostComment).filter(SocialPostComment.post_id == post.id).count()
    sc = shares_counts.get(post.id, 0) if shares_counts is not None else db.query(SocialPostShare).filter(SocialPostShare.post_id == post.id).count()
    liked = (post.id in viewer_liked) if viewer_liked is not None else (
        db.query(SocialPostLike).filter(SocialPostLike.post_id == post.id, SocialPostLike.user_id == viewer_id).first() is not None
    )

    if post.author_id == viewer_id:
        conn = "self"
    elif connection_status is not None:
        conn = connection_status
    else:
        conn = _connection_status_between(db, viewer_id, post.author_id)

    if verified_lawyer_ids is not None:
        is_vl = post.author_id in verified_lawyer_ids
    else:
        is_vl = bool(
            db.query(LawyerApplication)
            .filter(LawyerApplication.user_id == post.author_id, LawyerApplication.status == "approved")
            .first()
        )

    return {
        "id": post.id,
        "author_id": post.author_id,
        "author_name": _display_name(author, prof),
        "author_subtitle": _title_company(prof),
        "author_avatar_url": _avatar_url_for_user(
            post.author_id,
            prof,
            image_base or _public_base_url(),
            profile_row=prow,
        ),
        "author_user_type": getattr(author, "user_type", "user"),
        "author_is_verified_lawyer": is_vl,
        "content": post.content,
        "tags": tags,
        "category": post.category,
        "created_at": post.created_at.isoformat() + "Z",
        "likes_count": lc,
        "comments_count": cc,
        "shares_count": sc,
        "liked": liked,
        "connection_status": conn,
        "image_url": _normalize_image_url(post.image_url, image_base or _public_base_url()),
    }


# ---------- Posts ----------


class PostCreateBody(BaseModel):
    content: str = Field(..., min_length=1, max_length=20000)
    tags: List[str] = Field(default_factory=list)
    category: str = Field(default="All Updates", max_length=64)


@router.get("/posts")
def list_posts(
    request: Request,
    category: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(SocialPost)
    if category and category.strip() and category.strip() != "All Updates":
        q = q.filter(SocialPost.category == category.strip())
    total = q.count()
    rows = (
        q.order_by(SocialPost.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # Batch-fetch all authors, profiles, and interaction counts in 7 queries total.
    post_ids = [p.id for p in rows]
    author_ids = list({p.author_id for p in rows})
    authors_map = {u.id: u for u in db.query(User).filter(User.id.in_(author_ids)).all()}
    profile_rows = db.query(LegatoProfile).filter(LegatoProfile.user_id.in_(author_ids)).all()
    profiles_map = {pr.user_id: _parse_profile_row(pr) for pr in profile_rows}
    profile_rows_map = {pr.user_id: pr for pr in profile_rows}
    likes_counts, comments_counts, shares_counts, viewer_liked = _batch_post_stats(
        db, post_ids, current_user.id
    )
    connection_map = _batch_connection_status(db, current_user.id, author_ids)
    verified_ids = _verified_lawyer_ids_batch(db, author_ids)
    image_base = _public_base_url(request)

    return {
        "items": [
            _serialize_post(
                p, db, current_user.id,
                authors_map=authors_map,
                profiles_map=profiles_map,
                profile_rows_map=profile_rows_map,
                likes_counts=likes_counts,
                comments_counts=comments_counts,
                shares_counts=shares_counts,
                viewer_liked=viewer_liked,
                image_base=image_base,
                connection_status=connection_map.get(p.author_id),
                verified_lawyer_ids=verified_ids,
            )
            for p in rows
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.post("/posts")
async def create_post(
    request: Request,
    content: str = Form(default="", max_length=20000),
    category: str = Form(default="All Updates"),
    tags: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tag_list = [t.strip() for t in tags.split(",") if t.strip()][:20]
    text = content.strip()

    image_url: Optional[str] = None
    image_bytes_val: Optional[bytes] = None
    image_mime: Optional[str] = None
    image_ext = ".jpg"
    if image is not None:
        MAX_IMG_BYTES = 5 * 1024 * 1024  # 5 MB
        data = await image.read(MAX_IMG_BYTES + 1)
        if len(data) > MAX_IMG_BYTES:
            raise HTTPException(status_code=413, detail="Image too large. Maximum size is 5 MB.")
        if data:
            orig_name = (image.filename or "photo.jpg").strip() or "photo.jpg"
            ext = os.path.splitext(orig_name)[-1].lower()
            if ext not in _ALLOWED_IMAGE_EXT:
                ext = ".jpg"
            try:
                data, ext, image_mime = _normalize_uploaded_image(data, ext, image.content_type)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format. Please use JPG, PNG, or WebP.",
                )
            image_ext = ext
            image_bytes_val = data

    if not text and not image_bytes_val:
        raise HTTPException(status_code=400, detail="Post must include text or an image.")

    post = SocialPost(
        author_id=current_user.id,
        content=text,
        tags_json=json.dumps(tag_list, ensure_ascii=False),
        category=(category or "All Updates")[:64],
        image_url=None,
        image_bytes=image_bytes_val,
        image_mime_type=image_mime,
    )
    db.add(post)
    db.commit()
    db.refresh(post)

    if image_bytes_val:
        base = _public_base_url(request)
        _write_post_image_file(post.id, image_bytes_val, image_ext)
        post.image_url = f"{base.rstrip('/')}/static/post_images/post_{post.id}{image_ext}"
        db.add(post)
        db.commit()
        db.refresh(post)

    notify_connections_new_post(db, author_id=current_user.id, post_id=post.id)
    db.commit()

    return _serialize_post(post, db, current_user.id, image_base=_public_base_url(request))


def _static_path_from_public_url(url: Optional[str], marker: str = "/static/") -> Optional[str]:
    """Map a public image URL to a local path under ./static/ (best effort)."""
    if not url or marker not in url:
        return None
    rel = url[url.index(marker) + 1 :]  # static/post_images/abc.jpg
    return os.path.join(*rel.split("/"))


def _delete_local_static_file(url: Optional[str]) -> None:
    path = _static_path_from_public_url(url)
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


@router.get("/posts/{post_id}")
def get_post(
    post_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return _serialize_post(post, db, current_user.id, image_base=_public_base_url(request))


@router.delete("/posts/{post_id}")
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    is_admin = getattr(current_user, "role", "user") == "admin"
    if post.author_id != current_user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Not allowed to delete this post")

    _delete_local_static_file(post.image_url)
    if getattr(post, "image_bytes", None):
        ext = _ext_from_mime(getattr(post, "image_mime_type", None))
        legacy = os.path.join("static", "post_images", f"post_{post.id}{ext}")
        try:
            if os.path.isfile(legacy):
                os.remove(legacy)
        except OSError:
            pass

    db.query(SocialPostLike).filter(SocialPostLike.post_id == post_id).delete()
    db.query(SocialPostComment).filter(SocialPostComment.post_id == post_id).delete()
    db.query(SocialPostShare).filter(SocialPostShare.post_id == post_id).delete()
    db.query(UserNotification).filter(UserNotification.post_id == post_id).delete()
    db.delete(post)
    db.commit()
    return {"ok": True}


@router.post("/posts/{post_id}/like")
def toggle_like(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    existing = (
        db.query(SocialPostLike)
        .filter(SocialPostLike.post_id == post_id, SocialPostLike.user_id == current_user.id)
        .first()
    )
    if existing:
        db.delete(existing)
        liked = False
    else:
        db.add(SocialPostLike(post_id=post_id, user_id=current_user.id))
        liked = True
        notify_post_liked(
            db,
            actor_id=current_user.id,
            post_author_id=post.author_id,
            post_id=post_id,
        )
    db.commit()
    likes_count = db.query(SocialPostLike).filter(SocialPostLike.post_id == post_id).count()
    return {"liked": liked, "likes_count": likes_count}


class CommentBody(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)


@router.post("/posts/{post_id}/comment")
def add_comment(
    post_id: int,
    body: CommentBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    c = SocialPostComment(post_id=post_id, author_id=current_user.id, content=body.content.strip())
    db.add(c)
    notify_post_commented(
        db,
        actor_id=current_user.id,
        post_author_id=post.author_id,
        post_id=post_id,
        comment_preview=body.content.strip(),
    )
    db.commit()
    db.refresh(c)
    return {"id": c.id, "ok": True}


@router.get("/posts/{post_id}/comments")
def list_comments(
    post_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    q = db.query(SocialPostComment).filter(SocialPostComment.post_id == post_id)
    total = q.count()
    rows = (
        q.order_by(SocialPostComment.created_at.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    out: List[Dict[str, Any]] = []
    base = _public_base_url()
    comment_author_ids = list({c.author_id for c in rows})
    vl_ids = _verified_lawyer_ids_batch(db, comment_author_ids)
    for c in rows:
        u = db.query(User).filter(User.id == c.author_id).first()
        if not u:
            continue
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == c.author_id).first()
        pr = _parse_profile_row(prow)
        out.append(
            {
                "id": c.id,
                "author_id": c.author_id,
                "author_name": _display_name(u, pr),
                "author_avatar_url": _avatar_url_for_user(c.author_id, pr, base, profile_row=prow),
                "author_user_type": getattr(u, "user_type", "user"),
                "author_is_verified_lawyer": c.author_id in vl_ids,
                "content": c.content,
                "created_at": c.created_at.isoformat() + "Z",
            }
        )
    return {"items": out, "page": page, "page_size": page_size, "total": total}


@router.post("/posts/{post_id}/share")
def share_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    db.add(SocialPostShare(post_id=post_id, user_id=current_user.id))
    db.commit()
    shares_count = db.query(SocialPostShare).filter(SocialPostShare.post_id == post_id).count()
    return {"ok": True, "shares_count": shares_count}


# ---------- Profile helpers (merge with LegatoProfile JSON) ----------


def _merge_profile_payload(
    db: Session,
    user_id: int,
    mutator: Any,
) -> Dict[str, Any]:
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).first()
    if not row:
        row = LegatoProfile(user_id=user_id, payload_json="{}")
        db.add(row)
        db.flush()
    data = _parse_profile_row(row)
    mutator(data)
    row.payload_json = json.dumps(data, ensure_ascii=False)
    db.add(row)
    db.commit()
    return data


class ExperienceItem(BaseModel):
    title: str
    company: str
    start_date: str = ""
    end_date: str = ""
    description: str = ""


class EducationItem(BaseModel):
    school: str
    degree: str = ""
    year: str = ""


def _connection_status_between(db: Session, viewer_id: int, profile_user_id: int) -> str:
    """Return connection_status for profile viewer: connected, pending, or none."""
    if viewer_id == profile_user_id:
        return "self"
    inv = (
        db.query(NetworkInvite)
        .filter(
            or_(
                and_(
                    NetworkInvite.requester_id == viewer_id,
                    NetworkInvite.addressee_id == profile_user_id,
                ),
                and_(
                    NetworkInvite.requester_id == profile_user_id,
                    NetworkInvite.addressee_id == viewer_id,
                ),
            ),
            NetworkInvite.status.in_(("accepted", "pending")),
        )
        .order_by(NetworkInvite.id.desc())
        .first()
    )
    if not inv:
        return "none"
    if inv.status == "accepted":
        return "connected"
    return "pending"


@router.get("/profile/{user_id}")
def get_profile(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).first()
    prof = _parse_profile_row(prow)
    base = _public_base_url()
    if prow is not None and getattr(prow, "avatar_bytes", None):
        ext = _ext_from_mime(getattr(prow, "avatar_mime_type", None))
        avatar_url = f"{base.rstrip('/')}/static/profile_avatars/user_{user_id}{ext}"
    else:
        avatar_url = _avatar_url_for_user(user_id, prof, base, profile_row=prow)
    endorsements = (
        db.query(SkillEndorsement).filter(SkillEndorsement.recipient_id == user_id).count()
    )
    connections = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(
                NetworkInvite.requester_id == user_id,
                NetworkInvite.addressee_id == user_id,
            ),
        )
        .count()
    )
    return {
        "user_id": user_id,
        "email": u.email if user_id == current_user.id else None,
        "display_name": _display_name(u, prof),
        "title": prof.get("title") or "",
        "company": prof.get("company") or "",
        "location": prof.get("location") or "",
        "bio": prof.get("bio") or "",
        "avatar_url": avatar_url,
        "cover_url": _normalize_image_url(
            prof.get("coverUrl") or prof.get("cover_url") or "",
            _public_base_url(),
        ) or "",
        "skills": prof.get("skills") if isinstance(prof.get("skills"), list) else [],
        "experience": prof.get("experience") if isinstance(prof.get("experience"), list) else [],
        "education": prof.get("education") if isinstance(prof.get("education"), list) else [],
        "stats": {
            "connections": connections,
            "endorsements": endorsements,
        },
        "is_self": user_id == current_user.id,
        "connection_status": _connection_status_between(db, current_user.id, user_id),
        "user_type": getattr(u, "user_type", "user"),
        "is_verified_lawyer": bool(
            db.query(LawyerApplication)
            .filter(LawyerApplication.user_id == user_id, LawyerApplication.status == "approved")
            .first()
        ),
    }


# ---------- Profile: experience / education (stored in LegatoProfile JSON) ----------


@router.post("/profile/me/experience")
def add_experience(
    body: ExperienceItem,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    def mut(data: Dict[str, Any]) -> None:
        exp = data.get("experience")
        if not isinstance(exp, list):
            exp = []
        exp.append(
            {
                "title": body.title,
                "company": body.company,
                "startDate": body.start_date,
                "endDate": body.end_date,
                "description": body.description,
            }
        )
        data["experience"] = exp

    return _merge_profile_payload(db, current_user.id, mut)


@router.put("/profile/me/experience/{idx}")
def edit_experience(
    idx: int,
    body: ExperienceItem,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    prof = _parse_profile_row(row)
    exp = prof.get("experience")
    if not isinstance(exp, list) or idx < 0 or idx >= len(exp):
        raise HTTPException(status_code=400, detail="Invalid experience index")

    def mut(data: Dict[str, Any]) -> None:
        ex = data.get("experience")
        if not isinstance(ex, list) or idx < 0 or idx >= len(ex):
            return
        ex[idx] = {
            "title": body.title,
            "company": body.company,
            "startDate": body.start_date,
            "endDate": body.end_date,
            "description": body.description,
        }
        data["experience"] = ex

    return _merge_profile_payload(db, current_user.id, mut)


@router.delete("/profile/me/experience/{idx}")
def delete_experience(
    idx: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    prof = _parse_profile_row(row)
    exp = prof.get("experience")
    if not isinstance(exp, list) or idx < 0 or idx >= len(exp):
        raise HTTPException(status_code=400, detail="Invalid experience index")

    def mut(data: Dict[str, Any]) -> None:
        ex = data.get("experience")
        if not isinstance(ex, list) or idx < 0 or idx >= len(ex):
            return
        ex.pop(idx)
        data["experience"] = ex

    return _merge_profile_payload(db, current_user.id, mut)


@router.post("/profile/me/education")
def add_education(
    body: EducationItem,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    def mut(data: Dict[str, Any]) -> None:
        edu = data.get("education")
        if not isinstance(edu, list):
            edu = []
        edu.append(
            {
                "school": body.school,
                "degree": body.degree,
                "year": body.year,
            }
        )
        data["education"] = edu

    return _merge_profile_payload(db, current_user.id, mut)


@router.put("/profile/me/education/{idx}")
def edit_education(
    idx: int,
    body: EducationItem,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    prof = _parse_profile_row(row)
    edu = prof.get("education")
    if not isinstance(edu, list) or idx < 0 or idx >= len(edu):
        raise HTTPException(status_code=400, detail="Invalid education index")

    def mut(data: Dict[str, Any]) -> None:
        ed = data.get("education")
        if not isinstance(ed, list) or idx < 0 or idx >= len(ed):
            return
        ed[idx] = {
            "school": body.school,
            "degree": body.degree,
            "year": body.year,
        }
        data["education"] = ed

    return _merge_profile_payload(db, current_user.id, mut)


@router.delete("/profile/me/education/{idx}")
def delete_education(
    idx: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    prof = _parse_profile_row(row)
    edu = prof.get("education")
    if not isinstance(edu, list) or idx < 0 or idx >= len(edu):
        raise HTTPException(status_code=400, detail="Invalid education index")

    def mut(data: Dict[str, Any]) -> None:
        ed = data.get("education")
        if not isinstance(ed, list) or idx < 0 or idx >= len(ed):
            return
        ed.pop(idx)
        data["education"] = ed

    return _merge_profile_payload(db, current_user.id, mut)


@router.put("/profile/me")
def put_profile_me(
    fields: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Merge fields into LegatoProfile JSON (displayName, title, company, skills, bio, ...)."""
    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    if not row:
        row = LegatoProfile(user_id=current_user.id, payload_json="{}")
        db.add(row)
    try:
        cur = json.loads(row.payload_json or "{}")
    except Exception:
        cur = {}
    for k, v in (fields or {}).items():
        cur[k] = v
    row.payload_json = json.dumps(cur, ensure_ascii=False)
    db.add(row)
    db.commit()
    return {"ok": True, "profile": cur}


@router.post("/profile/me/avatar")
async def upload_profile_avatar(
    request: Request,
    avatar: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload profile photo; bytes stored in DB and synced to /static/profile_avatars/."""
    MAX_IMG_BYTES = 5 * 1024 * 1024
    data = await avatar.read(MAX_IMG_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file.")
    if len(data) > MAX_IMG_BYTES:
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 5 MB.")

    orig_name = (avatar.filename or "avatar.jpg").strip() or "avatar.jpg"
    ext = os.path.splitext(orig_name)[-1].lower()
    if ext not in _ALLOWED_IMAGE_EXT:
        ext = ".jpg"
    try:
        data, ext, mime = _normalize_uploaded_image(data, ext, avatar.content_type)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please use JPG, PNG, or WebP.",
        )

    row = db.query(LegatoProfile).filter(LegatoProfile.user_id == current_user.id).first()
    if not row:
        row = LegatoProfile(user_id=current_user.id, payload_json="{}")
        db.add(row)
        db.flush()

    try:
        cur = json.loads(row.payload_json or "{}")
    except Exception:
        cur = {}
    for key in ("avatar_url", "avatarUrl"):
        old = cur.get(key)
        if isinstance(old, str) and old.strip():
            _delete_local_static_file(old.strip())

    _write_avatar_image_file(current_user.id, data, ext)
    base = _public_base_url(request)
    avatar_url = f"{base.rstrip('/')}/static/profile_avatars/user_{current_user.id}{ext}"

    row.avatar_bytes = data
    row.avatar_mime_type = mime
    cur["avatar_url"] = avatar_url
    cur["avatarUrl"] = avatar_url
    row.payload_json = json.dumps(cur, ensure_ascii=False)
    db.add(row)
    db.commit()

    return {
        "ok": True,
        "avatar_url": avatar_url,
        "profile": cur,
    }


# ---------- Endorsements / documents / recommendations ----------


class EndorseBody(BaseModel):
    skill: str = Field(..., min_length=1, max_length=128)


@router.get("/profile/{user_id}/endorsements")
def list_endorsements(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    rows = db.query(SkillEndorsement).filter(SkillEndorsement.recipient_id == user_id).all()
    endorser_ids = [r.endorser_id for r in rows]
    vl_set: set = set()
    if endorser_ids:
        vl_set = {
            r.user_id
            for r in db.query(LawyerApplication.user_id)
            .filter(LawyerApplication.user_id.in_(endorser_ids), LawyerApplication.status == "approved")
            .all()
        }
    out = []
    for r in rows:
        eu = db.query(User).filter(User.id == r.endorser_id).first()
        pr = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == r.endorser_id).first())
        out.append(
            {
                "id": r.id,
                "endorser_id": r.endorser_id,
                "endorser_name": _display_name(eu, pr) if eu else "?",
                "endorser_is_verified_lawyer": r.endorser_id in vl_set,
                "skill": r.skill,
                "created_at": r.created_at.isoformat() + "Z",
            }
        )
    return {"items": out}


@router.post("/profile/{user_id}/endorse")
def endorse_skill(
    user_id: int,
    body: EndorseBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot endorse yourself")
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    skill = body.skill.strip()
    if (
        db.query(SkillEndorsement)
        .filter(
            SkillEndorsement.endorser_id == current_user.id,
            SkillEndorsement.recipient_id == user_id,
            SkillEndorsement.skill == skill,
        )
        .first()
    ):
        return {"ok": True, "already": True}
    db.add(
        SkillEndorsement(
            endorser_id=current_user.id,
            recipient_id=user_id,
            skill=skill,
        )
    )
    db.commit()
    return {"ok": True}


class DocumentBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    file_url: str = Field(..., min_length=1)


@router.get("/profile/{user_id}/documents")
def list_documents(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    rows = (
        db.query(ProfileUserDocument)
        .filter(ProfileUserDocument.user_id == user_id)
        .order_by(ProfileUserDocument.created_at.desc())
        .all()
    )
    return {
        "items": [
            {
                "id": r.id,
                "title": r.title,
                "file_url": r.file_url,
                "created_at": r.created_at.isoformat() + "Z",
            }
            for r in rows
        ]
    }


@router.post("/profile/me/documents")
def add_document(
    body: DocumentBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    d = ProfileUserDocument(
        user_id=current_user.id,
        title=body.title.strip(),
        file_url=body.file_url.strip(),
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return {"id": d.id, "ok": True}


@router.post("/profile/me/documents/upload")
def upload_document(
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image/PDF and store bytes in DB (SQLite BLOB) under profile_user_documents.
    Use GET /profile/me/documents/{id}/download to retrieve.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    d = ProfileUserDocument(
        user_id=current_user.id,
        title=title.strip()[:512],
        file_url="",  # stored in DB; use download endpoint
        mime_type=(file.content_type or "application/octet-stream")[:255],
        file_bytes=raw,
    )
    db.add(d)
    db.commit()
    db.refresh(d)
    return {"id": d.id, "ok": True}


@router.get("/profile/me/documents/{doc_id}/download")
def download_document_me(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    d = db.query(ProfileUserDocument).filter(ProfileUserDocument.id == doc_id).first()
    if not d or d.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    if not getattr(d, "file_bytes", None):
        raise HTTPException(status_code=404, detail="No stored file")
    mt = getattr(d, "mime_type", None) or "application/octet-stream"
    return Response(content=d.file_bytes, media_type=mt)


class RecommendationBody(BaseModel):
    content: str = Field(..., min_length=1, max_length=20000)


@router.get("/profile/{user_id}/recommendations")
def list_recommendations(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    rows = (
        db.query(ProfileRecommendation)
        .filter(ProfileRecommendation.recipient_id == user_id)
        .order_by(ProfileRecommendation.created_at.desc())
        .all()
    )
    out = []
    base = _public_base_url()
    rec_author_ids = list({r.author_id for r in rows})
    vl_ids = _verified_lawyer_ids_batch(db, rec_author_ids)
    for r in rows:
        au = db.query(User).filter(User.id == r.author_id).first()
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == r.author_id).first()
        pr = _parse_profile_row(prow)
        out.append(
            {
                "id": r.id,
                "author_id": r.author_id,
                "author_name": _display_name(au, pr) if au else "?",
                "author_avatar_url": _avatar_url_for_user(r.author_id, pr, base, profile_row=prow) if au else "",
                "author_is_verified_lawyer": r.author_id in vl_ids,
                "content": r.content,
                "created_at": r.created_at.isoformat() + "Z",
            }
        )
    return {"items": out}


@router.post("/profile/{user_id}/recommendations")
def add_recommendation(
    user_id: int,
    body: RecommendationBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot recommend yourself")
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    rec = ProfileRecommendation(
        author_id=current_user.id,
        recipient_id=user_id,
        content=body.content.strip(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {"id": rec.id, "ok": True}


# ---------- Network ----------


def _peer_ids_from_invites(
    invites: List[tuple[int, int, str]],
    user_id: int,
) -> set[int]:
    """Peers linked by accepted or pending invites (either direction)."""
    excluded: set[int] = set()
    for requester_id, addressee_id, status in invites:
        if status not in ("accepted", "pending"):
            continue
        if requester_id == user_id:
            excluded.add(addressee_id)
        elif addressee_id == user_id:
            excluded.add(requester_id)
    return excluded


def _network_excluded_peer_ids(db: Session, user_id: int) -> set[int]:
    """User IDs to omit from suggestions: connected or invite pending either way."""
    rows = (
        db.query(
            NetworkInvite.requester_id,
            NetworkInvite.addressee_id,
            NetworkInvite.status,
        )
        .filter(
            or_(NetworkInvite.requester_id == user_id, NetworkInvite.addressee_id == user_id),
            NetworkInvite.status.in_(("accepted", "pending")),
        )
        .all()
    )
    return _peer_ids_from_invites(
        [(r.requester_id, r.addressee_id, r.status) for r in rows],
        user_id,
    )


class InviteBody(BaseModel):
    to_user_id: int


@router.get("/network/stats")
def network_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    uid = current_user.id
    connections = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(NetworkInvite.requester_id == uid, NetworkInvite.addressee_id == uid),
        )
        .count()
    )
    endorsements = (
        db.query(SkillEndorsement).filter(SkillEndorsement.recipient_id == uid).count()
    )
    pending_in = (
        db.query(NetworkInvite)
        .filter(NetworkInvite.addressee_id == uid, NetworkInvite.status == "pending")
        .count()
    )
    return {
        "connections": connections,
        "endorsements": endorsements,
        "profile_views": 0,
        "invitations_pending": pending_in,
    }


@router.get("/network/search")
def network_search(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Search people by email prefix, display name, title, company, location, or skills.
    Returns minimal cards used by the mobile Network tab.
    """
    query = (q or "").strip().lower()
    if not query:
        return {"items": []}

    # Pull profiles first (fast + includes title/company/location/skills)
    prof_rows = (
        db.query(LegatoProfile)
        .filter(LegatoProfile.user_id != current_user.id)
        .limit(500)
        .all()
    )
    out: List[Dict[str, Any]] = []
    base = _public_base_url()
    for pr in prof_rows:
        if len(out) >= limit:
            break
        uid = pr.user_id
        u = db.query(User).filter(User.id == uid).first()
        if not u:
            continue
        prof = _parse_profile_row(pr)
        blob = " ".join(
            [
                str(u.email or ""),
                str(prof.get("displayName") or prof.get("name") or ""),
                str(prof.get("title") or ""),
                str(prof.get("company") or ""),
                str(prof.get("location") or ""),
                " ".join([str(s) for s in (prof.get("skills") or [])]) if isinstance(prof.get("skills"), list) else "",
            ]
        ).lower()
        if query not in blob:
            continue
        out.append(
            {
                "user_id": uid,
                "name": _display_name(u, prof),
                "subtitle": _title_company(prof),
                "location": prof.get("location") or "",
                "avatar_url": _avatar_url_for_user(uid, prof, base, profile_row=pr),
            }
        )

    # Fallback: match users without profile rows
    if len(out) < limit:
        user_rows = (
            db.query(User)
            .filter(User.id != current_user.id)
            .order_by(User.id.desc())
            .limit(200)
            .all()
        )
        seen = {x.get("user_id") for x in out}
        for u in user_rows:
            if len(out) >= limit:
                break
            if u.id in seen:
                continue
            email = (u.email or "").lower()
            if query in email:
                out.append(
                    {
                        "user_id": u.id,
                        "name": email.split("@")[0],
                        "subtitle": "Legal professional",
                        "location": "",
                        "avatar_url": "",
                    }
                )
    # Batch-annotate verified lawyer status in one query.
    result_ids = [item["user_id"] for item in out]
    vl_ids = _verified_lawyer_ids_batch(db, result_ids)
    for item in out:
        item["is_verified_lawyer"] = item["user_id"] in vl_ids
    return {"items": out}


@router.get("/network/suggestions")
def network_suggestions(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    excluded = _network_excluded_peer_ids(db, current_user.id)
    excluded.add(current_user.id)

    q = db.query(LegatoProfile).filter(LegatoProfile.user_id.notin_(excluded))
    rows = q.limit(limit * 5).all()
    out: List[Dict[str, Any]] = []
    base = _public_base_url()
    for r in rows:
        if len(out) >= limit:
            break
        uid = r.user_id
        u = db.query(User).filter(User.id == uid).first()
        if not u:
            continue
        prof = _parse_profile_row(r)
        out.append(
            {
                "user_id": uid,
                "name": _display_name(u, prof),
                "subtitle": _title_company(prof),
                "location": prof.get("location") or "",
                "avatar_url": _avatar_url_for_user(uid, prof, base, profile_row=r),
            }
        )
    suggestion_ids = [item["user_id"] for item in out]
    vl_ids = _verified_lawyer_ids_batch(db, suggestion_ids)
    for item in out:
        item["is_verified_lawyer"] = item["user_id"] in vl_ids
    return {"items": out}


@router.post("/network/invites")
def send_invite(
    body: InviteBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.to_user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Invalid recipient")
    peer = db.query(User).filter(User.id == body.to_user_id).first()
    if not peer:
        raise HTTPException(status_code=404, detail="User not found")
    existing = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.requester_id == current_user.id,
            NetworkInvite.addressee_id == body.to_user_id,
        )
        .first()
    )
    if existing:
        return {"id": existing.id, "status": existing.status, "ok": True}
    inv = NetworkInvite(
        requester_id=current_user.id,
        addressee_id=body.to_user_id,
        status="pending",
    )
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return {"id": inv.id, "status": inv.status, "ok": True}


@router.post("/network/invites/{invite_id}/accept")
def accept_invite(
    invite_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inv = db.query(NetworkInvite).filter(NetworkInvite.id == invite_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    if inv.addressee_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    inv.status = "accepted"
    db.add(inv)
    db.commit()
    return {"ok": True}


@router.get("/network/invites")
def list_pending_invites(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(NetworkInvite)
        .filter(NetworkInvite.addressee_id == current_user.id, NetworkInvite.status == "pending")
        .all()
    )
    out = []
    base = _public_base_url()
    requester_ids = list({inv.requester_id for inv in rows})
    vl_ids = _verified_lawyer_ids_batch(db, requester_ids)
    for inv in rows:
        u = db.query(User).filter(User.id == inv.requester_id).first()
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == inv.requester_id).first()
        pr = _parse_profile_row(prow)
        out.append(
            {
                "id": inv.id,
                "requester_id": inv.requester_id,
                "requester_name": _display_name(u, pr) if u else "?",
                "requester_avatar_url": _avatar_url_for_user(inv.requester_id, pr, base, profile_row=prow) if u else "",
                "requester_is_verified_lawyer": inv.requester_id in vl_ids,
                "created_at": inv.created_at.isoformat() + "Z",
            }
        )
    return {"items": out}


@router.get("/network/connections")
def list_connections(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return accepted connections for the current user."""
    uid = current_user.id
    rows = (
        db.query(NetworkInvite)
        .filter(
            NetworkInvite.status == "accepted",
            or_(NetworkInvite.requester_id == uid, NetworkInvite.addressee_id == uid),
        )
        .order_by(NetworkInvite.id.desc())
        .limit(limit)
        .all()
    )
    out: List[Dict[str, Any]] = []
    base = _public_base_url()
    for inv in rows:
        peer_id = inv.addressee_id if inv.requester_id == uid else inv.requester_id
        u = db.query(User).filter(User.id == peer_id).first()
        if not u:
            continue
        prow = db.query(LegatoProfile).filter(LegatoProfile.user_id == peer_id).first()
        pr = _parse_profile_row(prow)
        out.append(
            {
                "invite_id": inv.id,
                "user_id": peer_id,
                "name": _display_name(u, pr),
                "email": u.email,
                "subtitle": _title_company(pr),
                "location": pr.get("location") or "",
                "avatar_url": _avatar_url_for_user(peer_id, pr, base, profile_row=prow),
                "connected_at": (
                    inv.updated_at.isoformat() + "Z"
                    if getattr(inv, "updated_at", None)
                    else inv.created_at.isoformat() + "Z"
                ),
            }
        )
    conn_ids = [item["user_id"] for item in out]
    vl_ids = _verified_lawyer_ids_batch(db, conn_ids)
    for item in out:
        item["is_verified_lawyer"] = item["user_id"] in vl_ids
    return {"items": out}
