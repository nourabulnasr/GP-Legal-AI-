# -*- coding: utf-8 -*-
"""Professional networking API: /api/posts, /api/profile/*, /api/network/* — SQLite + SQLAlchemy."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import (
    User,
    LegatoProfile,
    SocialPost,
    SocialPostLike,
    SocialPostComment,
    SocialPostShare,
    NetworkInvite,
    SkillEndorsement,
    ProfileRecommendation,
    ProfileUserDocument,
)
from app.db.session import get_db

router = APIRouter(prefix="/api", tags=["social"])

_IMAGE_BASE_URL = os.environ.get("IMAGE_BASE_URL", "http://localhost:8000")


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


def _serialize_post(
    post: SocialPost,
    db: Session,
    viewer_id: int,
    *,
    authors_map: Optional[Dict[int, User]] = None,
    profiles_map: Optional[Dict[int, Dict[str, Any]]] = None,
    likes_counts: Optional[Dict[int, int]] = None,
    comments_counts: Optional[Dict[int, int]] = None,
    shares_counts: Optional[Dict[int, int]] = None,
    viewer_liked: Optional[set] = None,
) -> Dict[str, Any]:
    # Use pre-fetched data when available (batch path), else fall back to single queries.
    if authors_map is not None:
        author = authors_map.get(post.author_id)
    else:
        author = db.query(User).filter(User.id == post.author_id).first()
    if not author:
        raise HTTPException(status_code=500, detail="Post author missing")

    if profiles_map is not None:
        prof = profiles_map.get(post.author_id, {})
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

    return {
        "id": post.id,
        "author_id": post.author_id,
        "author_name": _display_name(author, prof),
        "author_subtitle": _title_company(prof),
        "content": post.content,
        "tags": tags,
        "category": post.category,
        "created_at": post.created_at.isoformat() + "Z",
        "likes_count": lc,
        "comments_count": cc,
        "shares_count": sc,
        "liked": liked,
        "image_url": f"{_IMAGE_BASE_URL}/{post.image_url}" if post.image_url else None,
    }


# ---------- Posts ----------


class PostCreateBody(BaseModel):
    content: str = Field(..., min_length=1, max_length=20000)
    tags: List[str] = Field(default_factory=list)
    category: str = Field(default="All Updates", max_length=64)


@router.get("/posts")
def list_posts(
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

    # Batch-fetch all authors, profiles, and interaction counts in 6 queries total.
    post_ids = [p.id for p in rows]
    author_ids = list({p.author_id for p in rows})
    authors_map = {u.id: u for u in db.query(User).filter(User.id.in_(author_ids)).all()}
    profiles_map = {
        pr.user_id: _parse_profile_row(pr)
        for pr in db.query(LegatoProfile).filter(LegatoProfile.user_id.in_(author_ids)).all()
    }
    likes_counts, comments_counts, shares_counts, viewer_liked = _batch_post_stats(
        db, post_ids, current_user.id
    )

    return {
        "items": [
            _serialize_post(
                p, db, current_user.id,
                authors_map=authors_map,
                profiles_map=profiles_map,
                likes_counts=likes_counts,
                comments_counts=comments_counts,
                shares_counts=shares_counts,
                viewer_liked=viewer_liked,
            )
            for p in rows
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.post("/posts")
async def create_post(
    content: str = Form(..., min_length=1, max_length=20000),
    category: str = Form(default="All Updates"),
    tags: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tag_list = [t.strip() for t in tags.split(",") if t.strip()][:20]

    image_url: Optional[str] = None
    if image and image.filename:
        ext = os.path.splitext(image.filename)[-1].lower()
        safe_ext = ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".jpg"
        raw_name = f"{uuid.uuid4().hex}_{image.filename[:64]}{'' if ext else safe_ext}"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
        save_path = os.path.join("static", "post_images", filename)
        os.makedirs("static/post_images", exist_ok=True)
        data = await image.read()
        with open(save_path, "wb") as f:
            f.write(data)
        image_url = f"static/post_images/{filename}"

    post = SocialPost(
        author_id=current_user.id,
        content=content.strip(),
        tags_json=json.dumps(tag_list, ensure_ascii=False),
        category=(category or "All Updates")[:64],
        image_url=image_url,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return _serialize_post(post, db, current_user.id)


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
    for c in rows:
        u = db.query(User).filter(User.id == c.author_id).first()
        if not u:
            continue
        pr = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == c.author_id).first())
        out.append(
            {
                "id": c.id,
                "author_id": c.author_id,
                "author_name": _display_name(u, pr),
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


@router.get("/profile/{user_id}")
def get_profile(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    prof = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).first())
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
        "avatar_url": prof.get("avatarUrl") or prof.get("avatar_url") or "",
        "cover_url": prof.get("coverUrl") or prof.get("cover_url") or "",
        "skills": prof.get("skills") if isinstance(prof.get("skills"), list) else [],
        "experience": prof.get("experience") if isinstance(prof.get("experience"), list) else [],
        "education": prof.get("education") if isinstance(prof.get("education"), list) else [],
        "stats": {
            "connections": connections,
            "endorsements": endorsements,
        },
        "is_self": user_id == current_user.id,
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
    out = []
    for r in rows:
        eu = db.query(User).filter(User.id == r.endorser_id).first()
        pr = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == r.endorser_id).first())
        out.append(
            {
                "id": r.id,
                "endorser_id": r.endorser_id,
                "endorser_name": _display_name(eu, pr) if eu else "?",
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
    for r in rows:
        au = db.query(User).filter(User.id == r.author_id).first()
        pr = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == r.author_id).first())
        out.append(
            {
                "id": r.id,
                "author_id": r.author_id,
                "author_name": _display_name(au, pr) if au else "?",
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
                    }
                )
    return {"items": out}


@router.get("/network/suggestions")
def network_suggestions(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(LegatoProfile)
        .filter(LegatoProfile.user_id != current_user.id)
        .limit(limit * 3)
        .all()
    )
    out: List[Dict[str, Any]] = []
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
            }
        )
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
    for inv in rows:
        u = db.query(User).filter(User.id == inv.requester_id).first()
        pr = _parse_profile_row(db.query(LegatoProfile).filter(LegatoProfile.user_id == inv.requester_id).first())
        out.append(
            {
                "id": inv.id,
                "requester_id": inv.requester_id,
                "requester_name": _display_name(u, pr) if u else "?",
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
    for inv in rows:
        peer_id = inv.addressee_id if inv.requester_id == uid else inv.requester_id
        u = db.query(User).filter(User.id == peer_id).first()
        if not u:
            continue
        pr = _parse_profile_row(
            db.query(LegatoProfile).filter(LegatoProfile.user_id == peer_id).first()
        )
        out.append(
            {
                "invite_id": inv.id,
                "user_id": peer_id,
                "name": _display_name(u, pr),
                "subtitle": _title_company(pr),
                "location": pr.get("location") or "",
                "connected_at": (
                    inv.updated_at.isoformat() + "Z"
                    if getattr(inv, "updated_at", None)
                    else inv.created_at.isoformat() + "Z"
                ),
            }
        )
    return {"items": out}
