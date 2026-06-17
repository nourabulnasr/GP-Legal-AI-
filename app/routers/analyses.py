from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Analysis, User
from app.schemas.analyses import (
    AnalysisCreateRequest,
    AnalysisResponse,
    AnalysisDetailResponse,
    AdminUpdateRoleRequest,
    FlagRequest,
)
from app.core.deps import get_current_user, require_admin  # ✅ IMPORTANT
from app.services.user_deletion import delete_user_account

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisResponse)
def create_analysis(
    payload: AnalysisCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    a = Analysis(
        user_id=current_user.id,
        filename=payload.filename,
        result_json=payload.result_json,
        mime_type=payload.mime_type,
        sha256=payload.sha256,
        page_count=payload.page_count,
        ocr_used=payload.ocr_used,
        detected_lang=payload.detected_lang,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return AnalysisResponse(id=a.id, filename=a.filename, created_at=a.created_at)


@router.get("", response_model=list[AnalysisResponse])
def list_analyses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return [AnalysisResponse(id=r.id, filename=r.filename, created_at=r.created_at) for r in rows]


@router.get("/admin/all", response_model=list[AnalysisDetailResponse])
def admin_list_all_analyses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    return [
        AnalysisDetailResponse(
            id=r.id,
            user_id=r.user_id,
            filename=r.filename,
            created_at=r.created_at,
            result_json=r.result_json,
        )
        for r in rows
    ]


@router.get("/admin/user/{user_id}", response_model=list[AnalysisResponse])
def admin_list_user_analyses(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    rows = (
        db.query(Analysis)
        .filter(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return [AnalysisResponse(id=r.id, filename=r.filename, created_at=r.created_at) for r in rows]


@router.get("/admin/users", response_model=list[dict])
def admin_list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """List all users for admin with role, user_type, and lawyer_status."""
    from app.db.models import LawyerApplication
    rows = db.query(User).order_by(User.id.desc()).offset(skip).limit(limit).all()
    user_ids = [u.id for u in rows]
    lawyer_apps = (
        db.query(LawyerApplication.user_id, LawyerApplication.status)
        .filter(LawyerApplication.user_id.in_(user_ids))
        .all()
    ) if user_ids else []
    lawyer_status_map = {uid: status for uid, status in lawyer_apps}
    return [
        {
            "id": u.id,
            "email": u.email or "",
            "role": getattr(u, "role", "user"),
            "user_type": getattr(u, "user_type", "user"),
            "lawyer_status": lawyer_status_map.get(u.id, ""),
            "is_verified_lawyer": lawyer_status_map.get(u.id) == "approved",
        }
        for u in rows
    ]


@router.patch("/admin/users/{user_id}")
def admin_update_user_role(
    user_id: int,
    payload: AdminUpdateRoleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update a user's role, user_type, or lawyer_status. Admin only."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    changed = False

    if payload.role is not None:
        role = payload.role.strip().lower()
        if role not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
        if target.id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot change your own role")
        target.role = role
        changed = True

    if payload.user_type is not None:
        target.user_type = payload.user_type.strip()
        changed = True

    if payload.lawyer_status is not None:
        # lawyer_status lives on the LawyerApplication record, not the User.
        from app.db.models import LawyerApplication
        from datetime import datetime, timezone
        app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == user_id).first()
        if app_record:
            app_record.status = payload.lawyer_status.strip()
            app_record.reviewed_at = datetime.now(timezone.utc)
            db.add(app_record)
        changed = True

    if not changed:
        raise HTTPException(status_code=400, detail="No updatable fields provided")

    db.add(target)
    db.commit()
    db.refresh(target)
    return {"id": target.id, "email": target.email, "role": target.role, "user_type": target.user_type}


@router.delete("/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Permanently delete a user account and related data. Admin only."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if getattr(target, "role", "user") == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin account")
    try:
        deleted = delete_user_account(db, user_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user")
    return {"status": "ok", "deleted_id": user_id, "email": deleted.email}


@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")

    is_admin = getattr(current_user, "role", "user") == "admin"
    if row.user_id != current_user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    return AnalysisDetailResponse(
        id=row.id,
        user_id=row.user_id,
        filename=row.filename,
        created_at=row.created_at,
        result_json=row.result_json,
    )


@router.patch("/{analysis_id}/flag")
def flag_analysis(
    analysis_id: int,
    payload: FlagRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Flag an analysis for lawyer review, or clear the flag. Owner or admin only."""
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Analysis not found")

    is_admin = (getattr(current_user, "role", "user") or "").lower() == "admin"
    if a.user_id != current_user.id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")

    a.needs_review = payload.needs_review
    if payload.lawyer_note is not None:
        a.lawyer_note = payload.lawyer_note
    db.add(a)
    db.commit()
    db.refresh(a)
    return {
        "id": a.id,
        "needs_review": a.needs_review,
        "lawyer_note": a.lawyer_note,
    }


@router.delete("/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # 1) fetch
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # 2) permissions: owner OR admin
    user_id = getattr(current_user, "id", None)
    role = (getattr(current_user, "role", None) or "").lower()
    is_admin = bool(getattr(current_user, "is_admin", False)) or (role == "admin")

    if (a.user_id != user_id) and (not is_admin):
        raise HTTPException(status_code=403, detail="Not allowed to delete this analysis")

    # 3) delete
    db.delete(a)
    db.commit()

    return {"status": "ok", "deleted_id": analysis_id}

