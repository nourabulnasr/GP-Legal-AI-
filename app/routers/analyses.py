from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Analysis, User
from app.schemas.analyses import AnalysisCreateRequest, AnalysisResponse, AnalysisDetailResponse
from app.core.deps import get_current_user, require_admin  # ✅ IMPORTANT

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
        # Requirement: forbid reading others' analyses
        raise HTTPException(status_code=403, detail="Forbidden")

    return AnalysisDetailResponse(
        id=row.id,
        user_id=row.user_id,
        filename=row.filename,
        created_at=row.created_at,
        result_json=row.result_json,
    )


@router.get("/admin/all", response_model=list[AnalysisDetailResponse])
def admin_list_all_analyses(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
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

