from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.db.models import LawyerApplication, User, UserNotification
from app.db.session import get_db

router = APIRouter(prefix="/admin/lawyers", tags=["admin-lawyers"])


class AdminReviewRequest(BaseModel):
    action: str = Field(..., pattern="^(approve|reject|negotiate)$")
    admin_note: Optional[str] = None
    negotiated_hourly_rate: Optional[float] = Field(None, gt=0)


@router.get("")
def list_lawyer_applications(
    status: str = Query("pending", description="Filter by status: pending / approved / rejected / all"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """List lawyer applications filtered by status."""
    q = db.query(LawyerApplication, User).join(User, LawyerApplication.user_id == User.id)
    if status != "all":
        q = q.filter(LawyerApplication.status == status)
    rows = q.order_by(LawyerApplication.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": app_record.id,
            "user_id": app_record.user_id,
            "user_email": user.email,
            "bar_license_number": app_record.bar_license_number,
            "years_of_experience": getattr(app_record, "years_of_experience", None),
            "hourly_rate": getattr(app_record, "hourly_rate", None),
            "negotiated_hourly_rate": getattr(app_record, "negotiated_hourly_rate", None),
            "document_filename": app_record.document_filename,
            "has_document": bool(app_record.document_bytes),
            "cv_filename": getattr(app_record, "cv_filename", None),
            "has_cv": bool(getattr(app_record, "cv_bytes", None)),
            "id_card_filename": getattr(app_record, "id_card_filename", None),
            "has_id_card": bool(getattr(app_record, "id_card_bytes", None)),
            "id_card_back_filename": getattr(app_record, "id_card_back_filename", None),
            "has_id_card_back": bool(getattr(app_record, "id_card_back_bytes", None)),
            "status": app_record.status,
            "admin_note": app_record.admin_note,
            "created_at": app_record.created_at,
            "reviewed_at": app_record.reviewed_at,
        }
        for app_record, user in rows
    ]


@router.patch("/{application_id}/review")
def review_lawyer_application(
    application_id: int,
    payload: AdminReviewRequest,
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin),
):
    """Approve or reject a lawyer application. Notifies the applicant."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")

    applicant = db.query(User).filter(User.id == app_record.user_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant user not found")

    if payload.action == "negotiate":
        if payload.negotiated_hourly_rate is None:
            raise HTTPException(status_code=400, detail="negotiated_hourly_rate is required for negotiate action")
        app_record.negotiated_hourly_rate = payload.negotiated_hourly_rate
        app_record.status = "rate_pending"
        app_record.admin_note = payload.admin_note
        db.add(app_record)
        db.commit()
        db.add(UserNotification(
            recipient_id=applicant.id,
            actor_id=admin_user.id,
            type="lawyer_rate_negotiation",
            reference_id=app_record.id,
            message=(
                f"Admin proposed hourly rate: {payload.negotiated_hourly_rate}."
                f"{f' Note: {payload.admin_note}' if payload.admin_note else ''}"
            ),
            read=False,
        ))
        db.commit()
        return {
            "status": "ok",
            "application_status": app_record.status,
            "negotiated_hourly_rate": app_record.negotiated_hourly_rate,
            "user_email": applicant.email,
        }

    if payload.action == "approve":
        app_record.status = "approved"
        applicant.user_type = "lawyer"
        if app_record.negotiated_hourly_rate is not None:
            app_record.hourly_rate = app_record.negotiated_hourly_rate
    else:
        app_record.status = "rejected"
        applicant.user_type = "user"

    app_record.admin_note = payload.admin_note
    app_record.reviewed_at = datetime.now(timezone.utc)
    db.add(app_record)
    db.add(applicant)
    db.commit()

    # Notify the lawyer about the decision
    action_text = "approved" if payload.action == "approve" else "rejected"
    note_part = f" Note: {payload.admin_note}" if payload.admin_note else ""
    db.add(UserNotification(
        recipient_id=applicant.id,
        actor_id=admin_user.id,
        type="lawyer_review",
        reference_id=app_record.id,
        message=f"Your lawyer application has been {action_text}.{note_part}",
        read=False,
    ))
    db.commit()

    return {
        "status": "ok",
        "application_status": app_record.status,
        "user_type_updated_to": applicant.user_type,
        "user_email": applicant.email,
    }


@router.get("/document/{application_id}")
def admin_download_lawyer_document(
    application_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Download the bar license document for a lawyer application."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    if not app_record.document_bytes:
        raise HTTPException(status_code=404, detail="No document uploaded for this application")
    return Response(
        content=app_record.document_bytes,
        media_type=app_record.document_mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{app_record.document_filename or "license_document"}"'
        },
    )


@router.get("/cv/{application_id}")
def admin_download_lawyer_cv(
    application_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Download the CV for a lawyer application."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    cv_bytes = getattr(app_record, "cv_bytes", None)
    if not cv_bytes:
        raise HTTPException(status_code=404, detail="No CV uploaded for this application")
    cv_filename = getattr(app_record, "cv_filename", None) or "cv"
    cv_mime = getattr(app_record, "cv_mime_type", None) or "application/octet-stream"
    return Response(
        content=cv_bytes,
        media_type=cv_mime,
        headers={"Content-Disposition": f'attachment; filename="{cv_filename}"'},
    )


@router.get("/id-card/{application_id}")
def admin_download_lawyer_id_card(
    application_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Download the ID card for a lawyer application."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    id_card_bytes = getattr(app_record, "id_card_bytes", None)
    if not id_card_bytes:
        raise HTTPException(status_code=404, detail="No ID card uploaded for this application")
    id_card_filename = getattr(app_record, "id_card_filename", None) or "id_card"
    id_card_mime = getattr(app_record, "id_card_mime_type", None) or "application/octet-stream"
    return Response(
        content=id_card_bytes,
        media_type=id_card_mime,
        headers={"Content-Disposition": f'attachment; filename="{id_card_filename}"'},
    )


@router.get("/id-card-back/{application_id}")
def admin_download_lawyer_id_card_back(
    application_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Download the ID card back for a lawyer application."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    id_card_back_bytes = getattr(app_record, "id_card_back_bytes", None)
    if not id_card_back_bytes:
        raise HTTPException(status_code=404, detail="No ID card back uploaded for this application")
    id_card_back_filename = getattr(app_record, "id_card_back_filename", None) or "id_card_back"
    id_card_back_mime = getattr(app_record, "id_card_back_mime_type", None) or "application/octet-stream"
    return Response(
        content=id_card_back_bytes,
        media_type=id_card_back_mime,
        headers={"Content-Disposition": f'attachment; filename="{id_card_back_filename}"'},
    )
