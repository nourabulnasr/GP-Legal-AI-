from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.models import LawyerApplication, User, UserNotification
from app.db.session import get_db

router = APIRouter(prefix="/lawyer", tags=["lawyer"])

_MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/apply")
async def apply_as_lawyer(
    bar_license_number: str = Form(""),
    years_of_experience: int = Form(None),
    hourly_rate: float = Form(None),
    document: UploadFile = File(None),
    cv: UploadFile = File(None),
    id_card: UploadFile = File(None),
    id_card_back: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit a lawyer verification application with an optional license document."""
    existing = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()

    # Gate: lawyer-account holders can apply. Rejected applicants can re-apply even
    # though admin_lawyers.review reverts user_type to "user" on rejection.
    is_lawyer_account = getattr(current_user, "user_type", "user") == "lawyer"
    is_reapply_after_rejection = bool(existing and existing.status == "rejected")
    if not is_lawyer_account and not is_reapply_after_rejection:
        raise HTTPException(
            status_code=400,
            detail="Only accounts registered as 'lawyer' can submit an application.",
        )

    if existing and existing.status == "approved":
        raise HTTPException(status_code=400, detail="Your lawyer status is already approved.")
    if existing and existing.status == "pending":
        raise HTTPException(status_code=400, detail="You already have a pending application. Please wait for admin review.")

    doc_bytes = None
    doc_mime = None
    doc_filename = None
    if document and document.filename:
        raw = await document.read()
        if len(raw) > _MAX_DOC_BYTES:
            raise HTTPException(status_code=413, detail="Document too large (max 10 MB)")
        doc_bytes = raw
        doc_mime = document.content_type or "application/octet-stream"
        doc_filename = document.filename

    cv_bytes_data = cv_mime_data = cv_fn_data = None
    if cv and cv.filename:
        raw = await cv.read()
        if len(raw) > _MAX_DOC_BYTES:
            raise HTTPException(status_code=413, detail="CV too large (max 10 MB)")
        cv_bytes_data = raw
        cv_mime_data = cv.content_type or "application/octet-stream"
        cv_fn_data = cv.filename

    id_card_bytes_data = id_card_mime_data = id_card_fn_data = None
    id_card_back_bytes_data = id_card_back_mime_data = id_card_back_fn_data = None
    if id_card and id_card.filename:
        raw = await id_card.read()
        if len(raw) > _MAX_DOC_BYTES:
            raise HTTPException(status_code=413, detail="ID card too large (max 10 MB)")
        id_card_bytes_data = raw
        id_card_mime_data = id_card.content_type or "application/octet-stream"
        id_card_fn_data = id_card.filename

    if id_card_back and id_card_back.filename:
        raw = await id_card_back.read()
        if len(raw) > _MAX_DOC_BYTES:
            raise HTTPException(status_code=413, detail="ID card back too large (max 10 MB)")
        id_card_back_bytes_data = raw
        id_card_back_mime_data = id_card_back.content_type or "application/octet-stream"
        id_card_back_fn_data = id_card_back.filename

    if existing:
        # Re-apply after a rejection — restore the lawyer-account state so the rest
        # of the app (badges, /lawyer/status, Flutter UI) treats them consistently
        # while waiting for re-review.
        if not is_lawyer_account:
            current_user.user_type = "lawyer"
            db.add(current_user)
        existing.bar_license_number = (bar_license_number.strip() or existing.bar_license_number) or None
        if years_of_experience is not None:
            existing.years_of_experience = years_of_experience
        if hourly_rate is not None:
            existing.hourly_rate = hourly_rate
        if doc_bytes:
            existing.document_bytes = doc_bytes
            existing.document_mime_type = doc_mime
            existing.document_filename = doc_filename
        if cv_bytes_data:
            existing.cv_bytes = cv_bytes_data
            existing.cv_mime_type = cv_mime_data
            existing.cv_filename = cv_fn_data
        if id_card_bytes_data:
            existing.id_card_bytes = id_card_bytes_data
            existing.id_card_mime_type = id_card_mime_data
            existing.id_card_filename = id_card_fn_data
        if id_card_back_bytes_data:
            existing.id_card_back_bytes = id_card_back_bytes_data
            existing.id_card_back_mime_type = id_card_back_mime_data
            existing.id_card_back_filename = id_card_back_fn_data
        existing.status = "pending"
        existing.admin_note = None
        existing.reviewed_at = None
        db.add(existing)
    else:
        app_record = LawyerApplication(
            user_id=current_user.id,
            bar_license_number=bar_license_number.strip() or None,
            years_of_experience=years_of_experience,
            hourly_rate=hourly_rate,
            document_bytes=doc_bytes,
            document_mime_type=doc_mime,
            document_filename=doc_filename,
            cv_bytes=cv_bytes_data,
            cv_mime_type=cv_mime_data,
            cv_filename=cv_fn_data,
            id_card_bytes=id_card_bytes_data,
            id_card_mime_type=id_card_mime_data,
            id_card_filename=id_card_fn_data,
            id_card_back_bytes=id_card_back_bytes_data,
            id_card_back_mime_type=id_card_back_mime_data,
            id_card_back_filename=id_card_back_fn_data,
            status="pending",
        )
        db.add(app_record)

    db.commit()

    # Notify all admins about the new/updated application
    admins = db.query(User).filter(User.role == "admin").all()
    for admin in admins:
        db.add(UserNotification(
            recipient_id=admin.id,
            actor_id=current_user.id,
            type="lawyer_application",
            message=f"{current_user.email} submitted a lawyer application.",
            read=False,
        ))
    if admins:
        db.commit()

    return {"status": "ok", "message": "Application submitted. Awaiting admin review."}


@router.get("/status")
def my_lawyer_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the current user's lawyer application status."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()
    if not app_record:
        return {
            "status": "not_applied",
            "user_type": getattr(current_user, "user_type", "user"),
        }
    return {
        "status": app_record.status,
        "bar_license_number": app_record.bar_license_number,
        "years_of_experience": app_record.years_of_experience,
        "hourly_rate": app_record.hourly_rate,
        "negotiated_hourly_rate": getattr(app_record, "negotiated_hourly_rate", None),
        "document_filename": app_record.document_filename,
        "has_document": bool(app_record.document_bytes),
        "cv_filename": app_record.cv_filename,
        "has_cv": bool(app_record.cv_bytes),
        "id_card_filename": app_record.id_card_filename,
        "has_id_card": bool(app_record.id_card_bytes),
        "id_card_back_filename": getattr(app_record, "id_card_back_filename", None),
        "has_id_card_back": bool(getattr(app_record, "id_card_back_bytes", None)),
        "admin_note": app_record.admin_note,
        "created_at": app_record.created_at,
        "reviewed_at": app_record.reviewed_at,
        "user_type": getattr(current_user, "user_type", "user"),
    }


@router.get("/document/{application_id}")
def download_own_lawyer_document(
    application_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the uploaded license document (own application only, or admin)."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.id == application_id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    if app_record.user_id != current_user.id and getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    if not app_record.document_bytes:
        raise HTTPException(status_code=404, detail="No document uploaded")
    return Response(
        content=app_record.document_bytes,
        media_type=app_record.document_mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{app_record.document_filename or "license_document"}"'
        },
    )


@router.get("/cv")
def download_my_cv(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the current user's own uploaded CV."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="No application found")
    cv_bytes = getattr(app_record, "cv_bytes", None)
    if not cv_bytes:
        raise HTTPException(status_code=404, detail="No CV uploaded")
    return Response(
        content=cv_bytes,
        media_type=getattr(app_record, "cv_mime_type", None) or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{getattr(app_record, "cv_filename", None) or "cv"}"'
        },
    )


@router.get("/id-card")
def download_my_id_card(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the current user's own uploaded ID card (front)."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="No application found")
    id_card_bytes = getattr(app_record, "id_card_bytes", None)
    if not id_card_bytes:
        raise HTTPException(status_code=404, detail="No ID card uploaded")
    return Response(
        content=id_card_bytes,
        media_type=getattr(app_record, "id_card_mime_type", None) or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{getattr(app_record, "id_card_filename", None) or "id_card"}"'
        },
    )


@router.get("/id-card-back")
def download_my_id_card_back(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the current user's own uploaded ID card (back)."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="No application found")
    id_card_back_bytes = getattr(app_record, "id_card_back_bytes", None)
    if not id_card_back_bytes:
        raise HTTPException(status_code=404, detail="No ID card back uploaded")
    return Response(
        content=id_card_back_bytes,
        media_type=getattr(app_record, "id_card_back_mime_type", None) or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{getattr(app_record, "id_card_back_filename", None) or "id_card_back"}"'
        },
    )


class RateResponseBody(BaseModel):
    accept: bool


@router.post("/rate-response")
def respond_to_negotiated_rate(
    body: RateResponseBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lawyer accepts or rejects admin's proposed hourly rate."""
    app_record = db.query(LawyerApplication).filter(LawyerApplication.user_id == current_user.id).first()
    if not app_record:
        raise HTTPException(status_code=404, detail="No application found")
    if app_record.status != "rate_pending":
        raise HTTPException(status_code=400, detail="No pending rate negotiation")

    now = datetime.now(timezone.utc)
    app_record.reviewed_at = now

    if body.accept:
        if app_record.negotiated_hourly_rate is None:
            raise HTTPException(status_code=400, detail="No negotiated rate on file")
        app_record.hourly_rate = app_record.negotiated_hourly_rate
        app_record.status = "approved"
        current_user.user_type = "lawyer"
        db.add(current_user)
        msg = f"You accepted the proposed rate of {app_record.hourly_rate}/hr. Your lawyer account is now verified."
    else:
        app_record.status = "rejected"
        current_user.user_type = "user"
        db.add(current_user)
        msg = "You declined the proposed hourly rate. Your application was rejected."

    db.add(app_record)
    db.commit()
    return {"status": "ok", "application_status": app_record.status, "hourly_rate": app_record.hourly_rate, "message": msg}
