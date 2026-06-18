"""Paymob payment webhooks (test and live)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.db.models import ConsultationRequest, UserNotification
from app.db.session import get_db
from app.routers.consultations import _confirm_consultation_after_payment
from app.services.consultation_helpers import format_scheduled_egypt
from app.services.paymob import extract_consultation_id_from_transaction, verify_transaction_hmac

router = APIRouter(prefix="/api/payments", tags=["payments"])
_LOG = logging.getLogger(__name__)


def _transaction_obj(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if payload.get("type") == "TRANSACTION" and isinstance(payload.get("obj"), dict):
        return payload["obj"]
    if isinstance(payload.get("obj"), dict) and payload["obj"].get("id") is not None:
        return payload["obj"]
    if payload.get("id") is not None and payload.get("success") is not None:
        return payload
    return None


@router.post("/paymob/webhook")
async def paymob_webhook(
    request: Request,
    hmac: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Paymob transaction processed callback — activates consultation after successful test/live payment."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    obj = _transaction_obj(payload if isinstance(payload, dict) else {})
    if not obj:
        return {"ok": True, "ignored": True, "reason": "not_a_transaction"}

    incoming_hmac = hmac or request.query_params.get("hmac")
    if incoming_hmac and not verify_transaction_hmac(obj, incoming_hmac):
        _LOG.warning("Paymob webhook HMAC mismatch for transaction %s", obj.get("id"))
        raise HTTPException(status_code=401, detail="Invalid HMAC")

    if not obj.get("success"):
        return {"ok": True, "paid": False, "reason": "transaction_not_successful"}

    consultation_id = extract_consultation_id_from_transaction(obj)
    if consultation_id is None:
        _LOG.warning("Paymob webhook: could not map transaction %s to consultation", obj.get("id"))
        return {"ok": True, "paid": False, "reason": "consultation_not_found_in_payload"}

    row = db.query(ConsultationRequest).filter(ConsultationRequest.id == consultation_id).first()
    if not row:
        return {"ok": True, "paid": False, "reason": "consultation_missing"}

    if row.payment_status == "paid" and row.status in ("active", "confirmed"):
        return {"ok": True, "paid": True, "consultation_id": consultation_id, "already_processed": True}

    if row.status != "awaiting_payment":
        return {"ok": True, "paid": False, "reason": "consultation_not_awaiting_payment"}

    row.paymob_transaction_id = str(obj.get("id") or "")
    order = obj.get("order") or {}
    if order.get("id") is not None:
        row.paymob_order_id = str(order.get("id"))
    row.paid_at = datetime.now(timezone.utc)
    db.add(row)

    conv = _confirm_consultation_after_payment(db, row)
    sched_label = format_scheduled_egypt(row.scheduled_at)
    db.add(
        UserNotification(
            recipient_id=row.lawyer_id,
            actor_id=row.requester_id,
            type="consultation_response",
            reference_id=row.id,
            message=f"Consultation payment received — session starts at {sched_label}.",
            read=False,
        )
    )
    db.add(
        UserNotification(
            recipient_id=row.requester_id,
            actor_id=row.lawyer_id,
            type="consultation_response",
            reference_id=row.id,
            message=f"Payment confirmed — consultation booked for {sched_label}.",
            read=False,
        )
    )
    db.commit()
    return {
        "ok": True,
        "paid": True,
        "consultation_id": consultation_id,
        "conversation_id": conv.id,
    }
