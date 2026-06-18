"""Paymob Accept — test (sandbox) and live payment helpers."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

_CONSULTATION_REF_PREFIX = "consultation-"


def paymob_configured() -> bool:
    return bool(_secret_key() and _public_key() and _integration_ids())


def paymob_mode() -> str:
    raw = (os.getenv("PAYMOB_MODE") or "test").strip().lower()
    return "live" if raw == "live" else "test"


def _base_url() -> str:
    return (os.getenv("PAYMOB_BASE_URL") or "https://accept.paymob.com").rstrip("/")


def _secret_key() -> str:
    return (os.getenv("PAYMOB_SECRET_KEY") or "").strip()


def _public_key() -> str:
    return (os.getenv("PAYMOB_PUBLIC_KEY") or "").strip()


def _hmac_secret() -> str:
    return (os.getenv("PAYMOB_HMAC_SECRET") or "").strip()


def _integration_ids() -> List[int]:
    raw = (os.getenv("PAYMOB_INTEGRATION_ID") or "").strip()
    if not raw:
        return []
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def _frontend_url() -> str:
    return (os.getenv("FRONTEND_URL") or "https://legatoappgp2026.web.app").rstrip("/")


def _api_public_url() -> str:
    return (os.getenv("IMAGE_BASE_URL") or os.getenv("API_PUBLIC_URL") or "https://srv1723974.hstgr.cloud").rstrip("/")


def consultation_reference(consultation_id: int) -> str:
    return f"{_CONSULTATION_REF_PREFIX}{consultation_id}"


def parse_consultation_reference(reference: Optional[str]) -> Optional[int]:
    if not reference:
        return None
    m = re.fullmatch(rf"{re.escape(_CONSULTATION_REF_PREFIX)}(\d+)", reference.strip())
    if not m:
        return None
    return int(m.group(1))


def _billing_data(email: str, first_name: str, last_name: str, phone: str = "01000000000") -> Dict[str, str]:
    return {
        "apartment": "NA",
        "email": email,
        "floor": "NA",
        "first_name": first_name or "Legato",
        "street": "NA",
        "building": "NA",
        "phone_number": phone,
        "shipping_method": "NA",
        "postal_code": "NA",
        "city": "Cairo",
        "country": "EG",
        "last_name": last_name or "User",
        "state": "Cairo",
    }


def create_consultation_checkout(
    *,
    consultation_id: int,
    amount_egp: float,
    customer_email: str,
    customer_name: str,
) -> Dict[str, Any]:
    """Create a Paymob payment intention and return hosted checkout URL."""
    if not paymob_configured():
        raise RuntimeError(
            "Paymob is not configured. Set PAYMOB_SECRET_KEY, PAYMOB_PUBLIC_KEY, and PAYMOB_INTEGRATION_ID (test keys from Accept dashboard)."
        )

    amount_cents = int(round(amount_egp * 100))
    if amount_cents <= 0:
        raise ValueError("Payment amount must be greater than zero")

    name_parts = (customer_name or "Legato User").strip().split(None, 1)
    first_name = name_parts[0] if name_parts else "Legato"
    last_name = name_parts[1] if len(name_parts) > 1 else "User"
    reference = consultation_reference(consultation_id)
    success_url = (os.getenv("PAYMOB_SUCCESS_URL") or f"{_frontend_url()}/").strip()
    failure_url = (os.getenv("PAYMOB_FAILURE_URL") or f"{_frontend_url()}/").strip()
    webhook_url = (os.getenv("PAYMOB_WEBHOOK_URL") or f"{_api_public_url()}/api/payments/paymob/webhook").strip()

    payload: Dict[str, Any] = {
        "amount": amount_cents,
        "currency": "EGP",
        "payment_methods": _integration_ids(),
        "items": [
            {
                "name": f"Consultation #{consultation_id}",
                "amount": amount_cents,
                "description": "Legato legal consultation booking",
                "quantity": 1,
            }
        ],
        "billing_data": _billing_data(customer_email, first_name, last_name),
        "special_reference": reference,
        "notification_url": webhook_url,
        "redirection_url": success_url,
        "extras": {"consultation_id": consultation_id, "reference": reference},
    }

    headers = {"Authorization": f"Token {_secret_key()}", "Content-Type": "application/json"}
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{_base_url()}/v1/intention/", json=payload, headers=headers)
        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise RuntimeError(f"Paymob intention failed ({resp.status_code}): {detail}")
        data = resp.json()

    client_secret = data.get("client_secret") or data.get("payment_keys", [{}])[0].get("key")
    if not client_secret:
        raise RuntimeError("Paymob did not return a client_secret")

    public_key = quote(_public_key(), safe="")
    secret = quote(str(client_secret), safe="")
    payment_url = f"{_base_url()}/unifiedcheckout/?publicKey={public_key}&clientSecret={secret}"

    return {
        "provider": "paymob",
        "mode": paymob_mode(),
        "checkout_ready": True,
        "payment_url": payment_url,
        "client_secret": client_secret,
        "intention_id": data.get("id"),
        "special_reference": reference,
        "success_url": success_url,
        "failure_url": failure_url,
        "webhook_url": webhook_url,
    }


def verify_transaction_hmac(obj: Dict[str, Any], incoming_hmac: str) -> bool:
    secret = _hmac_secret()
    if not secret or not incoming_hmac:
        return False

    hmac_dict = {
        "amount_cents": obj.get("amount_cents"),
        "created_at": obj.get("created_at"),
        "currency": obj.get("currency"),
        "error_occured": obj.get("error_occured"),
        "has_parent_transaction": obj.get("has_parent_transaction"),
        "id": obj.get("id"),
        "integration_id": obj.get("integration_id"),
        "is_3d_secure": obj.get("is_3d_secure"),
        "is_auth": obj.get("is_auth"),
        "is_capture": obj.get("is_capture"),
        "is_refunded": obj.get("is_refunded"),
        "is_standalone_payment": obj.get("is_standalone_payment"),
        "is_voided": obj.get("is_voided"),
        "order.id": (obj.get("order") or {}).get("id"),
        "owner": obj.get("owner"),
        "pending": obj.get("pending"),
        "source_data.pan": (obj.get("source_data") or {}).get("pan"),
        "source_data.sub_type": (obj.get("source_data") or {}).get("sub_type"),
        "source_data.type": (obj.get("source_data") or {}).get("type"),
        "success": obj.get("success"),
    }

    message = ""
    for value in hmac_dict.values():
        if isinstance(value, bool):
            value = str(value).lower()
        if value is None:
            value = ""
        message += str(value)

    calculated = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha512).hexdigest()
    return hmac.compare_digest(calculated.lower(), incoming_hmac.lower())


def extract_consultation_id_from_transaction(obj: Dict[str, Any]) -> Optional[int]:
    order = obj.get("order") or {}
    for candidate in (
        order.get("merchant_order_id"),
        order.get("special_reference"),
        (obj.get("payment_key_claims") or {}).get("extra", {}).get("reference"),
        (obj.get("data") or {}).get("special_reference"),
    ):
        cid = parse_consultation_reference(str(candidate) if candidate is not None else None)
        if cid is not None:
            return cid

    extras = order.get("extras") or obj.get("extras") or {}
    if isinstance(extras, dict):
        raw = extras.get("consultation_id") or extras.get("reference")
        if raw is not None:
            if isinstance(raw, int):
                return raw
            cid = parse_consultation_reference(str(raw))
            if cid is not None:
                return cid
            if str(raw).isdigit():
                return int(raw)
    return None
