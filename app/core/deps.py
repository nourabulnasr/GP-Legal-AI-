from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.db.models import LawyerApplication, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/form")



def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # في أغلب implementations بتبقى sub = user_id أو email
    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(status_code=401, detail="Token missing subject")

    # جرّب sub كـ id
    user = None
    try:
        uid = int(sub)
        user = db.query(User).filter(User.id == uid).first()
    except Exception:
        # لو sub كان email
        user = db.query(User).filter(User.email == str(sub)).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency that allows only admin users."""
    if getattr(current_user, "role", "user") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user


def require_verified_lawyer(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that allows only admin-approved lawyers (or admins).

    Checks for a LawyerApplication row with status == "approved" — the same
    source of truth used inline by the offer-send endpoint. A pending or
    rejected application does NOT pass this check.
    """
    if getattr(current_user, "role", "user") == "admin":
        return current_user
    approved = (
        db.query(LawyerApplication)
        .filter(
            LawyerApplication.user_id == current_user.id,
            LawyerApplication.status == "approved",
        )
        .first()
    )
    if not approved:
        raise HTTPException(status_code=403, detail="Verified lawyer access required")
    return current_user
