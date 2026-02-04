from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict

from jose import jwt, JWTError
from passlib.context import CryptContext

# =========================
# Password hashing (PBKDF2)
# =========================
# IMPORTANT: We intentionally avoid bcrypt for now because bcrypt/passlib backend is broken in your image.
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],  # <-- stable, no bcrypt
    deprecated="auto",
)

def hash_password(password: str) -> str:
    # basic guard
    password = (password or "").strip()
    if not password:
        raise ValueError("Password is empty")
    return pwd_context.hash(password)

def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify((plain_password or "").strip(), password_hash)
    except Exception:
        return False


# =========================
# JWT
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

def create_access_token(subject: str | None = None, *, sub: str | None = None, role: str = "user", expires_minutes: int | None = None):
    """
    Create JWT access token.
    Accepts subject in 3 ways:
      - create_access_token("2")
      - create_access_token(subject="2")
      - create_access_token(sub="2")
    """
    user_id = sub or subject
    if not user_id:
        raise ValueError("create_access_token requires subject/sub")

    exp_minutes = expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=exp_minutes)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode JWT access token.

    Returns payload dict if valid, otherwise None (expired/invalid).
    This prevents expired tokens from causing 500s in dependencies.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
