from __future__ import annotations

import hashlib
import secrets
import smtplib
import os
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.session import get_db
from app.db.models import User
from app.core.security import hash_password, verify_password, create_access_token
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    MeResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from app.core.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

# SMTP config for password reset emails
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@legato.local")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
RESET_TOKEN_EXPIRE_HOURS = 24


def _token_hash(token: str) -> str:
    """SHA-256 hash of token for secure storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def _send_reset_email(to_email: str, reset_link: str) -> bool:
    """Send password reset email via SMTP. Returns True if sent."""
    if not SMTP_HOST or not SMTP_USER:
        print(f"[DEV] Reset link for {to_email}: {reset_link}")
        return True
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Legato – Reset your password"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        html = f"""
        <p>You requested a password reset for Legato.</p>
        <p><a href="{reset_link}">Reset your password</a></p>
        <p>This link expires in {RESET_TOKEN_EXPIRE_HOURS} hours.</p>
        <p>If you didn't request this, ignore this email.</p>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[SMTP] Failed to send: {e}")
        return False


@router.post("/register", response_model=MeResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    exists = db.query(User).filter(User.email == email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    u = User(email=email, password_hash=hash_password(payload.password), role="user")
    db.add(u)
    db.commit()
    db.refresh(u)
    return MeResponse(id=u.id, email=u.email, role=getattr(u, "role", "user"))


# ✅ JSON login (زي ما انت بتستخدمه في curl حاليا)
@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # ✅ خليها sub بدل subject عشان تمنع الخطأ اللي ظهر عندك
    token = create_access_token(sub=str(u.id), role=getattr(u, "role", "user"))
    return TokenResponse(access_token=token)


# ✅ Swagger Authorize login (form-data)
@router.post("/login/form", response_model=TokenResponse)
def login_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Swagger يبعت username + password
    email = (form.username or "").lower().strip()
    password = form.password or ""

    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(sub=str(u.id), role=getattr(u, "role", "user"))
    return TokenResponse(access_token=token)


# Google OAuth config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://127.0.0.1:8000/auth/google/callback",
)


@router.get("/google")
def google_oauth_redirect():
    """Google OAuth - redirect to Google sign-in."""
    if not GOOGLE_CLIENT_ID:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_not_configured",
            status_code=302,
        )
    from fastapi.responses import RedirectResponse
    scope = "openid email profile"
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scope}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/google/callback")
async def google_oauth_callback(code: str = "", error: str = "", db: Session = Depends(get_db)):
    """Google OAuth callback - exchange code for token, create/get user, redirect to frontend with JWT."""
    from fastapi.responses import RedirectResponse

    if error or not code:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_denied",
            status_code=302,
        )

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_not_configured",
            status_code=302,
        )

    import httpx
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_resp.status_code != 200:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_token_failed",
            status_code=302,
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_token_failed",
            status_code=302,
        )

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if user_resp.status_code != 200:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_user_failed",
            status_code=302,
        )

    user_info = user_resp.json()
    email = (user_info.get("email") or "").lower().strip()
    if not email:
        return RedirectResponse(
            url=f"{FRONTEND_URL}/login?error=google_no_email",
            status_code=302,
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash="",  # No password for OAuth-only users
            role="user",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(sub=str(user.id), role=getattr(user, "role", "user"))
    return RedirectResponse(
        url=f"{FRONTEND_URL}/auth/google/callback?token={token}",
        status_code=302,
    )


@router.get("/me", response_model=MeResponse)
def me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=current_user.id,
        email=current_user.email,
        role=getattr(current_user, "role", "user"),
    )


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset link. Always returns 200 to avoid email enumeration."""
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return {"status": "ok", "message": "If an account exists, you will receive a reset link."}

    token = secrets.token_urlsafe(32)
    token_hash_val = _token_hash(token)
    expires = datetime.now(timezone.utc) + timedelta(hours=RESET_TOKEN_EXPIRE_HOURS)

    # Create table if missing (SQLite)
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash VARCHAR(64) NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        db.commit()
    except Exception:
        db.rollback()

    # Delete old tokens and insert new one
    try:
        db.execute(
            text("DELETE FROM password_reset_tokens WHERE user_id = :uid"),
            {"uid": user.id},
        )
        db.execute(
            text("""
                INSERT INTO password_reset_tokens (user_id, token_hash, expires_at)
                VALUES (:uid, :th, :exp)
            """),
            {"uid": user.id, "th": token_hash_val, "exp": expires.isoformat()},
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create reset token")

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    _send_reset_email(email, reset_link)

    return {"status": "ok", "message": "If an account exists, you will receive a reset link."}


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password using token from email link."""
    token = (payload.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    token_hash_val = _token_hash(token)
    now = datetime.now(timezone.utc).isoformat()

    try:
        row = db.execute(
            text("""
                SELECT user_id FROM password_reset_tokens
                WHERE token_hash = :th AND expires_at > :now
            """),
            {"th": token_hash_val, "now": now},
        ).fetchone()
    except Exception:
        raise HTTPException(status_code=500, detail="Reset token table not available")

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user_id = row[0]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    user.password_hash = hash_password(payload.new_password)
    db.add(user)
    db.execute(
        text("DELETE FROM password_reset_tokens WHERE user_id = :uid"),
        {"uid": user_id},
    )
    db.commit()

    return {"status": "ok", "message": "Password has been reset. You can sign in now."}
