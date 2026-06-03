import hashlib
import secrets
import smtplib
import os
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordRequestForm
from app.core.limiter import limiter
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
    VerifyResetCodeRequest,
    VerifyResetCodeResponse,
    VerifyEmailRequest,
    ResendVerificationRequest,
)
from app.core.deps import get_current_user
from app.email_templates import verification_email_html, reset_link_email_html

router = APIRouter(prefix="/auth", tags=["auth"])

RESET_TOKEN_EXPIRE_HOURS = 24
VERIFICATION_CODE_EXPIRE_MINUTES = int(os.getenv("VERIFICATION_CODE_EXPIRE_MINUTES", "30"))
RESET_TOKEN_AFTER_CODE_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_AFTER_CODE_EXPIRE_MINUTES", "30"))
RESEND_VERIFICATION_COOLDOWN_SECONDS = 60
_resend_last_sent: dict[str, float] = {}


def _smtp_settings():
    """Read SMTP config from env at send-time. Returns (host, port, user, password, from_addr).
    For Gmail, From is forced to SMTP_USER so the server accepts the message."""
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    from_env = (os.getenv("SMTP_FROM", "").strip() or user or "noreply@legato.local")
    # Gmail rejects if From != authenticated user; use user as From for gmail
    if host and "gmail" in host.lower() and user:
        from_addr = user
    else:
        from_addr = from_env
    return host, port, user, password, from_addr


def _smtp_configured() -> bool:
    """True only when SMTP can actually send (host + user + password)."""
    host, _, user, password, _ = _smtp_settings()
    return bool(host and user and password)


def _allow_dev_auto_verify() -> bool:
    env = os.getenv("ENV", "dev").strip().lower()
    flag = os.getenv("AUTO_VERIFY_EMAIL_IN_DEV", "1").strip().lower()
    enabled = flag in {"1", "true", "yes", "on"}
    return env != "prod" and enabled


def _smtp_send_message(host: str, port: int, user: str, password: str, msg: MIMEMultipart) -> None:
    """Send email via SMTP. Supports port 465 (SSL) and 587 (STARTTLS)."""
    if port == 465:
        with smtplib.SMTP_SSL(host, port) as server:
            if user and password:
                server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if user and password:
                server.login(user, password)
            server.send_message(msg)


def _token_hash(token: str) -> str:
    """SHA-256 hash of token for secure storage."""
    return hashlib.sha256(token.encode()).hexdigest()


def _code_hash(code: str) -> str:
    """SHA-256 hash of verification code for secure storage. Normalize to lowercase (codes are hex)."""
    return hashlib.sha256(code.strip().lower().encode()).hexdigest()


def _generate_verification_code() -> str:
    """6 hex chars."""
    return secrets.token_hex(3)


def _ensure_verification_codes_table(db: Session) -> None:
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS verification_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(255) NOT NULL,
                user_id INTEGER,
                code_hash VARCHAR(64) NOT NULL,
                purpose VARCHAR(32) NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """))
        db.commit()
    except Exception:
        db.rollback()


def _send_reset_email(to_email: str, reset_link: str) -> bool:
    """Send password reset link email via SMTP. Plain + HTML. Returns True if sent."""
    host, port, user, password, from_addr = _smtp_settings()
    if not host or not user:
        print("[DEV] Reset link for", to_email, "(SMTP not configured)")
        return True
    expire_hours = RESET_TOKEN_EXPIRE_HOURS
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").strip()
    html = reset_link_email_html(
        reset_link=reset_link,
        frontend_url=frontend_url,
        expire_hours=expire_hours,
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Legato – Reset your password"
        msg["From"] = from_addr
        msg["To"] = to_email
        plain = (
            f"You requested a password reset.\n\n"
            f"Open this link: {reset_link}\n\n"
            f"This link expires in {expire_hours} hours.\n"
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))
        _smtp_send_message(host, port, user, password, msg)
        return True
    except Exception as e:
        print("[SMTP] Failed to send reset email:", e)
        return False


def _send_verification_email(to_email: str, code: str, purpose: str) -> bool:
    """Send verification code email (signup or password_reset). Returns True if sent."""
    host, port, user, password, from_addr = _smtp_settings()
    if purpose == "signup":
        subject = "Legato – Verify your email"
    else:
        subject = "Legato – Password reset code"
    if not host or not user:
        print("[DEV] Verification code for", to_email, ":", code, "(SMTP not configured)")
        return True
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173").strip()
    expire_minutes = VERIFICATION_CODE_EXPIRE_MINUTES
    html = verification_email_html(
        code=code,
        purpose=purpose,
        frontend_url=frontend_url,
        expire_minutes=expire_minutes,
    )
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_email
        plain = f"Your code is: {code}\nThis code expires in {expire_minutes} minutes.\n"
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))
        _smtp_send_message(host, port, user, password, msg)
        print("[SMTP] Verification email sent to", to_email, "(purpose:", purpose, ")")
        return True
    except Exception as e:
        print("[SMTP] Failed to send verification email to", to_email, ":", e)
        return False


@router.post("/register", response_model=MeResponse)
@limiter.limit("5/minute")
def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    exists = db.query(User).filter(User.email == email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    auto_verify = _allow_dev_auto_verify() and not _smtp_configured()

    u = User(
        email=email,
        password_hash=hash_password(payload.password),
        role="user",
        email_verified=bool(auto_verify),
    )
    db.add(u)
    db.commit()
    db.refresh(u)

    if auto_verify:
        return MeResponse(id=u.id, email=u.email, role=getattr(u, "role", "user"))

    code = _generate_verification_code()
    code_hash_val = _code_hash(code)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)
    _ensure_verification_codes_table(db)
    try:
        db.execute(
            text("""
                DELETE FROM verification_codes WHERE email = :email AND purpose = 'signup'
            """),
            {"email": email},
        )
        db.execute(
            text("""
                INSERT INTO verification_codes (email, user_id, code_hash, purpose, expires_at)
                VALUES (:email, :uid, :ch, 'signup', :exp)
            """),
            {"email": email, "uid": u.id, "ch": code_hash_val, "exp": expires_at.isoformat()},
        )
        db.commit()
    except Exception:
        db.rollback()
        db.delete(u)
        db.commit()
        raise HTTPException(status_code=500, detail="Failed to create verification code")

    sent = _send_verification_email(email, code, "signup")
    if not sent:
        # Code is in DB; user can use "Resend code" on verify step or login page
        pass
    return MeResponse(id=u.id, email=u.email, role=getattr(u, "role", "user"))


# ✅ JSON login (زي ما انت بتستخدمه في curl حاليا)
@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()

    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not getattr(u, "email_verified", True):
        if _allow_dev_auto_verify() and not _smtp_configured():
            try:
                u.email_verified = True
                db.add(u)
                db.commit()
                db.refresh(u)
            except Exception:
                db.rollback()
        if not getattr(u, "email_verified", True):
            raise HTTPException(
                status_code=403,
                detail="Please verify your email first. Check your inbox for the verification code.",
            )

    token = create_access_token(sub=str(u.id), role=getattr(u, "role", "user"))
    return TokenResponse(access_token=token)


@router.post("/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Verify the 6-character code sent after registration. Marks email as verified so user can sign in."""
    email = payload.email.lower().strip()
    code = (payload.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    code_hash_val = _code_hash(code.lower())
    now = datetime.now(timezone.utc).isoformat()

    try:
        row = db.execute(
            text("""
                SELECT user_id FROM verification_codes
                WHERE email = :email AND code_hash = :ch AND purpose = 'signup' AND expires_at > :now
            """),
            {"email": email, "ch": code_hash_val, "now": now},
        ).fetchone()
    except Exception:
        raise HTTPException(status_code=500, detail="Verification code table not available")

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")

    user_id = row[0]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    try:
        user.email_verified = True
        db.add(user)
        db.execute(
            text("DELETE FROM verification_codes WHERE email = :email AND purpose = 'signup'"),
            {"email": email},
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to verify email")

    return {"status": "ok", "message": "Email verified. You can sign in now."}


@router.post("/resend-verification")
@limiter.limit("5/minute")
def resend_verification(request: Request, payload: ResendVerificationRequest, db: Session = Depends(get_db)):
    """Send a new verification code to an unverified user. Rate-limited per email."""
    global _resend_last_sent
    email = payload.email.lower().strip()
    u = db.query(User).filter(User.email == email).first()
    if not u:
        raise HTTPException(status_code=404, detail="No account found with this email")
    if getattr(u, "email_verified", True):
        raise HTTPException(status_code=400, detail="Email is already verified. You can sign in.")
    now = datetime.now(timezone.utc).timestamp()
    last = _resend_last_sent.get(email, 0)
    if now - last < RESEND_VERIFICATION_COOLDOWN_SECONDS:
        raise HTTPException(
            status_code=429,
            detail="Please wait a minute before requesting another code.",
        )
    code = _generate_verification_code()
    code_hash_val = _code_hash(code)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)
    _ensure_verification_codes_table(db)
    try:
        db.execute(
            text("DELETE FROM verification_codes WHERE email = :email AND purpose = 'signup'"),
            {"email": email},
        )
        db.execute(
            text("""
                INSERT INTO verification_codes (email, user_id, code_hash, purpose, expires_at)
                VALUES (:email, :uid, :ch, 'signup', :exp)
            """),
            {"email": email, "uid": u.id, "ch": code_hash_val, "exp": expires_at.isoformat()},
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create verification code")
    sent = _send_verification_email(email, code, "signup")
    _resend_last_sent[email] = now
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="Verification email could not be sent. Check SMTP settings and try again later.",
        )
    return {"status": "ok", "message": "Verification code sent. Check your inbox."}


# ✅ Swagger Authorize login (form-data)
@router.post("/login/form", response_model=TokenResponse)
def login_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Swagger يبعت username + password
    email = (form.username or "").lower().strip()
    password = form.password or ""

    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not getattr(u, "email_verified", True):
        raise HTTPException(
            status_code=403,
            detail="Please verify your email first. Check your inbox for the verification code.",
        )

    token = create_access_token(sub=str(u.id), role=getattr(u, "role", "user"))
    return TokenResponse(access_token=token)


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", "http://localhost:5173").strip()


def _google_client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "").strip()


@router.get("/google/config")
def google_oauth_config_public():
    """Public OAuth client id for Flutter web (no secret)."""
    cid = _google_client_id()
    web_origin = _frontend_url().rstrip("/")
    return {
        "client_id": cid,
        "enabled": bool(cid),
        "web_origin": web_origin,
        "web_redirect_uri": web_origin,
    }


class GoogleIdTokenBody(BaseModel):
    id_token: str = Field(..., min_length=20)


@router.post("/google/id-token", response_model=TokenResponse)
def google_id_token_login(body: GoogleIdTokenBody, db: Session = Depends(get_db)):
    """Verify Google ID token from Flutter web (legatoappgp2026.web.app origin)."""
    client_id = _google_client_id()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured on the server")

    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        idinfo = google_id_token.verify_oauth2_token(
            body.id_token.strip(),
            google_requests.Request(),
            client_id,
            clock_skew_in_seconds=30,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Google sign-in token")

    email = (idinfo.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash="",
            role="user",
            email_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(sub=str(user.id), role=getattr(user, "role", "user"))
    return TokenResponse(access_token=token, token_type="bearer")


def _google_redirect_uri() -> str:
    """OAuth callback on the public API host (Google rejects raw IP redirect URIs)."""
    explicit = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    for key in ("IMAGE_BASE_URL", "PUBLIC_API_URL", "API_PUBLIC_URL"):
        base = (os.getenv(key) or "").strip().rstrip("/")
        if base:
            return f"{base}/auth/google/callback"
    return "http://127.0.0.1:8000/auth/google/callback"


def _google_oauth_config():
    return (
        os.getenv("GOOGLE_CLIENT_ID", "").strip(),
        os.getenv("GOOGLE_CLIENT_SECRET", "").strip(),
        _google_redirect_uri(),
    )


@router.get("/google")
def google_oauth_redirect():
    """Google OAuth - redirect to Google sign-in."""
    from fastapi.responses import RedirectResponse
    client_id, _, redirect_uri = _google_oauth_config()
    if not client_id:
        return RedirectResponse(
            url=f"{_frontend_url().rstrip('/')}/?error=google_not_configured",
            status_code=302,
        )
    qs = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"
    return RedirectResponse(url=auth_url, status_code=302)


class GoogleCodeBody(BaseModel):
    code: str = Field(..., min_length=5)
    redirect_uri: str = Field(..., min_length=8)


async def _google_oauth_login_from_code(code: str, redirect_uri: str, db: Session) -> str:
    """Exchange Google auth code for app JWT. Raises HTTPException on failure."""
    client_id, client_secret, _ = _google_oauth_config()
    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured on the server")

    import httpx

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Google authentication failed")

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Google authentication failed")

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if user_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Could not retrieve Google account details")

    user_info = user_resp.json()
    email = (user_info.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            password_hash="",
            role="user",
            email_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return create_access_token(sub=str(user.id), role=getattr(user, "role", "user"))


@router.post("/google/code", response_model=TokenResponse)
async def google_oauth_code_exchange(body: GoogleCodeBody, db: Session = Depends(get_db)):
    """Exchange Google OAuth code from Flutter web (redirect URI = Firebase hosting)."""
    token = await _google_oauth_login_from_code(body.code.strip(), body.redirect_uri.strip(), db)
    return TokenResponse(access_token=token, token_type="bearer")


@router.get("/google/callback")
async def google_oauth_callback(code: str = "", error: str = "", db: Session = Depends(get_db)):
    """Google OAuth callback - exchange code for token, create/get user, redirect to frontend with JWT."""
    from fastapi.responses import RedirectResponse

    frontend_url = _frontend_url()
    _, _, redirect_uri = _google_oauth_config()

    if error or not code:
        return RedirectResponse(
            url=f"{frontend_url.rstrip('/')}/?error=google_denied",
            status_code=302,
        )

    try:
        token = await _google_oauth_login_from_code(code, redirect_uri, db)
    except HTTPException as exc:
        err = "google_token_failed"
        if exc.status_code == 503:
            err = "google_not_configured"
        elif "email" in (exc.detail or "").lower():
            err = "google_no_email"
        elif "retrieve" in (exc.detail or "").lower():
            err = "google_user_failed"
        return RedirectResponse(
            url=f"{frontend_url.rstrip('/')}/?error={err}",
            status_code=302,
        )

    return RedirectResponse(
        url=f"{frontend_url.rstrip('/')}/?token={token}",
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
@limiter.limit("5/minute")
def forgot_password(request: Request, payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset. Sends a verification code by email. Same response to avoid enumeration."""
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return {"status": "ok", "message": "If an account exists, you will receive a reset link."}

    code = _generate_verification_code()
    code_hash_val = _code_hash(code)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)

    _ensure_verification_codes_table(db)
    try:
        db.execute(
            text("""
                DELETE FROM verification_codes WHERE email = :email AND purpose = 'password_reset'
            """),
            {"email": email},
        )
        db.execute(
            text("""
                INSERT INTO verification_codes (email, user_id, code_hash, purpose, expires_at)
                VALUES (:email, :uid, :ch, 'password_reset', :exp)
            """),
            {"email": email, "uid": user.id, "ch": code_hash_val, "exp": expires_at.isoformat()},
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create verification code")

    _send_verification_email(email, code, "password_reset")
    return {"status": "ok", "message": "If an account exists, you will receive a reset link."}


@router.post("/verify-reset-code", response_model=VerifyResetCodeResponse)
def verify_reset_code(payload: VerifyResetCodeRequest, db: Session = Depends(get_db)):
    """Verify the code sent by email and return a reset_token for /reset-password."""
    email = payload.email.lower().strip()
    code = (payload.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")

    code_hash_val = _code_hash(code.lower())
    now = datetime.now(timezone.utc).isoformat()

    try:
        row = db.execute(
            text("""
                SELECT user_id FROM verification_codes
                WHERE email = :email AND code_hash = :ch AND purpose = 'password_reset' AND expires_at > :now
            """),
            {"email": email, "ch": code_hash_val, "now": now},
        ).fetchone()
    except Exception:
        raise HTTPException(status_code=500, detail="Verification code table not available")

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")

    user_id = row[0]
    token = secrets.token_urlsafe(32)
    token_hash_val = _token_hash(token)
    expires = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_AFTER_CODE_EXPIRE_MINUTES)

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
        db.execute(
            text("DELETE FROM verification_codes WHERE email = :email AND purpose = 'password_reset'"),
            {"email": email},
        )
        db.execute(
            text("DELETE FROM password_reset_tokens WHERE user_id = :uid"),
            {"uid": user_id},
        )
        db.execute(
            text("""
                INSERT INTO password_reset_tokens (user_id, token_hash, expires_at)
                VALUES (:uid, :th, :exp)
            """),
            {"uid": user_id, "th": token_hash_val, "exp": expires.isoformat()},
        )
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create reset token")

    return VerifyResetCodeResponse(reset_token=token)


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
