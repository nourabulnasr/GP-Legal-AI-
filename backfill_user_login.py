#!/usr/bin/env python3
"""
Ensure a user can log in: set email (lowercase), password hash, email_verified.
Uses DATABASE_URL from env (Docker: sqlite:////data/legalai.db).

Usage (host, project root):
  docker exec legalai-backend python backfill_user_login.py Aly@test.com 123456789

  # Local:
  set DATABASE_URL=sqlite:///./legalai.db
  python backfill_user_login.py AlyTEST.com 123456789

  # Create or update user and set role=admin (for testing admin APIs / Law RAG tab):
  python backfill_user_login.py admin@test.com YourPassword admin
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure app imports resolve when run from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env before app.db.session reads DATABASE_URL
try:
    from dotenv import load_dotenv

    for p in (Path(ROOT) / ".env", Path(os.getcwd()) / ".env"):
        if p.is_file():
            load_dotenv(p, override=False)
except Exception:
    pass

from app.core.security import hash_password
from app.db.models import User
from app.db.session import SessionLocal


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: backfill_user_login.py <email> <password> [admin]",
            file=sys.stderr,
        )
        return 1
    raw_email = sys.argv[1].strip()
    password = sys.argv[2]
    want_admin = len(sys.argv) > 3 and sys.argv[3].strip().lower() == "admin"
    email = raw_email.lower().strip()
    if not email or not password:
        print("Email and password required.", file=sys.stderr)
        return 1

    db_url = os.getenv("DATABASE_URL", "")
    print("DATABASE_URL:", db_url or "(default from app config)")

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        ph = hash_password(password)
        role = "admin" if want_admin else "user"
        if u:
            u.password_hash = ph
            u.email_verified = True
            if want_admin:
                u.role = "admin"
            db.add(u)
            print(
                f"Updated user id={u.id} email={u.email} email_verified=True role={getattr(u, 'role', 'user')}"
            )
        else:
            u = User(email=email, password_hash=ph, role=role, email_verified=True)
            db.add(u)
            print(f"Created user email={email} role={role} email_verified=True")
        db.commit()
        db.refresh(u)
        print(f"OK: you can log in with {email!r} / <password>")
        return 0
    except Exception as e:
        db.rollback()
        print("Error:", e, file=sys.stderr)
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
