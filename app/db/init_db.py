from __future__ import annotations

from sqlalchemy import text
from app.db.session import engine
from app.db.session import Base  # noqa
from app.db import models  # noqa


def init_db() -> None:
    """
    Creates tables if missing, and applies ultra-light schema tweaks safely.
    We do NOT drop anything.
    """
    # Create tables if not exist (won't override existing)
    Base.metadata.create_all(bind=engine)

    # Safe schema tweak: add role column if missing (SQLite supports ADD COLUMN)
    # If it already exists, ignore.
    try:
        with engine.begin() as conn:
            # Check if role column exists
            cols = conn.execute(text("PRAGMA table_info(users);")).fetchall()
            col_names = {c[1] for c in cols}  # (cid, name, type,...)
            if "role" not in col_names:
                conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user';"))
            if "email_verified" not in col_names:
                conn.execute(text("ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 1;"))
            # Seed admin account: always treat as verified so it can log in without email verification
            conn.execute(
                text("UPDATE users SET email_verified = 1 WHERE email = 'admin@test.com';")
            )
    except Exception:
        pass

    # Safe schema tweaks: add optional analyses metadata columns if missing
    # (SQLite supports ADD COLUMN; ignore if already exists)
    try:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(analyses);"))
            col_names = {c[1] for c in cols.fetchall()}

            if "mime_type" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN mime_type VARCHAR;"))
            if "sha256" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN sha256 VARCHAR;"))
            if "page_count" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN page_count INTEGER;"))
            if "ocr_used" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN ocr_used INTEGER;"))
            if "detected_lang" not in col_names:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN detected_lang VARCHAR;"))
    except Exception:
        pass
