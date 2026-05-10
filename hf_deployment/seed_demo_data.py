"""
seed_demo_data.py — Legato HuggingFace Spaces demo data seeder.

Usage (from any directory):
    python hf_deployment/seed_demo_data.py
    python hf_deployment/seed_demo_data.py --db /path/to/legalai.db

What it does:
  1. Creates admin user  (admin@legato.com)
  2. Creates demo user   (demo@legato.com)
  3. Creates 1 sample Analysis for the demo user (realistic contract result)
  4. Creates 2 social feed posts by the admin user
  5. Creates 1 deal thread linked to the demo analysis
  All operations are idempotent: safe to re-run without duplicating data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root (parent of hf_deployment/) is importable so that
# `app.*` packages resolve correctly when running from hf_deployment/.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy env bootstrap: load .env from project root before importing app modules
# (mirrors what app/main.py does at startup).
# ---------------------------------------------------------------------------
_env_path = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv  # type: ignore

    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except Exception:
    pass  # dotenv is optional; DATABASE_URL can be set via CLI arg instead

# ---------------------------------------------------------------------------
# App imports (after sys.path and .env are set up)
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.db.models import (
    Analysis,
    LegatoDealThread,
    SocialPost,
    User,
)
from app.core.security import hash_password


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)


def _resolve_db_url(cli_db_path: str | None) -> str:
    """
    Priority:
      1. --db CLI argument (raw file path → converted to sqlite:/// URL)
      2. DATABASE_URL environment variable
      3. Default: legalai.db in project root
    """
    import os

    if cli_db_path:
        p = Path(cli_db_path).resolve()
        return f"sqlite:///{p.as_posix()}"

    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url

    default_path = PROJECT_ROOT / "legalai.db"
    return f"sqlite:///{default_path.as_posix()}"


# ---------------------------------------------------------------------------
# Demo content
# ---------------------------------------------------------------------------

SAMPLE_RESULT_JSON: dict = {
    "rule_hits": [
        {
            "rule_id": "LABOR25_EMPLOYER_PLACEHOLDER",
            "severity": "error",
            "description": "بيانات صاحب العمل ما زالت placeholders",
            "article": "1",
            "law": "قانون العمل رقم 14 لسنة 2025",
        },
        {
            "rule_id": "LABOR25_ANNUAL_LEAVE",
            "severity": "error",
            "description": "مدة الإجازة السنوية أقل من الحد الأدنى 15 يوم",
            "article": "47",
            "law": "قانون العمل رقم 14 لسنة 2025",
        },
        {
            "rule_id": "LABOR25_EMPLOYER_INFO",
            "severity": "info",
            "description": "بيانات صاحب العمل موجودة",
            "article": "1",
        },
    ],
    "needs_review": True,
    "labor_summary": {
        "status": "violations_detected",
        "ml_violations": 2,
        "source": "rule_engine",
    },
    "ml_predictions": [
        {
            "rule_id": "LABOR25_ANNUAL_LEAVE",
            "score": 0.87,
            "passed_threshold": True,
        }
    ],
    "full_text_unified_risk": 0.78,
    "law_scope_used": ["labor"],
}

SOCIAL_POSTS = [
    {
        "content": (
            "تم إطلاق Legato — منصة الذكاء الاصطناعي القانوني لتحليل عقود العمل المصرية "
            "وفقًا لقانون العمل رقم 14 لسنة 2025. 🚀⚖️\n\n"
            "Legato is now live! AI-powered employment contract analysis under Egyptian Labor Law 14/2025. "
            "Upload your contract and get instant compliance insights."
        ),
        "tags_json": json.dumps(["launch", "legaltech", "egypt", "laborlaw"]),
        "category": "All Updates",
    },
    {
        "content": (
            "هل تعرف حقوقك في عقد العمل؟ 🔍\n"
            "قانون العمل المصري رقم 14 لسنة 2025 يلزم كل عقد بذكر:\n"
            "• بيانات صاحب العمل كاملة\n"
            "• الأجر وطريقة الصرف\n"
            "• مدة الإجازة السنوية (15 يوم حد أدنى)\n"
            "• فترة الاختبار (3 أشهر حد أقصى)\n\n"
            "Do you know your employment contract rights? "
            "Egyptian Labor Law 14/2025 mandates these key clauses in every contract."
        ),
        "tags_json": json.dumps(["laborlaw", "rights", "contracts", "egypt"]),
        "category": "Legal Updates",
    },
]


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

def seed_users(db: Session) -> tuple[User, User]:
    """Create admin and demo users. Returns (admin_user, demo_user)."""

    admin_email = "admin@legato.com"
    demo_email = "demo@legato.com"

    # --- Admin ---
    admin = db.query(User).filter(User.email == admin_email).first()
    if admin is None:
        admin = User(
            email=admin_email,
            password_hash=hash_password("LegatoAdmin2026!"),
            role="admin",
            email_verified=True,
            created_at=datetime.utcnow(),
        )
        db.add(admin)
        db.flush()  # assign id without committing
        print(f"  [+] Created admin user: {admin_email}")
    else:
        print(f"  [~] Admin user already exists: {admin_email} (id={admin.id})")

    # --- Demo user ---
    demo = db.query(User).filter(User.email == demo_email).first()
    if demo is None:
        demo = User(
            email=demo_email,
            password_hash=hash_password("LegatoDemo2026!"),
            role="user",
            email_verified=True,
            created_at=datetime.utcnow(),
        )
        db.add(demo)
        db.flush()
        print(f"  [+] Created demo user: {demo_email}")
    else:
        print(f"  [~] Demo user already exists: {demo_email} (id={demo.id})")

    db.commit()
    db.refresh(admin)
    db.refresh(demo)
    return admin, demo


def seed_analysis(db: Session, demo_user: User) -> Analysis:
    """Create 1 sample Analysis for the demo user. Idempotent (checks filename)."""

    existing = (
        db.query(Analysis)
        .filter(
            Analysis.user_id == demo_user.id,
            Analysis.filename == "sample_employment_contract.pdf",
        )
        .first()
    )

    if existing is not None:
        print(f"  [~] Sample analysis already exists (id={existing.id})")
        return existing

    analysis = Analysis(
        user_id=demo_user.id,
        filename="sample_employment_contract.pdf",
        result_json=json.dumps(SAMPLE_RESULT_JSON, ensure_ascii=False),
        mime_type="application/pdf",
        sha256="a3f1c2d4e5b6789012345678abcdef1234567890abcdef1234567890abcdef12",
        page_count=3,
        ocr_used=0,
        detected_lang="ar",
        needs_review=True,
        lawyer_note=(
            "عقد نموذجي للعرض التجريبي. يحتوي على مخالفتين رئيسيتين: "
            "بيانات صاحب العمل غير مكتملة، وإجازة سنوية أقل من الحد القانوني. "
            "| Demo contract with 2 violations: incomplete employer info and insufficient annual leave."
        ),
        created_at=datetime.utcnow(),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    print(f"  [+] Created sample analysis (id={analysis.id}) for demo user")
    return analysis


def seed_social_posts(db: Session, admin_user: User) -> list[SocialPost]:
    """Create 2 social posts by the admin user. Idempotent (checks content prefix)."""

    created: list[SocialPost] = []
    for post_data in SOCIAL_POSTS:
        content_prefix = post_data["content"][:60]
        existing = (
            db.query(SocialPost)
            .filter(
                SocialPost.author_id == admin_user.id,
                SocialPost.content.startswith(content_prefix),
            )
            .first()
        )
        if existing is not None:
            print(f"  [~] Social post already exists (id={existing.id})")
            created.append(existing)
            continue

        post = SocialPost(
            author_id=admin_user.id,
            content=post_data["content"],
            tags_json=post_data["tags_json"],
            category=post_data["category"],
            created_at=datetime.utcnow(),
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        snippet = post.content[:60].encode("ascii", "replace").decode("ascii")
        print(f"  [+] Created social post (id={post.id}): {snippet}...")
        created.append(post)

    return created


def seed_deal_thread(db: Session, demo_user: User, analysis: Analysis) -> LegatoDealThread:
    """Create 1 deal thread linked to the demo analysis. Idempotent."""

    existing = (
        db.query(LegatoDealThread)
        .filter(
            LegatoDealThread.analysis_id == analysis.id,
            LegatoDealThread.user_id == demo_user.id,
        )
        .first()
    )

    if existing is not None:
        print(f"  [~] Deal thread already exists (id={existing.id})")
        return existing

    thread = LegatoDealThread(
        analysis_id=analysis.id,
        user_id=demo_user.id,
        title="مراجعة عقد العمل النموذجي — Demo Contract Review",
        created_at=datetime.utcnow(),
    )
    db.add(thread)
    db.commit()
    db.refresh(thread)
    print(f"  [+] Created deal thread (id={thread.id}) linked to analysis (id={analysis.id})")
    return thread


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Legato demo data into legalai.db for HuggingFace Spaces deployment."
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=(
            "Path to legalai.db (e.g. ./legalai.db). "
            "Defaults to DATABASE_URL env var, then project-root/legalai.db."
        ),
    )
    args = parser.parse_args()

    db_url = _resolve_db_url(args.db)
    print(f"\n[seed_demo_data] Connecting to: {db_url}")

    engine = _get_engine(db_url)

    # Ensure all tables exist (safe on existing DB — does not drop data)
    from app.db.models import User as _U  # noqa: F401 — triggers Base metadata population
    from app.db.session import Base

    Base.metadata.create_all(bind=engine)
    print("[seed_demo_data] Schema verified (create_all is idempotent).\n")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db: Session = SessionLocal()

    try:
        print("--- Users ---")
        admin_user, demo_user = seed_users(db)

        print("\n--- Analysis ---")
        analysis = seed_analysis(db, demo_user)

        print("\n--- Social Posts ---")
        seed_social_posts(db, admin_user)

        print("\n--- Deal Thread ---")
        seed_deal_thread(db, demo_user, analysis)

        print("\n" + "=" * 60)
        print("  Legato demo data seeded successfully.")
        print("=" * 60)
        print(f"  Admin:     admin@legato.com  /  LegatoAdmin2026!")
        print(f"  Demo user: demo@legato.com   /  LegatoDemo2026!")
        print(f"  Analysis id: {analysis.id} (needs_review=True, 2 violations)")
        print("=" * 60 + "\n")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Seeding failed: {str(exc).encode('ascii', 'replace').decode('ascii')}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
