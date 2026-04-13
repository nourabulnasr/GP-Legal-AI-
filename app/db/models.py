from __future__ import annotations

from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user", nullable=False)
    email_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="user")


class AnalysisShare(Base):
    """Read-only share link for an analysis (token in URL)."""

    __tablename__ = "analysis_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    owner_user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DealThread(Base):
    """Party-to-party thread around a contract/analysis."""

    __tablename__ = "deal_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_by_user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DealMessage(Base):
    __tablename__ = "deal_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    thread_id: Mapped[int] = mapped_column(Integer, ForeignKey("deal_threads.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TimelineEvent(Base):
    """Manual or extracted milestone for an analysis (admin/owner visibility)."""

    __tablename__ = "timeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    event_date: Mapped[str] = mapped_column(String(32), nullable=False)  # ISO date string
    source: Mapped[str] = mapped_column(String(32), default="manual", nullable=False)  # manual | extracted
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LegalProfile(Base):
    """Lightweight “legal network” profile (MVP)."""

    __tablename__ = "legal_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), unique=True, index=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    headline: Mapped[str | None] = mapped_column(String(512), nullable=True)
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SignatureRecord(Base):
    """In-app signature capture metadata (not a third-party e-sign provider)."""

    __tablename__ = "signature_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    signer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    consent_version: Mapped[str] = mapped_column(String(64), default="legato-v1", nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)  # optional image preview hash, user agent, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional document metadata (helps defense + admin queries; safe to be null)
    mime_type: Mapped[str | None] = mapped_column(String, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ocr_used: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0/1
    detected_lang: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="analyses")
