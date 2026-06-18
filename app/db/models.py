from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Boolean, Integer, String, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="user", nullable=False)
    user_type: Mapped[str] = mapped_column(String, default="user", nullable=False)  # "user" or "lawyer"
    email_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="user")


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
    contract_category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    lawyer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship("User", back_populates="analyses")


class LawyerApplication(Base):
    __tablename__ = "lawyer_applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), unique=True, index=True, nullable=False)
    bar_license_number: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    document_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    document_mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    document_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    cv_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    cv_mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cv_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    id_card_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    id_card_mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    id_card_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    id_card_back_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    id_card_back_mime_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    id_card_back_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    years_of_experience: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hourly_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    negotiated_hourly_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    admin_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class LegatoShare(Base):
    __tablename__ = "legato_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class LegatoDealThread(Base):
    __tablename__ = "legato_deal_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=True)
    contract_category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class LegatoDealMessage(Base):
    __tablename__ = "legato_deal_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    thread_id: Mapped[int] = mapped_column(Integer, ForeignKey("legato_deal_threads.id"), index=True, nullable=False)
    author_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class LegatoDealThreadMember(Base):
    __tablename__ = "legato_deal_thread_members"

    thread_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("legato_deal_threads.id"), primary_key=True, index=True
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True, index=True)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class LegatoTimelineEvent(Base):
    __tablename__ = "legato_timeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    event_date: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class LegatoProfile(Base):
    __tablename__ = "legato_profiles"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")  # JSON object as string
    avatar_mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)


class LegatoSignature(Base):
    __tablename__ = "legato_signatures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    signer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    consent_acknowledged: Mapped[bool] = mapped_column(default=False, nullable=False)
    signature_png_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# --- Professional networking (feed, network, profile extensions) ---


class SocialPost(Base):
    __tablename__ = "social_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="All Updates", index=True)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)
    image_mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class SocialPostLike(Base):
    __tablename__ = "social_post_likes"
    __table_args__ = (UniqueConstraint("post_id", "user_id", name="uq_social_post_like_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    post_id: Mapped[int] = mapped_column(Integer, ForeignKey("social_posts.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SocialPostComment(Base):
    __tablename__ = "social_post_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    post_id: Mapped[int] = mapped_column(Integer, ForeignKey("social_posts.id"), index=True, nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class SocialPostShare(Base):
    __tablename__ = "social_post_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    post_id: Mapped[int] = mapped_column(Integer, ForeignKey("social_posts.id"), index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class PostServiceProposal(Base):
    __tablename__ = "post_service_proposals"
    __table_args__ = (UniqueConstraint("post_id", "lawyer_id", name="uq_post_service_proposal_lawyer"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    post_id: Mapped[int] = mapped_column(Integer, ForeignKey("social_posts.id"), index=True, nullable=False)
    lawyer_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    hourly_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    counter_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class UserNotification(Base):
    __tablename__ = "user_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    recipient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    actor_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    post_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("social_posts.id"), index=True, nullable=True)
    reference_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class ConsultationRequest(Base):
    __tablename__ = "consultation_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    requester_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    lawyer_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    hourly_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    payment_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    paymob_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    paymob_intention_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    paymob_transaction_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user_conversations.id"), index=True, nullable=True
    )
    session_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    session_ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class NetworkInvite(Base):
    __tablename__ = "network_invites"
    __table_args__ = (UniqueConstraint("requester_id", "addressee_id", name="uq_network_invite_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    requester_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    addressee_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SkillEndorsement(Base):
    __tablename__ = "skill_endorsements"
    __table_args__ = (UniqueConstraint("endorser_id", "recipient_id", "skill", name="uq_skill_endorse_triple"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    endorser_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    recipient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    skill: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProfileRecommendation(Base):
    __tablename__ = "profile_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    recipient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ProfileUserDocument(Base):
    __tablename__ = "profile_user_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    file_url: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_bytes: Mapped[Optional[bytes]] = mapped_column(nullable=True)  # SQLite BLOB
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# --- User messaging (private + group, connected users only) ---


class UserConversation(Base):
    __tablename__ = "user_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="direct", index=True)
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class UserConversationMember(Base):
    __tablename__ = "user_conversation_members"
    __table_args__ = (UniqueConstraint("conversation_id", "user_id", name="uq_conv_member"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_conversations.id"), index=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_read_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class UserMessage(Base):
    __tablename__ = "user_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_conversations.id"), index=True, nullable=False
    )
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    msg_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text", server_default="text")
    offer_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    offer_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
