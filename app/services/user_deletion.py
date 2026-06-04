from __future__ import annotations

from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from app.db.models import (
    Analysis,
    LegatoDealMessage,
    LegatoDealThread,
    LegatoDealThreadMember,
    LegatoProfile,
    LegatoShare,
    LegatoSignature,
    LegatoTimelineEvent,
    NetworkInvite,
    ProfileRecommendation,
    ProfileUserDocument,
    SkillEndorsement,
    SocialPost,
    SocialPostComment,
    SocialPostLike,
    SocialPostShare,
    User,
    UserConversation,
    UserConversationMember,
    UserMessage,
    UserNotification,
)


def delete_user_account(db: Session, user_id: int) -> User:
    """Remove a user and related rows that reference them."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise LookupError("User not found")

    email = (user.email or "").lower().strip()
    analysis_ids = [
        row[0]
        for row in db.query(Analysis.id).filter(Analysis.user_id == user_id).all()
    ]

    thread_filters = [LegatoDealThread.user_id == user_id]
    if analysis_ids:
        thread_filters.append(LegatoDealThread.analysis_id.in_(analysis_ids))
    thread_ids = [
        row[0]
        for row in db.query(LegatoDealThread.id).filter(or_(*thread_filters)).all()
    ]
    if thread_ids:
        db.query(LegatoDealMessage).filter(
            or_(
                LegatoDealMessage.thread_id.in_(thread_ids),
                LegatoDealMessage.author_id == user_id,
            )
        ).delete(synchronize_session=False)
        db.query(LegatoDealThreadMember).filter(
            LegatoDealThreadMember.thread_id.in_(thread_ids)
        ).delete(synchronize_session=False)
        db.query(LegatoDealThread).filter(LegatoDealThread.id.in_(thread_ids)).delete(
            synchronize_session=False
        )
    db.query(LegatoDealThreadMember).filter(LegatoDealThreadMember.user_id == user_id).delete(
        synchronize_session=False
    )

    if analysis_ids:
        db.query(LegatoShare).filter(LegatoShare.analysis_id.in_(analysis_ids)).delete(
            synchronize_session=False
        )
        db.query(LegatoTimelineEvent).filter(
            LegatoTimelineEvent.analysis_id.in_(analysis_ids)
        ).delete(synchronize_session=False)
        db.query(LegatoSignature).filter(LegatoSignature.analysis_id.in_(analysis_ids)).delete(
            synchronize_session=False
        )
        db.query(Analysis).filter(Analysis.id.in_(analysis_ids)).delete(synchronize_session=False)

    db.query(LegatoShare).filter(LegatoShare.user_id == user_id).delete(synchronize_session=False)
    db.query(LegatoTimelineEvent).filter(LegatoTimelineEvent.user_id == user_id).delete(
        synchronize_session=False
    )
    db.query(LegatoSignature).filter(LegatoSignature.user_id == user_id).delete(
        synchronize_session=False
    )

    post_ids = [
        row[0] for row in db.query(SocialPost.id).filter(SocialPost.author_id == user_id).all()
    ]
    if post_ids:
        db.query(SocialPostLike).filter(SocialPostLike.post_id.in_(post_ids)).delete(
            synchronize_session=False
        )
        db.query(SocialPostComment).filter(SocialPostComment.post_id.in_(post_ids)).delete(
            synchronize_session=False
        )
        db.query(SocialPostShare).filter(SocialPostShare.post_id.in_(post_ids)).delete(
            synchronize_session=False
        )
        db.query(SocialPost).filter(SocialPost.id.in_(post_ids)).delete(synchronize_session=False)

    db.query(SocialPostLike).filter(SocialPostLike.user_id == user_id).delete(
        synchronize_session=False
    )
    db.query(SocialPostComment).filter(SocialPostComment.author_id == user_id).delete(
        synchronize_session=False
    )
    db.query(SocialPostShare).filter(SocialPostShare.user_id == user_id).delete(
        synchronize_session=False
    )
    db.query(UserNotification).filter(
        or_(UserNotification.recipient_id == user_id, UserNotification.actor_id == user_id)
    ).delete(synchronize_session=False)
    db.query(NetworkInvite).filter(
        or_(NetworkInvite.requester_id == user_id, NetworkInvite.addressee_id == user_id)
    ).delete(synchronize_session=False)
    db.query(SkillEndorsement).filter(
        or_(SkillEndorsement.endorser_id == user_id, SkillEndorsement.recipient_id == user_id)
    ).delete(synchronize_session=False)
    db.query(ProfileRecommendation).filter(
        or_(
            ProfileRecommendation.author_id == user_id,
            ProfileRecommendation.recipient_id == user_id,
        )
    ).delete(synchronize_session=False)
    db.query(ProfileUserDocument).filter(ProfileUserDocument.user_id == user_id).delete(
        synchronize_session=False
    )
    db.query(LegatoProfile).filter(LegatoProfile.user_id == user_id).delete(
        synchronize_session=False
    )

    conv_ids = [
        row[0]
        for row in db.query(UserConversationMember.conversation_id)
        .filter(UserConversationMember.user_id == user_id)
        .all()
    ]
    if conv_ids:
        db.query(UserMessage).filter(UserMessage.conversation_id.in_(conv_ids)).delete(
            synchronize_session=False
        )
        db.query(UserConversationMember).filter(UserConversationMember.user_id == user_id).delete(
            synchronize_session=False
        )
        for cid in conv_ids:
            remaining = (
                db.query(UserConversationMember)
                .filter(UserConversationMember.conversation_id == cid)
                .count()
            )
            if remaining == 0:
                db.query(UserConversation).filter(UserConversation.id == cid).delete(
                    synchronize_session=False
                )
        db.query(UserConversation).filter(UserConversation.created_by == user_id).delete(
            synchronize_session=False
        )

    try:
        db.execute(
            text(
                "DELETE FROM verification_codes WHERE user_id = :uid OR lower(email) = :email"
            ),
            {"uid": user_id, "email": email},
        )
        db.execute(
            text("DELETE FROM password_reset_tokens WHERE user_id = :uid"),
            {"uid": user_id},
        )
    except Exception:
        pass

    db.delete(user)
    db.commit()
    return user
