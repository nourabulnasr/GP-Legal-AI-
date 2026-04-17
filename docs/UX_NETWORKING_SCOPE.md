# Professional networking UI (LexConnect-style) — scope note

This document tracks the **social / networking** surface **separately** from fine-tuning or LFM training work.

## Product intent

- **Feed**: legal professional posts, filters, likes, comments, shares (pagination).
- **My Network**: stats, invitations, suggested members, connect flow.
- **Profile**: cover/avatar, headline, stats, skills, about, experience, education, links to skills/documents/recommendations, and entry to the **12 Legato tools** grid.
- **Contracts** tab: shortcuts to analyze, history, and full tools (does not duplicate tool implementations).
- **Alerts**: placeholder until notification APIs exist.

## Backend

- Routes under **`/api/*`** (`app/routers/social.py`): posts, profile merge (`PUT /api/profile/me`), network invites, endorsements, documents, recommendations.
- Tables: `social_posts`, `social_post_likes`, `social_post_comments`, `social_post_shares`, `network_invites`, `skill_endorsements`, `profile_recommendations`, `profile_user_documents`; profile display fields remain in **`legato_profiles.payload_json`** (same DB as the rest of Legato).

## Mobile

- Bottom nav: **Feed | Network | Contracts | Alerts | Profile** (`legato_mobile/lib/screens/home/home_shell.dart`).
- Implementation files live under `lib/screens/social/`.

## Relation to fine-tuning

Fine-tune / Colab / LFM weights are **orthogonal**: networking uses the same auth and DB, not the ML artifact pipeline.
