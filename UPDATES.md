# Updates — Changes vs. `final_80%` Branch

This document records every change made to this codebase relative to the original
`final_80%` branch at [`nourabulnasr/GP-Legal-AI-`](https://github.com/nourabulnasr/GP-Legal-AI-).

---

## Table of Contents

1. [New Features Overview](#new-features-overview)
2. [New Files Added](#new-files-added)
3. [Backend Changes](#backend-changes)
   - [Database Models](#database-models)
   - [Database Migrations](#database-migrations)
   - [messaging.py](#approutersmessagingpy)
   - [social.py](#approuterssocialpy)
   - [notifications.py](#approutersnotificationspy)
   - [analyses.py](#approutersanalysespy)
   - [legato_mobile.py](#approuterslegato_mobilepy)
4. [Flutter Changes](#flutter-changes)
   - [Verified Lawyer Badge — All Locations](#verified-lawyer-badge--all-locations)
   - [Lawyer Offer Feature](#lawyer-offer-feature)
   - [Lawyer Application Screen](#lawyer-application-screen)
   - [Admin Screen](#admin-screen)
5. [API Reference — New Endpoints](#api-reference--new-endpoints)

---

## New Features Overview

| # | Feature | Scope |
|---|---------|-------|
| 1 | **Verified Lawyer Badge** — `Icons.verified` (LinkedIn blue `#0A66C2`) shown next to every verified lawyer's name across the entire app | Full-stack |
| 2 | **Lawyer Application System** — Users can apply to be verified lawyers; admins review/approve/reject; users can edit rejected or unsubmitted applications | Full-stack |
| 3 | **Lawyer Offer in DMs** — Verified lawyers can send structured service offers inside direct messages; recipients see a Pay button; payment intent is recorded | Full-stack |
| 4 | **Offer Payment Flow** — Offer card shows service title, description, price, and a "Pay" / "Payment confirmed" state | Full-stack |
| 5 | **Admin User Management** — Admin screen now correctly shows `user_type` and `lawyer_status` for every user | Backend |

---

## New Files Added

These files **did not exist** in the original `final_80%` branch and were created from scratch.

### `app/routers/lawyer.py`
FastAPI router (`/lawyer`) for the lawyer application lifecycle:
- `GET /lawyer/status` — returns the current user's application status, pre-filled form data (bar licence number, years of experience, filenames), so the Flutter screen can pre-populate fields.
- `POST /lawyer/apply` — submit a new application (multipart: `bar_license_number`, `years_of_experience`, `cv` file, `id_card` file). Allows re-application after rejection.
- `PATCH /lawyer/application` — update a pending or rejected application (any field is optional; files are only re-uploaded if a new file is provided, otherwise the server keeps the existing one).

### `app/routers/admin_lawyers.py`
FastAPI router (`/admin/lawyers`) for admin review:
- `GET /admin/lawyers/applications` — paginated list of all applications with status, file metadata, submission date.
- `GET /admin/lawyers/applications/{id}` — single application detail.
- `GET /admin/lawyers/applications/{id}/cv` — stream the CV file as a download.
- `GET /admin/lawyers/applications/{id}/id-card` — stream the ID card file.
- `PATCH /admin/lawyers/{id}/review` — approve or reject an application (`action: "approve" | "reject"`), with optional `admin_note`.

### `lib/screens/lawyer/lawyer_application_screen.dart`
Flutter screen for the end-user lawyer application flow:
- Shows current status (`not_applied`, `pending`, `approved`, `rejected`) with appropriate messaging.
- Pre-populates all text fields and shows green "Already on server" labels for uploaded files when editing a rejected/existing application.
- File upload fields for CV and ID card (optional re-upload when a server copy already exists).
- Submits via multipart form; handles both create and update paths.
- "View Status" mode for pending/approved states.

---

## Backend Changes

### Database Models

**File:** `app/db/models.py`

#### `UserMessage` — three new columns

| Column | Type | Description |
|--------|------|-------------|
| `msg_type` | `VARCHAR(32)` NOT NULL default `'text'` | Message type — `'text'` or `'lawyer_offer'` |
| `offer_json` | `TEXT` nullable | JSON blob with offer fields when `msg_type = 'lawyer_offer'` |
| `offer_status` | `VARCHAR(20)` nullable | Recipient response — `null` (pending), `'accepted'`, `'rejected'` |

#### `LawyerApplication` — new table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer PK | |
| `user_id` | Integer FK → users | Unique per user |
| `status` | VARCHAR(20) | `'pending'` / `'approved'` / `'rejected'` |
| `bar_license_number` | VARCHAR | |
| `years_of_experience` | Integer | |
| `cv_bytes` | BLOB | Stored file bytes |
| `cv_filename` | VARCHAR | Original filename |
| `id_card_bytes` | BLOB | Stored file bytes |
| `id_card_filename` | VARCHAR | Original filename |
| `submitted_at` | DateTime | |
| `reviewed_at` | DateTime nullable | Set when admin approves/rejects |
| `admin_note` | Text nullable | Admin's review note |

---

### Database Migrations

**File:** `app/db/init_db.py`

All migrations use `PRAGMA table_info()` guard — they only run if the column does not already exist, so they are safe to run on existing databases.

| Table | Column added |
|-------|-------------|
| `user_messages` | `msg_type VARCHAR(32) NOT NULL DEFAULT 'text'` |
| `user_messages` | `offer_json TEXT` |
| `user_messages` | `offer_status VARCHAR(20)` |
| `lawyer_applications` | Entire table created on startup |

---

### `app/routers/messaging.py`

**Original state:** No lawyer-awareness anywhere. `_user_public()` did not return `is_verified_lawyer`. `list_messages` returned only `id`, `author_id`, `author_name`, `email`, `body`, `created_at`, `is_mine`. No offer functionality existed.

#### Changes

**`_user_public()`**
- Added `LawyerApplication` import.
- Now returns `is_verified_lawyer: bool` — single DB query per user, cached by caller as a batch set.

**`list_messages` (`GET /api/messages/conversations/{id}/messages`)**
- Batch-queries `LawyerApplication` for all author IDs in one query (avoids N+1).
- Each message object now includes:
  - `author_is_verified_lawyer: bool`
  - `msg_type: str` (`"text"` or `"lawyer_offer"`)
  - `offer_data: dict | null` (parsed from `offer_json`)
  - `offer_status: str | null`

**`post_message` (`POST /api/messages/conversations/{id}/messages`)**
- Response now includes `author_is_verified_lawyer`, `msg_type: "text"`, `offer_data: null`, `offer_status: null`.

**New: `PostOfferBody` schema**
```python
class PostOfferBody(BaseModel):
    service_title: str   # 1–256 chars
    description:  str   # 1–2000 chars
    price:        float  # > 0
    currency:     str   # default "USD", 1–8 chars
```

**New endpoint: `POST /api/messages/conversations/{id}/offer`**
- Requires caller to be a verified lawyer (403 if not).
- Only works in direct (non-group) conversations (400 if group).
- Creates a `UserMessage` with `msg_type='lawyer_offer'` and `offer_json=<serialized payload>`.
- Returns full message dict including `offer_data` and `offer_status: null`.

**New: `OfferStatusBody` schema**
```python
class OfferStatusBody(BaseModel):
    status: str  # must be "accepted" or "rejected"
```

**New endpoint: `PATCH /api/messages/{message_id}/offer-status`**
- Caller must be a member of the conversation that contains the message.
- Caller must NOT be the message author (403 — cannot accept your own offer).
- Updates `UserMessage.offer_status` and returns `{id, offer_status}`.

---

### `app/routers/social.py`

**Original state:** No `is_verified_lawyer` field anywhere in the social API.

#### Changes

**`_serialize_post()` (internal helper)**
- Now includes `author_is_verified_lawyer: bool` in the returned dict.

**`list_posts` (`GET /posts`)**
- Batch-queries `LawyerApplication` for all post author IDs in one query.
- Each post now includes `author_is_verified_lawyer`.

**`GET /posts/{post_id}/comments`**
- Each comment now includes `author_is_verified_lawyer: bool`.

**`GET /network/search`**
- Each result now includes `is_verified_lawyer: bool`.

**`GET /network/suggestions`**
- Each suggestion now includes `is_verified_lawyer: bool`.

**`GET /network/connections`**
- Each connection now includes `is_verified_lawyer: bool`.

**`GET /network/invites`** (pending invites)
- Each invite now includes `requester_is_verified_lawyer: bool`.

**`GET /profile/{user_id}`**
- Response now includes `is_verified_lawyer: bool` and `user_type: str`.

**`GET /profile/{user_id}/endorsements`**
- Each endorsement item now includes `endorser_is_verified_lawyer: bool` via batch query.

**`GET /profile/{user_id}/recommendations`**
- Each recommendation item now includes `author_is_verified_lawyer: bool`.

> **Performance note:** All verified-lawyer lookups use a single batch query per endpoint (`LawyerApplication.user_id.in_(id_list)`) rather than one query per user, avoiding N+1 database calls.

---

### `app/routers/notifications.py`

**Original state:** `_serialize()` returned `id`, `type`, `message`, `post_id`, `actor_id`, `actor_name`, `actor_avatar_url`, `read`, `created_at`.

#### Changes

**`_serialize()`**
- Added `LawyerApplication` import.
- Now returns `actor_is_verified_lawyer: bool` — per-notification DB query.

---

### `app/routers/analyses.py`

**Original state:** `admin_list_users` returned `[{id, email, role}]` — no user type or lawyer status.

#### Changes

**`admin_list_users` (`GET /analyses/admin/users`)**
- Batch-queries `LawyerApplication` for all returned user IDs.
- Each user dict now includes:
  - `user_type: str` (from `User.user_type`, default `"user"`)
  - `lawyer_status: str` (`"pending"` / `"approved"` / `"rejected"` / `""`)
  - `is_verified_lawyer: bool`

---

### `app/routers/legato_mobile.py`

**Original state:** Deal thread member payloads and deal message lists had no lawyer verification data.

#### Changes

**`_deal_thread_member_payload()` (internal helper)**
- Batch-queries `LawyerApplication` for all member user IDs.
- Each member dict now includes `is_verified_lawyer: bool`.

**`deal_messages_list` (`GET /legato/deal-threads/{id}/messages`)**
- Batch-queries `LawyerApplication` for all message author IDs.
- Each message now includes `author_is_verified_lawyer: bool`.

**`deal_message_post` (`POST /legato/deal-threads/{id}/messages`)**
- Per-user query for the posting user.
- Response now includes `author_is_verified_lawyer: bool`.

---

## Flutter Changes

### Verified Lawyer Badge — All Locations

A `Tooltip(message: 'Verified Lawyer', child: Icon(Icons.verified, size: 14, color: Color(0xFF0A66C2)))` is now shown next to the user's name in every location across the app. The original branch had no badge anywhere.

| File | Location | Badge data key |
|------|----------|---------------|
| `lib/screens/social/feed_screen.dart` | Post author name | `is_verified_lawyer` |
| `lib/screens/social/feed_screen.dart` | Comment author name | `author_is_verified_lawyer` |
| `lib/screens/social/network_screen.dart` | "People you may know" (main list) | `is_verified_lawyer` |
| `lib/screens/social/network_screen.dart` | "People you may know" (bottom sheet) | `is_verified_lawyer` |
| `lib/screens/social/network_screen.dart` | `_PendingTile` requester name | `requester_is_verified_lawyer` |
| `lib/screens/social/network_screen.dart` | Connections list | `is_verified_lawyer` |
| `lib/screens/social/network_screen.dart` | Search results | `is_verified_lawyer` |
| `lib/screens/social/alerts_screen.dart` | Activity notification actor | `actor_is_verified_lawyer` |
| `lib/screens/social/alerts_screen.dart` | Invite requester name | `requester_is_verified_lawyer` |
| `lib/screens/social/member_profile_screen.dart` | Profile header | `is_verified_lawyer` |
| `lib/screens/social/profile_screen.dart` | Own profile header | from `AuthProvider` |
| `lib/screens/social/profile_recommendations_screen.dart` | Recommender name | `author_is_verified_lawyer` |
| `lib/screens/social/profile_skills_screen.dart` | Endorser name in Skills & Endorsements | `endorser_is_verified_lawyer` |
| `lib/screens/home/dashboard_tab.dart` | Dashboard header name | from `AuthProvider` |
| `lib/screens/messaging/messages_hub_screen.dart` | Direct chat peer name in hub | `is_verified_lawyer` |
| `lib/screens/messaging/conversation_screen.dart` | Group chat author name (via `UserChatBubble`) | `author_is_verified_lawyer` |
| `lib/screens/messaging/create_group_screen.dart` | Connection names in group creation picker | `is_verified_lawyer` |
| `lib/screens/messaging/group_members_sheet.dart` | Existing group members list | `is_verified_lawyer` |
| `lib/screens/messaging/group_members_sheet.dart` | "Add members" dialog picker | `is_verified_lawyer` |
| `lib/screens/messaging/deal_room_members_sheet.dart` | Deal room member names | `is_verified_lawyer` |
| `lib/screens/messaging/deal_contract_chat_screen.dart` | Deal chat author name (via `UserChatBubble`) | `author_is_verified_lawyer` |
| `lib/screens/messaging/start_private_chat_screen.dart` | Connection names in new DM picker | `is_verified_lawyer` |

---

### Lawyer Offer Feature

The entire offer flow is new — it did not exist in the original branch.

#### `lib/widgets/user_chat_bubble.dart`

**`UserChatMessage`** — three new fields:
```dart
final String  msgType;         // default 'text'
final Map<String, dynamic>? offerData;
final String? offerStatus;     // null | 'accepted' | 'rejected'
final bool    authorIsVerified; // default false
```

**`UserChatBubble`** — one new parameter:
```dart
final VoidCallback? onOfferPay;
```

**New widget `_LawyerOfferCard`**
Rendered inside `UserChatBubble` when `msgType == 'lawyer_offer'`:
- Shows gavel icon + service title (bold, blue).
- Shows description.
- Shows price/currency badge.
- For the **recipient** (`!isMine`):
  - If `offerStatus != 'accepted'` → blue filled "Pay" button that calls `onOfferPay`.
  - If `offerStatus == 'accepted'` → green "Payment confirmed" row with check icon.
- The sender sees no action button (offer is read-only for them).

#### `lib/screens/messaging/conversation_screen.dart`

**Send offer (verified lawyers only):**
- A gavel `IconButton` appears in the input bar for verified lawyers in direct conversations.
- Tapping it opens `_showOfferDialog()` — an `AlertDialog` with a `Form` containing service title, description, price, and currency fields.
- On confirm, calls `legato.postLawyerOffer()` and reloads messages.

**Receive and pay offer:**
- `offerStatus` from the message data is passed to `UserChatMessage`.
- `onOfferPay` is wired to `_payOffer(messageId, offerData)` for non-mine offer messages.
- `_payOffer()` shows a confirmation dialog ("Confirm Payment — Service: … Amount: …"), then calls `legato.acceptOffer(messageId)` on confirm, and refreshes messages.

#### `lib/api/legato_api.dart`

Two new methods:
```dart
// Verified lawyer sends an offer in a direct conversation
Future<Map<String, dynamic>> postLawyerOffer(int conversationId, {
  required String serviceTitle,
  required String description,
  required double price,
  String currency = 'USD',
});

// Recipient accepts the offer (calls PATCH /api/messages/{id}/offer-status)
Future<Map<String, dynamic>> acceptOffer(int messageId);
```

---

### Lawyer Application Screen

**File:** `lib/screens/lawyer/lawyer_application_screen.dart` (new file)

The complete lifecycle for end-users:

| Status | UI shown |
|--------|----------|
| `not_applied` | Application form (bar licence, years experience, CV upload, ID card upload) |
| `pending` | Status card — "Application under review", read-only fields |
| `approved` | Success card — "You are a Verified Lawyer", badge preview |
| `rejected` | Editable form pre-populated with previous submission; file pickers show "Already on server: filename (tap to replace)" in green; re-submission calls PATCH |

Key implementation details:
- Form pre-population on load via `GET /lawyer/status`.
- Files are only required on first submission; on re-edit, the server keeps existing files if no new file is selected.
- `_buildDocPicker()` accepts an `isServerFile` flag to render the green "on server" state.

---

### Admin Screen

**File:** `lib/screens/admin/admin_screen.dart`

No code changes to this file, but it now **works correctly** because:

- The Admin screen reads `m['user_type']` and `m['lawyer_status']` for each user.
- In the original `final_80%` branch, `admin_list_users` only returned `{id, email, role}`, so `user_type` and `lawyer_status` were always empty strings.
- After the `analyses.py` fix, both fields are populated from the database.

---

## API Reference — New Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/lawyer/status` | User | Get own application status + form data |
| `POST` | `/lawyer/apply` | User | Submit new lawyer application (multipart) |
| `PATCH` | `/lawyer/application` | User | Update pending/rejected application |
| `GET` | `/admin/lawyers/applications` | Admin | List all applications (paginated) |
| `GET` | `/admin/lawyers/applications/{id}` | Admin | Single application detail |
| `GET` | `/admin/lawyers/applications/{id}/cv` | Admin | Download CV file |
| `GET` | `/admin/lawyers/applications/{id}/id-card` | Admin | Download ID card file |
| `PATCH` | `/admin/lawyers/{id}/review` | Admin | Approve or reject an application |
| `POST` | `/api/messages/conversations/{id}/offer` | Verified Lawyer | Send a lawyer offer in a direct DM |
| `PATCH` | `/api/messages/{message_id}/offer-status` | User (recipient) | Accept or reject a received offer |

---

*Generated 2026-06-17 against branch `final_80%` of `nourabulnasr/GP-Legal-AI-`.*
