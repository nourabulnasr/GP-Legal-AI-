# Legato — Full Application Testing Plan

**Version:** 2026-06-02  
**Scope:** Backend (82 FastAPI routes), Flutter Web (Firebase Hosting), React admin/legacy frontend, VPS production, Firebase MCP verification  
**Primary production URLs:**

| Layer | URL |
|-------|-----|
| Flutter Web (Firebase Hosting) | https://legatoappgp2026.web.app |
| Backend API (Hostinger) | https://srv1723974.hstgr.cloud |
| Backend IP (direct) | http://76.13.4.148 |
| API docs | `{API_BASE}/docs` |

---

## 1. Goals & success criteria

### Goals

- Verify every mounted API route returns expected status codes, auth behavior, and payload shape.
- Verify all user-facing flows work end-to-end from Firebase-hosted Flutter Web → VPS API.
- Confirm Firebase Hosting serves the correct build with correct `API_BASE_URL`.
- Catch regressions in OCR/RAG/LFM, translation, social feed, and auth before demos/releases.

### Release sign-off (minimum)

- [ ] All **P0** API smoke tests pass on production.
- [ ] All **P0** frontend journeys pass on https://legatoappgp2026.web.app (mobile + desktop browser).
- [ ] Firebase Hosting deploy verified via MCP (`firebase_deploy_status`).
- [ ] No 5xx on core paths; 401/403 behave correctly for protected routes.
- [ ] Post images load (`IMAGE_BASE_URL` matches API public URL).
- [ ] Document chat (LFM) returns substantive Arabic for a known analysis.
- [ ] Contract upload + analysis completes within acceptable timeout (≤ 3 min on VPS).

---

## 2. Test pyramid

```
                    ┌─────────────────┐
                    │  Manual UAT     │  Demo script, committee walkthrough
                    └────────┬────────┘
               ┌─────────────┴─────────────┐
               │  E2E (browser / Flutter)   │  Feed, analyze, chat, network
               └─────────────┬─────────────┘
          ┌──────────────────┴──────────────────┐
          │  API integration (curl / pytest / Postman) │  All 82 routes
          └──────────────────┬──────────────────┘
     ┌───────────────────────┴───────────────────────┐
     │  Unit tests (pytest, mocked LLM/MT/Google)      │  app/tests/*
     └─────────────────────────────────────────────────┘
```

---

## 3. Environments & configuration matrix

Run the **same test checklist** on each environment you care about; record pass/fail per column.

| Variable / check | Local dev | VPS production |
|------------------|-----------|----------------|
| `SECRET_KEY`, `JWT_SECRET_KEY` | set | set |
| `DATABASE_URL` | sqlite local | `/data/legalai.db` in Docker |
| `CORS_ORIGINS` | localhost | includes `https://legatoappgp2026.web.app` |
| `IMAGE_BASE_URL` | `http://127.0.0.1:8000` | `https://srv1723974.hstgr.cloud` |
| `GEMINI_API_KEY` | optional | optional (assistant only) |
| `GOOGLE_TRANSLATION_API_KEY` | optional | set for MT |
| `LOCAL_LLM_PATH` / LFM mount | optional | `/models/lfm` |
| `WARMUP_LFM_AT_STARTUP` | 0 or 1 | 1 |
| `DOCUMENT_CHAT_GEMINI_FALLBACK` | 0 | 0 (LFM-only chat) |
| `ENABLE_STARTUP_RAG` | 0 or 1 | 1 |

### Firebase (Hosting only — not app auth)

Legato uses **custom JWT auth on the VPS**, not Firebase Authentication. Firebase is used to **host the Flutter Web build**.

**MCP verification (Cursor Firebase plugin):**

| Step | MCP tool | Expected |
|------|----------|----------|
| Auth | `firebase_login` | Logged in as project owner |
| Project | `firebase_list_projects` / `firebase_get_project` | Project hosting `legatoappgp2026` |
| Apps | `firebase_list_apps` | Web app registered |
| Hosting config | `firebase_read_resources` | Hosting site + deploy history |
| After deploy | `firebase_deploy_status` | Last deploy succeeded |
| SDK config | `firebase_get_sdk_config` | Matches Flutter web config if used |

> **Note:** Repo may not contain `firebase.json` locally; hosting config may live in a separate deploy repo or machine. Use MCP to inspect the live Firebase project.

---

## 4. Test accounts & data

| Role | Purpose | How to create |
|------|---------|---------------|
| `user_a` | Standard flows | `POST /auth/register` + verify email |
| `user_b` | Network invites, second device | Same |
| `admin` | Admin law + analyses admin | `PATCH /analyses/admin/users/{id}` → `role=admin` |
| Sample PDF | OCR pipeline | Arabic labor contract (e.g. cook employment template) |
| Sample image | Social post | JPG/PNG &lt; 5 MB |

Store JWT tokens from login for reuse in curl/Postman collections.

---

## 5. Backend API testing

### 5.1 Automation (existing)

```bash
# From repo root
python -m pytest app/tests/ -q
```

| File | Covers |
|------|--------|
| `test_language_and_translation.py` | LID, glossary, Google MT mock, LFM mock, labor gating |
| `test_ocr_check_and_search_integration.py` | Full analyze pipeline (env-dependent) |
| `test_document_chat.py` | Document chat helpers, low-quality detection |
| `test_pdf_text_pipeline.py` | PDF text extraction |
| `test_chunks_loader.py` | RAG chunk loading |

**Gap:** No automated tests yet for `/auth`, `/api/*`, `/legato/*`, `/analyses/*` — add incrementally (see §9).

### 5.2 Smoke template (every environment)

```bash
API=https://srv1723974.hstgr.cloud
curl -sf "$API/health" | jq .
curl -sf "$API/" | jq .
```

### 5.3 API checklist by domain

Use **Postman**, **curl**, or **pytest + httpx**. For JWT routes: `Authorization: Bearer $TOKEN`.

#### System (P0)

| # | Method | Path | Auth | Test | Expected |
|---|--------|------|------|------|----------|
| S1 | GET | `/health` | No | Smoke | 200 `{"status":"ok"}` |
| S2 | GET | `/` | No | Smoke | 200 API name |
| S3 | GET | `/docs` | No | Open Swagger | 200 HTML |

#### Auth (P0)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| A1 | POST | `/auth/register` | Valid email/password | 200, verification required or user created |
| A2 | POST | `/auth/verify-email` | Valid code | 200 |
| A3 | POST | `/auth/login` | Valid credentials | 200 + `access_token` |
| A4 | GET | `/auth/me` | With JWT | 200 user object |
| A5 | POST | `/auth/login` | Wrong password | 401 |
| A6 | POST | `/auth/forgot-password` | Known email | 200 |
| A7 | POST | `/auth/verify-reset-code` | Valid code | 200 + reset token |
| A8 | POST | `/auth/reset-password` | With token | 200 |
| A9 | GET | `/auth/google` | Browser redirect | 302 to Google |
| A10 | GET | `/auth/me` | No JWT | 401 |

#### OCR & analysis (P0)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| O1 | POST | `/ocr` | Upload PNG/PDF | 200 + text |
| O2 | POST | `/ocr_check_and_search` | Arabic contract PDF, defaults | 200 + `ocr_chunks`, `rule_hits` |
| O3 | POST | `/ocr_check_and_search` | `translate_to_ar=true`, `translation_target_lang=ar` | 200 + `translated_ar_text` or `translation_provider` |
| O4 | POST | `/ocr_check_and_search` | `save=true` without JWT | 401 |
| O5 | POST | `/ocr_check_and_search` | `save=true` with JWT | 200 + persisted analysis |
| O6 | POST | `/check_clause` | Single clause text | 200 scores/hits |
| O7 | POST | `/ocr_check_and_search` | `translation_only=true` | 200, minimal analysis |

#### Analyses CRUD (P0)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| N1 | GET | `/analyses` | JWT | 200 list |
| N2 | GET | `/analyses/{id}` | Owner JWT | 200 full result |
| N3 | GET | `/analyses/{id}` | Other user | 403 |
| N4 | PATCH | `/analyses/{id}/flag` | Set `needs_review` | 200 |
| N5 | DELETE | `/analyses/{id}` | Owner | 200/204 |
| N6 | GET | `/analyses/admin/all` | Admin JWT | 200 |
| N7 | GET | `/analyses/admin/all` | User JWT | 403 |

#### Chat (P0)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| C1 | POST | `/chat/document` | `analysis_id` + Arabic question | 200, Arabic answer, LFM path |
| C2 | POST | `/chat/document` | No analysis / empty context | 400 |
| C3 | POST | `/chat/message` | Gemini configured | 200 contract-aware reply |
| C4 | POST | `/chat/assistant` | General question | 200 or “not configured” if no Gemini |
| C5 | GET | `/chat/document` | GET method | 405 |

#### Legato mobile (P1)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| L1 | POST | `/legato/explain-clause` | Clause + rule context | 200 explanation |
| L2 | GET | `/legato/risk/{analysis_id}` | Saved analysis | 200 risk payload |
| L3 | POST | `/legato/summarize-clauses` | Clause list | 200 summaries |
| L4 | POST | `/legato/compare` | Two texts/analyses | 200 comparison |
| L5 | POST | `/legato/negotiation-chat` | Message | 200 (Gemini or LFM fallback) |
| L6 | POST | `/legato/shares` | Create share | 200 + token |
| L7 | GET | `/legato/shares/public/{token}` | No auth | 200 read-only analysis |
| L8 | POST | `/legato/deal-threads` | Create thread | 200 |
| L9 | POST | `/legato/deal-threads/{id}/messages` | Post message | 200 |
| L10 | POST | `/legato/timeline/events` | Add event | 200 |
| L11 | GET | `/legato/timeline/me` | JWT | 200 |
| L12 | GET/PUT | `/legato/profile/me` | Read/update profile | 200 |
| L13 | POST | `/legato/signatures` | Sign analysis | 200 |

#### Social / network (P0)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| F1 | GET | `/api/posts?page=1` | JWT | 200 `items`, counts |
| F2 | POST | `/api/posts` | Text only | 200 new post |
| F3 | POST | `/api/posts` | Text + image multipart | 200 + `image_url` public URL |
| F4 | GET | `/static/post_images/{file}` | No auth | 200 image bytes |
| F5 | POST | `/api/posts/{id}/like` | Toggle | 200 liked state |
| F6 | POST | `/api/posts/{id}/comment` | Add comment | 200 |
| F7 | GET | `/api/posts/{id}/comments` | List | 200 |
| F8 | POST | `/api/posts/{id}/share` | Share | 200 |
| F9 | GET | `/api/network/stats` | JWT | 200 |
| F10 | GET | `/api/network/suggestions` | JWT | 200 |
| F11 | POST | `/api/network/invites` | user_b → user_a | 200 |
| F12 | POST | `/api/network/invites/{id}/accept` | Accept | 200 |
| F13 | PUT | `/api/profile/me` | Update headline | 200 |
| F14 | POST | `/api/profile/{id}/endorse` | Endorse skill | 200 |

#### Admin law (P2 — admin only)

| # | Method | Path | Test | Expected |
|---|--------|------|------|----------|
| D1 | POST | `/admin/law/preview-pdf` | Small PDF | 200 preview |
| D2 | POST | `/admin/law/reindex` | Trigger job | 200 job id |
| D3 | GET | `/admin/law/jobs/{job_id}` | Poll status | 200 completed/failed |

---

## 6. Frontend testing

### 6.1 Flutter Web (primary — Firebase Hosting)

**Build under test:** deployed at https://legatoappgp2026.web.app  
**API must point to:** `https://srv1723974.hstgr.cloud` (verify in compiled `main.dart.js` or build flags).

#### Navigation shell (P0)

| Tab | Screen | Key actions |
|-----|--------|-------------|
| Feed | `feed_screen.dart` | List posts, create post, like/comment/share, **photo post displays image** |
| Network | `network_screen.dart` | Stats, invites, connect, suggestions |
| Contracts | `contracts_tab_screen.dart` | Open analyze, history |
| Alerts | `alerts_screen.dart` | Placeholder loads without crash |
| Profile | profile screens | Edit profile, tools grid |

#### P0 user journeys (manual or browser automation)

| ID | Journey | Steps | Pass criteria |
|----|---------|-------|---------------|
| FE-1 | Register & login | Register → verify (if SMTP) → login | Lands on home shell |
| FE-2 | Upload & analyze | Contracts → upload PDF → wait → violations tab | Results visible, no infinite spinner |
| FE-3 | Translate | Analyze with Arabic translation on | Translated text on chunks |
| FE-4 | Save analysis | Save to account | Appears in history |
| FE-5 | Document chat LFM | Select analysis → chat → `اشرح العقد` | Arabic explanation, not garbage |
| FE-6 | Feed text post | New post with tags/category | Appears in feed |
| FE-7 | Feed photo post | Attach image → post | **Image visible in card** |
| FE-8 | Network invite | User A invites B → B accepts | Connection count updates |
| FE-9 | Share link | Create share → open public URL | Read-only analysis loads |
| FE-10 | Logout / re-login | Session persists or clears correctly | JWT refresh behavior OK |

#### Cross-browser (P1)

- Chrome desktop, Safari iOS (as in production logs), Firefox.
- Check CORS: no blocked preflight on `/api/*` from Firebase origin.

### 6.2 React frontend (`legalai-frontend/`) — P2

Used for admin/analysis web UI if deployed separately.

| Page | Tests |
|------|-------|
| Login / Register / Forgot password | Auth flows |
| AnalyzePage | Upload, tabs, document chat LFM |
| ChatPage | General + contract chat modes |
| HistoryPage | Saved analyses list |
| AdminPage | Admin-only gate |

```bash
cd legalai-frontend/legalai-frontend
npm ci && npm run build
# Manual test against API with VITE_API_BASE_URL set
```

---

## 7. Firebase & MCP verification checklist

Run before/after each Flutter Web deploy.

| # | Action | Tool / command |
|---|--------|----------------|
| FB-1 | Confirm Firebase login | MCP `firebase_login` |
| FB-2 | List projects | MCP `firebase_list_projects` |
| FB-3 | Confirm web app | MCP `firebase_list_apps` |
| FB-4 | Read hosting resources | MCP `firebase_read_resources` |
| FB-5 | Deploy web build | MCP `firebase_deploy` (hosting only) or CI |
| FB-6 | Verify deploy | MCP `firebase_deploy_status` |
| FB-7 | Live URL loads | Browser → https://legatoappgp2026.web.app |
| FB-8 | API calls from hosted app | DevTools Network → requests to `srv1723974.hstgr.cloud` |
| FB-9 | CORS | No CORS errors on authenticated API calls |
| FB-10 | Cache bust | Hard refresh after deploy shows new version |

**Important:** Firebase MCP manages **Hosting/project metadata**, not VPS backend health. Always test API separately (§5).

---

## 8. Non-functional testing

Reference: `deploy/hostinger/CONCURRENCY_REPORT.md`

| # | Area | Test | Target |
|---|------|------|--------|
| NF-1 | Health latency | 10× `GET /health` | &lt; 50 ms each |
| NF-2 | Concurrent light API | 2 users browse feed + profile | Both 200 |
| NF-3 | Heavy analyze queue | 2 simultaneous full scans | Second completes (may wait); no OOM |
| NF-4 | LFM serialization | 2 concurrent `/chat/document` | Both 200; second slower |
| NF-5 | Image upload max | 6 MB image | 413 |
| NF-6 | SQLite WAL | Concurrent writes (invite + post) | No `database is locked` |
| NF-7 | Docker restart | `docker restart legalai-backend` | RAG + LFM warmup; health OK |
| NF-8 | Static persistence | Post image survives restart | URL still 200 |

---

## 9. Recommended automation backlog

Priority order for pytest/CI:

1. `tests/test_auth_api.py` — register/login/me/forgot-password (test DB)
2. `tests/test_social_api.py` — posts CRUD, image upload with TestClient
3. `tests/test_analyses_api.py` — owner/admin authorization
4. `tests/test_legato_api.py` — explain-clause mocked LFM
5. GitHub Action: `pytest app/tests/` on push
6. Optional: Playwright against staging Firebase URL for FE-1–FE-7

---

## 10. Execution schedule (suggested)

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1 — Smoke** | 1 h | S1–S3, A3–A4, O2, F1, FB-7–FB-9 on production |
| **Phase 2 — API depth** | 4 h | Full §5.3 checklists with Postman collection |
| **Phase 3 — Frontend UAT** | 3 h | FE-1–FE-10 on iPhone + Chrome |
| **Phase 4 — AI/RAG** | 2 h | O3, C1, L1–L5 with real LFM; translation Google |
| **Phase 5 — Admin & edge** | 2 h | D1–D3, NF tests, error paths |
| **Phase 6 — Sign-off** | 1 h | Demo script (`docs/DEMO_SCRIPT.md`) dry run |

---

## 11. Defect severity

| Severity | Definition | Example |
|----------|------------|---------|
| **P0** | Blocks core demo / data loss / security | Login broken, analyze 500, JWT bypass |
| **P1** | Major feature broken | Photo posts, document chat garbage, translation off |
| **P2** | Minor / workaround exists | Alerts placeholder, admin UI polish |
| **P3** | Cosmetic | Typo, slow but acceptable latency |

---

## 12. Reporting template

For each failed test, record:

```
ID: F7
Environment: production
Steps: ...
Expected: image visible in feed card
Actual: image_url null / localhost URL
Logs: docker logs legalai-backend --tail 50
Screenshot: attached
Severity: P1
```

---

## 13. Quick reference commands

```bash
# Backend tests
python -m pytest app/tests/ -q

# Production smoke
API=https://srv1723974.hstgr.cloud
curl -sf "$API/health"

# Login (get token)
curl -s -X POST "$API/auth/login" -H "Content-Type: application/json" \
  -d '{"email":"USER","password":"PASS"}' | jq -r .access_token

# Feed with JWT
curl -s "$API/api/posts?page=1" -H "Authorization: Bearer $TOKEN" | jq .

# Compare DB schemas (after migrations)
python scripts/compare_db_schemas.py old.db new.db

# Firebase MCP (in Cursor)
# firebase_get_environment → firebase_list_projects → firebase_deploy_status
```

---

## 14. Related docs

- `docs/DEMO_SCRIPT.md` — Committee demo flow (subset of UAT)
- `docs/API_CONTRACT.md` — OCR/translation request fields
- `deploy/hostinger/CONCURRENCY_REPORT.md` — Load expectations
- `docs/UX_NETWORKING_SCOPE.md` — Social UI scope
- `docs/ARCHITECTURE.md` — Route index and architecture
