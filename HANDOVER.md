# LEGATO — GRADUATION PROJECT HANDOVER
**For:** Teammate taking over to complete, test, and submit  
**From:** Nour Abulnasr (developer)  
**Date:** 2026-05-11  
**Branch:** `final_80%` → `https://github.com/nourabulnasr/GP-Legal-AI-`  
**Flutter repo:** `C:\dev\legato_mobile1` (separate local folder — see Section 12 for GitHub status)

---

## 1. PROJECT OVERVIEW

**Legato** is a three-layer AI-powered legal platform for the Egyptian employment law market. It is a graduation project at **Misr International University (MIU)**, Faculty of Computer Science, AI track, class of 2026.

**Team:**
- **Nour Abulnasr** — lead developer (backend + Flutter + architecture). Contact for any blocked question.
- **[TEAMMATE_NAME]** — taking over for testing, deployment, and submission.

**Graduation deadline:** [ASK NOUR — confirm exact date with committee]

### The Three Layers

**Layer 1 — AI Contract Intelligence**
A user uploads an Arabic/English employment contract (PDF, DOCX, or scanned image). The system runs a 10-stage pipeline: OCR text extraction → Arabic normalization → clause splitting → cross-border jurisdiction detection → Rule Engine (YAML rules for Egyptian Labor Law 14/2025) → ML assist (probability scores) → RAG retrieval (452 law document chunks from ChromaDB + FAISS) → LFM2.5-1.2B local model (on-device Arabic explanation grounded in law text) → structured JSON response. The output: every violation with law article citation, severity, LFM-generated Arabic explanation, and a `needs_review` flag for contracts that need a human lawyer.

**Layer 2 — Professional Networking**
A LinkedIn-style professional network scoped to Egypt's legal community. Social feed, posts, likes, comments, shares, connection invitations, skill endorsements, peer recommendations, profile pages. All data is real and stored in SQLite.

**Layer 3 — 12 Power Tools**
Tools for lawyers and HR professionals: clause explainer, clause summarizer, contract comparison, risk dashboard, negotiation coach, deal threads, contract timelines, e-signature, shareable analysis links, document chat (Q&A over a specific contract using LFM), public share view.

### The LFM/Gemini Boundary (NEVER violate)
**LFM (local, on-device):** explain-clause, summarize-clauses, compare, document chat (primary), negotiation chat (fallback).  
**Gemini (cloud):** general chat assistant, negotiation chat (primary only).  
LFM is used for all legal reasoning because it is grounded in retrieved law text — it cannot invent citations. Gemini has no access to the law corpus. Mixing them would produce hallucinated legal advice.

---

## 2. CURRENT STATUS — HONEST

| Component | % Done | What's left |
|-----------|--------|-------------|
| **Backend (FastAPI)** | ~95% | 2 polish fields (updated_at, avatar_url), gitignore one folder |
| **Flutter Mobile** | ~90% | Real device testing not started, 38-step checklist not run |
| **Web Frontend (React/Vite)** | ~80% | Functional pages exist, not deployed, not tested post-refactor |
| **HF Spaces Deployment** | 0% | Files ready in `hf_deployment/`, actual push not done |
| **Oracle Cloud Deployment** | 0% | Instance not created |
| **Real Device APK Test** | 0% | APK builds (50MB release), but no physical device test done |
| **Demo dry-run** | 0% | DEMO_SCRIPT.md exists, no committee rehearsal done |
| **Marketing video** | 0% | Not started |

**Do not inflate these numbers for the committee.** The backend is genuinely strong. The testing gap is real.

---

## 3. THE GRADUATION DELIVERABLES

What must exist on submission day:

| Deliverable | Status | Owner |
|-------------|--------|-------|
| **Live backend URL** (HF Spaces or Oracle Cloud) | ❌ Not deployed | YOU |
| **Flutter APK file** (Android, tested on real device) | ❌ Not tested on device | YOU |
| **Flutter Web URL** (Firebase Hosting) | ❌ Not deployed | YOU |
| **GitHub repo** with clean README on `main` branch | ⚠️ README exists on `final_80%`, not merged to `main` | YOU |
| **docs/ARCHITECTURE.md** | ✅ Done | Nour |
| **docs/DEMO_SCRIPT.md** | ✅ Done | Nour |
| **Demo data seeded** (admin + demo accounts) | ✅ Done locally | YOU (re-seed on deployed server) |
| **1-minute marketing video** | ❌ Not started | [DECIDE WITH NOUR] |
| **Live demo** (committee presentation, ~10-12 min) | ❌ Not rehearsed | YOU + Nour |
| MIU-specific submission forms | [ASK NOUR] | Nour |

---

## 4. ARCHITECTURE

Full system design is in `docs/ARCHITECTURE.md`. Read it before touching any file.

**Summary:**

```
Flutter App (42 Dart files, 38+ screens)
    +
React/Vite Web Frontend (9 pages)
         |
         | HTTPS/REST
         v
    FastAPI Backend (73 endpoints, 7 routers)
    app/main.py — entry point + core pipeline
         |
    ┌────┴────────────────────────────────┐
    │                                      │
Rule Engine         ML Assist         RAG Layer         LFM Explanation
(YAML rules,        (joblib            (ChromaDB          (LFM2.5-1.2B
 authoritative)     multilabel         154 docs +         local model,
                    classifier)        FAISS 298 docs)    2.23 GB)
    │
    └────────────────────────────────────────────────────┐
                                                         │
                                              SQLite DB (16 tables)
                                              legalai.db
```

**Core pipeline endpoint:** `POST /ocr_check_and_search`  
**Stages:** OCR → normalize → split clauses → cross-border detect → Rule Engine → ML → RAG → LFM → response

**LFM/Gemini boundary:** See Section 1 above and `docs/ARCHITECTURE.md` Section 3.

---

## 5. WHAT IS DONE

### Backend (FastAPI)

**Core pipeline (100%):**
- `POST /ocr_check_and_search` — full 10-stage pipeline working
- OCR chain: Google DocumentAI → PyMuPDF → Tesseract fallback
- Arabic normalization (diacritics, Arabic-Indic numerals, OCR mis-reads)
- Clause splitting (bilingual Arabic/English)
- Cross-border detection (UAE/MOHRE signals, 20+ regex patterns)
- Rule Engine (3 YAML files: labor.yaml, labor_mandatory.yaml, labor_cross_border.yaml)
- ML assist (joblib multilabel classifier + binary model)
- RAG retrieval (ChromaDB primary → FAISS fallback, 452 total chunks)
- LFM2.5-1.2B explanation (float32, max_new_tokens=256, Arabic sentinel stripped)
- `needs_review` flag on Analysis (set when any rule hit has `severity="error"`)
- `POST /check_clause` — single clause risk check
- `POST /ocr` — text extraction only (no analysis)

**All 73 endpoints across 7 routers (100%):**
- `/auth/*` — register, email verify, login, JWT, Google OAuth SSO, forgot-password, reset
- `/analyses/*` — history CRUD, `needs_review` flag, `PATCH /analyses/{id}/flag`
- `/chat/*` — Gemini assistant, threaded messages, LFM document chat
- `/legato/*` — all 12 power tools (explain, summarize, compare, risk, share, deal threads, timeline, signature, profile, negotiation)
- `/api/*` — social feed, posts, likes, comments, shares, network invites, endorsements, recommendations, profile documents
- `/admin/law/*` — law PDF upload, corpus update, RAG rebuild (zero downtime)
- `GET /health`

**Performance fixes (all committed):**
- Social feed N+1 fixed: `_batch_post_stats()` — 6 queries for any feed size (commit e4bd7d3)
- LFM speed: `torch.float32` (was float16), `max_new_tokens=256` (was 512) (commit 83c4869)
- Gemini 3-attempt retry with linear backoff 2s/4s (commit e4bd7d3)
- Negotiation fallback to LFM when Gemini fails (commit e4bd7d3)
- `used_fallback: bool` in `DocumentChatResponse` (commit e4bd7d3)
- CORS `allow_origin_regex` for Flutter Web localhost ports (commit 0fcb513)
- LFM `_build_context()` rewritten: OCR capped 10k, violations as Arabic bullet list, no raw JSON (commit 83c4869)

**Config (all set in `.env`):**
- `SECRET_KEY` — 64-char hex set
- `ACCESS_TOKEN_EXPIRE_MINUTES=720` — 12h tokens for demo sessions
- `WARMUP_LFM_AT_STARTUP=1` — LFM loads at startup in background thread
- `ENABLE_STARTUP_RAG=1` — ChromaDB + FAISS warm at startup
- `DOCUMENT_CHAT_GEMINI_FALLBACK=0` — LFM only for doc chat (no silent Gemini fallback)

**Artifacts (all present):**
- `LFM2.5-1.2B-Instruct/model.safetensors` — 2.23 GB real file
- `model_ML/law_aware_multilabel_model.joblib` — 10.2 MB
- `model_ML/law_aware_binary_model.joblib` — 3.3 MB
- `app/ml/artifacts/unified/` — vectorizer, model, thresholds, metrics, config
- All 3 rule YAML files
- `Legal Rag/chroma_db/` — ChromaDB persistent store (154 docs)
- `chunks/labor_law_chunks.cleaned.jsonl` — FAISS source (298 docs)

**Documentation (all written):**
- `README.md` — architecture, API reference, run instructions
- `docs/ARCHITECTURE.md` — full system design
- `docs/DEMO_SCRIPT.md` — 10-min committee demo with curl commands
- `LEGATO_AGENT_CONTEXT.md` — engineering context for AI agents
- `LEGATO_BACKEND_TASKS.md` — audit with task list

**Demo data (seeded locally):**
- `admin@legato.com` / `LegatoAdmin2026!`
- `demo@legato.com` / `LegatoDemo2026!`
- Sample analysis with violations, social posts, deal thread

**HF deployment files (ready, not pushed):**
- `hf_deployment/Dockerfile` — Python 3.11-slim, Tesseract Arabic, port 7860
- `hf_deployment/README.md` — HF Spaces YAML front-matter
- `hf_deployment/seed_demo_data.py` — idempotent demo data seeder
- `hf_deployment/.gitattributes` — Git LFS tracking

### Flutter Mobile App

**Stack:** Flutter 3.11+, Provider, `http` package, SQLite-free (all state from backend)  
**Location:** `C:\dev\legato_mobile1`  
**Build:** `flutter analyze`: 0 errors, 0 warnings. Release APK: 50MB, builds clean.

**All features implemented:**
- Auth: login, register, email verify, forgot-password 3-step, session expiry dialog on 401
- Contract analysis: file picker (PDF/DOCX/image), Android 13+ permissions, 10-min timeout, WakeLock
- Full result display: violations, risk score, OCR text, RAG hits, LFM explanation
- Analysis history: list, detail, delete
- Chat hub: assistant (Gemini), document chat (LFM, dropdown for analysis selection)
- Social: infinite scroll feed, create post, like/unlike, comments, share sheet, author → profile
- Network: stats, suggestions, connect button with sent-state, pending invites, search
- Profile: cinematic SliverAppBar (navy-to-gold gradient), edit dialog, skills, experience, education
- All 12 Legato tool screens (e-sign canvas, risk dashboard, explain, summarize, compare, negotiation, share, deal threads, timeline, clause checker)
- Admin panel (role-gated), Settings (sign out, runtime API URL override), Dashboard tab

**Critical fixes applied (2026-05-11):**
1. `chat_analysis_screen.dart` → POST `/chat/document` (was wrongly calling `/chat/message`)
2. Feed comments: handles both `items` and `comments` response keys
3. Feed share: custom bottom sheet (no `share_plus` dependency)
4. Network: suggestions handles `items` + `suggestions` response keys
5. Network: `_sentInvites` Set → "Sent ✓" feedback on connect button
6. Profile: cinematic SliverAppBar redesign
7. Feed: UTC → device local time via `.toLocal()`
8. `feed_screen.dart` → `_goToProfile()` → `MemberProfileScreen` navigation

### Web Frontend (React/Vite)

**Location:** `legalai-frontend/legalai-frontend/`  
**Pages:** Login, Register, ForgotPassword, ResetPassword, GoogleCallback, Analyze (5-tab: Summary/Violations/OCR/RAG/Document-chat), History, Chat, Admin  
**Auth:** JWT via axios interceptor, Bearer header  
**Status:** Functional pages built. Not deployed. Not tested after most recent backend changes.

---

## 6. WHAT IS NOT DONE — PRIORITIZED

### CRITICAL — Graduation blockers

**C1 — HF Spaces deployment** (30-60 min, Nour's HF credentials needed)
The backend has never been deployed. Every demo and test is local. Instructions are in Section 8.

**C2 — Real device APK test** (1-2 hours)
The APK builds (50MB) and Flutter analyze is clean, but it has never been run on a physical Android device. The 38-step checklist in Section 10 has never been executed. Do this before the committee demo.

**C3 — Rebuild APK with deployed server URL**
When the backend is deployed, rebuild:
```bash
flutter build apk --release \
  --dart-define=API_BASE_URL=https://nourabulnasr-legato.hf.space \
  --dart-define=SHARE_BASE_URL=https://nourabulnasr-legato.hf.space
```

**C4 — Re-seed demo data on deployed server**
After deploying backend:
```bash
python hf_deployment/seed_demo_data.py
# Set API_BASE_URL env var to point at deployed server first
```

**C5 — Demo dry-run with committee setup**
Read `docs/DEMO_SCRIPT.md` and run every curl command end-to-end. Time it. First LFM call takes 30-60 seconds — do a warmup run before the committee enters.

**C6 — 1-minute marketing video** (ask Nour — he owns the Higgsfield + Kling workflow)
Not started. Nour needs to make this using his AI video tools. It is a graduation deliverable.

**C7 — Merge `final_80%` to `main`** before repo submission
All work is on branch `final_80%`. GitHub repo `main` branch has older code. The committee will look at the default branch.

---

### IMPORTANT — Would be embarrassing to ship without

**I1 — Port mismatch between backend and Flutter default**
Backend DEMO_SCRIPT.md runs on port `8001`. Flutter's `app_config.dart` defaults to `http://10.0.2.2:8002`. Confirm which port the backend actually runs on and make sure the Flutter APK is built with the correct `API_BASE_URL`.

**I2 — B-P1: `updated_at` missing from Analysis model**
File: `app/db/models.py`. Only `created_at` exists. Add `updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)`. One line.

**I3 — B-P2: `avatar_url` not top-level in profile response**
File: `app/routers/social.py` line ~288. The `/api/profile/{user_id}` response has no standardized `avatar_url` field. Frontend profile cards show placeholders. Define it at the top level of the profile payload.

**I4 — `GP-Legal-AI-/` nested repo copy is not gitignored**
There is a 63MB folder `GP-Legal-AI-/` in the project root that is an old copy of the entire repo. It is not committed (git sees it as untracked) but it is not in `.gitignore`. Add one line to `.gitignore`:
```
GP-Legal-AI-/
```
Then commit and push.

**I5 — Real device test of the social screens**
The social feed, network, and profile screens are functional but have not been tested on a real device. They look generic (LinkedIn-style). If the committee pulls up the mobile app, these are the screens they'll spend most time on.

---

### NICE TO HAVE — Graduation works without these

- Additional demo data variety (more contracts, more social posts)
- Multilingual UI testing (Arabic RTL layout on all screens)
- `flutter_secure_storage` for JWT (blocked by Windows build path issue)
- Deep linking so share links open the app directly
- Named routes / go_router migration

---

## 7. HOW TO RUN LOCALLY

### Backend (FastAPI)

**Prerequisites:** Python 3.11+, `LFM2.5-1.2B-Instruct/` folder at project root (2.23 GB)

```bash
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"

# Install dependencies (first time only)
pip install -r requirements.txt

# Start with warmup flags (RECOMMENDED — avoids cold-start delays)
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# Watch for these startup lines before proceeding:
# [OK] LFM warmup: model loaded and ready.
# [OK] ChromaDB RAG ready: 154 docs.
# [OK] Retriever initialized with 298 docs.
# [OK] Auth router mounted
# [OK] Analyses router mounted
# [OK] Chat router mounted
# [OK] Legato mobile router mounted
# [OK] Social API router mounted
# [OK] Admin law router mounted

# Verify
curl http://127.0.0.1:8001/health
# Expected: {"status":"ok"}

# Swagger UI (interactive API browser)
# Open: http://127.0.0.1:8001/docs

# Seed demo data (already done locally — re-run on deployed server)
python hf_deployment/seed_demo_data.py
```

**Demo credentials (after seeding):**

| Role | Email | Password |
|------|-------|---------|
| Admin | admin@legato.com | LegatoAdmin2026! |
| Demo  | demo@legato.com  | LegatoDemo2026!  |

### Web Frontend (React/Vite)

```bash
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\legalai-frontend\legalai-frontend"
npm install
npm run dev
# Opens at http://localhost:5173
# Backend must be running first
```

### Flutter Mobile

```bash
cd C:\dev\legato_mobile1

# Run on Android emulator (backend at default 10.0.2.2:8002)
flutter run -d emulator-5554

# Run on real device (replace 192.168.1.X with your PC's LAN IP)
flutter run -d <device-id> --dart-define=API_BASE_URL=http://192.168.1.X:8001

# Build release APK (replace URL with deployed server)
flutter build apk --release \
  --dart-define=API_BASE_URL=https://nourabulnasr-legato.hf.space \
  --dart-define=SHARE_BASE_URL=https://nourabulnasr-legato.hf.space
# Output: build/app/outputs/flutter-apk/app-release.apk (~50MB)

# Build Flutter Web
flutter build web --dart-define=API_BASE_URL=https://nourabulnasr-legato.hf.space
```

**Runtime URL override (no rebuild needed for demo switching):**
Open the app → Settings → enter the new base URL → save. Persists across restarts.

---

## 8. HOW TO DEPLOY TO HUGGING FACE SPACES

The deployment files are ready in `hf_deployment/`. Requires Nour's HuggingFace account.

**Estimated time: 30-60 minutes** (most of it is the 2.23 GB LFM upload over your connection).

```bash
# STEP 1 — Create the Space
# Go to: https://huggingface.co/new-space
# Name: legato  |  Owner: nourabulnasr  |  SDK: Docker  |  Hardware: CPU Basic (free)

# STEP 2 — Clone the empty Space locally
git clone https://huggingface.co/spaces/nourabulnasr/legato hf_space
cd hf_space

# STEP 3 — Copy all files from the backend project
# Run from project root:
cp hf_deployment/Dockerfile .
cp hf_deployment/README.md .          # HF Spaces YAML metadata
cp hf_deployment/seed_demo_data.py .
cp hf_deployment/.gitattributes .
cp -r app .
cp -r rules .
cp -r chunks .
cp -r laws .
cp requirements.txt .
cp -r "Legal Rag" .
cp model_ML/law_aware_multilabel_model.joblib .
cp model_ML/law_aware_binary_model.joblib .
cp model_ML/labor14_2025_clean.jsonl .
cp -r app/ml/artifacts .
cp -r LFM2.5-1.2B-Instruct .          # 2.23 GB — this is the slow upload

# STEP 4 — Set up Git LFS for large files
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.joblib"
git lfs track "*.pt"
git add .gitattributes

# STEP 5 — Commit and push (the push will take 10-60 min for the LFM model)
git add .
git commit -m "Initial Legato deployment"
git push
# Watch upload percentage — LFM2.5-1.2B-Instruct/model.safetensors is 2.23 GB

# STEP 6 — Set HF Secrets
# Go to: Space Settings → Repository secrets → Add each:
# SECRET_KEY           = <same value from .env>
# GEMINI_API_KEY       = <get from aistudio.google.com>
# HF_TOKEN             = <get from huggingface.co/settings/tokens>
# ENABLE_STARTUP_RAG   = 1
# WARMUP_LFM_AT_STARTUP = 1
# DOCUMENT_CHAT_GEMINI_FALLBACK = 0
# ACCESS_TOKEN_EXPIRE_MINUTES = 720

# STEP 7 — Wait for the Space to build (5-15 min)
# Monitor at: https://huggingface.co/spaces/nourabulnasr/legato
# You'll see build logs — wait for "Application startup complete"

# STEP 8 — Seed demo data on deployed server
# Set env var then run:
API_BASE_URL=https://nourabulnasr-legato.hf.space python hf_deployment/seed_demo_data.py
```

**Live URL (once deployed):** `https://nourabulnasr-legato.hf.space`

---

## 9. KEY FILES MAP

### Backend

| File | What it does |
|------|-------------|
| `app/main.py` | Entry point. Core pipeline (`POST /ocr_check_and_search`). All startup hooks. |
| `app/rules.py` | Rule Engine — loads YAML files, runs text matching, returns `RuleHit` list |
| `app/local_llm.py` | LFM2.5 wrapper — singleton load, `generate()`, Arabic sentinel stripping |
| `app/legato_service.py` | Business logic for all 12 power tools |
| `app/legal_rag_bridge.py` | RAG facade — ChromaDB primary, calls `rag_chromadb.py` |
| `app/rag_chromadb.py` | ChromaDB access layer |
| `app/rag_utils.py` | FAISS in-memory retriever (fallback) |
| `app/cross_border.py` | UAE/MOHRE signal detector |
| `app/pdf_text_pipeline.py` | OCR chain (DocumentAI → PyMuPDF → Tesseract) |
| `app/model_ml_predictor.py` | Law-aware ML inference (joblib) |
| `app/routers/auth.py` | `/auth/*` — login, register, OAuth, reset |
| `app/routers/analyses.py` | `/analyses/*` — history CRUD, `needs_review` flag |
| `app/routers/chat.py` | `/chat/*` — Gemini assistant + LFM document chat |
| `app/routers/legato_mobile.py` | `/legato/*` — all 12 power tools |
| `app/routers/social.py` | `/api/*` — feed, network, profiles |
| `app/routers/admin_law.py` | `/admin/law/*` — law corpus update |
| `app/db/models.py` | 16 SQLAlchemy table definitions |
| `app/db/session.py` | SQLite engine + `get_db` dependency |
| `app/core/security.py` | JWT + bcrypt |
| `rules/labor_mandatory.yaml` | Main rule file for Labor Law 14/2025 |
| `rules/labor.yaml` | General labor rules |
| `rules/labor_cross_border.yaml` | UAE/MOHRE rules |
| `LFM2.5-1.2B-Instruct/model.safetensors` | 2.23 GB LFM model weights |
| `model_ML/law_aware_multilabel_model.joblib` | Trained ML classifier (10.2 MB) |
| `Legal Rag/chroma_db/` | ChromaDB persistent vector store (154 docs) |
| `chunks/labor_law_chunks.cleaned.jsonl` | FAISS source — 298 law chunks |
| `legalai.db` | SQLite database (runtime, not committed) |
| `.env` | All secrets and config flags |
| `hf_deployment/` | HF Spaces deployment artifacts |
| `docs/ARCHITECTURE.md` | Full system design document |
| `docs/DEMO_SCRIPT.md` | Committee demo script with curl commands |

### Flutter

| File | What it does |
|------|-------------|
| `lib/main.dart` | App entry, providers, lifecycle observer |
| `lib/api/api_client.dart` | Base HTTP client (JWT injection, 401 hook, timeout) |
| `lib/api/legato_api.dart` | All typed API methods |
| `lib/config/app_config.dart` | Compile-time API URL + timeout constants |
| `lib/config/runtime_config.dart` | SharedPreferences URL override (demo switching) |
| `lib/providers/auth_provider.dart` | Auth state (user, JWT, session expiry) |
| `lib/screens/home/home_shell.dart` | 6-tab NavigationBar shell |
| `lib/screens/analyze/analyze_screen.dart` | Contract upload + result display |
| `lib/screens/chat/chat_analysis_screen.dart` | LFM document chat (POST /chat/document) |
| `lib/screens/social/feed_screen.dart` | Social feed (posts/likes/comments/share) |
| `lib/screens/social/profile_screen.dart` | Own profile (cinematic SliverAppBar) |
| `lib/screens/features/phase5_screens.dart` | All 12 tool screens |
| `android/app/src/main/AndroidManifest.xml` | Permissions + network security config |

---

## 10. TESTING PROTOCOL — 38-STEP REAL DEVICE CHECKLIST

**Run this on a real Android device before the committee demo. Log PASS/FAIL for each step.**

**Prerequisites:**
1. Backend deployed and healthy (`GET /health` returns ok)
2. APK rebuilt with deployed server URL (`API_BASE_URL=https://nourabulnasr-legato.hf.space`)
3. APK installed on device (`adb install app-release.apk`)
4. Real PDF contract file on the device (a real Arabic employment contract works best)

---

### Auth (steps 1-5)
1. Register with a real email address → verify `[PASS/FAIL]`
2. Receive verification email → paste code → account activated `[PASS/FAIL]`
3. Login with correct credentials → home screen appears `[PASS/FAIL]`
4. Login with wrong password → error message shown (not crash) `[PASS/FAIL]`
5. Logout → login screen appears `[PASS/FAIL]`

### Contract Analysis (steps 6-13)
6. Upload PDF from device storage → 10 min timeout → results appear `[PASS/FAIL]`
7. Upload DOCX → results appear `[PASS/FAIL]`
8. View violation detail → LFM explanation text present (not empty) `[PASS/FAIL]`
9. Save analysis → appears in History tab `[PASS/FAIL]`
10. Open saved analysis from History `[PASS/FAIL]`
11. Delete analysis from History → removed `[PASS/FAIL]`
12. Run Explain Clause on a violation → LFM explanation in Arabic `[PASS/FAIL]`
13. Run Summarize Clauses → summaries appear `[PASS/FAIL]`

### Chat (steps 14-17)
14. Chat Hub → Assistant → send message → Gemini reply appears `[PASS/FAIL]`
15. Document Chat → select analysis from dropdown → send question → LFM reply `[PASS/FAIL]`
16. Ask follow-up question → response references prior context `[PASS/FAIL]`
17. Disconnect network → send message → graceful error (not crash) `[PASS/FAIL]`

### Social (steps 18-24)
18. Create post with tags + category → appears in feed `[PASS/FAIL]`
19. Like a post → count increments → unlike → count decrements `[PASS/FAIL]`
20. Expand comments → add comment → comment appears `[PASS/FAIL]`
21. Tap share → copy link → verify URL format `[PASS/FAIL]`
22. Tap author name → MemberProfileScreen opens `[PASS/FAIL]`
23. Network tab → "Connect" button → changes to "Sent ✓" `[PASS/FAIL]`
24. Network search → search by name → results appear `[PASS/FAIL]`

### Features / Tools (steps 25-32)
25. E-sign: draw signature → submit → success message `[PASS/FAIL]`
26. Risk dashboard: select analysis → risk badge + violations appear `[PASS/FAIL]`
27. Compare contracts: upload two files → comparison text appears `[PASS/FAIL]`
28. Negotiation coach: enter clause text → advice appears `[PASS/FAIL]`
29. Share analysis: tap share → token link shown `[PASS/FAIL]`
30. Deal thread: create thread → add message → message appears `[PASS/FAIL]`
31. Timeline: view timeline entries `[PASS/FAIL]`
32. Admin panel (admin account): view users list `[PASS/FAIL]`

### Profile (steps 33-35)
33. Edit profile (name, title, company) → save → changes reflected `[PASS/FAIL]`
34. Add education entry → appears in profile list `[PASS/FAIL]`
35. Navigate to Skills & Endorsements → screen loads `[PASS/FAIL]`

### Settings (steps 36-38)
36. Settings → change API URL → save → restart app → new URL persists `[PASS/FAIL]`
37. Sign out → login screen appears `[PASS/FAIL]`
38. Sign back in → session resumes, data loads `[PASS/FAIL]`

---

**Known expected delays (not failures):**
- LFM responses (explain-clause, document chat): 30-120 seconds on CPU backend. Expected.
- First backend request after cold start: up to 60 seconds while LFM loads. Expected.
- Phone must not sleep during analysis — WakeLock is active but manual screen-off bypasses it.

---

## 11. DEMO SCRIPT

Full demo script with timing, curl commands, and talking points is at:
**`docs/DEMO_SCRIPT.md`**

**Read it in full before the presentation.** It covers 7 steps in 10-12 minutes:
1. Contract upload + analysis (2 min) — core pipeline demo
2. Clause tools (2 min) — explain, summarize, risk
3. Document chat (1 min) — LFM vs Gemini boundary
4. Social features (1.5 min) — feed, post, network
5. Negotiation coach (1 min) — Gemini + LFM fallback
6. Admin: law RAG update (1.5 min) — zero-downtime rebuild
7. Auth + profiles (1 min) — JWT, endorsements

**Key talking points the committee will remember:**
- End-to-end pipeline: one PDF → OCR → rules → ML → RAG 452 docs → 2.2GB local LFM. All on Egyptian Law 14/2025.
- `needs_review` flag: auto-set on any `severity="error"` violation. PATCH endpoint lets a lawyer close the flag. Human-in-the-loop by design.
- LFM/Gemini boundary: LFM cannot invent citations because it only sees retrieved law text. Gemini never touches legal reasoning.
- Social feed in 6 queries: `_batch_post_stats()` eliminates N+1 for any feed size.
- Live law update: admin uploads new law PDF, system rebuilds RAG in background, no downtime.

**Pre-demo checklist (run before committee enters):**
```bash
# 1. Start backend (if not running)
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# 2. Wait for all 6 [OK] lines

# 3. Warm up LFM (first call is slow — do this early)
TOKEN=$(curl -s -X POST http://127.0.0.1:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@legato.com","password":"LegatoDemo2026!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://127.0.0.1:8001/legato/explain-clause \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"clause_text":"مدة العقد سنة واحدة.","language":"ar"}'
# Wait for response — this primes LFM in memory. All subsequent calls are fast.

# 4. Verify health
curl http://127.0.0.1:8001/health
```

---

## 12. CREDENTIALS AND ACCESS

**DO NOT put actual secrets here.** This section tells you WHERE to find them.

| Secret | Where to get it |
|--------|----------------|
| `SECRET_KEY` | Already set in `.env` (64-char hex). For HF deployment: copy same value to HF Secrets. |
| `GEMINI_API_KEY` | `aistudio.google.com` → Create API key. Paste into `.env` and HF Secrets. Ask Nour if an existing key is available. |
| `HF_TOKEN` | `huggingface.co/settings/tokens` → New token (Write access). Needed to push to HF Spaces. Nour's account: `nourabulnasr`. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google Cloud Console → APIs & Credentials. Already set in `.env`. Ask Nour. |
| GitHub repo | `https://github.com/nourabulnasr/GP-Legal-AI-` — branch `final_80%` |
| Flutter repo | `C:\dev\legato_mobile1` — git initialized locally, remote NOT set yet (ask Nour for where to push) |
| HF Spaces URL | `https://huggingface.co/spaces/nourabulnasr/legato` (once created) |
| Firebase project | Not created yet — needed for Flutter Web hosting. Create at `console.firebase.google.com`. |
| Oracle Cloud | Not created yet — alternative to HF for backend. Free Ampere A1 tier (4 OCPUs, 24GB RAM). |

**`.env` file location:** `C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\.env`  
**Never commit `.env` to git.** It is already in `.gitignore`.

---

## 13. TROUBLESHOOTING

### LFM shows "[LFM not loaded]" or first request takes 60+ seconds
`WARMUP_LFM_AT_STARTUP=1` is set in `.env`. On a cold start, the model loads in background — first LFM request waits for it. Watch for `[OK] LFM warmup: model loaded and ready.` in the server log. All subsequent calls are fast (model stays in memory).

### Gemini quota error (429) on chat
`DOCUMENT_CHAT_GEMINI_FALLBACK=0` is set — document chat uses LFM only. Negotiation chat falls back to LFM automatically. General assistant (`/chat/assistant`) returns a clear quota message. Gemini retries 3× with 2s/4s linear backoff before returning error.

### ChromaDB shows 0 hits / RAG not working
`ENABLE_STARTUP_RAG=1` is set. If RAG still shows no hits after startup: trigger a manual reindex as admin:
```bash
curl -X POST http://127.0.0.1:8001/admin/law/reindex \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```
Poll `GET /admin/law/jobs/{job_id}` until `"status":"done"`.

### Flutter app: API errors on every call
Check that the backend is running and the `API_BASE_URL` matches the backend's host + port. Use the Settings screen runtime override to update without rebuilding. Backend port is `8001` (not `8002`).

### Flutter app: file picker fails on Android 13+
All required permissions are in `AndroidManifest.xml` as of commit 2026-05-09. If still failing, verify the device is Android 13+ and check that the app was installed from the latest APK.

### DB missing tables on startup
`app/db/init_db.py` calls `Base.metadata.create_all()` at startup. All 16 tables are created automatically. If schema changes were made: delete `legalai.db` and restart. Demo data will need to be re-seeded.

### Port conflict (8001 in use)
```bash
uvicorn app.main:app --port 8002
```
Update Flutter's `API_BASE_URL` via Settings screen or rebuild the APK.

### `legalai_backup.db` in project root
Already in `.gitignore`, already confirmed not committed. Safe to delete or ignore.

### `GP-Legal-AI-/` folder in project root
This is an old 63MB copy of the repo nested inside itself. It is untracked by git (not committed). Add to `.gitignore` and delete:
```
# Add to .gitignore:
GP-Legal-AI-/
```

---

## 14. CONTACT

**Nour Abulnasr** — developer, designer, primary point of contact.

Ask Nour about:
- The actual graduation submission deadline (committee date)
- All secrets: GEMINI_API_KEY, GOOGLE_CLIENT_ID/SECRET, HF credentials
- The marketing video (he owns the Higgsfield workflow)
- The Flutter GitHub remote URL (Flutter repo is local-only, no remote set yet)
- Any legal logic questions (rule severity, law article applicability)
- The MIU submission requirements (forms, portal, format)
- The `.env` file values for deployed server

---

*HANDOVER.md last updated: 2026-05-11. Branch: `final_80%`. Commits on GitHub: through `fe91a75`.*  
*Flutter project: `C:\dev\legato_mobile1`. Flutter commits: local only (no remote set).*
