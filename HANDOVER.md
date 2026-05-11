# LEGATO — Project Handover Document

**Date:** 2026-05-11  
**Branch:** `final_80%` on `https://github.com/nourabulnasr/GP-Legal-AI-`  
**Last commit:** `d69cc1c`  
**For:** Teammate picking up where Nour left off  

---

## 1. PROJECT OVERVIEW

Legato is a three-layer AI-powered legal platform built as a graduation project at Misr International University (MIU), Faculty of Computer Science, class of 2026. It is designed specifically for the Egyptian legal market and targets compliance with Egyptian Labor Law No. 14/2025. The platform allows any user — employee, HR manager, or lawyer — to upload an Arabic/English employment contract (PDF, DOCX, or scanned image) and receive: an automatic compliance analysis with law article citations, severity ratings for every violation, RAG-grounded explanations from the actual law text, a professional networking layer (LinkedIn-style for Egypt's legal community), and a toolkit of 12 power tools for legal practitioners. The backend is a Python/FastAPI server; the mobile client is a Flutter app; there is also a React/Vite web frontend.

---

## 2. ARCHITECTURE

Full architecture is documented in `docs/ARCHITECTURE.md`. Summary:

**Layer 1 — AI Contract Intelligence:** A single `POST /ocr_check_and_search` endpoint drives a 10-stage pipeline: OCR (Google DocumentAI → PyMuPDF → Tesseract) → Arabic normalization → clause splitting → cross-border detection → Rule Engine (YAML, authoritative) → ML assist (joblib multilabel) → RAG retrieval (ChromaDB 154 docs + FAISS 298 docs) → LFM2.5-1.2B local explanation → response assembly. The Rule Engine is the truth source; ML adds probability scores; LFM generates grounded explanations (never from Gemini for legal reasoning).

**Layer 2 — Professional Networking:** A social feed, profiles, connection invites, skill endorsements, and peer recommendations — all scoped to Egyptian legal professionals.

**Layer 3 — 12 Power Tools:** Clause explainer, summarizer, contract comparison, risk summary, negotiation coach, deal threads, timelines, e-signatures, shareable analysis links, document chat.

The LFM/Gemini boundary is critical: LFM (local, on-device) handles all legal reasoning (explain, summarize, compare, document chat). Gemini (cloud) handles only conversational assistant and negotiation primary. Mixing them would produce hallucinated legal citations.

---

## 3. WHAT'S DONE

### Backend (FastAPI)
- **Full analysis pipeline** (`POST /ocr_check_and_search`) — OCR, normalization, clause split, cross-border detection, Rule Engine, ML assist, RAG, LFM explanation, structured JSON response
- **73 endpoints across 7 routers** — all implemented and confirmed working
- **6 routers mounted:** auth, analyses, chat, legato_mobile, social, admin_law
- **16 DB tables** — all defined in `app/db/models.py`, created at startup via `init_db.py`
- **Full auth flow** — register + email verify, login, JWT, Google OAuth SSO, forgot-password/reset
- **LFM singleton** — loaded once, reused; 2.23 GB `model.safetensors` at `./LFM2.5-1.2B-Instruct/`
- **LFM speed fixes (2026-05-11, commit 83c4869):** torch.float32 (was float16), max_new_tokens=256 (was 512), Arabic sentinel stripping, context cleanup
- **RAG subsystem** — ChromaDB (`Legal Rag/chroma_db/`, 154 docs) + FAISS in-memory (298 docs from `chunks/`)
- **Rule Engine** — 3 YAML files: `labor.yaml`, `labor_mandatory.yaml`, `labor_cross_border.yaml`
- **ML artifacts** — `model_ML/law_aware_multilabel_model.joblib`, binary model, label JSONL
- **Unified ML artifacts** — `app/ml/artifacts/unified/` (vectorizer, model, thresholds, metrics, config)
- **Social feed N+1 fix** — `_batch_post_stats()` does 6 queries for any feed size (commit e4bd7d3)
- **Gemini 3-attempt retry** — linear backoff 2s/4s on quota errors
- **Negotiation fallback** — LFM activates when Gemini fails or returns empty
- **`used_fallback` field** — `DocumentChatResponse` exposes whether Gemini was used
- **`needs_review` flag** — stored on `analyses` table, set `true` when any rule hit has `severity="error"`; writable via `PATCH /analyses/{id}/flag`
- **SECRET_KEY** — 64-char hex in `.env` (commit e4bd7d3)
- **CORS** — env-driven + `allow_origin_regex` for Flutter Web localhost ports (commit 0fcb513)
- **HF deployment artifacts** — `hf_deployment/` folder: Dockerfile, HF README, seed_demo_data.py, .gitattributes for Git LFS
- **Demo data seeded** — `legalai.db` has `admin@legato.com` / `LegatoAdmin2026!` and `demo@legato.com` / `LegatoDemo2026!`, sample analysis with violations, social posts, deal thread
- **Admin Law RAG rebuild** — zero-downtime corpus update via `POST /admin/law/upload-pdf`
- **Cross-border detection** — UAE/MOHRE signal matching, alternate Egyptian law scope

### Documentation
- `README.md` — architecture diagram, API reference, run instructions, demo credentials
- `docs/ARCHITECTURE.md` — full system design, DB schema diagrams, LFM/Gemini boundary
- `docs/DEMO_SCRIPT.md` — 10-minute committee demo with curl commands, talking points, backup plan
- `LEGATO_AGENT_CONTEXT.md` — master engineering context for any AI agent working on this project
- `LEGATO_BACKEND_TASKS.md` — full backend audit with remaining task list

### Mobile (Flutter)
- 42 Dart files, 38+ screens
- All API calls are real (no mock data)
- Auth flow, all 12 Legato tools, social layer, admin panel implemented
- See `LEGATO_AGENT_CONTEXT.md` for Flutter task list

---

## 4. WHAT'S NOT DONE

### 3 Backend Polish Items (low risk, non-blocking)
1. **B-P1:** `updated_at` field missing from `Analysis` model — only `created_at` exists. Simple SQLAlchemy column addition in `app/db/models.py`.
2. **B-P2:** `avatar_url` not a top-level field in `/api/profile/{user_id}` response — frontend shows placeholder avatars. Fix: standardize in `app/routers/social.py`.
3. **B-P3:** ~~`legalai_backup.db` not in `.gitignore`~~ — **DONE** (already in `.gitignore`, confirmed 2026-05-11). `legalai_backup.db` is present locally and correctly ignored.

### HF Spaces Deployment (Nour's hands — ~10-30 min upload)
The deployment files are ready in `hf_deployment/`. The actual push to HuggingFace has NOT been done because it requires uploading the 2.23 GB LFM model via Git LFS, which takes 10–30 minutes. This must be done by Nour manually (requires HF credentials). See Section 6.

### Flutter Items
Several Flutter polish items remain — see `LEGATO_AGENT_CONTEXT.md` Flutter task list for full details.

---

## 5. HOW TO RUN LOCALLY

**Prerequisites:** Python 3.11+, the `LFM2.5-1.2B-Instruct/` folder at project root (2.23 GB).

```bash
# 1. Navigate to project
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# .env is already present and configured.
# Key values already set: SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES=720,
# WARMUP_LFM_AT_STARTUP=1, ENABLE_STARTUP_RAG=1, DOCUMENT_CHAT_GEMINI_FALLBACK=0

# 4. Start the backend
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# Wait for these lines in the console:
# [OK] LFM warmup: model loaded and ready.
# [OK] ChromaDB RAG ready: 154 docs.
# [OK] Retriever initialized with 298 docs.
# [OK] All 6 routers mounted

# 5. Verify health
curl http://127.0.0.1:8001/health
# Expected: {"status":"ok"}

# 6. Swagger UI (full API browser)
# Open: http://127.0.0.1:8001/docs

# 7. Seed demo data (optional — already seeded in legalai.db)
python hf_deployment/seed_demo_data.py
```

**Demo credentials:**

| Role  | Email               | Password          |
|-------|---------------------|-------------------|
| Admin | admin@legato.com    | LegatoAdmin2026!  |
| Demo  | demo@legato.com     | LegatoDemo2026!   |

---

## 6. HOW TO DEPLOY TO HUGGINGFACE SPACES

The deployment artifacts are in `hf_deployment/`. The actual push has not been done yet.

```bash
# STEP 1: Create the HF Space
# Go to: https://huggingface.co/new-space
# Name: legato
# Owner: nourabulnasr
# SDK: Docker
# Hardware: CPU Basic (free) or T4 Small (GPU, paid)

# STEP 2: Clone the empty Space
git clone https://huggingface.co/spaces/nourabulnasr/legato hf_space
cd hf_space

# STEP 3: Copy deployment files
# From project root — copy these INTO hf_space/:
cp -r ../hf_deployment/* .             # Dockerfile, README.md (HF metadata), seed_demo_data.py, .gitattributes
cp -r ../app .
cp -r ../rules .
cp -r ../chunks .
cp -r ../requirements.txt .
cp -r "../Legal Rag" .
cp ../model_ML/*.joblib .
cp -r ../LFM2.5-1.2B-Instruct .        # 2.23 GB — this is the slow upload

# STEP 4: Set up Git LFS for large files
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.joblib"
git lfs track "*.pt"
git add .gitattributes

# STEP 5: Commit and push (the push will take 10–30 min for the LFM model)
git add .
git commit -m "Initial Legato deployment"
git push
# Monitor upload progress — it will show percentage for each LFS file

# STEP 6: Set HF Secrets (in Space Settings → Repository secrets)
# SECRET_KEY          = <same 64-char hex from your .env>
# GEMINI_API_KEY      = <your Gemini API key from aistudio.google.com>
# HF_TOKEN            = <your HuggingFace token>
# ENABLE_STARTUP_RAG  = 1
# WARMUP_LFM_AT_STARTUP = 1
# DOCUMENT_CHAT_GEMINI_FALLBACK = 0
# ACCESS_TOKEN_EXPIRE_MINUTES = 720

# STEP 7: Wait for the Space to build (5–15 min)
# Monitor at: https://huggingface.co/spaces/nourabulnasr/legato

# STEP 8: Seed demo data
# Once the Space is live:
curl -X POST https://nourabulnasr-legato.hf.space/seed_demo  # if endpoint exists
# OR: run seed_demo_data.py with API_BASE_URL pointed at the HF Space URL
```

The live URL (once deployed): `https://nourabulnasr-legato.hf.space`

---

## 7. KEY FILES MAP

| What | Where |
|------|-------|
| **Entry point** | `app/main.py` |
| **Core pipeline endpoint** | `app/main.py` — `POST /ocr_check_and_search` |
| **LFM model** | `./LFM2.5-1.2B-Instruct/model.safetensors` (2.23 GB) |
| **LFM wrapper** | `app/local_llm.py` (singleton load, generate, sentinel stripping) |
| **RAG: ChromaDB** | `Legal Rag/chroma_db/` (persistent, 154 docs) |
| **RAG: FAISS chunks** | `chunks/labor_law_chunks.cleaned.jsonl` + `chunks/labor_law_chunks.jsonl` |
| **RAG bridge** | `app/legal_rag_bridge.py` → `app/rag_chromadb.py` → `app/rag_utils.py` |
| **Rule YAMLs** | `rules/labor_mandatory.yaml`, `rules/labor.yaml`, `rules/labor_cross_border.yaml` |
| **ML artifacts (multilabel)** | `model_ML/law_aware_multilabel_model.joblib` (10.2 MB) |
| **ML artifacts (binary)** | `model_ML/law_aware_binary_model.joblib` (3.3 MB) |
| **ML unified artifacts** | `app/ml/artifacts/unified/` (vectorizer, model, thresholds, config, metrics) |
| **DB file** | `legalai.db` (SQLite, project root) |
| **DB models** | `app/db/models.py` (16 SQLAlchemy tables) |
| **Auth secrets** | `.env` — `SECRET_KEY`, `GEMINI_API_KEY`, `GOOGLE_CLIENT_ID/SECRET` |
| **All routers** | `app/routers/auth.py`, `analyses.py`, `chat.py`, `legato_mobile.py`, `social.py`, `admin_law.py` |
| **HF deployment files** | `hf_deployment/` — Dockerfile, README.md, seed_demo_data.py |
| **Web frontend** | `legalai-frontend/legalai-frontend/` (React + Vite) |
| **Demo script** | `docs/DEMO_SCRIPT.md` |
| **Architecture doc** | `docs/ARCHITECTURE.md` |

---

## 8. TROUBLESHOOTING

### "LFM not loaded" on first request
LFM takes 30–120 seconds to load from disk on first call. Verify `WARMUP_LFM_AT_STARTUP=1` in `.env`. Watch the server log for `[OK] LFM warmup: model loaded and ready.` If not set, the first request to `/legato/explain-clause` will block for up to 2 minutes.

### Gemini quota errors ("429")
Set `DOCUMENT_CHAT_GEMINI_FALLBACK=0` (already set in `.env`). Document chat uses LFM only. Negotiation chat falls back to LFM automatically. The Gemini helper retries 3 times with linear backoff before returning an error.

### ChromaDB not initialized
Set `ENABLE_STARTUP_RAG=1` in `.env` (already set). If RAG shows no hits, run `POST /admin/law/reindex` as admin to rebuild both ChromaDB and FAISS from the JSONL files in `chunks/`.

### DB missing tables on startup
`app/db/init_db.py` calls `Base.metadata.create_all()` at startup — all 16 tables are created automatically. If schema changes were made, delete `legalai.db` and restart to get a clean schema.

### Port conflict
Default port is `8001`. If port is busy: `uvicorn app.main:app --port 8002`. Update `CORS_ORIGINS` in `.env` if the frontend URL changes.

### Docker not running
Docker Desktop must be open (whale icon stable). The project can run without Docker — just use `uvicorn` directly.

### `legalai_backup.db` confusion
`legalai_backup.db` is in `.gitignore` and not committed. The authoritative DB is `legalai.db`. The backup is a stale local file, safe to ignore or delete.

### Flutter Web CORS
CORS is configured with `allow_origin_regex` covering all `http://localhost:<port>` origins. If Flutter Web still gets CORS errors, check that `CORS_ORIGINS` in `.env` includes the Flutter Web dev server URL explicitly.

---

## 9. CONTACT NOTES

**Developer:** Nour Abulnasr — MIU Computer Science / AI track, class of 2026.

Things to ask Nour about:
- The `.env` file — `GEMINI_API_KEY` and `SECRET_KEY` values (do not commit or share)
- The `GP-Legal-AI-/` folder in the project root — it is an old nested copy of the repo (63 MB). Safe to delete or add to `.gitignore`.
- The Flutter app source — Nour has the Flutter project separately. The mobile app connects to the backend via `API_BASE_URL` build flag.
- The HuggingFace account (`nourabulnasr`) — credentials needed for the actual HF Spaces push.
- The Google OAuth credentials — `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`.
- The examination committee date — target everything at that window.

---

*HANDOVER.md generated: 2026-05-11. Branch: final_80%. Commit: d69cc1c.*
