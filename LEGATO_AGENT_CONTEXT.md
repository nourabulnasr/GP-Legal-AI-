# LEGATO — MASTER ENGINEERING CONTEXT
# Read this completely before touching any file. This is the single source of truth.

---

## WHO YOU ARE

You are the lead engineering agent for Legato — a graduation project at Misr International University (MIU), Faculty of Computer Science, AI track. The developer is Nour (21, senior CS/AI student). This is not a toy project. The goal is to ship the most professional legal AI application built by any CS student in Egypt. Every decision you make should reflect that standard.

You have access to two audit files in this repo:
- `LEGATO_BACKEND_TASKS.md` — complete backend audit with all critical/medium/polish items
- `LEGATO_FLUTTER_TASKS.md` — complete Flutter audit with all critical/medium/polish items

Read both files before starting any work session.

---

## WHAT LEGATO ACTUALLY IS — THE REAL VISION

Legato is not just a contract checker. It is a **legal professional ecosystem for Egypt** — the intersection of three things that have never been combined properly in the Egyptian market:

### Layer 1 — AI-Powered Contract Intelligence
Egyptian employment contracts are bilingual (Arabic + English), inconsistent, and manually reviewed by HR departments and employees who often don't know their rights under Egyptian Labor Law 14/2025. Legato solves this with a pipeline that:
- Accepts any contract format (PDF, DOCX, image/scanned)
- Extracts text via OCR (Document AI > PyMuPDF > Tesseract fallback)
- Normalizes Arabic + removes OCR noise
- Splits into clauses
- Detects cross-border contracts (UAE, MOHRE signals) and applies the right law scope
- Runs the **Rule Engine** (authoritative — YAML rules for Labor Law 14/2025) → returns structured violations with law article references
- Runs **ML assist** (trained on rule engine outputs — adds probability scores, improves recall)
- Runs **RAG retrieval** (ChromaDB + FAISS over the actual law corpus) → grounds explanations in real law text
- Runs **LFM2.5-1.2B** (local, runs on-device/server) → generates natural language explanations GROUNDED IN retrieved law text. This model CANNOT be replaced by Gemini for legal reasoning. It is the reasoning layer.
- Uses **Gemini** ONLY for general conversational chat (assistant, not legal reasoning)

The output is: structured violations, law citations, severity levels, RAG-grounded explanations, summaries, and a needs_review flag when human lawyer review is warranted.

### Layer 2 — Professional Networking (LinkedIn for Legal Egypt)
Legal professionals, HR managers, employees, and lawyers need a professional network specific to their domain. Legato's social layer includes:
- Feed (posts, likes, comments, shares)
- Professional profiles (experience, education, skills)
- Endorsements and recommendations
- Connection invitations
- Document sharing between professionals
- All scoped to the Egyptian legal/employment space

### Layer 3 — 12 Power Tools for Professionals
Beyond the core analysis, Legato provides:
1. Contract analysis (full pipeline)
2. Clause check (single clause risk)
3. Explain clause (LFM + RAG, grounded)
4. Summarize clauses (LFM)
5. Compare two contracts (LFM)
6. Negotiation coach (Gemini primary, LFM fallback)
7. Risk summary (risk score visualization)
8. Share analysis (public tokenized link)
9. Deal threads (discussion per contract)
10. Timeline (case/deal milestone tracking)
11. Signatures (e-sign acknowledgement)
12. Document chat (Q&A over contract using LFM + RAG)

### The Auto-Update RAG Pipeline (Unique Differentiator)
When Egyptian Labor Law is officially updated, an admin uploads the new official PDF. The system:
1. Extracts and structures articles from the PDF
2. Preprocesses into JSONL chunks
3. Rebuilds ChromaDB vector index
4. Reloads in-memory FAISS retriever
All without downtime. This is working and confirmed by the examiner as a strength.

---

## COMPLETE SYSTEM ARCHITECTURE

### Backend (FastAPI + Python)
```
Entry: app/main.py
Core pipeline endpoint: POST /ocr_check_and_search

Pipeline execution order:
1. File ingestion → app/pdf_text_pipeline.py (DocAI > PyMuPDF > Tesseract)
2. Text normalization → app/main.py._normalize_contract_text()
3. Clause splitting → app/utils_text.py.split_into_clauses()
4. Cross-border detection → app/cross_border.py.detect_cross_border()
5. Labor applicability gate → app/main.py._labor_applicability()
6. Rule Engine → app/rules.py (labor.yaml + labor_cross_border.yaml + labor_mandatory.yaml)
7. ML assist → app/model_ml_predictor.py (law_aware_multilabel_model.joblib)
8. RAG search → app/main.py._rag_search_safe()
   └── primary: app/legal_rag_bridge.py (Legal Rag/src)
   └── fallback: app/rag_chromadb.py
   └── fallback: app/rag_utils.py (FAISS in-memory)
9. LFM explanations → app/local_llm.py (LFM2.5-1.2B-Instruct, singleton)
10. Response assembly → structured JSON with rule_hits, summaries, RAG hits, explanations

Routers:
- app/routers/auth.py → /auth/* (JWT + Google OAuth + email verify + password reset)
- app/routers/analyses.py → /analyses/* (saved history CRUD)
- app/routers/chat.py → /chat/* (Gemini assistant + LFM document chat)
- app/routers/legato_mobile.py → /legato/* (12 tools)
- app/routers/social.py → /api/* (networking)
- app/routers/admin_law.py → /admin/law/* (law corpus + RAG rebuild)

Database: SQLite via SQLAlchemy
Auth: JWT (python-jose) + bcrypt (passlib) + Google OAuth
```

### LFM / Gemini Boundary (CRITICAL — never violate this)
```
LFM2.5-1.2B (local, on-server reasoning):
  - /legato/explain-clause
  - /legato/summarize-clauses
  - /legato/compare
  - /chat/document (primary)
  - /legato/negotiation-chat (fallback when Gemini fails)

Gemini 2.0 Flash (cloud, conversational):
  - /chat/assistant (general questions)
  - /chat/message (analysis-scoped general chat)
  - /legato/negotiation-chat (primary)
  - /chat/document (ONLY if DOCUMENT_CHAT_GEMINI_FALLBACK=1 AND LFM fails)

This boundary exists because:
- LFM is grounded in retrieved law text — it cannot invent citations
- Gemini is for conversation — it has no law grounding
- Mixing them in legal reasoning would produce hallucinated legal advice
- For the examination: set DOCUMENT_CHAT_GEMINI_FALLBACK=0 so failures are explicit
```

### Flutter Mobile App
```
Stack: Flutter + Provider + http package
Auth: JWT via SharedPreferences + Bearer header
API: configurable base URL via --dart-define=API_BASE_URL=

42 Dart files organized:
- lib/screens/auth/ → login, register, forgot password, email verify
- lib/screens/home/ → home_shell (5-tab nav), dashboard_tab
- lib/screens/analyze/ → contract upload + analysis
- lib/screens/history/ → saved analyses + detail view
- lib/screens/chat/ → assistant + document chat
- lib/screens/social/ → feed, network, profile, skills, endorsements, documents, recommendations, alerts
- lib/screens/features/ → all 12 Legato tools (phase5_screens.dart)
- lib/screens/admin/ → admin panel
- lib/screens/settings/ → settings + sign out
- lib/api/ → api_client.dart + legato_api.dart
- lib/providers/ → auth_provider.dart
- lib/config/ → app_config.dart (base URL)
```

### Web Frontend (React/Vite)
```
Located: legalai-frontend/legalai-frontend/
Pages:
- LoginPage.tsx, RegisterPage.tsx, GoogleCallbackPage.tsx
- AnalyzePage.tsx (5 tabs: Summary/Violations/OCR/RAG hits/Document-chat)
- HistoryPage.tsx (saved analyses)
- ChatPage.tsx (general + contract modes)
- AdminPage.tsx (users + analyses + Law RAG tab)
API client: src/lib/api.ts (axios + Bearer token)
```

---

## CONFIRMED WORKING — DO NOT BREAK THESE

Every item below was verified in the audit. Do not refactor or touch them unless specifically fixing a listed issue:

- LFM2.5-1.2B real safetensors (2.23GB) at ./LFM2.5-1.2B-Instruct/ — loaded as singleton ✓
- All ML artifacts (joblib files + evidence JSONL) ✓
- All rule YAML files (labor.yaml + labor_cross_border.yaml + labor_mandatory.yaml) ✓
- All JSONL chunks for RAG ✓
- All DB models mapped (16 tables) ✓
- All 6 routers mounted ✓
- Complete auth flow (register/login/verify/reset/Google OAuth) ✓
- All 12 Legato tool endpoints implemented ✓
- Social networking full CRUD (posts/likes/comments/profiles/connections/endorsements) ✓
- Admin Law RAG rebuild pipeline ✓
- Cross-border detection pipeline ✓
- LFM/Gemini boundary correct ✓
- Flutter: 38 screens confirmed working (see LEGATO_FLUTTER_TASKS.md ✅ section) ✓
- Flutter: All API calls are real (no mock data) ✓

---

## WHAT IS ACTUALLY LEFT — COMPLETE TASK LIST

### 🔴 CRITICAL (must be done before any demo or deployment)

**BACKEND — estimated 8 hours total:**

B-C1: .env configuration (10 min)
```
GEMINI_API_KEY=<real key from aistudio.google.com>
SECRET_KEY=<random 64-char string>
ENABLE_STARTUP_RAG=1
ACCESS_TOKEN_EXPIRE_MINUTES=720
DOCUMENT_CHAT_GEMINI_FALLBACK=0
HF_TOKEN=  (clear placeholder)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://YOUR_DOMAIN
```

B-C2: LFM warmup at startup (2 hours)
File: app/main.py startup() event
Problem: LFM not loaded at startup → first request to explain-clause takes 30-120 seconds → demo timeout
Fix: Add background thread calling local_llm.load_model() at startup, same pattern as RAG warmup

B-C3: needs_review field + lawyer_override endpoint (4 hours) — EXAMINER REQUIREMENT
Files: app/db/models.py, app/main.py, app/routers/analyses.py
Add to Analysis model: needs_review: bool = False, lawyer_note: str = None
Add to OCRResponse: needs_review: bool
Logic: set needs_review=True when rule_hits contains severity "error" AND confidence above threshold
New endpoint: PATCH /analyses/{id}/flag with body {needs_review: bool, lawyer_note: str}
This is the #1 item the examiner specifically requested.

B-C4: Fix duplicate _normalize_contract_text stub (5 min)
File: app/main.py lines 33-36
Remove the stub. The real function is at line 620.

B-C5: Fix LABOR25_EMPLOYER_INFO / LABOR25_EMPLOYEE_INFO severity (30 min)
File: rules/labor_mandatory.yaml
Change severity from "high" to "info" for presence-detection rules
A well-formed contract should NOT show violations for containing employer info.

B-C6: Fix CORS to env-driven (30 min)
File: app/main.py
Replace hardcoded list with: os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

**FLUTTER — estimated 8 hours total:**

F-C1: Android 13+ file permissions (15 min)
File: android/app/src/main/AndroidManifest.xml
Add READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, READ_MEDIA_VISUAL_USER_SELECTED

F-C2: Real device API URL (build flag, no code change)
Command: flutter build apk --dart-define=API_BASE_URL=http://YOUR_SERVER_IP:8000

F-C3: Wire DashboardTab into HomeShell (30 min)
File: lib/screens/home/home_shell.dart
DashboardTab exists but is never mounted. Add as first tab.

F-C4: Session expiry dialog (1 hour)
Files: lib/api/api_client.dart, lib/providers/auth_provider.dart
On 401: show dialog "Session expired" → then redirect. Not silent crash.

F-C5: Fix or remove feed search bar (30 min)
File: lib/screens/social/feed_screen.dart
readOnly search with SnackBar error is unprofessional. Implement or remove.

F-C6: Risk visualization (2 hours)
File: lib/screens/features/phase5_screens.dart — RiskFeatureScreen
Parse response → show risk badge + top 3 violations as visual list

F-C7: Signature canvas (2 hours)
File: lib/screens/features/phase5_screens.dart — EsignFeatureScreen
Add hand_signature package → drawing canvas → base64 → POST /legato/signatures

F-C8: Analysis picker for chat (1 hour)
File: lib/screens/chat/chat_analysis_screen.dart
Replace manual ID input with dropdown loading GET /analyses

F-C9: Fix share URL (30 min)
File: lib/screens/features/phase5_screens.dart — ShareFeatureScreen
Use real server domain, not 10.0.2.2

F-C10: Network security config (1 hour)
File: android/app/src/main/AndroidManifest.xml + res/xml/network_security_config.xml
Scope cleartext to dev IPs only. Remove global usesCleartextTraffic=true.

### 🟡 MEDIUM (examiner will notice)

B-M1: Social feed N+1 query fix → app/routers/social.py _serialize_post()
B-M2: LFM singleton warmup flag → WARMUP_LFM_AT_STARTUP=1 in .env.example
B-M3: Add used_fallback: bool to DocumentChatResponse so examiner can see if Gemini was used
B-M4: Deferred circular imports in legato_service.py → extract to app/rag_helpers.py
F-M1: Remove duplicate VoiceAssistantFeatureScreen from Features Hub grid
F-M2: Profile API camelCase/snake_case alignment (displayName)
F-M3: Add API URL override to Settings screen for demo switching
F-M4: Add global FlutterError.onError handler in main.dart

### 🟢 POLISH (if time permits)

B-P1: Add updated_at field to Analysis model
B-P2: Add avatar_url as top-level field in profile payload
B-P3: legalai_backup.db → move to .gitignore
B-P4: AnalyzePage.tsx save toggle (currently force-saves everything)
F-P1: flutter_secure_storage for JWT (blocked by Windows path issue — fix by symlinking user folder)
F-P2: WakeLock during long analysis (prevents screen/Doze killing request on Android)
F-P3: Remove Biometrics stub from Features grid
F-P4: Loading state persistence across screen rotation

---

## DEPLOYMENT ARCHITECTURE

### Target: Oracle Cloud Always Free Tier (Ampere A1)
- 4 OCPUs + 24GB RAM → enough for LFM2.5-1.2B + FastAPI + ChromaDB + sentence-transformers
- Ubuntu 22.04 LTS
- Free forever (not a trial)
- Docker Compose deployment — the docker-compose.yml in repo is already configured

### Deployment Steps (Day 2)
```bash
# 1. Create Oracle A1 instance at cloud.oracle.com
# 2. SSH into instance
ssh ubuntu@YOUR_ORACLE_IP

# 3. Install Docker
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker ubuntu

# 4. Clone repo
git clone https://github.com/nourabulnasr/GP-Legal-AI-
cd GP-Legal-AI-

# 5. Upload LFM model (from your Windows machine — DO THIS FIRST, it's 2.2GB)
# Run this from your Windows machine:
scp -r "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\LFM2.5-1.2B-Instruct" ubuntu@ORACLE_IP:~/GP-Legal-AI-/

# 6. Configure environment
cp .env.example .env
nano .env  # fill in GEMINI_API_KEY, SECRET_KEY, etc.

# 7. Start
docker compose up --build -d

# 8. Verify
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

### Docker Desktop Issue (Your Local Machine)
Docker Desktop's Linux engine is not running.
Open Docker Desktop application → wait for the whale icon to stop animating → then run docker compose.
OR: You don't need Docker locally — just deploy directly to Oracle and test there.

### Flutter Deployment
```bash
# APK (for real device demo)
flutter build apk --release --dart-define=API_BASE_URL=http://ORACLE_IP:8000

# Flutter Web (for browser demo URL — no app store needed)
flutter build web --dart-define=API_BASE_URL=http://ORACLE_IP:8000
firebase deploy  # free Firebase hosting → you get a legato.web.app URL
```

---

## WHAT "MOST PROFESSIONAL LEGAL AI APPLICATION" MEANS — YOUR STANDARD

Every decision in this project should be evaluated against this:

### Technical Professionalism
- The LFM reasoning must be grounded. Every explanation must cite the specific law article that supports it. No hallucinated citations. The pipeline enforces this: LFM only sees text retrieved from the actual law corpus.
- The rule engine is the authoritative source. ML is assistance. They work in combination, not competition.
- Error handling must be explicit. When a feature fails, the user knows WHY (LFM not loaded vs network error vs invalid input). Silent fallbacks that hide failures are unprofessional.
- The needs_review flag is not just an examiner checkbox. It represents the system's intellectual humility — knowing when a contract is complex enough that a human lawyer should review it. This is core to the product's integrity.

### Product Professionalism
- A user (HR manager, employee, job seeker) should be able to upload a real Egyptian employment contract and understand their legal rights in under 60 seconds.
- The social layer should feel like a specialized LinkedIn — not a generic feed.
- The mobile app should work on a real Android phone without crashes. If it only works in the simulator, it's not a product.

### Demo Professionalism (for the examination committee)
- Every endpoint should respond in under 5 seconds for non-LFM calls
- LFM calls (explain-clause, summarize, compare) should respond in under 30 seconds
- The flow: login → upload contract → get analysis → explain a violation → see law citation → flag for review → should work end-to-end without any manual intervention
- The admin should be able to upload a new law PDF and rebuild the RAG index while the app stays live

---

## OPERATING INSTRUCTIONS FOR THIS AGENT

1. Read this file completely at the start of every session
2. Read LEGATO_BACKEND_TASKS.md and LEGATO_FLUTTER_TASKS.md before touching any file
3. Always verify: does the fix maintain the LFM/Gemini boundary? If not, stop.
4. Always verify: does the fix touch a "CONFIRMED WORKING" item? If yes, test it before and after.
5. Show the fix before applying it. One fix at a time. No batching multiple files in one edit unless explicitly instructed.
6. After each fix: restart the relevant service and verify it works.
7. When in doubt about legal logic (rule severity, law applicability): ask before changing.
8. Target platform: Android real device + Oracle Cloud Ubuntu. Not Windows. Not emulator.
9. Every response should end with: what was done, what's next, and how many critical items remain.

---

## SESSION STARTUP COMMAND

When starting a new Claude Code session on this project:

```
Read LEGATO_AGENT_CONTEXT.md completely.
Then read LEGATO_BACKEND_TASKS.md and LEGATO_FLUTTER_TASKS.md.
Tell me: how many 🔴 CRITICAL items remain unchecked, and what is the single highest-priority item to work on right now.
Then wait for my instruction before touching any file.
```

---

## REMAINING CRITICAL COUNT AT LAST AUDIT (2026-05-08)

Backend: 6 critical, 8 medium, 7 polish
Flutter: 5 critical, 9 medium, 4 polish
Deployment: not started

This number should decrease with every session. Track it.
