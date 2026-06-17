# Legato — AI-Powered Legal Professional Ecosystem for Egypt

Graduation project at **Misr International University (MIU)**.  
Legato is a three-layer legal AI platform built around Egyptian Labor Law 14/2025, combining automated contract analysis, professional networking, and a toolkit for practicing lawyers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│                                                                      │
│   Flutter Mobile App (42 Dart files, 38+ screens)                   │
│   React / Vite Web Frontend                                          │
└───────────────┬─────────────────────────────────────────────────────┘
                │ HTTPS / REST
┌───────────────▼─────────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                           │
│   73 endpoints  ·  7 routers  ·  JWT + Google OAuth                 │
│                                                                      │
│  /auth/*  /analyses/*  /chat/*  /legato/*  /api/*  /admin/law/*     │
└──────┬──────────────┬──────────────┬──────────────────┬─────────────┘
       │              │              │                  │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐ ┌──────▼──────────────┐
│ RULE ENGINE  │ │ ML ASSIST   │ │ RAG LAYER  │ │ LLM EXPLANATION     │
│              │ │             │ │            │ │                     │
│ YAML rules   │ │ joblib      │ │ ChromaDB   │ │ LFM2.5-1.2B         │
│ Labor Law    │ │ multilabel  │ │ 154 docs   │ │ (local, on-device)  │
│ 14/2025      │ │ classifier  │ │            │ │                     │
│ (authoritative│ │ (probability│ │ FAISS      │ │ Gemini 2.0 Flash   │
│ source of    │ │  scores)    │ │ 298 docs   │ │ (cloud fallback)    │
│ truth)       │ │             │ │ in-memory  │ │                     │
└──────────────┘ └─────────────┘ └────────────┘ └─────────────────────┘
                                                        │
┌───────────────────────────────────────────────────────▼─────────────┐
│                         DATA LAYER                                   │
│                                                                      │
│  SQLite  ·  SQLAlchemy ORM  ·  16 tables                            │
│  laws/   ·  rules/   ·  chunks/   ·  ml/artifacts/                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Product Layers

| Layer | Description |
|-------|-------------|
| **1 — Contract Analysis** | Upload PDF/DOCX/image contracts. OCR extracts text, the rule engine checks against Labor Law 14/2025, ML attaches probability scores, RAG retrieves supporting articles, and LFM generates grounded Arabic/English explanations. |
| **2 — Legal Networking** | Professional social feed, connection requests, skill endorsements, and peer recommendations — LinkedIn-style networking for Egypt's legal community. |
| **3 — Power Tools** | 12 tools for legal professionals: clause explainer, risk summary, contract comparison, negotiation chat, deal threads, timeline events, e-signature, document sharing, and more. |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11 |
| Database | SQLite via SQLAlchemy ORM (16 tables) |
| Local LLM | LFM2.5-1.2B-Instruct (on-device) |
| Cloud LLM | Google Gemini 2.0 Flash |
| Vector DB | ChromaDB (154 docs) + FAISS (298 docs, in-memory) |
| OCR | Google Document AI → PyMuPDF → Tesseract (fallback chain) |
| Auth | JWT + Google OAuth 2.0 + bcrypt |
| Mobile | Flutter (42 Dart files, 38+ screens) |
| Web | React + Vite |
| Deployment | HuggingFace Spaces (Docker) |

---

## Analysis Pipeline — `POST /ocr_check_and_search`

```
File Upload (PDF / DOCX / Image)
        │
        ▼
[ 1 ] OCR Extraction
      Google Document AI → PyMuPDF → Tesseract (fallback chain)
        │
        ▼
[ 2 ] Arabic Text Normalization
      Diacritic removal, common OCR error correction, unicode normalization
        │
        ▼
[ 3 ] Clause Splitting
      Sentence-boundary detection, minimum clause length filtering
        │
        ▼
[ 4 ] Cross-Border Detection
      UAE/MOHRE signal matching, governing-law jurisdiction check
        │
        ▼
[ 5 ] Rule Engine  ◄── YAML rules / Labor Law 14/2025  (authoritative)
      labor_mandatory.yaml + labor.yaml + labor_cross_border.yaml
        │
        ▼
[ 6 ] ML Assist
      joblib multilabel classifier — attaches probability scores,
      used as fallback when rule engine produces no hits
        │
        ▼
[ 7 ] RAG Retrieval
      ChromaDB primary → FAISS fallback
      Query boosted with rule descriptions + matched text
        │
        ▼
[ 8 ] LFM2.5 Explanation
      Grounded in retrieved law articles (top-k violations only)
        │
        ▼
[ 9 ] Structured Response
      rule_hits · labor_summary · cross_border_summary ·
      rag_by_violation · needs_review · pipeline_steps
```

---

## LFM / Gemini Boundary

| Endpoint | Primary | Fallback |
|----------|---------|---------|
| `/legato/explain-clause` | LFM (local) | — |
| `/legato/summarize-clauses` | LFM (local) | — |
| `/legato/compare` | LFM (local) | — |
| `/chat/document` | LFM (local) | — |
| `/legato/negotiation-chat` | Gemini (cloud) | LFM (local) |
| `/chat/assistant` | Gemini (cloud) | — |
| `/chat/message` | Gemini (cloud) | — |

---

## API Reference — 73 Endpoints, 7 Routers

| Router | Prefix | Endpoints | Responsibility |
|--------|--------|-----------|----------------|
| Auth | `/auth` | 11 | JWT login, Google OAuth, email verify, password reset |
| Analyses | `/analyses` | 9 | History CRUD, `needs_review` flag, lawyer notes |
| Chat | `/chat` | 4 | Assistant, threaded messages, document Q&A |
| Legato Power Tools | `/legato` | 18 | Explain, summarize, compare, risk, share, deal threads, timeline, signature, profile, negotiation |
| Social Networking | `/api` | 25 | Feed, posts, likes, comments, network invites, endorsements, recommendations, profile documents |
| Admin Law | `/admin/law` | 6 | Law PDF upload, corpus update, RAG rebuild (no downtime) |
| Core | `/` | — | `GET /health`, `POST /ocr_check_and_search`, `POST /check_clause`, `POST /ocr` |

Interactive docs: `http://localhost:8001/docs` (Swagger UI)

---

## Database Schema — 16 Tables

```
users                     analyses                  legato_shares
legato_deal_threads       legato_deal_messages      legato_timeline_events
legato_profiles           legato_signatures
social_posts              social_post_likes         social_post_comments
social_post_shares        network_invites           skill_endorsements
profile_recommendations   profile_user_documents
```

---

## Running Locally

**Prerequisites:** Python 3.11+, LFM2.5-1.2B-Instruct model at `./LFM2.5-1.2B-Instruct/`

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Required: GEMINI_API_KEY
# Optional: HF_TOKEN, SMTP_HOST, SMTP_USER, SMTP_PASSWORD,
#           GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
```

**Startup flags:**

| Variable | Value | Effect |
|----------|-------|--------|
| `WARMUP_LFM_AT_STARTUP` | `1` | LFM model loads in a background thread — eliminates cold-start delay on first request |
| `ENABLE_STARTUP_RAG` | `1` | ChromaDB + FAISS warm in background — API accepts requests immediately |

```bash
# Start the backend
WARMUP_LFM_AT_STARTUP=1 ENABLE_STARTUP_RAG=1 uvicorn app.main:app --reload --port 8001

# Seed demo data (optional)
python hf_deployment/seed_demo_data.py
```

**Demo credentials (after seeding):**

| Role | Email | Password |
|------|-------|---------|
| Admin | `admin@legato.com` | `LegatoAdmin2026!` |
| Demo | `demo@legato.com` | `LegatoDemo2026!` |

---

## Key Implementation Details

**Reliability**
- Gemini API calls use 3-attempt retry with linear backoff (2s, 4s)
- Social feed executes 6 batch queries regardless of feed size — constant query count at any scale
- Rule engine is the authoritative decision source; ML predictor adds probability scores and cannot override rule decisions

**Law Update Pipeline**
- Admin uploads a replacement PDF to `POST /admin/law/upload`
- System extracts articles, rebuilds ChromaDB and FAISS indices in-place
- No downtime — requests in flight are not interrupted during rebuild

**Cross-Border Logic**
- UAE/MOHRE signals matched via 20+ regex patterns
- Egyptian governing-law signals are checked first; Egyptian Labor Law scope stays active unless a foreign jurisdiction is explicitly stated
- Cross-border contracts still receive full Egyptian Labor Law analysis by default

**`needs_review` Flag**
- Automatically set to `true` when any `rule_hit` carries `severity = "error"`
- Stored on the `analyses` table and surfaced in the response for lawyer review queue workflows

---

## HuggingFace Spaces Deployment

Deployment artifacts live in `hf_deployment/`:

| File | Purpose |
|------|---------|
| `Dockerfile` | Container build configuration |
| `README.md` | HuggingFace Space metadata |
| `seed_demo_data.py` | Post-deploy database seeding |
| `.gitattributes` | Git LFS tracking for model weights |

Live space (when active): `https://nourabulnasr-legato.hf.space`

---

## Project Structure

```
GP-Legal-AI--main/
├── app/
│   ├── main.py                  # FastAPI app, startup hooks, core pipeline
│   ├── rules.py                 # Rule engine (YAML loader + text matcher)
│   ├── local_llm.py             # LFM2.5 inference wrapper
│   ├── legato_service.py        # Power tools business logic
│   ├── rag_chromadb.py          # ChromaDB retrieval
│   ├── legal_rag_bridge.py      # RAG facade (Chroma → FAISS fallback)
│   ├── cross_border.py          # UAE/MOHRE signal detector
│   ├── ml_predictor.py          # joblib multilabel classifier
│   ├── routers/
│   │   ├── auth.py              # /auth/*
│   │   ├── analyses.py          # /analyses/*
│   │   ├── chat.py              # /chat/*
│   │   ├── legato_mobile.py     # /legato/*
│   │   ├── social.py            # /api/*
│   │   └── admin_law.py         # /admin/law/*
│   └── db/
│       ├── models.py            # 16 SQLAlchemy table definitions
│       └── session.py           # SQLite engine + session factory
├── rules/
│   ├── labor_mandatory.yaml     # Mandatory field checks (Labor Law 14/2025)
│   ├── labor.yaml               # General labor rules
│   └── labor_cross_border.yaml  # Cross-border / UAE rules
├── laws/
│   ├── raw/                     # Source PDF (Labor Law 14/2025)
│   └── processed/               # Extracted articles (JSON)
├── ml/
│   └── artifacts/               # Trained model, vectorizer, label map (joblib)
├── hf_deployment/               # HuggingFace Spaces deployment files
└── legalai-frontend/            # React/Vite web frontend
```

---

## Team

Built as a graduation project at **Misr International University (MIU)**, Faculty of Computer Science, class of 2026.
