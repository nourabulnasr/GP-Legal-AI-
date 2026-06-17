# Legato Legal AI — System Architecture

**Version:** 1.0  
**Date:** 2026-05-10  
**Project:** Graduation Project — Egyptian Labor Law AI Ecosystem  
**Authors:** Legato Engineering Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Analysis Pipeline](#2-core-analysis-pipeline)
3. [LFM / Gemini Boundary](#3-lfm--gemini-boundary)
4. [RAG Architecture](#4-rag-architecture)
5. [Database Schema](#5-database-schema)
6. [Authentication and Security](#6-authentication-and-security)
7. [Performance Decisions](#7-performance-decisions)
8. [Deployment Architecture](#8-deployment-architecture)
9. [API Surface](#9-api-surface)

---

## 1. System Overview

Legato is a three-layer AI legal assistant designed specifically for the Egyptian labor law context. Its primary purpose is to analyze employment contracts for compliance with Egyptian Labor Law No. 14 of 2025, detect cross-border contract patterns (Egypt–UAE), and provide an AI-powered professional networking platform for legal practitioners.

```
+---------------------------------------------------------------+
|                        LEGATO PLATFORM                        |
|                                                               |
|  +------------------+  +----------------+  +---------------+ |
|  |    LAYER 1       |  |    LAYER 2     |  |   LAYER 3     | |
|  |                  |  |                |  |               | |
|  | AI CONTRACT      |  | PROFESSIONAL   |  | 12 POWER      | |
|  | INTELLIGENCE     |  | NETWORKING     |  | TOOLS         | |
|  |                  |  |                |  |               | |
|  | - OCR ingestion  |  | - Social feed  |  | - Deal rooms  | |
|  | - Rule engine    |  | - Profiles     |  | - Timelines   | |
|  | - ML predictor   |  | - Network      |  | - Signatures  | |
|  | - RAG retrieval  |  |   invitations  |  | - Sharing     | |
|  | - LFM explain    |  | - Endorsements |  | - Chat        | |
|  | - Gemini chat    |  | - Recommends   |  | - Negotiation | |
|  +------------------+  +----------------+  +---------------+ |
|                                                               |
|              FastAPI Backend  (app/main.py)                   |
|              SQLite via SQLAlchemy ORM                        |
+---------------------------------------------------------------+
```

### Layer 1: AI Contract Intelligence

The core analytical engine. Accepts a contract document (PDF, image, DOCX), runs it through OCR, normalizes Arabic text, detects cross-border jurisdiction signals, applies the rule engine against Labor Law No. 14/2025, augments results with a trained ML predictor, retrieves grounding evidence from the RAG corpus, and generates clause-level explanations via LFM2.5.

### Layer 2: Professional Networking

A LinkedIn-style feed and profile system scoped to legal professionals. Provides posts with category tags, likes, comments, shares, connection invitations, skill endorsements, and written recommendations. All interactions are persisted in SQLite and served through the `/api/*` router (`app/routers/social.py`).

### Layer 3: 12 Power Tools

Productivity tools that operate over analyzed contracts: deal collaboration threads, contract-event timelines, digital signature capture, shareable analysis links (tokenized), AI document chat (Gemini-powered, scoped to a specific analysis), and AI negotiation assistance.

---

## 2. Core Analysis Pipeline

The primary endpoint is `POST /ocr_check_and_search`. The pipeline executes the following ten stages in sequence:

```
+--------------------------------------------------------------+
|                FULL ANALYSIS PIPELINE                        |
+--------------------------------------------------------------+
|                                                              |
|  [1] FILE INGESTION                                          |
|       pdf_text_pipeline.py                                   |
|       Priority order:                                        |
|         (a) Google DocumentAI (DocumentAI.py)                |
|         (b) PyMuPDF  fitz  (text layer)                      |
|         (c) Tesseract OCR  (image/scanned fallback)          |
|       DOCX: python-docx paragraphs + tables;                 |
|             zipfile media OCR for scanned DOCX               |
|             SHA-256 deduplication on upload                  |
|                            |                                 |
|                            v                                 |
|  [2] ARABIC TEXT NORMALIZATION                               |
|       main.py :: _normalize_contract_text()                  |
|       - unicodedata NFKC normalization                       |
|       - Diacritic stripping                                  |
|       - Arabic-Indic numeral conversion                      |
|       - Common OCR mis-read corrections                      |
|         ("جميه" -> "جنيه", "ديب" -> "دبي", etc.)            |
|       - BIDI control character removal                       |
|                            |                                 |
|                            v                                 |
|  [3] CLAUSE SPLITTING                                        |
|       utils_text.py :: split_into_clauses()                  |
|       - Sentence / paragraph boundary detection              |
|       - Arabic + Latin punctuation aware                     |
|       - Language detection per clause                        |
|                            |                                 |
|                            v                                 |
|  [4] CROSS-BORDER DETECTION                                  |
|       cross_border.py :: detect_cross_border()               |
|       - Regex pattern matching on normalized text            |
|       - Signals: UAE / MOHRE / AED / Emirati law references  |
|       - Output: { enabled: bool, reason, matches[] }         |
|       - If enabled: labor_cross_border.yaml rules activated  |
|                            |                                 |
|                            v                                 |
|  [5] LABOR APPLICABILITY GATE                                |
|       main.py :: _labor_applicability()                      |
|       - Determines whether Labor Law No. 14/2025 applies     |
|       - Checks for employment relationship signals           |
|       - Sets applicability flag for downstream rule routing  |
|                            |                                 |
|                            v                                 |
|  [6] RULE ENGINE                                             |
|       rules.py :: RuleEngine                                 |
|       YAML rule files (rules/ directory):                    |
|         - labor.yaml            (standard labor rules)       |
|         - labor_mandatory.yaml  (mandatory provisions)       |
|         - labor_cross_border.yaml (UAE/MOHRE provisions)     |
|       Each rule: rule_id, law, article, severity,            |
|                  description, rationale, suggestion          |
|       Output: List[RuleHit] with matched_text excerpts       |
|                            |                                 |
|                            v                                 |
|  [7] ML ASSIST                                               |
|       model_ml_predictor.py                                  |
|       Model: model_ML/law_aware_multilabel_model.joblib      |
|              model_ML/law_aware_binary_model.joblib          |
|       Framework: joblib + sklearn (TF-IDF + OvR LogReg)      |
|       Law retrieval: BM25 over labor14_2025_clean.jsonl      |
|         BM25_TOP_K=30, TOP_K_EVIDENCE=6                      |
|       Trained on rule engine output labels                   |
|       Output: per-rule probability scores,                   |
|               clause-level violation risk (0.0-1.0)          |
|                            |                                 |
|                            v                                 |
|  [8] RAG RETRIEVAL                                           |
|       legal_rag_bridge.py (primary: ChromaDB 154 docs)       |
|         -> rag_chromadb.py (ChromaDB access layer)           |
|       rag_utils.py (fallback: FAISS in-memory 298 docs)      |
|       Embeddings: sentence-transformers (HuggingFace)        |
|       Query boosted with rule hit context + law anchors      |
|       Deduplication: by (law, article, id, text[:80])        |
|       Article alignment: hits ranked by rule.article match   |
|                            |                                 |
|                            v                                 |
|  [9] LFM EXPLANATION                                         |
|       local_llm.py :: LFM2.5-1.2B-Instruct (singleton)      |
|       - Loaded once; reused across requests                  |
|       - Device: CPU (or GPU if available)                    |
|       - Input: clause text + retrieved law chunks            |
|       - Output: Arabic/bilingual clause explanation          |
|       - MAX_PROMPT_CHARS = 6000 (prevents OOM)               |
|       - _llm_alignment_with_severity() checks                |
|         explanation tone vs rule severity                    |
|                            |                                 |
|                            v                                 |
| [10] RESPONSE ASSEMBLY                                       |
|       Structured JSON response containing:                   |
|         - rule_hits[]     (rule_id, severity, suggestion)    |
|         - ml_predictions  (per-rule scores)                  |
|         - unified_ml_risk (clause-level float 0-1)           |
|         - rag_hits[]      (retrieved law chunks)             |
|         - lfm_explanations (per-rule LFM text)               |
|         - labor_summary   (structured leave/salary/etc.)     |
|         - cross_border    (detection result)                 |
+--------------------------------------------------------------+
```

### Data Flow Diagram

```
  Client Request (multipart/form-data)
         |
         v
  +------+-------+
  |   FastAPI    |
  |   main.py    |
  +------+-------+
         |
         |-- bytes -------> [pdf_text_pipeline.py]
         |                       |
         |                  raw_text
         |                       |
         |<-- raw_text -----------+
         |
         |-- raw_text ----> [_normalize_contract_text()]
         |                       |
         |               normalized_text
         |                       |
         |<-- normalized_text ----+
         |
         |-- normalized_text --> [split_into_clauses()]
         |                           |
         |-- normalized_text --> [detect_cross_border()]
         |                           |
         |                    cross_border_flag
         |
         |-- clauses + flag --> [_labor_applicability()]
         |                           |
         |-- clauses ----------> [RuleEngine.check()]
         |                           |
         |                      rule_hits[]
         |
         |-- normalized_text --> [model_ml_predictor]
         |                           |
         |                      ml_scores + risk
         |
         |-- rule context -----> [_rag_search_safe()]
         |                           |
         |                      rag_chunks[]
         |
         |-- (clause + chunks) -> [local_llm.explain()]
         |                           |
         |                      lfm_explanations[]
         |
         |-- all above ----------> [Response Assembly]
         |
         v
  Structured JSON Response
```

---

## 3. LFM / Gemini Boundary

This is a critical design decision with direct implications for factual accuracy and hallucination risk.

```
+-----------------------------+-----------------------------+
|     LFM2.5-1.2B (Local)     |     Gemini 2.0 Flash        |
|                             |     (Cloud / Google API)     |
+-----------------------------+-----------------------------+
| explain-clause              | POST /chat/assistant        |
| summarize-clauses           | POST /chat/message          |
| compare (two contracts)     | negotiation-chat (primary)  |
| document-chat (primary)     |                             |
| negotiation-chat (fallback) |                             |
+-----------------------------+-----------------------------+
|                             |                             |
| GROUNDED in retrieved       | CONVERSATIONAL: free-form   |
| law chunks from RAG corpus  | dialogue scoped to the      |
| (ChromaDB / FAISS).         | stored analysis result_json |
|                             |                             |
| Cannot cite articles that   | Suitable for summarization  |
| are not in the retrieved    | and natural language Q&A    |
| context. Hallucinated        | over already-verified data. |
| citations are structurally  |                             |
| prevented.                  | NOT suitable for legal      |
|                             | reasoning or article        |
| Runs entirely on-device.    | citation — Gemini has no    |
| No network dependency for   | access to the law corpus.   |
| legal explanation tasks.    |                             |
+-----------------------------+-----------------------------+
```

### Exam Mode Configuration

```
DOCUMENT_CHAT_GEMINI_FALLBACK=0
```

When this environment variable is set to `0`, the document-chat endpoint does not silently fall back to Gemini if LFM fails. Failures are surfaced explicitly so that the examination committee can observe the on-device model behavior. This prevents the system from masking LFM unavailability during a live demonstration.

### Gemini Retry Strategy

Gemini API calls implement a linear backoff retry strategy to handle quota exhaustion and transient errors gracefully without failing the user request immediately:

```
Attempt 1  --->  (on 429 / quota error)  --->  wait 2s
Attempt 2  --->  (on 429 / quota error)  --->  wait 4s
Attempt 3  --->  final result or error
```

Model selection is configurable via `GEMINI_MODEL` environment variable (default: `gemini-2.0-flash`).

---

## 4. RAG Architecture

The Retrieval-Augmented Generation subsystem provides grounded law text to the LFM model and to the response payload. It operates on a two-store architecture.

```
+-----------------------------------------------------------+
|                   RAG SUBSYSTEM                           |
+-----------------------------------------------------------+
|                                                           |
|  Query                                                    |
|    |                                                      |
|    v                                                      |
|  build_rag_query()  -- rule context enrichment            |
|    |                   law anchor injection               |
|    |                   Arabic normalization               |
|    |                                                      |
|    v                                                      |
|  _rag_search_safe()                                       |
|    |                                                      |
|    +----> LEGAL_RAG_QUERY_BACKEND == "chroma_first"?      |
|    |              (default)                               |
|    |              YES                                     |
|    |               |                                      |
|    |               v                                      |
|    |      legal_rag_bridge.py                             |
|    |      (Bridge to Legal Rag/src)                       |
|    |               |                                      |
|    |               v                                      |
|    |      rag_chromadb.py                                 |
|    |      ChromaDB persistent store                       |
|    |      Collection: egyptian_labor_laws                 |
|    |      Corpus: 154 chunks from Labor Law 14/2025       |
|    |      Embeddings: sentence-transformers               |
|    |      Config: Legal Rag/config.yaml                   |
|    |      Persist dir: Legal Rag/chroma_db                |
|    |               |                                      |
|    |          hits returned?                              |
|    |          NO                                          |
|    |               |                                      |
|    |               v                                      |
|    +----> rag_utils.py :: Retriever                       |
|           In-memory FAISS flat index                      |
|           Corpus: 298 chunks (JSONL files in chunks/)     |
|           Embeddings: hash-based TF-like vectors          |
|           Built at startup from load_chunks_as_docs()     |
|                                                           |
|  Post-retrieval:                                          |
|    _filter_to_labor_only()    -- source/law filter        |
|    _dedupe_rag_hits_by_metadata()  -- (law,art,id,text)   |
|    _rag_prioritize_hits_for_rule_hit() -- article align   |
|    _dedupe_rag_blocks_by_rule()   -- per-violation dedup  |
|                                                           |
+-----------------------------------------------------------+
```

### Vector Store Comparison

| Property           | ChromaDB (Primary)                  | FAISS (Fallback)                      |
|--------------------|-------------------------------------|---------------------------------------|
| Storage            | Persistent on disk                  | In-memory (rebuilt at startup)        |
| Corpus size        | 154 chunks                          | 298 chunks                            |
| Source             | Labor Law 14/2025 via Legal Rag     | JSONL files in chunks/ directory      |
| Embedding model    | sentence-transformers (HuggingFace) | Hash-based TF cosine similarity       |
| Startup behavior   | Loads from chroma_db/ directory     | Rebuilt from JSONL on each startup    |
| Update path        | POST /admin/law/upload-pdf          | Auto-reload after JSONL regeneration  |
| Backend control    | LEGAL_RAG_QUERY_BACKEND=chroma_first| LEGAL_RAG_QUERY_BACKEND=memory_only   |

### Law Update Pipeline (Zero Downtime)

```
  Admin: POST /admin/law/upload-pdf
               |
               v
        law_update_service.py
               |
        +------+------+
        |             |
        v             v
  Extract text    Chunk text
  (PyMuPDF)       (article-aware
                   splitting)
               |
               v
        Write new JSONL
        to chunks/ dir
               |
        +------+------+
        |             |
        v             v
  Rebuild Chroma   Reload FAISS
  collection       retriever
  (Legal Rag       (reload_chunk_
   DataIngestion)   retriever())
               |
               v
        Both stores updated
        API continues serving
        (no restart required)
```

### Startup Warmup

When `ENABLE_STARTUP_RAG=1`, both ChromaDB and the FAISS in-memory retriever are loaded in a background daemon thread at application startup. The main event loop becomes available for HTTP requests immediately; the RAG backend becomes ready within seconds to minutes depending on corpus size and available hardware.

---

## 5. Database Schema

The application uses SQLite via SQLAlchemy ORM. The database file defaults to `legalai.db` in the project root. The schema is managed through `app/db/init_db.py`, which creates all tables at startup via `Base.metadata.create_all()`.

```
+----------------------------------------------------------------+
|                     DATABASE SCHEMA (16 tables)               |
+----------------------------------------------------------------+

  CORE USER AND ANALYSIS
  +------------------+        +-----------------------------+
  | users            |        | analyses                    |
  +------------------+        +-----------------------------+
  | id (PK)          |<---+   | id (PK)                     |
  | email (unique)   |    |   | user_id (FK -> users.id)    |
  | password_hash    |    +---| filename                    |
  | role             |        | result_json (TEXT)          |
  | email_verified   |        | mime_type                   |
  | created_at       |        | sha256 (indexed)            |
  +------------------+        | page_count                  |
                              | ocr_used (0/1)              |
                              | detected_lang               |
                              | needs_review (bool)         |
                              | lawyer_note                 |
                              | created_at                  |
                              +-----------------------------+

  SHARING AND COLLABORATION
  +-------------------+       +---------------------------+
  | legato_shares     |       | legato_deal_threads       |
  +-------------------+       +---------------------------+
  | id (PK)           |       | id (PK)                   |
  | token (unique,64) |       | analysis_id (FK)          |
  | analysis_id (FK)  |       | user_id (FK)              |
  | user_id (FK)      |       | title                     |
  | created_at        |       | created_at                |
  | expires_at        |       +---------------------------+
  +-------------------+                  |
                                         | 1:N
                              +---------------------------+
                              | legato_deal_messages      |
                              +---------------------------+
                              | id (PK)                   |
                              | thread_id (FK)            |
                              | author_id (FK, nullable)  |
                              | body (TEXT)               |
                              | created_at                |
                              +---------------------------+

  TIMELINE AND SIGNATURES
  +---------------------------+    +---------------------------+
  | legato_timeline_events    |    | legato_signatures         |
  +---------------------------+    +---------------------------+
  | id (PK)                   |    | id (PK)                   |
  | analysis_id (FK)          |    | analysis_id (FK)          |
  | user_id (FK)              |    | user_id (FK)              |
  | label (String 512)        |    | signer_name               |
  | event_date (String 64)    |    | consent_acknowledged      |
  | source (manual/auto)      |    | signature_png_base64      |
  | created_at                |    | created_at                |
  +---------------------------+    +---------------------------+

  PROFESSIONAL NETWORKING
  +------------------+       +---------------------------+
  | legato_profiles  |       | social_posts              |
  +------------------+       +---------------------------+
  | user_id (PK, FK) |       | id (PK)                   |
  | payload_json     |       | author_id (FK)            |
  +------------------+       | content (TEXT)            |
                             | tags_json                 |
                             | category (indexed)        |
                             | created_at (indexed)      |
                             +---------------------------+
                                          |
               +----------+--------------+---------------+
               |          |                              |
  +--------------------------+   +--------------------+  +--------------------+
  | social_post_likes        |   | social_post_       |  | social_post_shares |
  +--------------------------+   | comments           |  +--------------------+
  | id (PK)                  |   +--------------------+  | id (PK)            |
  | post_id (FK)             |   | id (PK)            |  | post_id (FK)       |
  | user_id (FK)             |   | post_id (FK)       |  | user_id (FK)       |
  | UNIQUE(post_id, user_id) |   | author_id (FK)     |  | created_at         |
  | created_at               |   | content (TEXT)     |  +--------------------+
  +--------------------------+   | created_at(indexed)|
                                 +--------------------+

  NETWORK AND ENDORSEMENTS
  +---------------------------+    +---------------------------+
  | network_invites           |    | skill_endorsements        |
  +---------------------------+    +---------------------------+
  | id (PK)                   |    | id (PK)                   |
  | requester_id (FK)         |    | endorser_id (FK)          |
  | addressee_id (FK)         |    | recipient_id (FK)         |
  | status (pending/accepted) |    | skill (String 128)        |
  | UNIQUE(req_id, addr_id)   |    | UNIQUE(endorser,recip,    |
  | created_at                |    |        skill)             |
  +---------------------------+    | created_at                |
                                   +---------------------------+

  RECOMMENDATIONS AND DOCUMENTS
  +---------------------------+    +---------------------------+
  | profile_recommendations   |    | profile_user_documents    |
  +---------------------------+    +---------------------------+
  | id (PK)                   |    | id (PK)                   |
  | author_id (FK)            |    | user_id (FK)              |
  | recipient_id (FK)         |    | title (String 512)        |
  | content (TEXT)            |    | file_url (TEXT)           |
  | created_at (indexed)      |    | mime_type                 |
  +---------------------------+    | file_bytes (BLOB)         |
                                   | created_at                |
                                   +---------------------------+
```

### Key Design Notes

- `analyses.result_json` stores the full pipeline output as a serialized JSON string. This avoids schema migrations as the analysis output evolves.
- `analyses.sha256` is indexed to support deduplication: if the same file bytes are uploaded twice, the system can return the cached result without re-running the pipeline.
- `legato_profiles.payload_json` uses a schema-free JSON blob for profile data, enabling flexible field evolution without migrations.
- `social_post_likes` and `network_invites` carry composite unique constraints to prevent duplicate relationships at the database level.

---

## 6. Authentication and Security

```
+------------------------------------------------------+
|            AUTHENTICATION SUBSYSTEM                  |
+------------------------------------------------------+
|                                                      |
|  +------------------+    +------------------------+ |
|  |  Email / Password |    |  Google OAuth 2.0 SSO  | |
|  +------------------+    +------------------------+ |
|          |                           |               |
|          v                           v               |
|    bcrypt password             Google token          |
|    verification                verification          |
|    (passlib)                   (google-auth)         |
|          |                           |               |
|          +-----------+---------------+               |
|                      |                               |
|                      v                               |
|             JWT issued via python-jose               |
|             Algorithm: HS256                         |
|             Key: JWT_SECRET_KEY (env var)            |
|             Expiry: ACCESS_TOKEN_EXPIRE_MINUTES       |
|             Default: 10080 min (7 days in config)    |
|             Exam-mode override: 720 min (12 hours)   |
|                      |                               |
|                      v                               |
|    Bearer token in Authorization header              |
|    Dependency: get_current_user() (core/deps.py)     |
|    Optional: get_current_user_optional()             |
|              (used by /ocr_check_and_search          |
|               to allow unauthenticated calls         |
|               when save=false)                       |
|                                                      |
|  Email Verification Flow:                            |
|    Register --> send verification email (SMTP)       |
|    --> User clicks link --> email_verified = True    |
|                                                      |
|  Password Reset Flow:                                |
|    Request reset --> SMTP email with token           |
|    --> Token validated --> new password set          |
|                                                      |
+------------------------------------------------------+
```

### Role System

The `users.role` column supports role-based access control. The `require_admin` dependency (used in `/analyses` admin endpoints and `/admin/law/upload-pdf`) restricts access to users with `role = "admin"`.

### Middleware Stack

```
  Incoming Request
       |
       v
  +--------------------+
  | CORS Middleware    |  -- allow_origins from CORS_ORIGINS env var
  +--------------------+
       |
       v
  +--------------------+
  | RequestLogging     |  -- optional, app/middleware/request_logging.py
  | Middleware         |
  +--------------------+
       |
       v
  +--------------------+
  | SlowAPI Rate Limit |  -- optional, app/middleware/rate_limit.py
  | Middleware         |
  +--------------------+
       |
       v
  Route Handler
```

All 4xx and 5xx responses include CORS headers via custom exception handlers. This prevents browser-side CORS errors masking the actual HTTP error during frontend development and demo sessions.

---

## 7. Performance Decisions

### LFM Singleton and Warmup

LFM2.5-1.2B-Instruct is approximately 2.2 GB on disk. Without warmup, the first request to any LFM-backed endpoint would block for 30 to 120 seconds while the model loads from disk into memory.

```
  WARMUP_LFM_AT_STARTUP=1

  Application startup
         |
         +-- (main thread) --> HTTP server ready (immediate)
         |
         +-- (daemon thread: "lfm-warmup")
                  |
                  v
           app.local_llm.load_model()
           Loads tokenizer + model weights
           into process memory
                  |
                  v
           Model resident in memory
           First LFM request: ~100ms
           Without warmup: 30-120s
```

The model is held as a module-level singleton (`_tokenizer`, `_model`) in `local_llm.py`. Subsequent calls reuse the loaded instance.

### Social Feed N+1 Query Fix

A naive implementation of the social feed would execute one query per post to retrieve its like count, comment count, and share count. For a feed of N posts, this produces 3N+1 database queries.

The `_batch_post_stats()` function resolves this with 6 fixed queries regardless of feed size:

```
  Feed query (1 query)
       +
  Batch like counts   (1 query: GROUP BY post_id)
  Batch comment counts(1 query: GROUP BY post_id)
  Batch share counts  (1 query: GROUP BY post_id)
  Batch user likes    (1 query: WHERE user_id = current)
  Batch user shares   (1 query: WHERE user_id = current)
  = 6 total queries for any feed size
```

### RAG Background Bootstrap

Both ChromaDB and the FAISS in-memory retriever can take several seconds to minutes to initialize (embedding model load + index build). When `ENABLE_STARTUP_RAG=1`, this work executes in a daemon thread named `rag-bootstrap`, leaving the HTTP server immediately responsive.

### Gemini Retry Backoff

Cloud API quota errors (HTTP 429) are retried up to three times with linear delay (2s, 4s) before returning an error to the client. This avoids surfacing transient quota bursts as user-facing failures.

---

## 8. Deployment Architecture

### HuggingFace Spaces (Production)

```
+------------------------------------------------------+
|               HUGGINGFACE SPACES                     |
+------------------------------------------------------+
|                                                      |
|  Docker container (Dockerfile at project root)       |
|  Base: Python + FastAPI + uvicorn                    |
|  Port: 7860 (HF Spaces standard)                     |
|                                                      |
|  +------------------------------------------------+  |
|  |  Persistent storage via HF Datasets / Space   |  |
|  |  or mounted volume for legalai.db + chroma_db  |  |
|  +------------------------------------------------+  |
|                                                      |
|  Large model files tracked via Git LFS:              |
|    *.safetensors  (LFM2.5 weights)                   |
|    *.bin          (legacy weights)                   |
|    *.joblib       (ML predictor bundles)             |
|                                                      |
|  Environment configuration via HF Secrets:          |
|    JWT_SECRET_KEY                                    |
|    GEMINI_API_KEY                                    |
|    GEMINI_MODEL                                      |
|    DATABASE_URL                                      |
|    ENABLE_STARTUP_RAG                                |
|    WARMUP_LFM_AT_STARTUP                             |
|    DOCUMENT_CHAT_GEMINI_FALLBACK                     |
|    CORS_ORIGINS                                      |
|    CHROMA_LEGAL_DIR                                  |
|    RAG_MIN_SCORE                                     |
|    LEGAL_RAG_QUERY_BACKEND                           |
|                                                      |
+------------------------------------------------------+
```

### Local Development

```
+------------------------------------------------------+
|               LOCAL DEVELOPMENT                      |
+------------------------------------------------------+
|                                                      |
|  uvicorn app.main:app --host 0.0.0.0 --port 8001    |
|                                                      |
|  .env file in project root for all secrets           |
|  legalai.db created automatically at startup         |
|                                                      |
|  Frontend: legalai-frontend/ (separate process)      |
|  Configured at http://localhost:5173                 |
|  CORS_ORIGINS includes localhost:5173                |
|                                                      |
+------------------------------------------------------+
```

### File and Directory Structure

```
  GP-Legal-AI--main/
  |
  +-- app/                    FastAPI application package
  |   +-- main.py             Entry point, pipeline orchestration
  |   +-- rules.py            Rule engine
  |   +-- local_llm.py        LFM2.5 wrapper (singleton)
  |   +-- model_ml_predictor.py  Law-aware ML inference
  |   +-- cross_border.py     UAE/MOHRE signal detector
  |   +-- pdf_text_pipeline.py   OCR pipeline
  |   +-- legal_rag_bridge.py ChromaDB bridge (primary RAG)
  |   +-- rag_chromadb.py     ChromaDB access layer
  |   +-- rag_utils.py        FAISS in-memory retriever
  |   +-- utils_text.py       Text normalization + splitting
  |   +-- routers/            FastAPI routers
  |   |   +-- auth.py         /auth/* endpoints
  |   |   +-- analyses.py     /analyses/* endpoints
  |   |   +-- chat.py         /chat/* (Gemini)
  |   |   +-- social.py       /api/* (networking)
  |   |   +-- admin_law.py    /admin/law/*
  |   |   +-- legato_mobile.py /legato/* (mobile tools)
  |   +-- db/
  |   |   +-- models.py       SQLAlchemy ORM models (16 tables)
  |   |   +-- session.py      DB engine + get_db dependency
  |   |   +-- init_db.py      create_all() on startup
  |   +-- core/
  |   |   +-- config.py       Settings dataclass
  |   |   +-- security.py     JWT + bcrypt
  |   |   +-- deps.py         get_current_user, require_admin
  |   +-- middleware/
  |       +-- request_logging.py
  |       +-- rate_limit.py
  |
  +-- rules/                  YAML rule definitions
  |   +-- labor.yaml
  |   +-- labor_mandatory.yaml
  |   +-- labor_cross_border.yaml
  |
  +-- model_ML/               Trained ML model bundles
  |   +-- law_aware_multilabel_model.joblib
  |   +-- law_aware_binary_model.joblib
  |   +-- labor14_2025_clean.jsonl
  |
  +-- LFM2.5-1.2B-Instruct/   Local LLM (HuggingFace snapshot)
  |   +-- config.json
  |   +-- *.safetensors
  |
  +-- Legal Rag/              ChromaDB RAG subsystem
  |   +-- src/                VectorStore, RAGEngine, DataIngestion
  |   +-- chroma_db/          Persistent ChromaDB store
  |   +-- config.yaml
  |
  +-- chunks/                 JSONL law chunks (FAISS source)
  +-- laws/                   Law text files
  +-- legalai.db              SQLite database (runtime)
  +-- Dockerfile
  +-- docker-compose.yml
```

---

## 9. API Surface

The following table lists the primary API routes organized by subsystem.

| Method | Path                          | Auth     | Description                                        |
|--------|-------------------------------|----------|----------------------------------------------------|
| GET    | /health                       | None     | Health check                                       |
| POST   | /ocr                          | None     | Extract text from uploaded file (no analysis)      |
| POST   | /ocr_check_and_search         | Optional | Full analysis pipeline (core endpoint)             |
| POST   | /check_clause                 | None     | Single-clause ML rule check                        |
| POST   | /auth/register                | None     | User registration with email verification          |
| POST   | /auth/login/form              | None     | Email + password login (returns JWT)               |
| POST   | /auth/google                  | None     | Google OAuth SSO                                   |
| POST   | /auth/verify-email            | None     | Email verification token confirmation              |
| POST   | /auth/forgot-password         | None     | Initiate password reset                            |
| GET    | /analyses                     | JWT      | List user's saved analyses                         |
| GET    | /analyses/{id}                | JWT      | Retrieve single analysis with full result_json     |
| POST   | /analyses                     | JWT      | Save analysis result to database                   |
| POST   | /chat/assistant               | JWT      | Start Gemini chat session for a contract           |
| POST   | /chat/message                 | JWT      | Send message in existing chat session              |
| POST   | /admin/law/upload-pdf         | Admin    | Upload new law PDF and rebuild RAG corpus          |
| GET    | /api/posts                    | JWT      | Paginated social feed                              |
| POST   | /api/posts                    | JWT      | Create new social post                             |
| POST   | /api/posts/{id}/like          | JWT      | Toggle like on a post                              |
| GET    | /api/profile/{user_id}        | JWT      | Get user profile                                   |
| POST   | /api/network/invite           | JWT      | Send connection invitation                         |
| POST   | /api/network/respond          | JWT      | Accept or decline invitation                       |
| POST   | /legato/share                 | JWT      | Generate shareable link for an analysis            |
| POST   | /legato/deal/thread           | JWT      | Create deal collaboration thread                   |
| POST   | /legato/timeline/event        | JWT      | Add event to contract timeline                     |
| POST   | /legato/sign                  | JWT      | Record digital signature for a contract            |

---

*End of architecture document.*
