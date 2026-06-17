# Legato — Examination Committee Demo Script
**Misr International University — Computer Science Department**
**Total runtime: 10–12 minutes**

---

## Talking Points (What the Committee Will Remember)

1. **End-to-end AI pipeline** — A single PDF upload triggers OCR, a YAML rule engine, a trained ML model, vector RAG over 452 law documents (ChromaDB 154 + FAISS 298), and a 2.2 GB local LFM — all in one request, all on Egyptian Law 14/2025.
2. **needs_review flag** — Contracts with severity="error" rule hits are automatically flagged for human lawyer review. The flag is writable via a dedicated PATCH endpoint, creating a defensible human-in-the-loop workflow.
3. **LFM/Gemini boundary** — The local LFM (LFM2.5-1.2B-Instruct) is used only where law grounding matters: explain, summarize, document chat, and negotiation fallback. Gemini handles the general assistant and negotiation primary. Mixing would produce hallucinated legal advice; the codebase enforces the split.
4. **Optimized social feed** — The professional networking layer resolves an entire paginated feed in exactly 6 SQL queries regardless of feed size, using a batch stats helper that eliminates the N+1 problem.
5. **Live RAG administration** — The law corpus can be rebuilt in production without downtime. The reindex job runs in a background thread and is tracked by a job-status endpoint. The server never blocks a request to update the law.

---

## Pre-Demo Checklist (Run Before the Committee Enters)

```bash
# Start the server
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# Confirm healthy startup — look for these lines in the log:
# [OK] LFM warmup: model loaded and ready.
# [OK] ChromaDB RAG ready (Legal RAG corpus): 154 docs.
# [OK] Retriever initialized with 298 docs.
# [OK] Auth router mounted
# [OK] Analyses router mounted
# [OK] Chat router mounted
# [OK] Admin law router mounted
# [OK] Legato mobile router mounted
# [OK] Social API router mounted

# Verify health endpoint
curl http://127.0.0.1:8001/health
```

Expected health response:
```json
{"status": "ok"}
```

Have `contracts-guide.pdf` (or any Arabic labor contract PDF) ready on your Desktop.

---

## Step 1 — Contract Upload and Analysis (2 min)

**What this demonstrates:** The full AI analysis pipeline. One PDF goes in; rule hits, RAG results, and LFM explanations come out.

### Request

```bash
curl -s -X POST http://127.0.0.1:8001/ocr_check_and_search \
  -F "file=@C:/Users/Aly ahmed/Desktop/GP-Legal-AI--main/contracts-guide.pdf" \
  -F "query=notice period" \
  -F "use_rag=true" \
  -F "use_llm=true" \
  -F "llm_top_k=2" \
  -F "llm_max_new_tokens=200" \
  -F "save=false" | python -m json.tool
```

### What to Point Out

- **`rule_hits`** — Each item has `rule_id`, `severity`, `description`, and `matched_text`. Rule IDs are named after the specific Labor Law 14/2025 article they enforce (e.g., `LABOR25_SALARY_PAYMENT_METHOD`, `LABOR25_PROBATION_PERIOD_EXCEEDED`). Severity `"error"` means a clear statutory violation.
- **`needs_review`** — This boolean is `true` whenever any rule hit has `severity="error"`. It is the signal that drives the lawyer review workflow. Ask the committee: "This is how we keep a human lawyer in the loop."
- **`rag_by_violation`** — For each error-level rule hit, the system retrieved the actual Egyptian law text that the rule is based on. The field `rag_query_used` shows the boosted query sent to ChromaDB. The `hits` array contains the exact law chunks with article metadata.
- **`llm_explanation`** — The LFM-generated explanation in Arabic (or English) grounded in the retrieved law text. Not Gemini — the 2.2 GB local model running on the same machine.
- **`pipeline_steps`** — Shows exactly which stages ran: `["ocr", "rule_engine", "ml_assist", "rag", "llm"]`.

---

## Step 2 — Clause Tools (2 min)

**What this demonstrates:** The three LFM-powered clause tools available to lawyers and HR professionals.

### Get a Demo Token First

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@legato.com","password":"LegatoDemo2026!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo $TOKEN
```

### 2a — Explain a Clause

```bash
curl -s -X POST http://127.0.0.1:8001/legato/explain-clause \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "clause_text": "يلتزم العامل بعدم الإفصاح عن أسرار الشركة خلال مدة العقد وبعد انتهائه بمدة خمس سنوات.",
    "language": "ar"
  }' | python -m json.tool
```

Point out: The response contains the LFM explanation referencing the relevant labor-law article. The `language` field controls output language; `ar` and `en` are both supported.

### 2b — Summarize Clauses

```bash
curl -s -X POST http://127.0.0.1:8001/legato/summarize-clauses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "clauses": [
      "يلتزم الطرف الثاني بمباشرة العمل في اليوم التالي لتوقيع هذا العقد.",
      "تحدد مدة العقد بسنة واحدة قابلة للتجديد باتفاق الطرفين.",
      "يستحق العامل إجازة سنوية مدتها 15 يوم بعد انتهاء السنة الأولى."
    ],
    "language": "ar"
  }' | python -m json.tool
```

Point out: The `summaries` array contains one LFM-generated summary per clause. The model is grounded in the contract text it was given — no invented facts.

### 2c — Risk Summary (GET, requires a saved analysis ID)

If you have a saved analysis (set `save=true` in Step 1 with a Bearer token), substitute the returned `analysis_id`:

```bash
curl -s http://127.0.0.1:8001/legato/risk/ANALYSIS_ID \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Point out: The response includes a `risk_score` (0–100), a `severity_breakdown` object (`error`, `warning`, `info` counts), and the top violations. This powers the risk gauge visualization in the mobile app.

---

## Step 3 — Document Chat (1 min)

**What this demonstrates:** Q&A over a specific contract using the local LFM, not a cloud model.

```bash
curl -s -X POST http://127.0.0.1:8001/chat/document \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "message": "What are my rights regarding the notice period?",
    "document_context": "يجوز لأي من الطرفين إنهاء هذا العقد بإخطار كتابي مدته ثلاثون يوماً. في حالة الإنهاء المفاجئ يستحق العامل تعويضاً يعادل أجر مدة الإخطار."
  }' | python -m json.tool
```

### What to Point Out

- The `content` field is the LFM answer, grounded in the `document_context` text provided.
- `used_fallback: false` confirms the local LFM handled the request. If it were `true`, Gemini was used as a fallback — this is controlled by `DOCUMENT_CHAT_GEMINI_FALLBACK` in `.env`. Our production config sets this to `0`, meaning the LFM is always primary for document chat.
- This is why the LFM/Gemini boundary matters: Gemini does not know the specific clause wording or the Egyptian law article numbers unless we retrieve and inject them from the RAG corpus.

---

## Step 4 — Social Features (1.5 min)

**What this demonstrates:** The professional networking layer — a legal-community feed for Egypt.

### 4a — Get the Feed (Optimized)

```bash
curl -s "http://127.0.0.1:8001/api/posts?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Point out: The response contains `items`, `page`, `page_size`, and `total`. Each item has `likes_count`, `comments_count`, `shares_count`, and `liked` (viewer-aware). This entire response — for any feed size — is built from exactly **6 SQL queries**: 1 for posts, 1 for authors, 1 for profiles, then 3 batch-aggregate queries for likes, comments, and shares. The `_batch_post_stats` helper eliminates the N+1 loop entirely.

### 4b — Create a Post

```bash
curl -s -X POST http://127.0.0.1:8001/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "content": "Reminder: under Egyptian Labor Law 14/2025, probation periods cannot exceed 3 months for the same employer. Any clause exceeding this limit is null by operation of law.",
    "tags": ["labor-law", "probation", "egypt"],
    "category": "Legal Updates"
  }' | python -m json.tool
```

### 4c — Show Network Invites

```bash
curl -s http://127.0.0.1:8001/api/network/invites \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Point out: The `items` array lists pending connection requests with requester name and timestamp.

---

## Step 5 — Negotiation Coach (1 min)

**What this demonstrates:** A Gemini-primary, LFM-fallback negotiation advisor.

```bash
curl -s -X POST http://127.0.0.1:8001/legato/negotiation-chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "message": "My employer wants to include a 6-month probation period. What is my negotiating position under Egyptian law?"
  }' | python -m json.tool
```

### What to Point Out

- The `content` field is the response. If a Gemini API key is configured, Gemini answered.
- If the key is absent or quota is exhausted, the fallback activates automatically. The fallback condition checks for an empty response or a response string starting with `"[Negotiation chat error:"`. When triggered, the local LFM generates a brief answer using any contract context provided.
- You can demonstrate the fallback by temporarily removing `GEMINI_API_KEY` from `.env` and restarting — the endpoint stays available, powered by the on-device model.

---

## Step 6 — Admin: Law RAG Update (1.5 min)

**What this demonstrates:** The system can update its knowledge of Egyptian law in production without downtime.

### Get an Admin Token

```bash
ADMIN_TOKEN=$(curl -s -X POST http://127.0.0.1:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@legato.com","password":"LegatoAdmin2026!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo $ADMIN_TOKEN
```

### 6a — Trigger a RAG Rebuild

```bash
curl -s -X POST http://127.0.0.1:8001/admin/law/reindex \
  -H "Authorization: Bearer $ADMIN_TOKEN" | python -m json.tool
```

Expected response:
```json
{"job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "status": "queued"}
```

Point out: The endpoint returns immediately with a `job_id`. The rebuild runs in a daemon background thread. The API never pauses — any ongoing request continues unaffected.

### 6b — Poll Job Status

Replace `JOB_ID` with the value returned above:

```bash
curl -s http://127.0.0.1:8001/admin/law/jobs/JOB_ID \
  -H "Authorization: Bearer $ADMIN_TOKEN" | python -m json.tool
```

The `status` field transitions through `"queued"` to `"running"` to `"done"` (or `"error"` with a reason string). Calling this GET endpoint any number of times causes no side effects — it is purely idempotent.

Point out: "When Egypt publishes an amendment to Labor Law 14/2025, a non-technical admin uploads the new PDF via `POST /admin/law/upload-pdf`. The system rebuilds its RAG index without redeploying any code and without taking the API offline."

---

## Step 7 — Auth and Profile (1 min)

**What this demonstrates:** JWT-based authentication and the profile/endorsement system.

### 7a — Login and Inspect the Token

```bash
curl -s -X POST http://127.0.0.1:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@legato.com","password":"LegatoDemo2026!"}' \
  | python -m json.tool
```

Point out: The response contains `access_token` (a signed JWT), `token_type: "bearer"`, and `expires_in`. All protected endpoints accept this token in the `Authorization: Bearer <token>` header. Token TTL is `ACCESS_TOKEN_EXPIRE_MINUTES=720` (12 hours), configured in `.env` so demo sessions never expire mid-presentation.

### 7b — Call a Protected Profile Endpoint

```bash
curl -s http://127.0.0.1:8001/api/profile/1 \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Point out: The profile payload includes `displayName`, `title`, `company`, `skills`, and nested `experience` and `education` arrays.

### 7c — View Skill Endorsements

```bash
curl -s http://127.0.0.1:8001/api/profile/1/endorsements \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Point out: Endorsements are peer-verified skill signals — the professional credibility layer of the network, equivalent to LinkedIn endorsements but scoped to the Egyptian legal profession.

---

## Architecture Summary (Verbal — 30 seconds)

If asked to summarize:

"A lawyer or HR professional uploads a contract PDF. The system extracts text with PyMuPDF or OCR, runs it through a deterministic YAML rule engine that encodes Egyptian Labor Law 14/2025 article by article, then a trained ML model adds probabilistic scoring. For every error-level violation, the vector RAG layer retrieves the exact law article from a 452-document corpus using ChromaDB and FAISS. The local LFM — a 2.2-billion-parameter Arabic-capable model running entirely on this machine — then writes a grounded explanation in Arabic. No cloud model touches legal advice. Gemini handles only general chat and negotiation coaching, where the hallucination risk is lower and the stakes are not a binding legal opinion. The entire pipeline is auditable via the `pipeline_steps` field in every response."

---

## Backup Plan

### If LFM is slow or shows "[LFM not loaded]"

Say: "LFM2.5-1.2B is a 2.2 GB model loaded in a background thread at startup, controlled by `WARMUP_LFM_AT_STARTUP=1` in `.env`. On a cold start the first inference request takes 30–60 seconds while the model loads from disk; every subsequent call is fast because the model stays in memory. In our HuggingFace Spaces deployment the model is pinned in GPU memory permanently."

To trigger the load and warm up:
```bash
curl -s -X POST http://127.0.0.1:8001/legato/explain-clause \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"clause_text": "مدة العقد سنة واحدة.", "language": "ar"}'
# First call may take up to 60 seconds. All subsequent calls are fast.
```

Check the server log for: `[Startup] LFM warmup: model loaded and ready.`

### If Gemini quota is exhausted

Say: "We designed every Gemini call site to degrade gracefully. The negotiation coach falls back to the local LFM automatically. Document chat is LFM-primary with Gemini disabled in production (`DOCUMENT_CHAT_GEMINI_FALLBACK=0`). The general assistant at `/chat/assistant` returns a human-readable quota message rather than a 500 error. The Gemini helper also retries three times with linear backoff before giving up."

### If the server is not running

```bash
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
uvicorn app.main:app --host 127.0.0.1 --port 8001
# Wait for all 7 "[OK] ... router mounted" lines before proceeding.
```

### If a request returns 401 Unauthorized

The demo token may have expired. Re-run the login command from Step 2 to get a fresh token:
```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@legato.com","password":"LegatoDemo2026!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### If `analysis_id` is not available for Steps 2c or 5

Re-run the Step 1 upload with `save=true` and a Bearer token to generate a persisted analysis record:

```bash
curl -s -X POST http://127.0.0.1:8001/ocr_check_and_search \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@C:/Users/Aly ahmed/Desktop/GP-Legal-AI--main/contracts-guide.pdf" \
  -F "save=true" | python -m json.tool
# Use the returned analysis_id value for subsequent steps.
```

### If no PDF contract is available

The clause tools (`/legato/explain-clause`, `/legato/summarize-clauses`) and document chat (`/chat/document`) all accept inline text directly — no file upload required. Use the example strings already in this script.

---

*Demo script version: 2026-05-10. Server: uvicorn on port 8001. Law corpus: Egyptian Labor Law 14/2025.*
