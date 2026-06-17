# LEGATO Backend Audit — Prioritized Task List
_Generated: 2026-05-08 | Auditor: Claude Sonnet 4.6 (full static analysis, 8 checks)_

---

## 🔴 CRITICAL (blocks deployment or core demo)

- [ ] **Set GEMINI_API_KEY in .env** — `.env:79` — The key is blank (`GEMINI_API_KEY=`). Without it: `/chat/assistant` returns a static "not configured" message, `/legato/negotiation-chat` falls back to LFM (which may also fail), and `/chat/document` Gemini fallback is silent. The entire chat demo is broken. Get a key from aistudio.google.com and paste it into `.env`.

- [ ] **Preload LFM at startup to avoid first-request timeout** — `app/main.py:1147-1205`, `app/local_llm.py:91-120` — The `startup()` event checks LFM availability but does NOT load it into memory. First call to `/legato/explain-clause`, `/legato/summarize-clauses`, `/legato/compare`, or `/chat/document` will block for 30–120 seconds loading 2.2 GB into CPU RAM — likely returning a 503 or gateway timeout before the demo examiner. Fix: add `app/local_llm.load_model()` inside the startup event (inside a background thread, same pattern as RAG bootstrap). Guard with `WARMUP_LFM_AT_STARTUP=1` env flag.

- [ ] **Implement `needs_review` / `lawyer_override` — examiner requirement** — No file — The entire `needs_review` / `flag_for_review` field is absent from `Analysis` model, `OCRResponse`, and all routers. There is no `lawyer_override` endpoint anywhere. Grep confirms zero occurrences. The examiner explicitly listed this as a requirement. Minimum: add `needs_review: bool = False` to `Analysis` model + migration, expose it in `/ocr_check_and_search` response, and add `PATCH /analyses/{id}/flag` endpoint.

- [ ] **Set SECRET_KEY in .env** — `app/core/security.py:36` — Defaults to literal `"dev-secret-change-me"`. Any attacker who knows the default can forge valid JWTs. For a demo/exam environment this must be changed. Also: current token expiry is 60 minutes (`ACCESS_TOKEN_EXPIRE_MINUTES=60`), meaning reviewers testing over a long session get kicked out. Set to at least 720 minutes for demo.

- [ ] **`_normalize_contract_text` defined twice in main.py** — `app/main.py:33-36` and `app/main.py:620-650` — The stub on line 33 shadows the full function on line 620. Python will use the last definition at module level, so the actual behavior depends on load order. This is a silent bug: at runtime the stub (which does nothing) exists in module scope first, then gets replaced. Any code that captures the reference before full load could get the stub. Clean up by removing the stub at line 33-36.

- [ ] **`ENABLE_STARTUP_RAG` not set — RAG cold on first analysis** — `.env` (not set) — Defaults to `"0"`, so RAG completely skips warmup. The first `/ocr_check_and_search` call triggers a cold ChromaDB + JSONL retriever init in-band, adding 10–60 seconds of latency to the first request. Set `ENABLE_STARTUP_RAG=1` in `.env` to background-thread warmup at start.

---

## 🟡 MEDIUM (examiner will notice, should fix)

- [ ] **LABOR25_EMPLOYER_INFO / LABOR25_EMPLOYEE_INFO have `severity: high` but fire on PRESENCE** — `rules/labor_mandatory.yaml:3-46`, `app/rules.py:165-195` — RuleEngine special-cases these rules to match when employer/employee labels ARE found in the contract (a good condition). But `severity: high` causes `_build_labor_summary()` to classify them as violations. A well-formed contract will show "violations_detected" because it contains employer info. Fix: change severity to `"presence"` or `"info"` for these informational presence rules; keep `"error"` only for PLACEHOLDER rules.

- [ ] **CORS hardcoded to `localhost:5173`** — `app/main.py:408` — `CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]`. If the frontend is served on any other port or domain (Docker, staging, different machine), all API calls fail with CORS errors. Drive from env: `CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")`.

- [ ] **`DOCUMENT_CHAT_GEMINI_FALLBACK` defaults to `"1"` — hides LFM failures** — `app/routers/chat.py:444` — When LFM fails to load, `/chat/document` silently falls back to Gemini with no visible indicator. For the examiner demo, this means the "local LFM document chat" may actually be Gemini cloud. Set `DOCUMENT_CHAT_GEMINI_FALLBACK=0` in `.env` for exam so failures are explicit, OR add a `used_fallback: bool` field to `DocumentChatResponse`.

- [ ] **No cloud migration flag to disable local LFM globally** — There is a `DOCUMENT_CHAT_GEMINI_FALLBACK` flag but it only covers `/chat/document`. Endpoints `/legato/explain-clause`, `/legato/summarize-clauses`, and `/legato/compare` call `local_llm.is_available()` and raise `503 RuntimeError` if LFM is missing — no Gemini fallback. For cloud deployment without the 2.2 GB model, these features go offline entirely. Add `FORCE_CLOUD_LLM=1` env that routes LFM calls to Gemini.

- [ ] **Deferred circular imports in `legato_service.py`** — `app/legato_service.py:33-41, 150-157` — `_live_rag_for_violation()` and `run_explain_clause()` import RAG helpers from `app.main` at call time inside function bodies. This works but makes stack traces unreadable and breaks if `app.main` is refactored. Extract shared RAG helpers into `app/rag_helpers.py` so both `main.py` and `legato_service.py` import from there.

- [ ] **Social feed N+1 query problem** — `app/routers/social.py:65-84` — `_serialize_post()` fires individual DB queries per post for: author lookup, profile lookup, likes count, comments count, shares count, liked status. For a feed of 20 posts this is 120+ queries. Examiner loading the social feed will see slow response. Fix with SQLAlchemy JOIN or batch queries.

- [ ] **No rate limiting on `/ocr_check_and_search`** — `app/middleware/rate_limit.py` exists but is only mounted if `slowapi` import succeeds. The most expensive endpoint (OCR + ML + RAG + LFM) has no per-user throttle. One stuck request can starve the server. Confirm `slowapi` is installed and add `@limiter.limit("10/minute")` to the endpoint.

- [ ] **`legalai_backup.db` leftover in root** — root directory — A stale backup database file exists at project root. It could contain different schema versions or test data. Rename to `legalai_backup.db.archive` or `.gitignore` it to avoid confusion about which DB is authoritative.

- [ ] **`HF_TOKEN=your_huggingface_token_here` placeholder in .env** — `.env:162` — The literal placeholder value will be sent as the actual HF token to Hugging Face APIs if `HF_TOKEN` is read, causing auth failures for HF model downloads. Clear it or replace with a real token.

---

## 🟢 POLISH (nice to have in 10 days)

- [ ] **Add `WARMUP_LFM_AT_STARTUP=1` flag** — `app/main.py:1147` — After fixing the cold-start issue above, expose this as a documented env flag in `.env.example` so teams with/without GPU can opt in.

- [ ] **Token expiry too short for demo sessions** — `app/core/security.py:38` — Default 60 minutes. Set `ACCESS_TOKEN_EXPIRE_MINUTES=720` in `.env` so reviewers aren't logged out mid-demo.

- [ ] **Remove stub `_normalize_contract_text` at module top** — `app/main.py:33-36` — Dead stub (3 lines that do nothing) above the real function at line 620. Clean up to remove confusion.

- [ ] **`app/legato_service.py` has no Gemini fallback for summarize/compare** — `app/legato_service.py:221-264` — When LFM unavailable, `summarize_clauses_llm()` and `compare_contracts_llm()` raise `RuntimeError("Local LFM not available")` → 503. For demos without a GPU: add `try: use_gemini_for_summary() except: raise` pattern.

- [ ] **AnalyzePage.tsx always sends `save: true`** — `legalai-frontend/legalai-frontend/src/pages/AnalyzePage.tsx:146` — Every analysis is force-saved to DB even if the user doesn't want it. This fills the DB with throwaway analysis results during testing. Add a user-facing "Save to history" toggle and only pass `save: true` when the user explicitly opts in.

- [ ] **`analysis_id` is sent back in OCR response** but AnalyzePage reads it with optional chaining — `AnalyzePage.tsx:150` — Clean, but `saveAnalysisToDb()` is also called redundantly when `save: true` already saved via the backend. The frontend calls `/analyses` POST as a second save. Review whether both save paths are needed or if one should be removed.

- [ ] **Add `analyzed_at` / `updated_at` to Analysis model** — `app/db/models.py:37` — Only `created_at` exists. Admin view shows "Date" which maps to `created_at`. For re-analyses or updated results, you want `updated_at`. Low effort SQLAlchemy column addition.

- [ ] **Social router profile endpoint missing avatar support** — `app/routers/social.py:288` — `/api/profile/{user_id}` returns profile JSON blob. No avatar URL field standardized. Frontend profile cards will show placeholders. Define `avatar_url` as a top-level field in the profile payload schema.

---

## ✅ CONFIRMED WORKING

- **LFM2.5-1.2B-Instruct model** — Real 2.23 GB safetensors at `./LFM2.5-1.2B-Instruct/` (not a Git LFS pointer), `config.json` has valid `model_type: lfm2`, `architectures: [Lfm2ForCausalLM]`. Auto-detected by both `app/local_llm.py` and `llm/lfm_model.py`.
- **LFM loaded as singleton** — `app/local_llm.py:20-22, 98-99` — Module-level `_tokenizer`, `_model`, `_loaded_path` with guard `if _model is not None and _loaded_path == path_str: return`. Safe across concurrent requests after first load.
- **LFM/Gemini boundary** — LFM: explain-clause, summarize, compare, doc-chat, negotiation-fallback. Gemini: /chat/assistant, /chat/message, negotiation-primary. Boundary is correct.
- **DOCUMENT_CHAT_GEMINI_FALLBACK logic** — `app/routers/chat.py:399-461` — Correctly conditional: Gemini only activates when `_local_document_failed(content)` returns True (LFM error prefix detected), not as a silent replacement.
- **All 6 routers present and conditionally mounted** — auth, analyses, chat, legato_mobile, social, admin_law — all files exist, all mounted with graceful try/except fallbacks in startup.
- **All model_ML artifacts** — `model_ML/law_aware_multilabel_model.joblib`, `model_ML/law_aware_binary_model.joblib`, `model_ML/labor14_2025_clean.jsonl` — all exist.
- **All chunks JSONL files** — `chunks/labor_law_chunks.cleaned.jsonl` + 3 others — exist and loaded by fallback retriever.
- **laws/processed/labor14_2025_articles.json** — exists, loaded by RuleEngine.
- **app/ml/artifacts/unified/** — vectorizer, model, thresholds, config — all 5 files exist.
- **RuleEngine loads all 3 YAML files** — `app/rules.py:40-49` — `sorted(Path(d).glob("*.y*ml"))` loads labor.yaml, labor_cross_border.yaml, labor_mandatory.yaml.
- **All DB models mapped** — User, Analysis, LegatoShare, LegatoDealThread, LegatoDealMessage, LegatoTimelineEvent, LegatoProfile, LegatoSignature, SocialPost, SocialPostLike, SocialPostComment, SocialPostShare, NetworkInvite, SkillEndorsement, ProfileRecommendation, ProfileUserDocument — all in `app/db/models.py`.
- **Auth endpoints complete** — register (with email verify), login JSON + form, /me, forgot-password, verify-reset-code, reset-password, Google OAuth (/auth/google + /auth/google/callback).
- **Legato endpoints complete** — explain-clause, risk/{id}, summarize-clauses, compare, negotiation-chat (Gemini+fallback), shares (CRUD+public), deal-threads (CRUD+messages), timeline (create+list), signatures, profile (get/put), network profiles.
- **AdminPage Law RAG tab** — implemented with upload-pdf, reindex, preview, job polling.
- **AnalyzePage.tsx 5 tabs** — Summary, Violations, OCR, RAG hits, Document-chat — all implemented.
- **ChatPage.tsx dual mode** — connects to /chat/assistant (general) AND /chat/document (contract) correctly.
- **PyMuPDF in wheels/** — Local .whl files for PyMuPDF 1.24.9 available for offline install.
- **Cross-border detection** — `app/cross_border.py` + `rules/labor_cross_border.yaml` + `_labor_applicability()` in main.py — complete pipeline.

---

## ❌ MISSING ARTIFACTS

_No critical artifacts are missing. All key model files, data files, and rule files exist._

| File | Status | Notes |
|------|--------|-------|
| `LFM2.5-1.2B-Instruct/model.safetensors` | ✓ EXISTS | 2.23 GB real file (not LFS pointer) |
| `model_ML/law_aware_multilabel_model.joblib` | ✓ EXISTS | |
| `model_ML/law_aware_binary_model.joblib` | ✓ EXISTS | |
| `model_ML/labor14_2025_clean.jsonl` | ✓ EXISTS | |
| `Legal Rag/data/labor14_2025_chunks.cleaned.jsonl` | ✓ EXISTS | |
| `chunks/labor_law_chunks.cleaned.jsonl` | ✓ EXISTS | |
| `laws/processed/labor14_2025_articles.json` | ✓ EXISTS | |
| `rules/labor.yaml` | ✓ EXISTS | Only 1 rule (old law); labor_mandatory.yaml has Labor 14/2025 rules |
| `rules/labor_cross_border.yaml` | ✓ EXISTS | |
| `rules/labor_mandatory.yaml` | ✓ EXISTS | Main rule file for Labor Law 14/2025 |
| `app/ml/artifacts/unified/` | ✓ EXISTS | 5 files: vectorizer, model, thresholds, metrics, config |
| `.env` | ✓ EXISTS | But essentially unconfigured — GEMINI key empty, SECRET_KEY default |
| `models/LFM2.5-1.2B-Instruct/` | ✗ MISSING | Not needed — root `LFM2.5-1.2B-Instruct/` is the fallback path |

---

## 🔍 CHECK 8 — LFM + Gemini Boundary Audit

| Question | Status | Details |
|----------|--------|---------|
| LFM used ONLY for explain/summarize/compare/doc-chat/negotiation-fallback | ✅ CORRECT | Verified in legato_service.py + chat.py |
| Gemini used ONLY for /chat/assistant, /chat/message, negotiation-primary | ✅ CORRECT | Verified in chat.py + legato_mobile.py |
| DOCUMENT_CHAT_GEMINI_FALLBACK only on LFM fail, not as replacement | ✅ CORRECT | `_local_document_failed(content)` guard at chat.py:443 |
| /chat/document graceful "LFM not available" message | ✅ CORRECT | Returns `[Local LLM not available...]` → caught by `_local_document_failed()` |
| LFM loaded as singleton (not per request) | ✅ CORRECT | Module-level cache in `app/local_llm.py:20-22` with path guard |

**One concern:** `DOCUMENT_CHAT_GEMINI_FALLBACK` defaults to `"1"` (on). If LFM fails to load for any reason, document chat silently uses Gemini. The response gives no indication that the local model was skipped. Examiner may think they're testing local LFM when actually hitting Gemini. Recommend setting `DOCUMENT_CHAT_GEMINI_FALLBACK=0` in `.env` for the examination session.

---

## 🚀 10-Day Sprint Priority Order

1. Set GEMINI_API_KEY, SECRET_KEY, ENABLE_STARTUP_RAG=1 in `.env` _(1 hour)_
2. Add LFM startup warmup in background thread _(2 hours)_
3. Implement `needs_review` field + `PATCH /analyses/{id}/flag` endpoint _(4 hours)_
4. Fix LABOR25_EMPLOYER_INFO / LABOR25_EMPLOYEE_INFO severity to `"presence"` _(30 min)_
5. Fix CORS to be env-driven _(30 min)_
6. Remove duplicate `_normalize_contract_text` stub _(5 min)_
7. Set `DOCUMENT_CHAT_GEMINI_FALLBACK=0` for exam _(1 min)_
8. Fix social feed N+1 query _(2 hours)_
9. Add `ACCESS_TOKEN_EXPIRE_MINUTES=720` to `.env` _(1 min)_
10. Add Gemini fallback for summarize/compare endpoints _(3 hours)_
