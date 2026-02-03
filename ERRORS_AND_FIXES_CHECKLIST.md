# Errors and Fixes Checklist (for Agent Mode)

Use this list when in **Agent mode** to fix all reported errors and warnings so the project runs with no errors or warnings. Focus: LLM (LFM), RAG, ML, frontend–backend sync.

**Status:** All items below have been addressed. Run backend from project root; see README "Environment variables" and `.env.example` for GEMINI_API_KEY, HF_TOKEN, DEVICE.

---

## 1. Where to run the backend (critical)

**Error:** `ModuleNotFoundError: No module named 'app'` when running uvicorn from `legalai-frontend\legalai-frontend`.

**Cause:** The backend `app` package lives at **project root** (`GP-Legal-AI--main`), not inside the frontend folder.

**Fix:**
- Always start the backend from project root:  
  `cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"` then  
  `uvicorn app.main:api --reload --host 0.0.0.0 --port 8000`
- In **README.md**, **HOW_TO_RUN.md**, and **RUN_AND_VERIFY.md**: add a clear note: “Start uvicorn from the **project root** (where `app/` and `requirements.txt` are), not from the frontend directory.”

---

## 2. Deprecation: `torch_dtype` → `dtype`

**Warning:** `torch_dtype is deprecated! Use dtype instead!`

**Files to update:**
- `llm/lfm_model.py`: in `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)` change to `dtype=...` (same value).
- `app/local_llm.py`: same change in `AutoModelForCausalLM.from_pretrained(...)`.
- `Legal Rag/src/llm_client.py`: in the `from_pretrained` call, replace `"torch_dtype": ...` with `"dtype": ...` in the kwargs dict (if that’s how it’s passed).

**Check:** Hugging Face `from_pretrained` now prefers `dtype=`; `torch_dtype=` may still work but triggers the warning.

---

## 3. Generation flag: `temperature` invalid / ignored

**Warning:** `The following generation flags are not valid and may be ignored: ['temperature'].`

**Cause:** For some models, `model.generate(..., temperature=...)` is ignored when `do_sample=False`, and the API may warn.

**Files to update:**
- `llm/generate.py`: in `generate_answer` and `generate`, when calling `model.generate(...)`:  
  - Either pass `temperature` only when `do_sample=True`, or  
  - Omit `temperature` when `do_sample=False` to avoid the warning.
- `app/local_llm.py`: same logic in `generate(...)`.

**Fix pattern:** Only pass `temperature` (and `do_sample=True`) when you want sampling; when `do_sample=False`, do not pass `temperature`.

---

## 4. Embedding model: two models loading (and UNEXPECTED keys)

**Observed:**  
- “No sentence-transformers model found with name aubmindlab/bert-base-arabertv2. Creating a new one with mean pooling.”  
- Then “BertModel LOAD REPORT” with UNEXPECTED keys.  
- Then “BertModel LOAD REPORT” for `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with UNEXPECTED keys.

**Clarification:**
- **Chroma (Legal RAG):** uses `aubmindlab/bert-base-arabertv2` (with mean pooling). UNEXPECTED keys for AraBERT are expected when wrapped by SentenceTransformer; document or log that once at INFO level so it doesn’t look like a failure.
- **In-memory Retriever (app/rag_utils):** uses `paraphrase-multilingual-MiniLM-L12-v2` and loads from `chunks/` (452 docs). So two different systems = two models. No bug, but should be clear in logs.

**Fixes:**
- In **Legal Rag/src/vector_store.py**: add a single INFO log when loading the primary model, e.g. “Using embedding model X with mean pooling; UNEXPECTED keys for non–sentence-transformers models (e.g. AraBERT) are expected.”
- Optionally: reduce or silence the verbose “LOAD REPORT” from transformers (e.g. env or logging level) so startup isn’t noisy, or document in README that these lines are expected.

---

## 5. 154 vs 452 docs (Chroma vs Retriever)

**Observed:** “ChromaDB RAG ready: 154 docs” and “Retriever initialized with 452 docs.”

**Cause:** Two different indexes:
- **154:** Legal RAG Chroma collection (e.g. `Legal Rag/data/labor14_2025_chunks.cleaned.jsonl` → `Legal Rag/chroma_db`).
- **452:** In-memory Retriever built from `chunks/` (e.g. `chunks/*.jsonl`) in `app/main.py` startup.

**Fix:** In **app/main.py** startup, after both are initialized, print one consolidated line, e.g.:  
`[Startup] RAG: Chroma (Legal RAG)=154 docs; in-memory Retriever (chunks)=452 docs.`  
So the two counts are clearly explained and not mistaken for an error.

---

## 6. Graceful shutdown (CancelledError / KeyboardInterrupt)

**Request:** “Stop scary tracebacks during reload (CancelledError / KeyboardInterrupt). Handle graceful shutdown so CancelledError doesn’t print a full traceback in dev.”

**Fix:**
- In **app/main.py** add a `@api.on_event("shutdown")` handler that performs any needed cleanup (e.g. closing DB, clearing caches) and optionally logs “Shutting down gracefully.”
- Ensure uvicorn isn’t logging full tracebacks for `asyncio.CancelledError` or `KeyboardInterrupt` in dev. This may require a custom logging config or wrapping the server run so these exceptions are caught and not printed as full tracebacks (implementation depends on your uvicorn/version).

---

## 7. Hugging Face token (HF_TOKEN)

**Request:** “Set Hugging Face token to avoid rate limits; document in README + .env.example; still says not present.”

**Fixes:**
- **.env.example** (project root): already has `# HF_TOKEN=`. Ensure it’s clearly documented: “Set HF_TOKEN for Hugging Face model downloads (avoids rate limits). Get token from https://huggingface.co/settings/tokens.”
- **README.md**: add a short “Environment variables” section that lists `GEMINI_API_KEY`, `HF_TOKEN`, and (if used) `LOCAL_LLM_PATH`, and point to `.env.example`.
- **Backend:** Loading of `.env` from project root (and cwd fallback) is already in place; ensure no code path overwrites or ignores `HF_TOKEN` when calling Hugging Face APIs (e.g. Legal Rag vector_store and any model download). If any script or module still prints “HF_TOKEN not present,” track that log and make it conditional (e.g. only warn when doing a download that would benefit from a token).

---

## 8. Frontend–backend sync (LLM, RAG, ML)

**Ensure:**
- **Analyze:** Frontend sends `use_llm=true` when “LLM explanations (LFM)” is checked (`analyzeContract` with `useLlm`, `llmTopK`, `llmMaxNewTokens`). Backend returns `rag_by_violation[*].llm_explanation`; frontend shows it in the Violations tab (already implemented; verify in UI).
- **Document chat (LFM):** Frontend “Chat with document (LFM)” tab calls `POST /chat/document` with `document_context` and `message`. Backend uses LFM (`llm/generate.py` or `app/local_llm.py`). Ensure backend is run from **project root** so `app` and `llm` are importable; then test one message and confirm no 404.
- **AI Chat (Gemini):** Frontend AI Chat page calls `POST /chat/message` with `analysis_id` and `message`. Backend uses `GEMINI_API_KEY` from `.env` at project root. Ensure `.env` exists at project root and contains `GEMINI_API_KEY`; restart backend after adding it.
- **RAG:** Analysis uses Legal RAG (Chroma) when available; retriever fallback uses in-memory index. No frontend change needed if backend startup and endpoints are correct.
- **ML:** Analysis uses `model_ml_predictor` for rule scores and violations. No frontend change needed if backend loads ML artifacts and `/check_clause` / `ocr_check_and_search` respond correctly.

---

## 9. DEVICE=auto (CUDA if available else CPU)

**Request:** “Add DEVICE=auto logic: cuda if available else cpu.”

**Status:** Already implemented in **Legal Rag/src/config.py** (`get_device()`: `DEVICE` env, default `auto` → cuda if `torch.cuda.is_available()` else cpu). Ensure:
- **Legal Rag** uses `config.get_device()` for embeddings and LLM (vector_store, llm_client).
- **app/local_llm.py** and **llm/lfm_model.py** use `device_map="auto"` or equivalent so GPU is used when available; no hardcoded `"cpu"` unless intended.
- **.env.example** documents: `# DEVICE=auto  # auto | cpu | cuda`.

---

## 10. Single consolidated startup report

**Request:** “Add a single consolidated startup report: DB ok, artifacts ok, chroma collection name + doc count, retriever doc count, device, embedding model. Avoid repeating it multiple times during reload.”

**Fix:** In **app/main.py** `startup()`, keep (or refine) the single `[Startup] health: ...` line so it includes:
- DB=ok (or n/a)
- artifacts=ok (or n/a)
- chroma_collection_name + chroma_docs
- retriever_docs (in-memory Retriever count)
- device (cpu/cuda)
- embedding_model (name)

Ensure this block runs only once per process (no duplicate logs on reload). If reload runs startup twice, consider a guard (e.g. a module-level “startup already logged” flag) so the full report is printed only once.

---

## 11. Remove or reduce debug instrumentation (after verification)

After you’ve confirmed in agent mode that:
- Gemini chat works (GEMINI_API_KEY loaded),
- Document chat (LFM) works (no 404, LFM responds),
- Env and routes are as expected,

remove or comment out the debug logging added for the investigation:
- **app/main.py:** `_debug_log`, the env_after_load log, the chat_routes log, and `_DebugChatLogMiddleware` (and its `api.add_middleware`).
- **app/routers/chat.py:** `_chat_debug_log` and the “entry” logs in `chat_message` and `chat_document`.
- **legalai-frontend/legalai-frontend/src/lib/api.ts:** the `fetch(...)` debug logs in `chatWithDocument` (before_post, after_post, post_error).

---

## 12. Optional: silence UNEXPECTED keys at startup

If the transformers “LOAD REPORT” and “UNEXPECTED” lines are too noisy:
- Set `TRANSFORMERS_VERBOSITY=error` (or similar) in the environment before loading models, or
- Configure logging so that the logger used by the model loading code is at WARNING or ERROR level for that library only.

Apply only if you want a quieter startup; functionally those messages are expected for AraBERT/MiniLM in this setup.

---

## Summary table

| # | Item                    | Area        | Status |
|---|-------------------------|------------|--------|
| 1 | Run backend from root  | Docs + UX  | Done – README/HOW_TO_RUN/RUN_AND_VERIFY. |
| 2 | torch_dtype → dtype    | LLM        | Done – llm/lfm_model.py, app/local_llm.py, Legal Rag llm_client. |
| 3 | temperature when do_sample=False | LLM | llm/generate.py, app/local_llm.py. |
| 4 | Embedding load / UNEXPECTED | RAG   | Done – Legal Rag vector_store INFO log; TRANSFORMERS_VERBOSITY in main. |
| 5 | 154 vs 452 docs        | Startup    | Done – Single consolidated [Startup] line (Chroma + Retriever). |
| 6 | Graceful shutdown      | Backend    | Done – @api.on_event("shutdown") in main.py. |
| 7 | HF_TOKEN docs          | Docs + .env| Done – README Environment variables; .env.example. |
| 8 | Frontend–backend sync   | E2E        | Verified – use_llm, /chat/document, /chat/message, .env. |
| 9 | DEVICE=auto            | Config     | Done – Legal Rag config.get_device(); .env.example DEVICE. |
| 10| Single startup report  | Startup    | Done – One line; _startup_report_done guard. |
| 11| Remove debug logs      | All        | Done – main.py, chat.py, api.ts instrumentation removed. |
| 12| Optional: quiet LOAD REPORT | Env/log | Done – TRANSFORMERS_VERBOSITY=error in main.py. |

Use this checklist in **Agent mode** and tick off items as you fix them so the project runs with no errors or warnings and frontend/backend stay in sync for LLM, RAG, and ML.
