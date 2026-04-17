# Legato demo script and acceptance criteria

Use this checklist for smoke tests and releases. All steps assume backend (`GP-Legal-AI--main`) is running with a configured DB and (for full LLM) `LOCAL_LLM_PATH` or `LFM2.5-1.2B-Instruct` present; mobile (`legato_mobile1`) points to the same API base URL.

## Journey A — Mobile user (core)

| Step | Action | Pass criteria |
|------|--------|----------------|
| A1 | Register or sign in | JWT stored; home/dashboard loads |
| A2 | Upload a PDF contract (analyze flow) | `POST /ocr_check_and_search` returns JSON with `rule_hits`, `ocr_chunks` |
| A3 | Open history | `GET /analyses` lists the saved analysis |
| A4 | Open Explain clause; paste clause text; optional analysis id | `POST /legato/explain-clause` returns `explanation` (non-empty or clear LLM error) |
| A5 | Set output language (AR/EN) if UI offers it | Response `language` matches selection; explanation text matches language |
| A6 | Chat — analysis-scoped or assistant | `POST /chat/message` or `/chat/assistant` returns `content` (Gemini when configured) |

## Journey B — Admin (law corpus)

| Step | Action | Pass criteria |
|------|--------|----------------|
| B1 | Sign in as admin | Role `admin` |
| B2 | Call `POST /admin/law/reindex` or upload PDF per `docs/NEW_LEGAL_RAG.md` | Job completes; Chroma/retriever reload per job `result` |
| B3 | Re-run A2/A4 | RAG-backed explanations still cite labor corpus |

## Journey C — Phase 5 features (optional)

| Step | Action | Pass criteria |
|------|--------|----------------|
| C1 | Risk summary for analysis id | `GET /legato/risk/{id}` returns JSON with risk fields from stored analysis |
| C2 | Summarize clauses | `POST /legato/summarize-clauses` returns `summaries` array |
| C3 | Create share | `POST /legato/shares` returns `token`; public URL loads payload |

Failure on any **required** row (A1–A5, B1–B2 for admin demo) blocks release until fixed.
