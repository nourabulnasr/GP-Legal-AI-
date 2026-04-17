# API contract: backend ↔ legato_mobile1

Base URL: `AppConfig.apiBaseUrl` in Flutter (e.g. `http://10.0.2.2:8000` for Android emulator).

## Auth

| Client | Endpoint | Notes |
|--------|----------|--------|
| `ApiClient` | `POST /auth/register`, `POST /auth/login`, `POST /auth/verify-email` | Bearer JWT on subsequent calls |

## Analyses (not under `/legato`)

| Dart `LegatoApi` | HTTP | Body / query |
|--------------------|------|----------------|
| `analyzeContract` | `POST /ocr_check_and_search` | multipart: `file`, `use_rag`, `use_ml`, `use_llm`, etc. |
| `listAnalyses` | `GET /analyses` | — |
| `getAnalysis` | `GET /analyses/{id}` | — |
| `saveAnalysisToDb` | `POST /analyses` | JSON: `filename`, `result_json`, optional metadata |
| `deleteAnalysis` | `DELETE /analyses/{id}` | — |
| Admin methods | `/analyses/admin/*` | admin only |

## Chat (Gemini = navigation / contract Q&A per product rules)

| Dart | HTTP | Notes |
|------|------|--------|
| `chatMessage` | `POST /chat/message` | `analysis_id`, `message`, `history` |
| `chatAssistant` | `POST /chat/assistant` | General Gemini assistant |
| `chatWithDocument` | `POST /chat/document` | Local LFM + optional Gemini fallback |

## Legato (`/legato/*`) — LFM for explain/summarize/compare/negotiation-fallback patterns

| Dart | HTTP | Response keys (success) |
|------|------|-------------------------|
| `explainClause` | `POST /legato/explain-clause` | `explanation`, `language`, optional `rule_id`, `sources` |
| `summarizeClauses` | `POST /legato/summarize-clauses` | `summaries` (list of `{clause_index, summary}`) |
| `compareContracts` | `POST /legato/compare` | `comparison` |
| `negotiationChat` | `POST /legato/negotiation-chat` | `content` |
| `riskSummary` | `GET /legato/risk/{analysisId}` | risk fields from stored analysis JSON |
| `createShare` | `POST /legato/shares` | `token`, `expires_at` |
| `publicShare` | `GET /legato/shares/public/{token}` | shared payload |
| Deal threads / messages | `/legato/deal-threads/*` | JSON lists / message objects |
| Timeline | `POST /legato/timeline/events`, `GET /legato/timeline/all` | admin for `all` |
| Profile | `GET/PUT /legato/profile/me` | profile JSON |
| `listNetworkProfiles` | `GET /legato/network/profiles` | list |
| `recordSignature` | `POST /legato/signatures` | acknowledgment |

## Law admin (web or HTTP client)

| Endpoint | Role |
|----------|------|
| `POST /admin/law/preview-pdf` | admin |
| `POST /admin/law/upload-pdf` | admin |
| `POST /admin/law/reindex` | admin |
| `GET /admin/law/jobs/{job_id}` | admin |
| `POST /admin/law/sync-from-url` | admin |

## Canonical LFM entry

Clause explanation and related **local** generation use **`app/local_llm.py`** (`generate`, `build_explanation_prompt`, `explain_violation`). The `llm/` package may delegate to the same path when `app.local_llm` is available.
