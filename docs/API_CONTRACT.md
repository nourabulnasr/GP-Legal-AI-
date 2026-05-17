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

### `POST /ocr_check_and_search` translation/language fields

Request (multipart) additions:
- `source_language_mode`: `auto` or `manual` (default `auto`).
- `source_language_override`: used only when `source_language_mode=manual` and must be a valid language tag.
- `translation_target_lang`: one of `ar`, `en`, `fr`, `de` (default `ar`).
- Translation provider is chosen **server-side** (Google Cloud Translation v2 first, then local LFM; Argos only if both fail for Arabic). Clients must not send a provider field.

Response additions (non-breaking):
- `source_language_mode`: effective source mode used by backend.
- `source_language_effective`: effective source language code after detect/override/fallback.
- `translation_target_lang`: effective output translation language.
- `translated_chunks`: generic translated chunk list (`id`, `page`, `translated_text`).

Existing compatibility fields are still returned:
- `ar_translated_chunks` remains for Arabic-target flows.
- `translation` object still includes `translation_status` and `translation_provider`.
- `language_detection` still includes `language_code`, `confidence`, and `is_mixed`.

Example request (multipart):

```bash
curl -X POST "http://127.0.0.1:8000/ocr_check_and_search" \
  -H "Authorization: Bearer <token>" \
  -F "file=@sample_contract.pdf" \
  -F "use_rag=true" \
  -F "use_ml=true" \
  -F "use_llm=false" \
  -F "translate_to_ar=true" \
  -F "translation_only=false" \
  -F "source_language_mode=auto" \
  -F "translation_target_lang=fr"
```

Example response (trimmed):

```json
{
  "language_detection": {
    "language_code": "de",
    "confidence": 0.78,
    "is_mixed": false,
    "lfm_fallback_enabled": true,
    "lfm_fallback_used": false
  },
  "translation": {
    "translation_status": "ok",
    "translation_provider": "google_cloud_translate_v2",
    "translation_target_lang": "fr",
    "per_chunk": false
  },
  "source_language_mode": "auto",
  "source_language_effective": "de",
  "translation_target_lang": "fr",
  "translated_chunks": [
    { "id": "page_0", "page": 0, "translated_text": "..." }
  ],
  "ar_translated_chunks": []
}
```

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
