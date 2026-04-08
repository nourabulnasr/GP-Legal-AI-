# Run & Verify (Hybrid Legal Pipeline)

This repo now supports a **hybrid legal analysis pipeline**:

OCR (Google Document AI / fallbacks) → Rule Engine (authoritative) → ML assist (severity only) → RAG (ChromaDB) → Local LLM (explanations only)

---

## 0) Prerequisites

- Python 3.11+
- (Optional) NVIDIA GPU + CUDA for faster LLM/embeddings
- Node.js 18+ (for the frontend)
- If you want **Google Document AI OCR**:
  - `GOOGLE_APPLICATION_CREDENTIALS` pointing to your service-account JSON
  - `DOCUMENT_AI_PROJECT_ID`, `DOCUMENT_AI_LOCATION` (default `us`), `DOCUMENT_AI_PROCESSOR_ID`

---

## 1) Install backend dependencies

From repo root:

```powershell
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
pip install -r requirements.txt
```

This installs (among others): `chromadb`, `sentence-transformers`, `faiss-cpu`, `transformers`, `torch`, `google-cloud-documentai`.

Quick check:

```powershell
python -c "import chromadb, sentence_transformers, transformers, torch; print('deps ok')"
```

---

## 2) Verify each functionality (step-by-step)

### Step A — Rule Engine loads

```powershell
python -c "from pathlib import Path; from app.rules import RuleEngine; re=RuleEngine(Path('rules'), Path('laws')); print('rules:', len(re.rules))"
```

Expected: prints a non-zero rules count.

---

### Step B — ChromaDB RAG (Legal Rag bridge or app/rag_chromadb)

Prefers **Legal Rag bridge** (uses `Legal Rag/src` VectorStore + DataIngestion; corpus `Legal Rag/data/labor14_2025_chunks.cleaned.jsonl`; index `Legal Rag/chroma_db`). Falls back to `app/rag_chromadb` (index `./chroma_legal`).

```powershell
python scripts/verify_chromadb_rag.py
```

Expected: prints `OK: ChromaDB RAG verified.` and shows 5 results for `ساعات العمل`.

**Note:** First run with the Legal Rag bridge may download the Arabert embedding model and ingest the corpus; this can take several minutes.

---

### Step C — Local LLM (LFM2.5-1.2B-Instruct) loads from disk

Prefers **llm/generate.py**; fallback: `app/local_llm`. Uses the local folder `./LFM2.5-1.2B-Instruct/` (or env `LOCAL_LLM_PATH`).

```powershell
python scripts/verify_local_llm.py
```

Expected: prints `OK: Local LLM verified (explanation-only path).`

**Note:** First run can take 1–2 minutes while the 1.2B model weights are loaded.

---

### Step C2 — ML model (model_ML predictor)

```powershell
python scripts/verify_ml_model.py
```

Expected: prints `OK: ML model verified.` and evidence lines (top rule_id, score).

---

### Step D — Document AI OCR path (optional)

If you configured Document AI env vars and credentials, start the API (next section) and test OCR using `/ocr_check_and_search` on a PDF. The backend will prefer `app/DocumentAI.py` for PDFs when configured.

---

### Step E — Full pipeline (OCR → Rules → ML → RAG → LLM)

1) Start backend (next section).
2) Call the full endpoint:

```powershell
$file = "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\data\contracts_raw\pdf\Labor contract (1).pdf"

# PowerShell 5.1 doesn't support Invoke-RestMethod -Form, so use curl.exe:
curl.exe -s -X POST "http://localhost:8000/ocr_check_and_search" `
  -F "file=@$file" `
  -F "use_rag=true" `
  -F "use_ml=true" `
  -F "use_llm=true" `
  -F "llm_top_k=2" `
  -F "llm_max_new_tokens=200"
```

Notes:
- This endpoint works **without login** as long as `save=false` (default).
- If you set `save=true`, you must call it with an `Authorization: Bearer <token>` header (login via the frontend or `/auth/login/form`).

Check response fields:
- `pipeline_steps` includes: `ocr`, `rule_engine`, `ml_assist`, `rag`, and `llm` (if enabled)
- `rule_hits` contains authoritative hits from the YAML rule engine
- `rag_by_violation[*].hits` contains retrieved law chunks (ChromaDB preferred)
- `rag_by_violation[*].llm_explanation` exists when `use_llm=true`

---

## 3) Run the backend API

**Important:** The backend must be started from the **project root** (where `app/` and `requirements.txt` are), not from the frontend folder. Running uvicorn from `legalai-frontend\legalai-frontend` will cause `ModuleNotFoundError: No module named 'app'`.

```powershell
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
uvicorn app.main:api --reload --host 0.0.0.0 --port 8000
```

If port **8000** is already in use on your machine, run on **8001** instead:

```powershell
uvicorn app.main:api --reload --host 0.0.0.0 --port 8001
```

Docs:
- Swagger: http://localhost:8000/docs (or http://localhost:8001/docs)

---

## 4) Run the frontend (optional)

```powershell
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main\legalai-frontend\legalai-frontend"
npm install
npm run dev
```

Then open the Vite URL shown in the terminal.

---

## 5) Verification evidence

After running the verification scripts, you should see **evidence** lines that confirm each component is working:

| Script | Evidence to expect |
|--------|--------------------|
| `python scripts/verify_chromadb_rag.py` | `[Evidence] document_count=N` and `[Evidence] first_result_snippet: "..."` (first search result text preview). |
| `python scripts/verify_ml_model.py` | `[Evidence] ml_used=True`, top 1–2 `rule_id` and `score`, and `rule_scores_to_rule_hits` count. |
| `python scripts/verify_local_llm.py` | `[Evidence] generate_output_preview: "..."` and `[Evidence] explain_violation_output_preview: "..."` (first ~150 chars of LFM output). |

To run all three from project root:

```powershell
python scripts/verify_chromadb_rag.py; python scripts/verify_ml_model.py; python scripts/verify_local_llm.py
```

Expect one `OK:` line per script when the component is available and working.

---

## 6) Common issues

### ChromaDB returns 0 docs
- Ensure the corpus exists:
  - `Legal Rag\data\labor14_2025_chunks.cleaned.jsonl`
  - or `chunks\labor14_2025_chunks.cleaned.jsonl`

### Local LLM is slow / uses too much RAM
- First load is slow on CPU. Prefer GPU if available.
- Reduce usage by setting `use_llm=false` except when needed.

### Document AI returns empty text
- Verify env vars and `GOOGLE_APPLICATION_CREDENTIALS`.

---

## 6) Integration (Legal Rag + llm folders)

- **RAG**: The app prefers `app/legal_rag_bridge`, which uses **Legal Rag/src** (`VectorStore`, `RAGEngine`, `DataIngestion`) with paths resolved from project root. Corpus: `Legal Rag/data/labor14_2025_chunks.cleaned.jsonl`; ChromaDB: `Legal Rag/chroma_db`. If the bridge is unavailable, the app falls back to `app/rag_chromadb` (index under `./chroma_legal`).
- **LLM**: The app prefers **llm/generate.py** (`explain_violation`, `generate`, `is_available`) for violation explanations; fallback: `app/local_llm`. Both use the same local model path (`LOCAL_LLM_PATH` or `./LFM2.5-1.2B-Instruct`).

---

## 8) GPU (CUDA)

- **LLM** and **embeddings** (Legal Rag’s Arabert / sentence-transformers) use CUDA when available (`torch.cuda.is_available()`). The local LLM uses `device_map="auto"` and `torch.float16` on GPU; `cudnn.benchmark=True` is set for faster inference.
- To force GPU: set `CUDA_VISIBLE_DEVICES` (e.g. `0`) before starting. To force CPU: set `CUDA_VISIBLE_DEVICES=""` or ensure no GPU drivers are visible to PyTorch.
- Heavy work is in PyTorch/transformers; numba/GIL alternatives are not used for this pipeline.

