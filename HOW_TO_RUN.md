# LegalAI – How to Run

## Run backend from project root (important)

Start uvicorn from the **project root** (where `app/` and `requirements.txt` are), not from the frontend directory. Running uvicorn from `legalai-frontend\legalai-frontend` will cause `ModuleNotFoundError: No module named 'app'`.

```powershell
cd "C:\Users\Aly ahmed\Desktop\GP-Legal-AI--main"
uvicorn app.main:api --reload --host 0.0.0.0 --port 8000
```

---

## Docker (recommended)

### Full stack (backend + frontend)

```bash
# Build and run everything
docker compose up --build

# Backend API:  http://localhost:8000
# Frontend UI:  http://localhost:5173
# Docs:         http://localhost:8000/docs
```

### Backend only

```bash
# Build
docker build -t legalai .

# Run
docker run -p 8000:8000 legalai

# With volume for persistence
docker run -p 8000:8000 -v $(pwd)/legalai.db:/app/legalai.db legalai
```

### PowerShell (Windows)

```powershell
# Full stack
docker compose up --build

# Backend only
docker build -t legalai .
docker run -p 8000:8000 legalai
```

### Test OCR + Check (PowerShell)

```powershell
$file = "data/contracts_raw/pdf/Labor contract (1).pdf"
Invoke-RestMethod -Uri "http://localhost:8000/ocr_check_and_search" -Method Post -Form @{
  file = Get-Item $file
  use_rag = $true
  use_ml = $true
  save = $false
} -Headers @{Authorization="Bearer YOUR_TOKEN"}  # if auth required
```

For unauthenticated testing (if auth is disabled), omit the `Authorization` header.

### Test with curl (Bash/WSL)

```bash
curl -X POST "http://localhost:8000/ocr_check_and_search" \
  -F "file=@data/contracts_raw/pdf/Labor contract (1).pdf" \
  -F "use_rag=true" \
  -F "use_ml=true" \
  -F "save=false"
```

---

## Evaluation

### Local

```bash
# All contracts
python scripts/run_evaluation.py

# Strict: held-out test contracts only (unbiased metrics)
python scripts/run_evaluation.py --strict
```

Output: `reports/evaluation_results.json` + console summary.

### Inside Docker

```bash
docker exec -it <container_name> python scripts/run_evaluation.py
docker exec -it <container_name> python scripts/run_evaluation.py --strict
```

Or run in a one-off container:

```bash
docker run --rm -v $(pwd):/app legalai python scripts/run_evaluation.py --strict
```

### Stricter evaluation workflow

1. Edit `data/held_out_test_contracts.txt` to list test-only contract IDs.
2. Retrain ML (excludes held-out): `python -m app.train_ml_predictor`
3. Run strict eval: `python scripts/run_evaluation.py --strict`

---

## Clause Classifier Training

```bash
python ml/scripts/train_clause_classifier.py
```

Artifacts: `app/ml/artifacts/clause_classifier/`

---

## Google Document AI (Optional OCR)

For better PDF text extraction (especially scanned documents), set these env vars:

```env
DOCUMENT_AI_PROJECT_ID=your-gcp-project-id
DOCUMENT_AI_LOCATION=us
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

1. Create a service account in GCP Console and download the JSON key.
2. Use **Document OCR** or **Form Parser** processor in Document AI.
3. Mount the key when running Docker: `-v C:\path\to\key.json:/app/key.json` and set `GOOGLE_APPLICATION_CREDENTIALS=/app/key.json`

Example Docker run with Document AI:

```powershell
docker run -p 8000:8000 -v ${PWD}:/app -v C:\path\to\key.json:/app/key.json -e DOCUMENT_AI_PROJECT_ID=my-project -e DOCUMENT_AI_PROCESSOR_ID=abc123 -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json legalai
```

When configured, the app uses Document AI for PDFs first; falls back to PyMuPDF + Tesseract if unavailable.

---

## RAG Embedding Upgrade

Requires `sentence-transformers` and `faiss-cpu` (in `requirements.txt`).

The app uses embedding-based retrieval when these packages are installed; otherwise it falls back to the hash-based retriever.
