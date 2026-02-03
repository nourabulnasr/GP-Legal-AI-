GP — Legal AI (Egypt)



AI assistant to review Egyptian legal contracts.

v0.1 delivers a 10% demo: /health, /ocr, /check\_clause with 1 labor rule + suggestion.



1\) Repo layout

legalai/

&nbsp; app/                      # FastAPI service

&nbsp; laws/                     # ✔ output: parsed laws as JSON (commit to git)

&nbsp; rules/                    # ✔ handwritten YAML rules (commit to git)

&nbsp; seyagh/                   # ✔ output: cleaned clause snippets .txt (commit to git)

&nbsp; data/

&nbsp;   raw/                    # ✖ raw PDFs (do NOT commit)

&nbsp;     laws/

&nbsp;     seyagh/

&nbsp;     contracts/

&nbsp;   processed/              # optional per-page text dumps (don’t commit)

&nbsp;   contracts\_jsonl/        # test clause corpora (ok to commit if small)

&nbsp; scripts/                  # preprocessing tools (commit)

&nbsp;   ocr\_pdf.py

&nbsp;   law\_pdf\_to\_json.py

&nbsp;   seyagh\_book\_to\_txt.py

&nbsp; Dockerfile

&nbsp; requirements.txt

&nbsp; .dockerignore

&nbsp; .gitignore

&nbsp; README.md





.gitignore must include:



data/raw/\*\*

\*.pdf

chroma/\*\*

\*.ipynb\_checkpoints



2\) Branching \& PR workflow



Main rule: Nobody commits to main directly.

Branch name: feat/<name>-<topic> or fix/<name>-<topic>

Examples:



feat/aly-ocr-baseline

feat/sondos-clause-splitter

feat/mariam-ner-regex

feat/nour-preprocess-labor

feat/youssef-status-export





Steps:



\# 1) get latest

git checkout main

git pull



\# 2) create branch

git checkout -b feat/aly-ocr-baseline



\# 3) work, then stage \& commit

git add .

git commit -m "ocr: baseline extractor (pymupdf + tesseract fallback)"



\# 4) push branch

git push --set-upstream origin feat/aly-ocr-baseline



\# 5) open Pull Request (PR) on GitHub

\# 6) at least 1 reviewer approves (Nour = default reviewer)

\# 7) squash merge to main





Commit style: short prefix + detail

Examples: rules: add 5 labor YAML rules, ocr: add normalization, docs: add run guide.



3\) Running the API (Docker)



You only need Docker to run the service. Preprocessing can be run either on host Python or inside the container (see §6).



Build image

cd C:\\Users\\noura\\legalai

docker buildx build --load -t legalai-10pc .



Run container

docker run --rm -p 8000:8000 --name legalai legalai-10pc





Open: http://127.0.0.1:8000/docs (Swagger) and http://127.0.0.1:8000/redoc.



Dev mode (live-edit with bind mount)



If you want container to read local files live (no rebuild):



docker run --rm -p 8000:8000 `

&nbsp; --mount type=bind,source="C:\\Users\\noura\\legalai",target=/app `

&nbsp; --name legalai legalai-10pc `

&nbsp; uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

(Run from **project root** where `app/` and `requirements.txt` are; do not run from the frontend directory or you will get `ModuleNotFoundError: No module named 'app'`.)





Edit files on your machine → container sees them immediately.



Use this when you’re tweaking .yaml, .json, .txt during development.



Stop container:



docker stop legalai  # or CTRL+C in the window running it



**Environment variables**

Copy `.env.example` to `.env` in the project root and set:

- **GEMINI_API_KEY** — For the contract-aware chatbot (Gemini). Get a key at https://aistudio.google.com/apikey
- **HF_TOKEN** — For Hugging Face model downloads (Legal RAG embeddings, LFM). Avoids rate limits; get a token at https://huggingface.co/settings/tokens
- **LOCAL_LLM_PATH** — (Optional) Path to local LFM model folder for document chat and LLM explanations
- **DEVICE** — (Optional) `auto` (default), `cpu`, or `cuda` for model inference

See `.env.example` for all options.



4\) Endpoints (10% demo)



GET /health → {"ok": true}



POST /check\_clause



{

&nbsp; "clause\_text": "يحق لصاحب العمل فصل العامل دون سبب.",

&nbsp; "law\_scope": \["labor"],

&nbsp; "language": "ar"

}





Returns matches with law/article \& suggestion.



POST /ocr → upload file (pdf/png/jpg) → returns extracted text.



Swagger gotcha: the page shows both 200 and 422 blocks. The 422 block is just an example of an error schema—not your result.



5\) What goes to Git vs stays local?

Item	Where to store	Commit to Git?

Raw PDFs (laws, Seyagh, contracts)	data/raw/\*\*	❌ NO

Parsed laws (articles JSON)	laws/\*.json	✅ YES

Seyagh snippets (.txt)	seyagh/\*\*	✅ YES

Rules (YAML)	rules/\*.yaml	✅ YES

Scripts (preprocessing tools)	scripts/\*\*	✅ YES

Per-page OCR dumps	data/processed	❌ usually NO



Reason: PDFs are heavy; Git will bloat. Outputs and scripts are needed by everyone → commit them.



6\) Preprocessing — how \& where to run



Preprocessing can be done in two ways:



Option A — Run scripts on your host (recommended for now)



Requirements: Python 3.11, pip install -r requirements.txt



Put PDFs locally:



data/raw/laws/قانون رقم ١٤ لسنة ٢٠٢٥ قانون العمل.pdf

data/raw/laws/egyptian civil code.pdf

data/raw/laws/commercial code.pdf

data/raw/seyagh/seyagh\_book.pdf





Extract text (debug):



python scripts/ocr\_pdf.py "data/raw/laws/قانون رقم ١٤ لسنة ٢٠٢٥ قانون العمل.pdf" > data/processed/labor\_pages.json





Law PDF → JSON (articles)



python scripts/law\_pdf\_to\_json.py \\

&nbsp; "data/raw/laws/قانون رقم ١٤ لسنة ٢٠٢٥ قانون العمل.pdf" \\

&nbsp; "قانون العمل" "14/2025" \\

&nbsp; laws/labor\_14\_2025.json





Seyagh book → snippets



python scripts/seyagh\_book\_to\_txt.py \\

&nbsp; "data/raw/seyagh/seyagh\_book.pdf" \\

&nbsp; employment \\

&nbsp; seyagh/employment





Then commit the outputs in laws/ and seyagh/.



Option B — Run scripts inside Docker (when Python setup is messy)



Start a shell in the image:



docker run --rm -it `

&nbsp; --mount type=bind,source="C:\\Users\\noura\\legalai",target=/app `

&nbsp; --name legalai-dev legalai-10pc bash





Now inside the container:



python scripts/law\_pdf\_to\_json.py "data/raw/laws/..." "قانون العمل" "14/2025" laws/labor\_14\_2025.json

python scripts/seyagh\_book\_to\_txt.py "data/raw/seyagh/seyagh\_book.pdf" employment seyagh/employment





Because we mounted the repo, the outputs appear on your Windows folder too.



When to use Docker for preprocessing?



If your local Tesseract/Arabic pack isn’t installed.



If you want the exact same environment as teammates.

Otherwise, run on host.



7\) Who preprocesses what (split across team)



We have 4 PDFs to process:



Source PDF	Owner (first pass)	Output

Labor Law (14/2025)	Nour	laws/labor\_14\_2025.json

Civil Code (131/1948)	Mariam	laws/civil\_131\_1948.json

Commercial Code (17/1999)	Sondos	laws/commercial\_17\_1999.json

Seyagh Book	Aly	seyagh/<family>/\*.txt + data/seyagh\_catalog.csv



After first pass, we review and fix article-splitting regex for each law (headers differ). Keep scripts simple; we’ll refine.



8\) Expanding rules (YAML)



Template (Arabic-friendly, UTF-8, no BOM):



\- id: LABOR\_69\_UNLAWFUL\_TERMINATION

&nbsp; scope: "labor"

&nbsp; law: "Labor Law 12/2003"      # or "Labor Law 14/2025" when confirmed

&nbsp; article: "69"

&nbsp; description: "حالة فصل العامل دون سبب مشروع"

&nbsp; language: "ar"

&nbsp; severity: "high"

&nbsp; match:

&nbsp;   any:

&nbsp;     - pattern: "فصل\\\\s+العامل.\*(?:دون|بدون)\\\\s+(?:سبب|مبرر)"

&nbsp;     - pattern: "فصل\\\\s+تعسف(?:ي|يًا)"

&nbsp;   flags: "iu"

&nbsp; suggestion\_ref: "seyagh/employment/termination\_fair\_cause.txt"

&nbsp; rationale: "الفصل دون سبب مشروع يعد باطلاً."





Save as: rules/labor.yaml.

Test quickly: call /check\_clause in Swagger.



9\) Updating files so Docker picks them up



If running normal docker run (no mount):

After you edit laws/, rules/, or seyagh/, you must rebuild:



docker buildx build --load -t legalai-10pc .

docker stop legalai

docker run --rm -p 8000:8000 --name legalai legalai-10pc





If running with bind mount (--mount … target=/app):

No rebuild needed. Files update live.



Copy a file out of a running container (rare):



docker cp legalai:/app/laws/labor\_14\_2025.json C:\\Users\\noura\\legalai\\laws\\



10\) Arabic text, JSON \& common pitfalls



Use UTF-8 (no BOM) for .json, .yaml, .txt. On Windows PowerShell:



(Get-Content .\\path\\file.txt) | Set-Content -Encoding utf8 .\\path\\file.txt





In Swagger, paste Arabic via Notepad to avoid invisible RTL characters.



422 JSON error means invalid JSON (often trailing comma or smart quotes).



11\) Verification checklist (for each PR)



&nbsp;Preprocessing script runs with command shown in PR description.



&nbsp;New files in laws/ or seyagh/ are small \& UTF-8 (no BOM).



&nbsp;If rules changed, /check\_clause tested with a sample clause and screenshot attached.



&nbsp;No files from data/raw/\*\* included in the PR.



&nbsp;Docker run tested: http://127.0.0.1:8000/docs works locally.





12\) Who does what (Sprint-1 + Preprocess in parallel)



Nour (Lead): rules + preprocessing scripts; integrate parsed laws; review PRs.



Aly (A): OCR baseline + Seyagh parsing; contribute snippets.



Sondos (B): clause splitter + commercial law JSON.



Mariam (C): regex/NER helpers + civil code JSON.



Youssef (D): /status, /result, /export/json (stubs now; real data later).



13\) FAQ



Q: I added a file inside the container but others don’t see it. Why?

A: The container FS is isolated. Either run with bind mount or docker cp the file out and commit to Git.



Q: When do I use Docker for preprocessing?

A: Only if your local Python/Tesseract is problematic. Otherwise run scripts on host \& commit outputs.



Q: Where are the “AI” parts?

A: This sprint is about data + rule engine plumbing. RAG + embeddings (MiniLM + ChromaDB) come after we have clean Seyagh snippets and reliable law JSON.

