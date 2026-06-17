# Fine-tune LFM for explain-clause (Google Colab)

Use this when you have **GPU on Colab** and want to improve clause-explanation quality. The **canonical inference path** in production is `app/local_llm.py` loading **`LiquidAI/LFM2.5-1.2B-Instruct`** (or your fine-tuned adapter merged into that folder).

## What to export from Colab

1. Train LoRA or full fine-tune on your `explain_clause` JSONL (instruction → completion).
2. Merge adapters into a **single HuggingFace-style directory** with `config.json`, tokenizer, and weights (same layout as a `transformers` snapshot).
3. Zip the folder and download, or push to a private Hugging Face model repo.

## Integrate locally

1. Replace or symlink **`LFM2.5-1.2B-Instruct/`** at the repo root, **or** set **`LOCAL_LLM_PATH`** to the new folder path.
2. Confirm **`config.json`** is real JSON (not Git LFS pointer): `python -c "import json; json.load(open('config.json'))"`.
3. Restart the API and run **`docs/DEMO_SCRIPT.md`** journey A4 (Explain clause).

## Rollback

Keep a copy of the previous model directory; swap folders or `LOCAL_LLM_PATH` back.

## Note

`llm/generate.py` **`explain_violation`** delegates to **`app.local_llm`** when available, so one checkpoint feeds both **`/ocr_check_and_search`** (embedded explain) and **`/legato/explain-clause`**.
