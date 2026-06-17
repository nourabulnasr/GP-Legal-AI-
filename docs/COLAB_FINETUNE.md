# Fine-tune LFM for explain-clause (Google Colab)

Use this when you have **GPU on Colab** and want to improve clause-explanation quality. Production loads **`LiquidAI/LFM2.5-1.2B-Thinking`** as the base model and applies the legal LoRA via **`LOCAL_LORA_PATH`** (`app/local_llm.py`).

## What to export from Colab

1. Train LoRA on your `explain_clause` JSONL (instruction → completion) using base **`liquidai/LFM2.5-1.2B-Thinking`** (not Instruct).
2. Export the **root adapter folder** only: `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.
3. Zip and upload to the VPS, or push to a private Hugging Face model repo.

Alternatively, merge adapters into a single HuggingFace directory and set **`LOCAL_LLM_PATH`** only (no `LOCAL_LORA_PATH`).

## Integrate locally / Docker

1. Base: `models/LFM2.5-1.2B-Thinking/` (HuggingFace snapshot).
2. Adapter: `models/out_adapter/` (from your zip’s `out_adapter/` folder).
3. `.env` / compose:
   - `LOCAL_LLM_HOST_PATH=./models/LFM2.5-1.2B-Thinking`
   - `LOCAL_LORA_HOST_PATH=./models/out_adapter`
4. Rebuild backend: `docker compose build backend && docker compose up -d backend`
5. Run **`python scripts/verify_local_llm.py`** and **`docs/DEMO_SCRIPT.md`** journey A4.

## VPS switch (existing deployment)

```bash
scp out_adapter.zip root@YOUR_VPS:/root/
scp deploy/hostinger/switch-to-thinking-lora.sh root@YOUR_VPS:/root/
ssh root@YOUR_VPS 'bash /root/switch-to-thinking-lora.sh /root/out_adapter.zip'
```

## Rollback

Keep `LFM2.5-1.2B-Instruct.backup` on the VPS. Set `LOCAL_LLM_HOST_PATH` back to Instruct, remove or unset `LOCAL_LORA_HOST_PATH`, restart backend.

## Note

`llm/generate.py` **`explain_violation`** delegates to **`app.local_llm`** when available, so one checkpoint feeds both **`/ocr_check_and_search`** (embedded explain) and **`/legato/explain-clause`**.

---

## Fine-tuning upgrades for better explain-clause quality

The current adapter (268 steps, ~608 train examples) is a solid start but production still differs from training in several ways. Address these in the **next Colab run**:

### 1. Match production prompt limits (fixed in code)

Training used **1500 chars** for clause text and law articles; production previously truncated to **300 / 200**. `app/local_llm.py` now uses **1500** to match `data/dataset/build_explain_clause_dataset.py`. Re-train after regenerating the dataset if you changed prompts.

### 2. Fix training labels — wrong law articles

Many examples use **keyword fallback** when a rule has no `article` in YAML, which pairs clauses with **irrelevant articles** (e.g. definitions in المادة 1 instead of المادة 89). Upgrade:

- Add explicit `article:` to every rule in `rules/*.yaml`.
- Regenerate: `python data/dataset/build_explain_clause_dataset.py`
- Drop rows where `meta.article` is null and law text is a weak keyword match.

### 3. Use production RAG context in training

Production retrieves law via **Chroma + live RAG** (`legato_service.run_explain_clause`), not only static YAML articles. Upgrade:

- Export real `(clause, rule_id, rag_hits[])` tuples from staging API logs or a script that calls `_rag_search_labor_corpus` for each training clause.
- Build instructions with the **same** `law_articles` the API would send (up to 6 hits, 1500 chars each).

### 4. Train on structured output format consistently

Target answers should always follow:

```
وفقًا للمادة X من …: «اقتباس قصير»

سبب المشكلة:
…

كيف يرتبط ذلك بالنص:
…

التصحيح المقترح:
…
```

Use `build_output_ar()` in `build_explain_clause_dataset.py` for all Arabic rows; add **real `suggestion_ref` text** in rules instead of generic “أعد صياغة البند…”.

### 5. Add negative / insufficient-context examples

~5% of val set should be `"لا يوجد نص قانوني كافٍ في المستند."` when `law_str` is empty (you have some; add more for cross-border / warning rules).

### 6. Include `analysis_id` flow

When users pass `analysis_id`, production fills **real `rule_id`, `description`, `matched_text`, RAG hits** from stored analysis. Add training rows where:

- `description` comes from rule engine hit (not generic “User requested explanation”).
- `matched_text` is the **hit span**, not the full pasted clause.

### 7. English explanations (optional)

If `language=en` is used, add **50+ English** instruction/output pairs; current training is Arabic-only.

### 8. Hyperparameters (next run)

| Setting | Current (approx.) | Suggested |
|---------|-------------------|-----------|
| Epochs | 2 | 3–4 with early stopping on eval_loss |
| LoRA r | 16 | 16–32 (try 32 if GPU allows) |
| Dataset size | ~608 | 1500+ after RAG-aligned rebuild |
| Checkpoint | step 268 | Use **checkpoint-250** (best eval_loss 0.21) or early-stop |
| Eval | every 50 steps | Keep; add **human eval** on 20 held-out real contracts |

### 9. Do not use Thinking chat template for SFT

Keep **flat instruction prompts** (same as `build_explanation_prompt`) — do not wrap training in `<|im_start|>` / thinking blocks unless you also change production to `tokenizer.apply_chat_template()`.

### 10. Regenerate dataset before re-training

```bash
python data/dataset/build_explain_clause_dataset.py
# Inspect explain_clause_train.jsonl — verify article/clause alignment
```

Then Colab: base `liquidai/LFM2.5-1.2B-Thinking`, PEFT 0.18.1, same `target_modules` as exported `adapter_config.json`.
