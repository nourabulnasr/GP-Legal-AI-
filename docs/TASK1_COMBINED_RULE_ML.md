# Task 1: Combined Rule-Based + ML Pipeline (Implementation Summary)

This document summarizes what was implemented to **combine the best of rule-based matching and the described ML system**, and how to **test accuracies and outputs**. The design is ready for **300+ contracts** (contract-disjoint split, full metrics).

---

## What Was Implemented

### 1. BM25 law retriever (`app/bm25_law_retriever.py`)

- **BM25-style** retriever over law article chunks (no extra dependency).
- Used to fetch **top-K relevant labor-law passages** per clause so the model can see both contract text and law (RAG-style).
- `load_law_docs_for_bm25(chunks_dir)` loads from `chunks/*.jsonl`; `BM25LawRetriever.build_index(docs)` then `search(query, top_k)`.

### 2. Unified training (`ml/scripts/train_unified.py`)

- **Clause-level** data from `data/dataset/train_all.jsonl`, `silver_dataset.jsonl`, `augmented.jsonl`.
- **Contract-disjoint split**: train / val / test use **different contracts** (e.g. 64% / 16% / 20% of contracts) so evaluation is on unseen documents.
- **Two modes**:
  - **Contracts-only**: input = clause text only.
  - **Law-aware** (`--law_aware`): BM25 retrieves top-K law chunks per clause; input = clause + `[LAW]` + law texts.
- **Features**: TF-IDF with **character n-grams** (3–5) for Arabic/OCR; **LogisticRegression** with `class_weight="balanced"`.
- **Target**: **Binary** “has violation” = any rule fired on the clause (silver labels from RuleEngine).
- **Threshold**: Tuned on **validation F1**.
- **Metrics**: **Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix**; **GroupKFold** (5 folds) on training only for stability (mean ± std F1).
- **Artifacts**: `app/ml/artifacts/unified/` — `vectorizer.joblib`, `model.joblib`, `thresholds.json`, `metrics.json`, `config.json`.

**Commands:**

```bash
# From project root
python ml/scripts/train_unified.py              # contracts-only
python ml/scripts/train_unified.py --law_aware  # law-aware (BM25 + clause)
```

### 3. Unified predictor (`app/unified_predictor.py`)

- Loads artifacts from `app/ml/artifacts/unified/`.
- **`predict_violation_risk(text)`** → `(score, above_threshold)`.
- Optional **law at inference**: if the model was trained with `--law_aware`, you can pass `use_law_at_inference=True` to append retrieved law to the clause before scoring.

### 4. API integration (`app/main.py`, `app/schema.py`)

- **`/check_clause`**: Response now includes **`unified_ml_risk`** and **`unified_ml_above_threshold`** (clause-level violation risk from the unified model), in addition to rule hits.
- **`/ocr_check_and_search`**: Response includes:
  - **`clause_level_unified_risks`**: list of `{start, end, text_preview, unified_ml_risk, unified_ml_above_threshold}` per clause.
  - **`full_text_unified_risk`**: document-level risk score (0–1).

**Flow:** Rules still define *what* is a violation (regex + logic). The unified model adds a **clause-level risk score** that can be used for ranking, filtering, or human review. Both outputs are combined in the same response.

---

## How to Test Accuracies and Outputs

### 1. Train and read metrics

After running:

```bash
python ml/scripts/train_unified.py
python ml/scripts/train_unified.py --law_aware --top_k_law 3
```

- **Test-set metrics** are printed in the terminal and saved in **`app/ml/artifacts/unified/metrics.json`** (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, GroupKFold F1 mean ± std).

### 2. Run the evaluation report script

```bash
python scripts/run_unified_evaluation.py
```

- Trains both modes and writes **`reports/unified_evaluation_report.json`** with metrics for each mode and a short summary (e.g. best F1 mode, F1/ROC-AUC/PR-AUC per mode).

### 3. Call the API

- **Single clause:** `POST /check_clause` with `clause_text` → response includes `unified_ml_risk`, `unified_ml_above_threshold`, and `matches` (rule hits).
- **Full document:** `POST /ocr_check_and_search` with a file → response includes `clause_level_unified_risks`, `full_text_unified_risk`, and `rule_hits`.

---

## Example Metrics (Current Data)

With the current dataset (~19 contracts, ~1117 clause samples), typical numbers after running `train_unified.py`:

| Mode            | F1 (test) | ROC-AUC | PR-AUC | GroupKFold F1 (mean ± std) |
|-----------------|-----------|---------|--------|----------------------------|
| contracts_only  | ~0.95     | ~0.94   | ~0.99  | ~0.94 ± 0.01               |
| law_aware       | ~0.96     | ~0.94   | ~0.99  | ~0.94 ± 0.00               |

When you add more contracts (e.g. 300+), use the same commands; the **contract-disjoint split** and **GroupKFold** will keep evaluation valid across contracts.

---

## Files Touched / Added

| Path | Role |
|------|------|
| `app/bm25_law_retriever.py` | BM25 over law chunks for law-aware training/inference |
| `ml/scripts/train_unified.py` | Unified clause-level training (contract-disjoint, binary, char n-grams, full metrics) |
| `app/unified_predictor.py` | Load unified model; expose `predict_violation_risk` |
| `app/schema.py` | `unified_ml_risk`, `unified_ml_above_threshold` on clause response |
| `app/main.py` | Unified predictor import; `/check_clause` and `/ocr_check_and_search` return unified risk |
| `scripts/run_unified_evaluation.py` | Run both modes and write `reports/unified_evaluation_report.json` |
| `docs/TASK1_COMBINED_RULE_ML.md` | This summary |

---

## Relation to the Described System (300+ Contracts)

- **Same ideas:** Clause-level binary “violation” signal, contract-disjoint split, law-aware input (BM25 + top-K law), char n-grams, logistic regression, class balancing, threshold on val F1, PR-AUC / ROC-AUC / F1 / confusion matrix, GroupKFold on training.
- **Current labels:** Silver (RuleEngine) until you have 300+ with human or marker-based labels; the pipeline and evaluation stay the same when you switch labels.
- **Combined with rules:** RuleEngine remains the source of *which* rule fired and *why*; the unified model adds a single **violation risk** per clause for prioritization and testing.
