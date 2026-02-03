# LegalAI – Requirements Checklist

## Target System Requirements

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| (1) | Extract text (PDF/OCR) | ✅ | Existing in `main.py`: PyMuPDF + OCR fallback |
| (2) | Normalize Arabic | ✅ | Existing in `utils_text.py`: `norm_ar`, `normalize_for_rules` |
| (3) | Split contracts into clauses | ✅ **Added** | `utils_text.split_into_clauses()`, used in `/ocr_check_and_search`, response includes `clauses` |
| (4) | ML entity/clause extraction | ✅ **Added** | Rule-level ML predictor (existing) + clause classifier training script `ml/scripts/train_clause_classifier.py` |
| (5) | Knowledge base of Egyptian labor law | ✅ | `laws/raw/`, `chunks/labor14_2025_chunks.cleaned.jsonl` |
| (6) | RAG: embedding + vector search | ✅ **Upgraded** | `rag_utils.py`: `EmbeddingRetriever` (sentence-transformers + FAISS) when deps installed |
| (6b) | RAG: LLM cites only retrieved text | ⚪ | No LLM in current pipeline; guard applies when LLM added |
| (6c) | RAG: return retrieved passages when LLM disabled | ✅ | `rag_legal_hits`, `rag_by_violation`, `rag_global_hits` always returned |
| (7) | ML + Rules violation detection | ✅ | Rule engine + ML shortlist; core rules always; ML adds extra rule IDs |
| (8) | Structured output (Violation/Missing/Compliant) | ✅ | `rule_hits`, `labor_summary`, `cross_border_summary` |
| (8b) | Contract clause text, law article, explanation in simple Arabic | ✅ | `rule_hits` (matched_text, law, article), summaries with bilingual messages |
| (9) | Evaluation (accuracy, precision, recall, F1) | ✅ **Added** | `scripts/run_evaluation.py` → `reports/evaluation_results.json` |

---

## Fixes Applied

### Task B – ML Predictor API
- **`app/ml_predictor.py`**:
  - Added `predict_rule_scores(text, top_k=40)` wrapper
  - Fixed `predict_rule_ids`: removed stray `return predict_rule_ids(...)` line
  - Exports: `predict_rule_scores_full`, `predict_rule_scores`, `predict_rule_ids`

### Task C – main.py Integration
- **`app/main.py`** (`/ocr_check_and_search`):
  - `response["ml_used"]` and `response["ml_predictions"]` set correctly
  - Uses `predict_rule_scores_full` with scope filter; `ml_rule_ids` drives extra rule engine runs
  - Keys: `rule_id`, `score`, `threshold`, `passed_threshold`, `rank`

### Task D – Clause Splitting
- **`app/utils_text.py`**: `split_into_clauses(text, min_clause_len=15)` with bilingual heuristics
- **`app/main.py`**: `clauses` in `/ocr_check_and_search` response

### Task E – ML Clause Classification
- **`ml/scripts/train_clause_classifier.py`**: TF-IDF + LogisticRegression, labels: salary, hours_overtime, duration, notice_termination, insurance, leave_holidays, other

### Task F – RAG Upgrade
- **`app/rag_utils.py`**: `EmbeddingRetriever` (paraphrase-multilingual-MiniLM-L12-v2 + FAISS) when `sentence-transformers` and `faiss-cpu` installed

### Task G – Evaluation
- **`scripts/run_evaluation.py`**: Runs pipeline on labeled contracts, outputs metrics JSON + table
