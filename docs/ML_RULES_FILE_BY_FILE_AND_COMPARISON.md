# Legal AI: File-by-File Function Summary (ML & Rules) and ML vs Rule Comparison

This document (1) explains the function of every relevant file in the ML and rule-matching pipeline, and (2) compares the **described violation-detection ML system** (your specification) with what the **codebase** actually does—same or not, differences, and which approach is better in what sense.

---

## Part 1: File-by-File Function Summary

### A. Rule engine and rule definitions

| File | Function |
|------|----------|
| **`rules/labor_mandatory.yaml`** | YAML list of labor-law rules. Each rule has `id`, `scope`, `law`, `article`, `severity`, `description`, `rationale`, and `match.any` (list of regex `pattern`s). Used by `RuleEngine` to detect mandatory contract elements and violations (e.g. employer/employee info, salary, working hours, annual leave, probation).

| **`rules/labor_cross_border.yaml`** | Same structure for cross-border employment (governing law, jurisdiction, currency, payment method, transfer fees, tax/insurance, remote tools). 
|
| **`app/rules.py`** | **RuleEngine**: loads all `*.yaml` from `rules_dir` and law JSONs from `laws_dir`. `check_text(text, law_scope=..., only_rule_ids=...)` runs regex (and special-case logic for e.g. working hours, probation, annual leave) and returns hits with `rule_id`, `severity`, `description`, `article_text`, `matched_text`. Violations are **defined by rules firing**, not by a separate ML “violation” label. |

### B. RAG and rule selection from law chunks

| File | Function |
|------|----------|
| **`app/rag_utils.py`** | **Retriever**: builds an index over law chunks (TF-style hashing + cosine similarity, or optional SentenceTransformer+FAISS). `build_index(docs)`, `search(query, top_k, filters, min_score)`. Used to retrieve relevant law passages for display and, together with `rag_rule_mapping`, to decide which rules to run. **Not BM25**; no article-level BM25 index over law text. |


| **`app/rag_rule_mapping.py`** | Maps RAG hits to rule IDs: loads static `keywords_to_rules` / `article_to_rules` from JSON (if present), and builds per-rule text (description + rationale + pattern keywords) from YAML. `chunks_to_rule_ids(rag_hits, rules_dir)` returns the set of rule_ids to run based on article, keywords, and token overlap with rule text. |


| **`app/chunks_loader.py`** | Reads `*.jsonl` under `chunks/` and returns list of `{page_content, metadata}` for law chunks. Used at startup to feed the Retriever. |

### C. ML training (app-level: document → rule_ids)

| File | Function |
|------|----------|
| **`app/train_ml_predictor.py`** | **Document-level** training. Loads contracts from `data/contracts_raw` (PDF/DOCX/TXT), normalizes text, runs **RuleEngine.check_text()** per scope (labor, cross_border) to get **silver labels** = set of rule_ids that fired. Builds X = normalized full contract text, Y = set of rule_ids. Uses `TfidfVectorizer(ngram_range=(1,2), max_features=50000)` (word n-grams, not char), `MultiLabelBinarizer`, `OneVsRestClassifier(LogisticRegression)`, `train_test_split` (random 80/20, no contract-level group split). Tunes per-rule thresholds on validation F1; saves vectorizer, model, mlb, thresholds.json, metrics.json to `app/ml/artifacts/`. Optional: excludes files listed in `data/held_out_test_contracts.txt`. |

### D. ML inference (app-level)

| File | Function |
|------|----------|
| **`app/ml_predictor.py`** | Loads artifacts from `app/ml/artifacts/` (or `ML_ARTIFACTS_DIR`). `predict_rule_scores_full(text)` returns list of `{rule_id, score, threshold, passed_threshold, rank}`. `predict_rule_ids(text, top_k=15)` returns top rule_ids (passed first, else by score). **ML does not decide “violation yes/no”**; it predicts **which rules are relevant**. The API then runs **RuleEngine** on that shortlist (plus fixed CORE rule_ids); **violations are still determined by regex rules**. |

### E. ML scripts under `ml/scripts/` (clause-level, alternate pipeline)

| File | Function |
|------|----------|
| **`ml/scripts/Build_dataset.py`** | **Clause-level** dataset builder. Reads contracts from `data/contracts_raw/pdf` and `docx`, normalizes with `_normalize_contract_text`, splits into clauses via `chunk_clauses()` (sentence splitting + sliding windows). For each clause, runs `RuleEngine.check_text(clause)` and records **silver labels** = rule_ids that fired. Writes `data/dataset/silver_dataset.jsonl` with `contract_id`, `chunk_id`, `text`, `labels`, `source=rule_engine_silver`. |
| **`ml/scripts/train_baseline_tfidf.py`** | Reads `data/dataset/train_all.jsonl` (text + labels + contract_id). **Char n-grams**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=200000)`. Multi-label (rule_ids), `OneVsRestClassifier(LogisticRegression(max_iter=400, class_weight="balanced"))`. `train_test_split(..., stratify=is_synth)` so SYNTH vs real are represented in test. Saves to `ml/artifacts/tfidf_baseline/` (vectorizer, model, mlb). No law retrieval, no threshold tuning in this script. |
| **`ml/scripts/tune_thresholds.py`** | Loads tfidf_baseline model and `train_all.jsonl`, same split as train_baseline_tfidf. For each label, searches a threshold grid on validation F1 and saves per-label thresholds to `thresholds.joblib`. |
| **`ml/scripts/eval_fixed_split.py`** | Loads tfidf_baseline + thresholds, splits data by contract_id (20% SYNTH to test, 20% real to test), evaluates with `classification_report`. |
| **`ml/scripts/eval_retrain_fixed_split.py`** | Same fixed contract-based split; retrains the char-ngram multi-label model on train portion, evaluates on test with fixed 0.5 threshold (no threshold tuning). Reports micro/macro F1. |
| **`ml/scripts/train_clause_classifier.py`** | **Different task**: clause **type** classification (salary, hours_overtime, duration, notice_termination, insurance, leave_holidays, other). Maps rule_ids to these types via `RULE_TO_CLAUSE`. Reads from `data/dataset/train_all.jsonl` (or silver_dataset / augmented). Same char n-gram TF-IDF + OneVsRestClassifier(LogisticRegression). Saves to `app/ml/artifacts/clause_classifier/`. Does **not** predict violation yes/no. |

### F. Main API and pipeline

| File | Function |
|------|----------|
| **`app/main.py`** | FastAPI app. **`/check_clause`**: normalizes clause, gets ML shortlist (`predict_rule_ids`) + fixed CORE rule_ids, runs **RuleEngine.check_text(..., only_rule_ids=...)**. Returns rule hits; violations = rules that fired. **`/ocr_check_and_search`**: OCR → normalize → **RAG-first** rule selection (`chunks_to_rule_ids(rag_hits)`) when labor applicable; else **ML** shortlist. Runs RuleEngine on CORE + RAG/ML-derived rule_ids; optionally adds “ML-only” hits as potential violations. Labor/cross-border summaries are derived from rule hits. RAG is used for evidence retrieval and for **which rules to run**, not to augment clause text before a classifier. |
| **`app/utils_text.py`** | `norm_ar`, `normalize_for_rules`, `split_into_clauses`, language detection. Used before rule matching and clause splitting. |

### G. Supporting (config, DB, routers)

| File | Function |
|------|----------|
| **`app/schema.py`** | Pydantic models for API requests/responses (e.g. ClauseCheckRequest, ClauseCheckResponseWithML). |
| **`app/core/config.py`** | Configuration (e.g. from env). |
| **`app/db/*`** | SQLite DB, Analysis/User models, session; optional persist for analysis. |
| **`app/routers/analyses.py`** | Analysis list/detail endpoints. |
| **`app/routers/chat.py`** | Chat endpoint (if present). |
| **`laws/processed/*.json`** | Structured labor law articles; RuleEngine uses them for `article_text` in hits. |
| **`chunks/*.jsonl`** | Law chunks for RAG index (loaded by chunks_loader). |

---

## Part 2: Described ML System vs Codebase — Same or Not?

### Your described system (summary)

- **Task**: Clause-level **binary** classifier: “unlawful employment provision” vs not.
- **Data**: Contracts segmented into **legal clauses**; **ground truth** = visually marked non-black text in original documents (violation markers); markers **removed** from training text to avoid leakage.
- **Splits**: **Strict contract-level** (group-based): train / validation / test contain **disjoint contracts**.
- **Models**: (1) **Contracts-only baseline**: clause text only. (2) **Law-aware**: BM25 index over **article-level** law text; **top-K retrieved articles** appended to clause (RAG-style); same classifier on “[clause] + [law evidence]”.
- **Features**: **TF-IDF with character n-grams** (Arabic morphology, OCR, formatting).
- **Classifier**: Regularized **logistic regression** with **class balancing**.
- **Threshold**: Chosen on **validation** to maximize **F1**.
- **Eval**: Held-out test set: **PR-AUC, ROC-AUC, precision, recall, F1, confusion matrices**; **GroupKFold** stability on training only.
- **Goals**: Scientifically valid generalization, auditability, detection on unseen contracts using both contract language and governing law.

### What the codebase does

| Aspect | Codebase behavior |
|--------|-------------------|
| **Task** | **Multi-label**: predict **which rule_ids** are relevant (or clause type in one script). **Violation = RuleEngine regex match**, not a single ML “unlawful” score. |
| **Unit** | **Document-level** in `app/train_ml_predictor` (full contract text); **clause-level** in `ml/scripts/` (from silver_dataset / train_all.jsonl). |
| **Labels** | **Silver**: RuleEngine regex hits → set of rule_ids. **No** human annotations; **no** visual markers; **no** removal of markers. |
| **Splits** | **Random** 80/20 in app trainer; **stratify by SYNTH** in ml/scripts; **fixed** 20% per group (SYNTH/real) in eval scripts. **No** strict “disjoint contracts” group split in the main app training. |
| **Law in model** | **None**. No BM25; no law text appended to clause inside the classifier. RAG is used **at inference** to (1) select which rules to run, (2) fetch evidence for the UI. |
| **Features** | **App**: word TF-IDF (1,2). **ml/scripts**: char n-grams (3,5). Both use LogisticRegression (with class_weight in scripts). |
| **Threshold** | Tuned on validation F1 in app (`tune_thresholds`) and in `tune_thresholds.py` for tfidf_baseline. |
| **Eval** | Micro/macro F1, precision, recall in app; classification_report + F1 in scripts. **No** PR-AUC, ROC-AUC, or GroupKFold in the app. |

### Direct comparison

| Criterion | Described system | This codebase |
|-----------|------------------|----------------|
| **Same task?** | No. Yours: binary “unlawful clause”. Ours: multi-label “which rules apply” + rule-based violation. |
| **Clause-level?** | Yes, strict clause segmentation. | Clause-level only in ml/scripts datasets; app training is document-level. |
| **Ground truth** | Human-like (visual markers, then removed). | Silver (regex rules); no human labels. |
| **Contract disjoint split** | Yes (group-based). | Partially (eval scripts); app trainer uses random split. |
| **Law inside model** | Yes (BM25 + top-K articles concatenated). | No; RAG only for rule selection and evidence. |
| **Char n-grams** | Yes. | Only in ml/scripts; app uses word n-grams. |
| **Threshold on val F1** | Yes. | Yes (app + tune_thresholds.py). |
| **PR-AUC / ROC-AUC / GroupKFold** | Yes. | No in current app. |

So: **the ML model and rule-based matching are not doing the same thing.**  
- **Rule-based matching**: defines violations by regex (and special logic) over contract text; it **is** the violation detector.  
- **ML in this codebase**: predicts **which rules are relevant** (and optionally shortlists rules or adds “ML-detected” suggestions); **violations are still decided by the RuleEngine**.

Your described system would **replace or complement** that with a **single clause-level binary classifier** (optionally law-aware) that directly predicts “unlawful vs not,” with human-or-marker-derived labels and strict evaluation.

---

## Part 3: Differences in Short

1. **Objective**: Yours = one binary “violation” label per clause. Ours = many rule_ids per text + violations = rules that fire.
2. **Labels**: Yours = high-quality (markers, no leakage). Ours = silver (rule hits only).
3. **Generalization**: Yours = group-based splits + PR/ROC + GroupKFold. Ours = random or stratified split, F1/precision/recall only.
4. **Law**: Yours = BM25 + concatenated law in the model. Ours = law only in RAG for rule selection and UI evidence.
5. **Detection**: Yours = ML outputs “violation” directly. Ours = ML suggests rules; RuleEngine outputs violations.

---

## Part 4: Who Is Better?

- **For scientific rigor and generalization claims**  
  **Your described system is better**: disjoint contract splits, PR-AUC/ROC-AUC, GroupKFold, and human-like labels avoid leakage and give a clearer picture of performance on unseen contracts.

- **For auditability and interpretability**  
  **Rule-based is stronger**: each violation is tied to a specific rule and matched span. Your design (threshold on F1, logistic regression, optional law evidence) is also auditable; the codebase’s “ML-only” hits are less so (no matched span).

- **For leveraging law inside the model**  
  **Your law-aware setup is better**: the model sees the actual law articles (BM25-retrieved) per clause. Here, the model never sees law; RAG only guides which rules to run.

- **For cold start and maintenance**  
  **Rule-based (current) is easier**: no need for labeled violation data; you add/change YAML and optionally retrain ML for rule relevance. Your system needs clause-level violation labels (e.g. from markers).

- **For end-to-end violation detection**  
  **Your system** can directly output “this clause is unlawful.” The **current pipeline** outputs “these rules fired” (and optionally “ML thinks these rules might be relevant”); you then interpret rule hits as violations. So for a **single scalar “violation risk” per clause**, your design is better; for **explainable, rule-grounded reports**, the current hybrid (rules + ML shortlist) is strong.

**Practical recommendation**:  
- Keep **rules** as the source of truth for *what* is a violation and for evidence (article, description, matched text).  
- Add your **clause-level binary (optionally law-aware) classifier** as a **screening or prior**: e.g. “only run rules on clauses above a violation-risk threshold,” or “flag clauses for human review.”  
- Optionally use your **evaluation protocol** (disjoint contracts, PR-AUC, ROC-AUC, GroupKFold) to measure both the new classifier and the existing rule-based pipeline on a shared test set.

---

## Part 5: Summary Table

| Dimension | Described ML system | This codebase (ML + rules) |
|-----------|---------------------|-----------------------------|
| **Task** | Binary: unlawful clause | Multi-label rule relevance; violations = rules |
| **Labels** | Human-like (markers, removed) | Silver (RuleEngine) |
| **Split** | Disjoint contracts | Random / stratified |
| **Law in model** | BM25 + top-K articles | No (RAG for rule selection only) |
| **Features** | TF-IDF char n-grams | App: word; scripts: char |
| **Eval** | PR-AUC, ROC-AUC, F1, GroupKFold | F1, precision, recall |
| **Violation output** | ML “violation” per clause | RuleEngine hits (regex) |

They are **not the same**. The **described system** is better for **generalization and law-aware clause-level violation detection**; the **codebase** is better for **explainable, rule-driven violations** and lower dependency on labeled data. Combining both (your classifier as a filter/prior, rules as the authority) is a strong direction.
