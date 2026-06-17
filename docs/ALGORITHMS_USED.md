# Algorithms Used in GP-Legal-AI

This document lists all **named algorithms** used in the project and their roles (e.g. TF-IDF, Logistic Regression, BM25), not code function/class names.

---

## Information Retrieval & Search

| Algorithm | Where Used | Role |
|-----------|------------|------|
| **BM25** (Okapi BM25) | `app/bm25_law_retriever.py`, `app/model_ml_predictor.py`, `model_ML/training_only.py`, `model_ML/infer_ocr_to_llm_payload.py`, `ml/scripts/train_unified.py`, `app/unified_predictor.py` | Rank law articles/chunks by relevance to a contract clause; parameters k1=1.5, b=0.75; IDF formula: log((N−df+0.5)/(df+0.5)+1). |
| **TF-IDF** (term frequency–inverse document frequency) | `app/train_ml_predictor.py`, `app/model_ml_predictor.py`, `model_ML/training_only.py`, `model_ML/infer_ocr_to_llm_payload.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_unified.py`, `ml/scripts/train_baseline_tfidf.py`, `ml/scripts/eval_*` | (1) **Feature extraction** for ML: word or char n-grams for rule/clause classification. (2) **Reranking**: char_wb (3–6 or 3–7) n-grams with sublinear_tf; used after BM25 to rerank law candidates. |
| **Cosine similarity** | `app/rag_utils.py`, `app/rag_chromadb.py`, `app/model_ml_predictor.py`, `model_ML/training_only.py`, `model_ML/infer_ocr_to_llm_payload.py`, Legal Rag `vector_store.py` | Compare query vs document vectors (TF-IDF or embeddings); ChromaDB uses cosine distance (similarity = 1 − distance); TF-IDF reranker uses dot product on normalized vectors. |
| **FAISS** (IndexFlatIP) | `app/rag_utils.py` | Exact nearest-neighbor search on normalized embeddings (inner product = cosine similarity) for RAG retrieval. |
| **ChromaDB** (HNSW, cosine) | `app/rag_chromadb.py`, Legal Rag `vector_store.py` | Vector store over law chunks; `hnsw:space": "cosine"`; used for RAG over Legal RAG corpus. |
| **SentenceTransformer** (paraphrase-multilingual-MiniLM-L12-v2, etc.) | `app/rag_utils.py`, `app/rag_chromadb.py`, Legal Rag `vector_store.py` | Dense embeddings for Arabic/English; used for ChromaDB and FAISS retrieval. |

---

## Machine Learning (Classification & Training)

| Algorithm | Where Used | Role |
|-----------|------------|------|
| **Logistic Regression** | `app/train_ml_predictor.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_unified.py`, `ml/scripts/train_baseline_tfidf.py`, `ml/scripts/eval_retrain_fixed_split.py`, `model_ML/training_only.py` | Binary/multilabel classifier for rule violation and clause type; solver liblinear; often wrapped in OneVsRest. |
| **One-vs-Rest (OvR)** | `app/train_ml_predictor.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_baseline_tfidf.py`, `ml/scripts/eval_retrain_fixed_split.py` | Multi-label rule/clause classification: one binary LogisticRegression per label. |
| **MultiLabelBinarizer** | `app/train_ml_predictor.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_baseline_tfidf.py`, `ml/scripts/eval_retrain_fixed_split.py`, `model_ML/training_only.py` | Encode multi-label targets as binary matrix for multilabel training/eval. |
| **CalibratedClassifierCV** (isotonic / sigmoid) | `model_ML/training_only.py` | Probability calibration for multilabel/binary models (isotonic or sigmoid). |
| **Train–test split** | `app/train_ml_predictor.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_unified.py`, `ml/scripts/tune_thresholds.py`, `ml/scripts/train_baseline_tfidf.py`, `ml/scripts/eval_retrain_fixed_split.py` | Random 80/20 (or similar) split for training vs validation/test. |
| **GroupKFold** | `ml/scripts/train_unified.py`, `model_ML/training_only.py` | Cross-validation by contract_id so the same contract never appears in both train and validation (generalization check). |
| **Threshold tuning (grid search on F1)** | `app/train_ml_predictor.py`, `ml/scripts/tune_thresholds.py`, `ml/scripts/train_unified.py` | Per-label (or binary) threshold chosen to maximize validation F1 over a grid (e.g. 0.10–0.90). |

---

## Evaluation Metrics

| Metric | Where Used | Role |
|--------|------------|------|
| **F1 (micro / macro)** | `app/train_ml_predictor.py`, `scripts/run_evaluation.py`, `ml/scripts/train_clause_classifier.py`, `ml/scripts/train_unified.py`, `ml/scripts/eval_retrain_fixed_split.py`, `model_ML/training_only.py` | Primary classification metric for rule/clause and unified models. |
| **Precision / Recall** | Same as F1 | `precision_score`, `recall_score` (micro/macro) for classification evaluation. |
| **ROC-AUC** | `ml/scripts/train_unified.py`, `model_ML/training_only.py`, `scripts/run_unified_evaluation.py` | Ranking/confidence evaluation for binary or multilabel. |
| **PR-AUC (average_precision_score)** | `ml/scripts/train_unified.py`, `model_ML/training_only.py` | Precision–recall area for imbalanced/binary evaluation. |
| **Confusion matrix** | `ml/scripts/train_unified.py`, `model_ML/training_only.py` | Per-class counts for analysis. |

---

## Hashing & Security

| Algorithm | Where Used | Role |
|-----------|------------|------|
| **FNV-1a** (32-bit) | `app/rag_utils.py` (`_hash_idx`: offset 2166136261, prime 16777619) | Map token to dimension index for lightweight bag-of-words–style vectors in fallback retriever. |
| **PBKDF2-SHA256** | `app/core/security.py` (passlib `pbkdf2_sha256`) | Password hashing and verification (no bcrypt in this codebase). |
| **SHA-256** | `app/routers/auth.py` (reset token hash), `app/main.py` (file sha256) | Hashing reset tokens and file integrity. |
| **JWT (HS256)** | `app/core/security.py` (jose) | Sign/verify access tokens (sub, role, iat, exp). |

---

## Text & Normalization

| Algorithm / Method | Where Used | Role |
|--------------------|------------|------|
| **Unicode NFKC** | `app/main.py`, `app/utils_text.py`, `app/model_ml_predictor.py`, `model_ML/training_only.py`, `model_ML/infer_ocr_to_llm_payload.py`, `model_ML/process_labor_from_txt.py`, `scripts/preprocess_labor14_2025.py`, `scripts/build_articles_from_pdf_v3.py` | Normalize Arabic/Unicode (e.g. presentation forms) before rules, ML, and RAG. |
| **Regex (PCRE-style)** | `app/rules.py`, `app/utils_text.py`, `app/cross_border.py`, `app/rag_rule_mapping.py`, `app/numeric_checks.py`, and others | Rule matching (YAML patterns), clause splitting, cross-border keywords, number extraction. |
| **SequenceMatcher** (difflib) | `app/main.py` (`_FallbackRetriever._score`) | Fuzzy string similarity ratio for fallback RAG scoring. |

---

## Sigmoid

| Use | Where | Role |
|-----|--------|------|
| **Sigmoid (1/(1+e^(-z)))** | `app/train_ml_predictor.py`, `app/ml_predictor.py` | Map decision_function output to [0,1] when predict_proba is not used. |
| **Sigmoid calibration** | `model_ML/training_only.py` | Optional method in CalibratedClassifierCV. |

---

## Summary Table

| Algorithm Name | Primary Function in This Project |
|----------------|----------------------------------|
| **BM25** | Rank law chunks by relevance to contract clause. |
| **TF-IDF** | Features for ML (word/char n-grams) and reranking after BM25 (char n-grams, sublinear_tf). |
| **Cosine similarity** | Compare query and documents (embeddings or TF-IDF). |
| **FAISS (IndexFlatIP)** | Exact k-NN on normalized embeddings for RAG. |
| **ChromaDB (HNSW, cosine)** | Persistent vector index for legal RAG. |
| **SentenceTransformer** | Dense multilingual embeddings for retrieval. |
| **Logistic Regression** | Binary/multilabel classifier for violations and clause types. |
| **One-vs-Rest** | Multi-label classification with one LR per label. |
| **MultiLabelBinarizer** | Encode multi-label targets. |
| **CalibratedClassifierCV (isotonic/sigmoid)** | Probability calibration. |
| **Train_test_split** | Random train/validation split. |
| **GroupKFold** | CV by contract_id for generalization. |
| **F1 / precision / recall** | Classification evaluation. |
| **ROC-AUC / PR-AUC** | Ranking and binary/multilabel evaluation. |
| **Confusion matrix** | Per-class evaluation. |
| **FNV-1a** | Token→dimension hashing in fallback retriever. |
| **PBKDF2-SHA256** | Password hashing. |
| **SHA-256** | Token and file hashing. |
| **JWT (HS256)** | Access token signing. |
| **Unicode NFKC** | Text normalization. |
| **SequenceMatcher** | Fuzzy string similarity in fallback RAG. |
| **Sigmoid** | Score calibration and decision_function mapping. |
