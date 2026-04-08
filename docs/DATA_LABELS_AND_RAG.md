# Contract Labels and RAG (No ChromaDB)

Short answers to: **Is the contracts data labeled?** and **How does RAG work without ChromaDB?**

---

## 1. Is the contracts data labeled?

**Yes, but only with “silver” (automatic) labels — not human-labeled.**

- **How labels are created**
  - Contract/clause text is run through the **RuleEngine** (`app/rules.py`): regex + logic over your YAML rules.
  - Whatever **rule_ids** fire on that text become the **labels** for that text.
  - So: “labels” = “which rules matched this text” (e.g. `LABOR25_SALARY`, `LABOR25_PROBATION_LIMIT`).

- **Where this happens**
  - **`ml/scripts/Build_dataset.py`**: splits contracts into clauses, runs `RuleEngine.check_text(clause)` per clause, writes `data/dataset/silver_dataset.jsonl` with `text`, `labels` (list of rule_ids), `contract_id`, `source: "rule_engine_silver"`.
  - **`app/train_ml_predictor.py`**: uses full contract text + RuleEngine to get silver labels, then trains the **rule-relevance** ML model.
  - **`ml/scripts/train_unified.py`**: uses clause-level data from `train_all.jsonl` / `silver_dataset.jsonl` / `augmented.jsonl`; binary label = “has violation” = “any rule fired” (again from RuleEngine).

- **Summary**
  - There is ** separate human-annotated** dataset ( “expert said this clause is a violation”).
  - There is **no ChromaDB (or any DB) for labels**; labels are computed on the fly from rules when building datasets and when training.
  - For a **300+ human-labeled** setup (e.g. visual markers → violation yes/no), you would add a new labeling pipeline and point the unified trainer at that dataset; the rest of the pipeline (splits, metrics, API) can stay the same.

---

## 2. How does RAG work if there is no ChromaDB?

**RAG in this project does not use ChromaDB. It uses an in-memory index built at startup.**

- **Where RAG lives**
  - **`app/rag_utils.py`**: defines the retriever.
  - **`app/chunks_loader.py`**: loads law chunks from **`chunks/*.jsonl`** into a list of `{page_content, metadata}`.
  - **`app/main.py`** (startup): calls `load_chunks_as_docs(CHUNKS_DIR)`, then `retriever.build_index(docs)`. So all law chunks are kept **in memory**.

- **How retrieval works (no ChromaDB)**
  - **Default `Retriever`** (no extra dependencies):
    - Tokenizes query and each document (Arabic + English, words ≥ 2 chars).
    - Maps tokens to a fixed-size vector via **hashing** (dim 2048), then `log(1 + count)`.
    - **Cosine similarity** between query vector and all document vectors.
    - Returns top-K docs above `min_score`; supports metadata filters (e.g. `source: "labor14_2025"`).
  - **Optional upgrade** (if `sentence-transformers` and `faiss-cpu` are installed):
    - **`EmbeddingRetriever`** replaces the default: same API, but uses a **multilingual MiniLM** model and a **FAISS in-memory** index (still no ChromaDB).

- **Summary**
  - **No ChromaDB** is used anywhere.
  - Law content comes from **`chunks/*.jsonl`** (e.g. `chunks/labor14_2025_chunks.cleaned.jsonl`).
  - Index is built **in memory** at app startup (TF-style hashing or FAISS); search is query vs in-memory vectors.
  - The README’s “RAG + embeddings (MiniLM + ChromaDB) come after” was a **future plan**; the current implementation uses **MiniLM + FAISS** (optional) or the built-in hashing retriever, and no ChromaDB.

---

## Quick reference

| Question | Answer |
|----------|--------|
| Are contracts human-labeled? | No. Labels are **silver** from RuleEngine (which rules fired). |
| Is there a label DB (e.g. ChromaDB)? | No. Labels are computed when building datasets / training. |
| Does RAG use ChromaDB? | No. |
| Where does RAG get law content? | **`chunks/*.jsonl`** (loaded at startup). |
| How does RAG search? | **In-memory** index: hashed token vectors + cosine similarity (default), or FAISS + MiniLM embeddings if installed. |
