#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training: combine rule-based labels + ML (best of both).

- Clause-level data from train_all.jsonl, silver_dataset.jsonl, augmented.jsonl.
- Contract-disjoint split (train/val/test = disjoint contracts) for valid generalization.
- Two modes: contracts_only (clause text only) and law_aware (clause + top-K BM25 law chunks).
- Char n-grams (Arabic/OCR friendly), LogisticRegression, class balancing.
- Binary target: has_violation = (any rule fired).
- Threshold tuned on validation F1; metrics: PR-AUC, ROC-AUC, F1, precision, recall, confusion matrix.
- GroupKFold on training only for stability.

Run from project root:
  python ml/scripts/train_unified.py
  python ml/scripts/train_unified.py --law_aware
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold

# Project root (ml/scripts -> ml -> root)
BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

DATA_PATHS = [
    BASE / "data" / "dataset" / "train_all.jsonl",
    BASE / "data" / "dataset" / "silver_dataset.jsonl",
    BASE / "data" / "dataset" / "augmented.jsonl",
]
CHUNKS_DIR = BASE / "chunks"
ARTIFACTS_DIR = BASE / "app" / "ml" / "artifacts" / "unified"
SEP_LAW = "\n[LAW]\n"


def load_dataset() -> tuple[list[str], list[int], list[str]]:
    """Load all dataset jsonl; return texts, binary labels (has_violation), contract_ids."""
    texts: list[str] = []
    labels_bin: list[int] = []
    contract_ids: list[str] = []
    seen = set()
    for dp in DATA_PATHS:
        if not dp.exists():
            continue
        with open(dp, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = (r.get("text") or "").strip()
                if len(t) < 15:
                    continue
                labs = r.get("labels") or []
                cid = (r.get("contract_id") or "unknown").strip()
                key = (t[:200], cid)
                if key in seen:
                    continue
                seen.add(key)
                texts.append(t)
                labels_bin.append(1 if len(labs) > 0 else 0)
                contract_ids.append(cid)
    return texts, labels_bin, contract_ids


def contract_disjoint_split(
    texts: list[str],
    labels_bin: list[int],
    contract_ids: list[str],
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    """Split by contract_id so train/val/test contain disjoint contracts."""
    unique_cids = list(dict.fromkeys(contract_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_cids)
    n = len(unique_cids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    train_cids = set(unique_cids[:n_train])
    val_cids = set(unique_cids[n_train : n_train + n_val]) if n_val else set()
    test_cids = set(unique_cids[n_train + n_val :])

    X_train, y_train, g_train = [], [], []
    X_val, y_val, g_val = [], [], []
    X_test, y_test, g_test = [], [], []

    for i, (text, lab, cid) in enumerate(zip(texts, labels_bin, contract_ids)):
        if cid in train_cids:
            X_train.append(text)
            y_train.append(lab)
            g_train.append(cid)
        elif cid in val_cids:
            X_val.append(text)
            y_val.append(lab)
            g_val.append(cid)
        elif cid in test_cids:
            X_test.append(text)
            y_test.append(lab)
            g_test.append(cid)

    return (
        (X_train, y_train, g_train),
        (X_val, y_val, g_val),
        (X_test, y_test, g_test),
        (len(train_cids), len(val_cids), len(test_cids)),
    )


def augment_with_law(texts: list[str], top_k: int = 5) -> list[str]:
    """Retrieve top_k law chunks per text and append (RAG-style)."""
    try:
        from app.bm25_law_retriever import BM25LawRetriever, load_law_docs_for_bm25
    except Exception:
        return texts

    docs = load_law_docs_for_bm25(CHUNKS_DIR)
    if not docs:
        return texts

    retriever = BM25LawRetriever(k1=1.5, b=0.75)
    retriever.build_index(docs)
    out: list[str] = []
    for t in texts:
        hits = retriever.search(t, top_k=top_k, min_score=0.0)
        law_parts = [h.get("text", "")[:800] for h in hits if h.get("text")]
        if law_parts:
            out.append(t + SEP_LAW + SEP_LAW.join(law_parts))
        else:
            out.append(t)
    return out


def tune_threshold_binary(y_true, y_score, grid=None):
    if grid is None:
        grid = np.arange(0.15, 0.86, 0.05)
    best_thr = 0.5
    best_f1 = -1.0
    for t in grid:
        yp = (y_score >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr


def main():
    parser = argparse.ArgumentParser(description="Unified rule+ML training (clause-level, contract-disjoint).")
    parser.add_argument("--law_aware", action="store_true", help="Append top-K BM25 law chunks to each clause.")
    parser.add_argument("--top_k_law", type=int, default=5, help="Number of law chunks to append (law_aware).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[1] Loading dataset...")
    texts, labels_bin, contract_ids = load_dataset()
    if len(texts) < 20:
        print("ERROR: Need at least 20 samples. Add more data to data/dataset/*.jsonl")
        return 1

    n_pos = sum(labels_bin)
    n_neg = len(labels_bin) - n_pos
    print(f"    Samples: {len(texts)} (positive={n_pos}, negative={n_neg}), unique contracts: {len(set(contract_ids))}")

    if args.law_aware:
        print("[2] Building law-aware inputs (BM25 over chunks)...")
        texts = augment_with_law(texts, top_k=args.top_k_law)
    else:
        print("[2] Using contracts-only (no law augmentation).")

    print("[3] Contract-disjoint split (64% train, 16% val, 20% test)...")
    (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test), (n_tr_c, n_va_c, n_te_c) = contract_disjoint_split(
        texts, labels_bin, contract_ids, seed=args.seed
    )
    print(f"    Train: {len(X_train)} samples, {n_tr_c} contracts | Val: {len(X_val)} samples, {n_va_c} contracts | Test: {len(X_test)} samples, {n_te_c} contracts")

    y_train_np = np.array(y_train, dtype=np.int32)
    y_val_np = np.array(y_val, dtype=np.int32)
    y_test_np = np.array(y_test, dtype=np.int32)

    print("[4] Vectorizing (char n-grams 3–5, max 200k features)...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val) if X_val else None
    Xte = vectorizer.transform(X_test)

    print("[5] Training LogisticRegression (class_weight=balanced)...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=args.seed)
    clf.fit(Xtr, y_train_np)

    # Validation threshold
    if Xva is not None and len(y_val_np) > 0:
        yva_score = clf.predict_proba(Xva)[:, 1]
        thr = tune_threshold_binary(y_val_np, yva_score)
    else:
        thr = 0.5

    # Test metrics
    yte_score = clf.predict_proba(Xte)[:, 1]
    yte_pred = (yte_score >= thr).astype(int)

    metrics = {
        "mode": "law_aware" if args.law_aware else "contracts_only",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_contracts_train": n_tr_c,
        "n_contracts_val": n_va_c,
        "n_contracts_test": n_te_c,
        "threshold": thr,
        "precision": float(precision_score(y_test_np, yte_pred, zero_division=0)),
        "recall": float(recall_score(y_test_np, yte_pred, zero_division=0)),
        "f1": float(f1_score(y_test_np, yte_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test_np, yte_score))
    except Exception:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_test_np, yte_score))
    except Exception:
        metrics["pr_auc"] = 0.0

    cm = confusion_matrix(y_test_np, yte_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    # GroupKFold on training only (5 folds)
    if len(g_train) >= 10 and len(set(g_train)) >= 5:
        groups = np.array(g_train)
        gkf = GroupKFold(n_splits=5)
        f1s = []
        for train_idx, _ in gkf.split(X_train, y_train_np, groups):
            X_fold = Xtr[train_idx]
            y_fold = y_train_np[train_idx]
            clf_fold = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=args.seed)
            clf_fold.fit(X_fold, y_fold)
            # Evaluate on val if available, else on a held-out part of train
            if Xva is not None and len(y_val_np) > 0:
                yva_fold = clf_fold.predict_proba(Xva)[:, 1]
                thr_f = tune_threshold_binary(y_val_np, yva_fold)
                yva_p = (yva_fold >= thr_f).astype(int)
                f1s.append(f1_score(y_val_np, yva_p, zero_division=0))
            else:
                f1s.append(0.0)
        if f1s:
            metrics["group_kfold_f1_mean"] = float(np.mean(f1s))
            metrics["group_kfold_f1_std"] = float(np.std(f1s))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    joblib.dump(clf, ARTIFACTS_DIR / "model.joblib")
    (ARTIFACTS_DIR / "thresholds.json").write_text(
        json.dumps({"threshold": thr, "mode": metrics["mode"]}, indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {"law_aware": args.law_aware, "top_k_law": args.top_k_law, "seed": args.seed}
    (ARTIFACTS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n[DONE] Artifacts saved to:", ARTIFACTS_DIR)
    print("Metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
