#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train clause classification model for LegalAI.

Tags clauses into: salary, hours_overtime, duration, notice_termination,
insurance, leave_holidays, other.

Uses TF-IDF + LogisticRegression (OneVsRest multi-label). No GPU required.
Artifacts saved to: app/ml/artifacts/clause_classifier/

Run (from project root):
    python ml/scripts/train_clause_classifier.py
Or:
    python -m ml.scripts.train_clause_classifier
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Clause type mapping from rule_ids
RULE_TO_CLAUSE = {
    "salary": [
        "LABOR25_SALARY", "LABOR25_SALARY_VALUE_PRESENT", "LABOR25_SALARY_STRUCTURE",
        "SALARY_AND_PAYMENT", "LABOR25_BENEFITS",
    ],
    "hours_overtime": [
        "LABOR25_WORKING_HOURS", "LABOR25_WORKING_HOURS_PRESENCE",
        "LABOR25_WORKING_HOURS_VIOLATION", "WORK_HOURS_LIMITS",
    ],
    "duration": [
        "LABOR25_CONTRACT_DURATION", "CONTRACT_DURATION_AND_START",
    ],
    "notice_termination": [
        "LABOR25_PROBATION_LIMIT", "LABOR25_PROBATION_PRESENCE",
        "LABOR25_PROBATION_PERIOD", "PROBATION_MAX_3_MONTHS",
    ],
    "insurance": [
        "CROSSBORDER_TAX_INSURANCE_CLARITY", "INSURANCE_NUMBER",
    ],
    "leave_holidays": [
        "LABOR25_ANNUAL_LEAVE", "LABOR25_ANNUAL_LEAVE_PRESENCE",
        "LABOR25_ANNUAL_LEAVE_VIOLATION", "LABOR25_ANNUAL_LEAVE_WAIVER",
        "LABOR25_ANNUAL_LEAVE_DEFERRAL", "ANNUAL_LEAVE_15_21",
    ],
}
CLAUSE_CLASSES = ["salary", "hours_overtime", "duration", "notice_termination", "insurance", "leave_holidays", "other"]


def _rule_ids_to_clause_types(rule_ids: list) -> list:
    """Map rule_ids to clause type labels."""
    out = set()
    for rid in rule_ids or []:
        rid = (rid or "").strip()
        found = False
        for ctype, rules in RULE_TO_CLAUSE.items():
            if rid in rules or any(r in rid for r in rules):
                out.add(ctype)
                found = True
                break
        if not found and rid:
            out.add("other")
    return list(out) if out else ["other"]


def main():
    base = Path(__file__).resolve().parent.parent.parent
    data_paths = [
        base / "data" / "dataset" / "train_all.jsonl",
        base / "data" / "dataset" / "silver_dataset.jsonl",
        base / "data" / "dataset" / "augmented.jsonl",
    ]
    texts = []
    clause_labels = []

    for dp in data_paths:
        if not dp.exists():
            continue
        with open(dp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = (r.get("text") or "").strip()
                if len(t) < 10:
                    continue
                labels = r.get("labels") or []
                clabels = _rule_ids_to_clause_types(labels)
                texts.append(t)
                clause_labels.append(clabels)

    if not texts:
        print("No training data found. Ensure data/dataset/*.jsonl exist.")
        return

    print(f"Loaded {len(texts)} samples with clause labels.")

    mlb = MultiLabelBinarizer(classes=CLAUSE_CLASSES)
    Y = mlb.fit_transform(clause_labels)

    X_train, X_test, Y_train, Y_test = train_test_split(
        texts, Y, test_size=0.2, random_state=42
    )

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150000,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=400, class_weight="balanced"))
    clf.fit(Xtr, Y_train)

    pred = clf.predict(Xte)
    print("\nClause classification report:")
    print(classification_report(Y_test, pred, target_names=CLAUSE_CLASSES, zero_division=0))

    metrics = {
        "precision_micro": float(precision_score(Y_test, pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(Y_test, pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(Y_test, pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(Y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(Y_test, pred, average="macro", zero_division=0)),
    }
    print("Metrics:", metrics)

    out_dir = base / "app" / "ml" / "artifacts" / "clause_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vec, out_dir / "vectorizer.joblib")
    joblib.dump(clf, out_dir / "model.joblib")
    joblib.dump(mlb, out_dir / "mlb.joblib")
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": CLAUSE_CLASSES}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
