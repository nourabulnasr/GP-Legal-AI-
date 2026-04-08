# -*- coding: utf-8 -*-
"""
Train ML predictor for LegalAI using weak labels from the rule engine.

Goal:
- Build a multi-label classifier that predicts *rule_id* directly (violation / clause IDs)
- Use the existing YAML rules in /rules (labor_mandatory.yaml, labor_cross_border.yaml, etc.)
- Generate labels by running RuleEngine.check_text() on each contract text (silver labels)
- Save artifacts to: app/ml/artifacts/
    - vectorizer.joblib
    - model.joblib
    - mlb.joblib
    - thresholds.json
    - metrics.json

Run (from project root inside Docker or venv):
    python -m app.train_ml_predictor

Or:
    python app/train_ml_predictor.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Project imports
from app.rules import RuleEngine
from app.utils_text import normalize_for_rules


# -----------------------------
# Text extraction helpers
# -----------------------------
def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Best-effort PDF text extraction.

    1) Try PyMuPDF (fitz) if available.
    2) Fallback to pypdf (pure Python) if installed.
    3) Otherwise, return empty string (training will continue).
    """
    # --- 1) PyMuPDF ---
    try:
        import fitz  # PyMuPDF
    except Exception:
        fitz = None  # type: ignore

    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            doc = None
        if doc is not None:
            text_parts: List[str] = []
            for page in doc:
                try:
                    t = page.get_text("text") or ""
                except Exception:
                    t = ""
                if t.strip():
                    text_parts.append(t)
            try:
                doc.close()
            except Exception:
                pass
            return "\n".join(text_parts).strip()

    # --- 2) pypdf fallback ---
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        PdfReader = None  # type: ignore

    if PdfReader is not None:
        try:
            reader = PdfReader(str(pdf_path))
            parts: List[str] = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    parts.append(t)
            return "\n".join(parts).strip()
        except Exception:
            return ""

    # --- 3) No PDF extractor available ---
    print(f"[WARN] PDF extractor unavailable; skipping: {pdf_path.name}")
    return ""



def _extract_text_from_docx(docx_path: Path) -> str:
    try:
        from docx import Document
    except Exception:
        return ""

    try:
        d = Document(str(docx_path))
    except Exception:
        return ""

    parts: List[str] = []
    for p in d.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts).strip()


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return _extract_text_from_docx(file_path)
    if suffix in (".txt",):
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


# -----------------------------
# Label generation
# -----------------------------
def build_silver_labels(
    rule_engine: RuleEngine,
    text_norm: str,
    scopes: List[List[str]],
) -> Set[str]:
    labels: Set[str] = set()
    for law_scope in scopes:
        try:
            hits = rule_engine.check_text(text_norm, law_scope=law_scope) or []
        except Exception:
            hits = []
        for h in hits:
            rid = h.get("rule_id") or h.get("id")
            if rid:
                labels.add(str(rid))
    return labels


# -----------------------------
# Threshold tuning
# -----------------------------
def tune_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: List[str],
    grid: List[float] | None = None,
) -> Dict[str, float]:
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]

    thresholds: Dict[str, float] = {}
    n_labels = y_true.shape[1]

    for j in range(n_labels):
        yt = y_true[:, j]
        ys = y_score[:, j]

        # If no positives in validation, keep a safe default
        if int(yt.sum()) == 0:
            thresholds[label_names[j]] = 0.50
            continue

        best_thr = 0.50
        best_f1 = -1.0
        for thr in grid:
            yp = (ys >= thr).astype(int)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        thresholds[label_names[j]] = best_thr

    return thresholds


# -----------------------------
# Main training
# -----------------------------
def main() -> None:
    # Layout: this file is in app/, so project root is parent of app/
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    rules_dir = project_root / "rules"
    laws_dir = project_root / "laws"
    data_dir = project_root / "data" / "contracts_raw"
    artifacts_dir = app_dir / "ml" / "artifacts"

    if not rules_dir.exists():
        raise FileNotFoundError(f"rules_dir not found: {rules_dir}")
    if not laws_dir.exists():
        raise FileNotFoundError(f"laws_dir not found: {laws_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(
            f"contracts_raw not found: {data_dir}\n"
            f"Expected: {project_root}/data/contracts_raw"
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) load rules
    rule_engine = RuleEngine(rules_dir, laws_dir)
    all_rule_ids: List[str] = sorted([r.get("id") for r in rule_engine.rules if r.get("id")])

    print(f"[INFO] Loaded rules: {len(all_rule_ids)}")

    # 2) build dataset
    files = sorted([p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".txt")])

    # Exclude held-out test contracts (for stricter evaluation)
    held_out_path = project_root / "data" / "held_out_test_contracts.txt"
    exclude_stems: Set[str] = set()
    if held_out_path.exists():
        for line in held_out_path.read_text(encoding="utf-8").splitlines():
            s = line.strip().split("#")[0].strip()
            if s:
                exclude_stems.add(s)
                exclude_stems.add(s.replace(".pdf", "").replace(".docx", ""))
        if exclude_stems:
            n_before = len(files)
            files = [p for p in files if p.stem not in exclude_stems]
            print(f"[INFO] Excluded {n_before - len(files)} held-out test contracts")

    print(f"[INFO] Found contract files: {len(files)}")

    scopes = [["labor"], ["cross_border"]]

    X: List[str] = []
    Y: List[Set[str]] = []

    for fp in files:
        raw = extract_text(fp)
        if not raw.strip():
            print(f"[WARN] empty text: {fp.name}")
            continue

        text_norm = normalize_for_rules(raw)
        labels = build_silver_labels(rule_engine, text_norm, scopes=scopes)

        # Keep only labels that exist in rule engine list (safety)
        labels = set([l for l in labels if l in set(all_rule_ids)])

        X.append(text_norm)
        Y.append(labels)

        print(f"[OK] {fp.name}: labels={len(labels)}")

    if len(X) < 5:
        raise RuntimeError("Not enough training samples after extraction. Check contracts_raw content / OCR.")

    # 3) split
    X_train, X_val, y_train_sets, y_val_sets = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    # 4) vectorize
    vectorizer = TfidfVectorizer(
        lowercase=False,  # Arabic
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)

    # 5) binarize labels
    mlb = MultiLabelBinarizer(classes=all_rule_ids)
    ytr = mlb.fit_transform(y_train_sets)
    yva = mlb.transform(y_val_sets)

    # 6) model (random_state for reproducibility)
    np.random.seed(42)
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            n_jobs=1,
            solver="liblinear",
            random_state=42,
        )
    )
    clf.fit(Xtr, ytr)

    # 7) scores & thresholds
    if hasattr(clf, "predict_proba"):
        yva_score = clf.predict_proba(Xva)
    else:
        # fallback: decision_function -> sigmoid-ish scaling
        scores = clf.decision_function(Xva)
        yva_score = 1 / (1 + np.exp(-scores))

    thresholds = tune_thresholds(yva, yva_score, label_names=list(mlb.classes_))

    # 8) evaluate with tuned thresholds
    thr_vec = np.array([thresholds[r] for r in mlb.classes_], dtype=float)
    yva_pred = (yva_score >= thr_vec).astype(int)

    metrics = {
        "n_rules": int(len(all_rule_ids)),
        "n_files_used": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "micro_f1": float(f1_score(yva, yva_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(yva, yva_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(yva, yva_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(yva, yva_pred, average="micro", zero_division=0)),
    }

    # 9) save artifacts
    joblib.dump(vectorizer, artifacts_dir / "vectorizer.joblib")
    joblib.dump(clf, artifacts_dir / "model.joblib")
    joblib.dump(mlb, artifacts_dir / "mlb.joblib")
    (artifacts_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Saved artifacts to:", artifacts_dir)
    print("[METRICS]", json.dumps(metrics, ensure_ascii=False, indent=2))



    # 10) persist metrics for demo/reporting
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ml_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Saved metrics to: {reports_dir / 'ml_metrics.json'}")

if __name__ == "__main__":
    main()
