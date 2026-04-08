from __future__ import annotations

"""
Lightweight ML predictor for LegalAI.

Artifacts produced by app/train_ml_predictor.py are expected in a directory containing:
- model.joblib
- vectorizer.joblib
- mlb.joblib
- thresholds.json
- (optional) label_map.json
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os

import joblib  # type: ignore
import numpy as np  # type: ignore


@dataclass
class _Artifacts:
    root: Path
    model: Any
    vectorizer: Any
    mlb: Any
    thresholds: Dict[str, float]
    classes: List[str]


def _candidate_artifact_dirs() -> List[Path]:
    here = Path(__file__).resolve()
    candidates = [
        Path(os.environ["ML_ARTIFACTS_DIR"]).expanduser().resolve()
        if os.environ.get("ML_ARTIFACTS_DIR")
        else None,
        here.parent / "ml" / "artifacts",
        here.parent.parent / "ml" / "artifacts",
        Path.cwd() / "app" / "ml" / "artifacts",
        Path.cwd() / "ml" / "artifacts",
    ]
    return [c for c in candidates if c is not None]


def _find_artifacts_dir() -> Optional[Path]:
    required = {"model.joblib", "vectorizer.joblib", "mlb.joblib"}
    for d in _candidate_artifact_dirs():
        try:
            if d.exists() and d.is_dir():
                files = {p.name for p in d.iterdir() if p.is_file()}
                if required.issubset(files):
                    return d
        except Exception:
            continue
    return None


def _load_thresholds(path: Path, classes: List[str]) -> Dict[str, float]:
    default_thr = float(os.environ.get("ML_DEFAULT_THRESHOLD", "0.5"))

    if not path.exists():
        return {c: default_thr for c in classes}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "thresholds" in data and isinstance(data["thresholds"], dict):
                data = data["thresholds"]
            out = {str(k): float(v) for k, v in data.items()}
            return {c: float(out.get(c, default_thr)) for c in classes}
    except Exception:
        pass

    return {c: default_thr for c in classes}


def _load_artifacts() -> Tuple[bool, Optional[_Artifacts], Optional[Exception]]:
    artifacts_dir = _find_artifacts_dir()
    if not artifacts_dir:
        missing = ", ".join([str(p) for p in _candidate_artifact_dirs()])
        return False, None, FileNotFoundError(
            f"ML artifacts not found. Expected files: model.joblib, vectorizer.joblib, mlb.joblib in one of: [{missing}]. "
            "Tip: run training, then mount/copy artifacts, or set ML_ARTIFACTS_DIR."
        )

    try:
        model = joblib.load(artifacts_dir / "model.joblib")
        vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")
        mlb = joblib.load(artifacts_dir / "mlb.joblib")

        classes = [str(c) for c in getattr(mlb, "classes_", [])]
        if not classes:
            lm = artifacts_dir / "label_map.json"
            if lm.exists():
                j = json.loads(lm.read_text(encoding="utf-8"))
                classes = [str(c) for c in j.get("classes", [])]

        thresholds = _load_thresholds(artifacts_dir / "thresholds.json", classes)

        return True, _Artifacts(
            root=artifacts_dir,
            model=model,
            vectorizer=vectorizer,
            mlb=mlb,
            thresholds=thresholds,
            classes=classes,
        ), None
    except Exception as e:
        return False, None, e


_HAS_ML_PREDICTOR, _ART, _LOAD_ERR = _load_artifacts()
if _HAS_ML_PREDICTOR and _ART:
    print(f"[ML] Loaded artifacts from: {_ART.root}")
else:
    print(f"[ML][WARN] ML predictor disabled: {_LOAD_ERR!r}")


def _scores_from_model(text: str) -> List[float]:
    if not _ART:
        raise RuntimeError("ML predictor not available")

    x = _ART.vectorizer.transform([text])

    if hasattr(_ART.model, "predict_proba"):
        probs = _ART.model.predict_proba(x)

        if isinstance(probs, np.ndarray) and probs.ndim == 2:
            row = probs[0]
            return [float(v) for v in row.tolist()]

        if isinstance(probs, list):
            out: List[float] = []
            for p in probs:
                arr = np.asarray(p)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    out.append(float(arr[0, 1]))
                else:
                    out.append(float(arr.reshape(-1)[0]))
            return out

    if hasattr(_ART.model, "decision_function"):
        z = _ART.model.decision_function(x)
        z = np.asarray(z)
        if z.ndim == 2:
            z = z[0]
        scores = 1.0 / (1.0 + np.exp(-z))
        return [float(v) for v in scores.tolist()]

    if hasattr(_ART.model, "predict"):
        y = _ART.model.predict(x)
        y = np.asarray(y)
        if y.ndim == 2:
            y = y[0]
        return [float(v) for v in y.tolist()]

    raise RuntimeError("Model doesn't support predict_proba/decision_function/predict")


def predict_rule_scores_full(text: str, *, sort: bool = True) -> List[Dict[str, Any]]:
    if not _HAS_ML_PREDICTOR or not _ART:
        return []

    scores = _scores_from_model(text)

    n = min(len(scores), len(_ART.classes))
    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        rid = _ART.classes[i]
        score = float(scores[i])
        thr = float(_ART.thresholds.get(rid, 0.5))
        pairs.append(
            {
                "rule_id": rid,
                "score": score,
                "threshold": thr,
                "passed_threshold": bool(score >= thr),
            }
        )

    if sort:
        pairs.sort(key=lambda d: d.get("score", 0.0), reverse=True)

    for idx, p in enumerate(pairs, start=1):
        p["rank"] = idx

    return pairs


def predict_rule_scores(text: str, top_k: int = 40) -> List[Dict[str, Any]]:
    """
    Wrapper: calls predict_rule_scores_full, returns top_k results.
    Prefers passed_threshold first, else highest scores.
    """
    full = predict_rule_scores_full(text, sort=True)
    if not full:
        return []
    passed = [p for p in full if p.get("passed_threshold") is True]
    if passed:
        return passed[: int(top_k)]
    return full[: int(top_k)]


def predict_rule_ids(text: str, *, top_k: int = 15) -> List[Dict[str, Any]]:
    full = predict_rule_scores_full(text, sort=True)
    if not full:
        return []
    passed = [p for p in full if p.get("passed_threshold") is True]
    return (passed[: int(top_k)]) if passed else (full[: int(top_k)])