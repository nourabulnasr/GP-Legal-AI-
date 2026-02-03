# -*- coding: utf-8 -*-
"""
Unified predictor: clause-level violation risk (rule+ML combined pipeline).

Loads artifacts from app/ml/artifacts/unified/ (trained by ml/scripts/train_unified.py).
Exposes violation_risk score per clause; can optionally use law-aware retrieval at inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

_UNIFIED_ARTIFACTS_DIRS = [
    Path(__file__).resolve().parent / "ml" / "artifacts" / "unified",
    Path(__file__).resolve().parent.parent / "app" / "ml" / "artifacts" / "unified",
    Path.cwd() / "app" / "ml" / "artifacts" / "unified",
]
if os.environ.get("UNIFIED_ML_ARTIFACTS_DIR"):
    _UNIFIED_ARTIFACTS_DIRS.insert(0, Path(os.environ["UNIFIED_ML_ARTIFACTS_DIR"]).expanduser().resolve())

_unified_dir: Optional[Path] = None
_unified_vectorizer = None
_unified_model = None
_unified_threshold: float = 0.5
_unified_config: Dict[str, Any] = {}
_unified_law_retriever = None


def _find_unified_artifacts() -> Optional[Path]:
    for d in _UNIFIED_ARTIFACTS_DIRS:
        try:
            d = d.resolve()
            if d.exists() and (d / "model.joblib").exists() and (d / "vectorizer.joblib").exists():
                return d
        except Exception:
            continue
    return None


def _load_unified() -> bool:
    global _unified_dir, _unified_vectorizer, _unified_model, _unified_threshold, _unified_config, _unified_law_retriever
    if _unified_model is not None:
        return True
    _unified_dir = _find_unified_artifacts()
    if not _unified_dir:
        return False
    try:
        _unified_vectorizer = joblib.load(_unified_dir / "vectorizer.joblib")
        _unified_model = joblib.load(_unified_dir / "model.joblib")
        _unified_threshold = 0.5
        if (_unified_dir / "thresholds.json").exists():
            data = json.loads((_unified_dir / "thresholds.json").read_text(encoding="utf-8"))
            _unified_threshold = float(data.get("threshold", 0.5))
        if (_unified_dir / "config.json").exists():
            _unified_config = json.loads((_unified_dir / "config.json").read_text(encoding="utf-8"))
        if _unified_config.get("law_aware"):
            try:
                from app.bm25_law_retriever import BM25LawRetriever, load_law_docs_for_bm25
                chunks_dir = Path(__file__).resolve().parent.parent / "chunks"
                docs = load_law_docs_for_bm25(chunks_dir)
                if docs:
                    _unified_law_retriever = BM25LawRetriever(k1=1.5, b=0.75)
                    _unified_law_retriever.build_index(docs)
            except Exception:
                _unified_law_retriever = None
        return True
    except Exception:
        _unified_vectorizer = None
        _unified_model = None
        return False


def predict_violation_risk(text: str, *, use_law_at_inference: bool = False) -> Tuple[float, bool]:
    """
    Return (score, is_above_threshold) for clause-level violation risk.
    If use_law_at_inference and model was trained law_aware, appends top-K law chunks.
    """
    if not _load_unified() or _unified_model is None or _unified_vectorizer is None:
        raise RuntimeError("Unified ML artifacts not found. Run ml/scripts/train_unified.py first.")

    inp = text.strip()
    if not inp:
        return 0.0, False

    if use_law_at_inference and _unified_law_retriever is not None:
        hits = _unified_law_retriever.search(inp, top_k=_unified_config.get("top_k_law", 5), min_score=0.0)
        law_parts = [h.get("text", "")[:800] for h in hits if h.get("text")]
        if law_parts:
            inp = inp + "\n[LAW]\n" + "\n[LAW]\n".join(law_parts)

    x = _unified_vectorizer.transform([inp])
    score = float(_unified_model.predict_proba(x)[0, 1])
    passed = score >= _unified_threshold
    return score, passed


def predict_violation_risk_safe(text: str, *, use_law_at_inference: bool = False) -> Optional[Tuple[float, bool]]:
    """Same as predict_violation_risk but returns None if unified model not loaded."""
    if not _load_unified() or _unified_model is None:
        return None
    try:
        return predict_violation_risk(text, use_law_at_inference=use_law_at_inference)
    except Exception:
        return None


def has_unified_predictor() -> bool:
    return _load_unified() and _unified_model is not None


def get_unified_metrics() -> Dict[str, Any]:
    """Return last saved metrics from unified training if available."""
    if not _unified_dir or not (_unified_dir / "metrics.json").exists():
        return {}
    try:
        return json.loads((_unified_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
