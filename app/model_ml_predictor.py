"""
Model-ML predictor: loads model_ML bundles (law_aware_multilabel, law_aware_binary),
uses labor14_2025_clean.jsonl for retrieval when needed, and exposes a rule_hits–compatible API.
No rule engine dependency.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Paths relative to project root (parent of app/)
_BASE = Path(__file__).resolve().parent.parent
MODEL_ML_DIR = _BASE / "model_ML"
MULTILABEL_BUNDLE_PATH = MODEL_ML_DIR / "law_aware_multilabel_model.joblib"
BINARY_BUNDLE_PATH = MODEL_ML_DIR / "law_aware_binary_model.joblib"
LAW_JSONL_PATH = MODEL_ML_DIR / "labor14_2025_clean.jsonl"

# Inference constants (aligned with model_ML/infer_ocr_to_llm_payload.py)
BM25_TOP_K = 30
TOP_K_EVIDENCE = 6
MIN_QUERY_TOKENS = 3
GLOBAL_MIN_LABEL_PROB = 0.0
HARD_MIN_PROB = 0.15

# Normalization (Arabic-friendly)
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EASTERN_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
BIDI_CHARS_RE = re.compile(r"[\u200e\u200f\u202a\u202b\u202c\u202d\u202e]")
VIOL_TOKEN_RE = re.compile(r"\[\[VIOLATION_\d+\]\]", re.IGNORECASE)


def _normalize_ar(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_TATWEEL, "")
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.translate(_ARABIC_INDIC).translate(_EASTERN_INDIC)
    t = BIDI_CHARS_RE.sub("", t)
    t = t.replace("\u00a0", " ").replace("ـ", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_leakage_tokens(text: str) -> str:
    return VIOL_TOKEN_RE.sub(" ", text or "")


def _normalize_for_tokens(text: str) -> str:
    t = _strip_leakage_tokens(text)
    t = _normalize_ar(t).lower()
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_simple(text: str) -> List[str]:
    t = _normalize_for_tokens(text)
    if not t:
        return []
    return [w for w in t.split(" ") if len(w) >= 2]


def _sget(d: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            if s != "":
                return s
    return default


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _build_retrieval_from_law_jsonl(law_jsonl: Path) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    law_rows = _read_jsonl(law_jsonl)
    docs: List[dict] = []
    df: Dict[str, int] = {}

    for r in law_rows:
        article = _sget(r, ["article", "article_no", "articleId", "id"])
        text = _sget(r, ["text", "article_text", "content"])
        heading = _sget(r, ["heading", "title", "name"], default="")
        if not article or not text:
            continue
        toks = _tokenize_simple((heading + " " + text).strip())
        if not toks:
            continue
        docs.append({"article": str(article), "heading": str(heading), "text": str(text), "tokens": toks})
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    N = len(docs)
    if N == 0:
        raise RuntimeError("Law docs empty. Check LAW_JSONL content/fields (article/text).")

    idf: Dict[str, float] = {}
    for w, dfi in df.items():
        idf[w] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))
    avgdl = sum(len(d["tokens"]) for d in docs) / max(1, N)
    corpus = [_normalize_for_tokens((d.get("heading", "") + " " + d.get("text", "")).strip()) for d in docs]
    tfidf_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_df=0.98, sublinear_tf=True)
    X_law = tfidf_vec.fit_transform(corpus)

    return {
        "bm25": {"docs": docs, "idf": idf, "avgdl": float(avgdl), "k1": 1.5, "b": 0.75},
        "tfidf_reranker": {"vectorizer": tfidf_vec, "X_law": X_law},
    }


def _bm25_score(
    idf: Dict[str, float], avgdl: float, k1: float, b: float, q: List[str], doc_tokens: List[str]
) -> float:
    if not q or not doc_tokens:
        return 0.0
    tf: Dict[str, int] = {}
    for w in doc_tokens:
        tf[w] = tf.get(w, 0) + 1
    dl = len(doc_tokens)
    score = 0.0
    for w in q:
        if w not in tf:
            continue
        idf_w = float(idf.get(w, 0.0))
        freq = tf[w]
        denom = freq + k1 * (1 - b + b * (dl / (avgdl + 1e-9)))
        score += idf_w * (freq * (k1 + 1) / (denom + 1e-9))
    return float(score)


def _retrieve_law_evidence(
    clause_text: str,
    bm25_pack: dict,
    tfidf_vec: Any,
    X_law: Any,
    top_k_evidence: int = TOP_K_EVIDENCE,
    bm25_top_k: int = BM25_TOP_K,
) -> Tuple[float, str]:
    q = _tokenize_simple(clause_text)
    if len(q) < MIN_QUERY_TOKENS:
        return 0.0, ""

    docs = bm25_pack["docs"]
    idf = bm25_pack["idf"]
    avgdl = float(bm25_pack["avgdl"])
    k1 = float(bm25_pack.get("k1", 1.5))
    b = float(bm25_pack.get("b", 0.75))

    scored: List[Tuple[int, float]] = []
    for i, d in enumerate(docs):
        s = _bm25_score(idf, avgdl, k1, b, q, d.get("tokens", []))
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: max(bm25_top_k, top_k_evidence)]
    if not scored:
        return 0.0, ""

    idxs = [i for (i, _) in scored]
    bm25_scores = np.array([s for (_, s) in scored], dtype=float)
    clause_norm = _normalize_for_tokens(clause_text)
    X_clause = tfidf_vec.transform([clause_norm])
    X_cand = X_law[idxs]
    cos = (X_clause @ X_cand.T).toarray().ravel().astype(float)

    def minmax(a: np.ndarray) -> np.ndarray:
        if len(a) == 0:
            return a
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-12:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    final = 0.35 * minmax(bm25_scores) + 0.65 * minmax(cos)
    ev_lines: List[str] = []
    total_chars = 0
    EVIDENCE_MAX_TOTAL_CHARS = 3800
    EVIDENCE_MAX_PER_ARTICLE = 900
    for j in np.argsort(-final):
        d = docs[idxs[j]]
        art = str(d.get("article", "")).strip()
        txt = (str(d.get("text", "") or ""))[:EVIDENCE_MAX_PER_ARTICLE]
        line = f"المادة {art}: {txt}"
        if total_chars + len(line) > EVIDENCE_MAX_TOTAL_CHARS:
            break
        total_chars += len(line)
        ev_lines.append(line)
    evidence_text = "\n\n".join(ev_lines) if ev_lines else ""
    bm25_sum = float(sum(bm25_scores))
    return bm25_sum, evidence_text


def _build_model_input_text(bm25_sum: float, clause_text: str, evidence_text: str) -> str:
    return (
        f"BM25_SUM={bm25_sum:.4f}\n"
        f"CONTRACT_META:\n\n"
        f"CLAUSE:\n{clause_text}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )


# Module-level state
_multilabel_bundle: Optional[dict] = None
_binary_bundle: Optional[dict] = None
_retrieval: Optional[Dict[str, Any]] = None
_HAS_MODEL_ML = False


def _load_bundles() -> None:
    global _multilabel_bundle, _binary_bundle, _retrieval, _HAS_MODEL_ML
    if _multilabel_bundle is not None:
        return
    if not MULTILABEL_BUNDLE_PATH.exists():
        return
    try:
        _multilabel_bundle = joblib.load(MULTILABEL_BUNDLE_PATH)
        if not isinstance(_multilabel_bundle, dict):
            _multilabel_bundle = None
            return
    except Exception:
        _multilabel_bundle = None
        return

    retr = (_multilabel_bundle.get("retrieval") or {}) or {}
    bm25_pack = retr.get("bm25", {}) or {}
    tfidf_pack = retr.get("tfidf_reranker", {}) or {}
    tfidf_vec = tfidf_pack.get("vectorizer")
    X_law = tfidf_pack.get("X_law")

    if not bm25_pack or tfidf_vec is None or X_law is None:
        if LAW_JSONL_PATH.exists():
            _retrieval = _build_retrieval_from_law_jsonl(LAW_JSONL_PATH)
        else:
            _retrieval = None
    else:
        _retrieval = {"bm25": bm25_pack, "tfidf_reranker": tfidf_pack}

    try:
        if BINARY_BUNDLE_PATH.exists():
            _binary_bundle = joblib.load(BINARY_BUNDLE_PATH)
            if not isinstance(_binary_bundle, dict):
                _binary_bundle = None
    except Exception:
        _binary_bundle = None

    _HAS_MODEL_ML = _multilabel_bundle is not None


def has_model_ml_predictor() -> bool:
    _load_bundles()
    return _HAS_MODEL_ML


def _get_retrieval_artifacts() -> Tuple[Any, Any, Any]:
    _load_bundles()
    if not _retrieval:
        return None, None, None
    bm25 = _retrieval.get("bm25", {})
    tfidf = _retrieval.get("tfidf_reranker", {})
    vec = tfidf.get("vectorizer")
    X_law = tfidf.get("X_law")
    return bm25, vec, X_law


def predict_rule_scores_full(
    text: str,
    *,
    sort: bool = True,
    use_law_retrieval: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns list of {rule_id, score, passed_threshold, ...} compatible with main.py
    rule_hits and ml_predictions. Uses multilabel bundle; no rule engine.
    """
    _load_bundles()
    if not _multilabel_bundle:
        return []

    vec = _multilabel_bundle.get("vectorizer")
    classifiers = _multilabel_bundle.get("classifiers") or {}
    label_names = list(_multilabel_bundle.get("label_names") or [])
    thresholds = dict(_multilabel_bundle.get("thresholds") or {})

    if not label_names or not classifiers:
        return []

    clause_text = _normalize_ar(_strip_leakage_tokens(text or ""))
    if not clause_text.strip():
        return []

    model_input = clause_text
    bm25_sum = 0.0
    if use_law_retrieval:
        bm25_pack, tfidf_vec, X_law = _get_retrieval_artifacts()
        if bm25_pack and tfidf_vec is not None and X_law is not None:
            bm25_sum, evidence_text = _retrieve_law_evidence(
                clause_text, bm25_pack, tfidf_vec, X_law,
                top_k_evidence=TOP_K_EVIDENCE, bm25_top_k=BM25_TOP_K,
            )
            model_input = _build_model_input_text(bm25_sum, clause_text, evidence_text)

    Xv = vec.transform([model_input])
    probs: List[float] = []
    for lab in label_names:
        clf = classifiers.get(lab)
        if clf is None:
            probs.append(0.0)
            continue
        p = clf.predict_proba(Xv)[0, 1]
        probs.append(float(p))

    out: List[Dict[str, Any]] = []
    for j, lab in enumerate(label_names):
        thr = float(thresholds.get(lab, 0.5))
        thr = max(thr, GLOBAL_MIN_LABEL_PROB)
        pr = probs[j] if j < len(probs) else 0.0
        passed = pr >= max(thr, HARD_MIN_PROB)
        out.append({
            "rule_id": lab,
            "score": pr,
            "prob": pr,
            "threshold": thr,
            "passed_threshold": passed,
        })
    if sort:
        out.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)
    return out


def predict_violation_risk_safe(
    text: str,
    *,
    use_law_at_inference: bool = True,
) -> Optional[Tuple[float, bool]]:
    """
    Returns (binary_risk_score, above_threshold). Prefers binary bundle if present;
    otherwise uses max multilabel prob as risk and threshold from meta or 0.5.
    """
    _load_bundles()

    if _binary_bundle:
        vec = _binary_bundle.get("vectorizer")
        clf = _binary_bundle.get("classifier")
        thr = float((_binary_bundle.get("meta") or {}).get("threshold", 0.5))
        if vec is not None and clf is not None:
            clause_text = _normalize_ar(_strip_leakage_tokens(text or ""))
            if not clause_text.strip():
                return (0.0, False)
            model_input = clause_text
            if use_law_at_inference:
                bm25_pack, tfidf_vec, X_law = _get_retrieval_artifacts()
                if bm25_pack and tfidf_vec is not None and X_law is not None:
                    bm25_sum, evidence_text = _retrieve_law_evidence(
                        clause_text, bm25_pack, tfidf_vec, X_law,
                        top_k_evidence=TOP_K_EVIDENCE, bm25_top_k=BM25_TOP_K,
                    )
                    model_input = _build_model_input_text(bm25_sum, clause_text, evidence_text)
            Xv = vec.transform([model_input])
            p = clf.predict_proba(Xv)[0, 1]
            return (float(p), float(p) >= max(thr, HARD_MIN_PROB))

    if _multilabel_bundle:
        preds = predict_rule_scores_full(text, sort=True, use_law_retrieval=use_law_at_inference)
        if not preds:
            return (0.0, False)
        max_prob = max(p.get("score") or 0.0 for p in preds)
        thr = 0.5
        return (max_prob, max_prob >= max(thr, HARD_MIN_PROB))

    return None


def rule_scores_to_rule_hits(
    preds: List[Dict[str, Any]],
    chunk_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert predict_rule_scores_full output to main.py rule_hits shape."""
    hits: List[Dict[str, Any]] = []
    for p in preds:
        rid = p.get("rule_id")
        if not rid:
            continue
        score = p.get("score") or p.get("prob") or 0.0
        if not p.get("passed_threshold"):
            continue
        hits.append({
            "rule_id": rid,
            "severity": "high",
            "description": f"ML-detected potential violation ({rid}). Verify with legal review.",
            "matched_text": None,
            "chunk_id": chunk_id,
            "law": None,
            "article": None,
            "ml_detected": True,
            "score": score,
        })
    return hits
