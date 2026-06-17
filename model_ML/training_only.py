# app/training_only.py
# ------------------------------------------------------------
# TRAINING-ONLY pipeline (RAG-style retrieval + ML) with STRONGER evaluation.
#
# What’s improved vs your version:
# 1) ✅ Auto-builds clause-level violation IDs using dataset_violations.jsonl
# 2) ✅ Proper metrics output (clause-level + contract-level)
# 3) ✅ Dataset integrity checks + clear report sections
# 4) ✅ Still splits by contract_id (GroupShuffleSplit) to prevent leakage
#
# ✅ NEW IN THIS VERSION (your request):
# A) Clause unit QA + optional re-splitting (NO OCR) BEFORE training examples
# B) Safe per-label calibration (prevents: "Requesting 3-fold CV but <3 examples")
#
# ✅ ADDITIONS (requested now - keep everything, add only):
# C) Retrieval evaluation (Recall@K / MRR / Precision@K) when GT law articles exist
# D) Evidence dedup + truncation safety (stable input size)
# E) Evidence ablation: evaluate with evidence vs without evidence on validation/test
# F) Save FULL inference bundle including retrieval artifacts (BM25 + TF-IDF reranker)
# G) Save LLM payload schema + safety prompt template in the bundle (for next stage)
#
# Usage:
#   (.venv) python -m app.training_only
#
# Outputs:
#   models/law_aware_multilabel_model.joblib
#   cases/processed/training_only_model_report.txt
#
# Requirements:
#   pip install scikit-learn joblib numpy
# ------------------------------------------------------------

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Iterable

import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)

# =========================
# PATHS
# =========================
LAW_JSONL = Path(r"laws\preprocessed\labor14_2025_clean.jsonl")

CLAUSES_JSONL = Path(r"cases\processed\dataset_clauses.jsonl")
CONTRACTS_JSONL = Path(r"cases\processed\dataset_contracts.jsonl")
VIOLATIONS_JSONL = Path(r"cases\processed\dataset_violations.jsonl")  # <-- optional but recommended

# OPTIONAL: if you have a mapping of violation_id -> law_article(s) for retrieval evaluation
# JSONL rows accepted examples:
#   {"violation_id":"VIOLATION_0012","law_articles":["111","112"]}
#   {"violation_id":"VIOLATION_0012","law_article":"111"}
VIOLATION_TO_LAW_JSONL = Path(r"cases\processed\violation_to_law.jsonl")  # optional

OUT_DIR = Path(r"cases\processed")
MODELS_DIR = Path("models")

MODEL_BUNDLE_PATH = MODELS_DIR / "law_aware_multilabel_model.joblib"
MODEL_REPORT_TXT = OUT_DIR / "training_only_model_report.txt"

# =========================
# Retrieval settings (RAG-ish)
# =========================
BM25_TOP_K = 30
MIN_QUERY_TOKENS = 3
TOP_K_EVIDENCE_FOR_MODEL = 6

# Evidence safety: prevent huge concatenation
EVIDENCE_MAX_CHARS_PER_ARTICLE = 900
EVIDENCE_MAX_TOTAL_CHARS = 3800
EVIDENCE_DEDUP = True

# Retrieval evaluation K values
RETR_EVAL_KS = [1, 3, 5, 10]

# =========================
# Train/val/test split (by contract_id)
# =========================
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

# =========================
# Classifier settings
# =========================
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
CALIBRATION_METHOD = "isotonic"  # "sigmoid" faster; "isotonic" often better

# Threshold objective per label (on validation)
THRESH_OBJECTIVE = "f05"  # "f05" | "f1" | "acc"

GLOBAL_MIN_LABEL_PROB = 0.0  # optional global min over per-label threshold

# =========================
# Clause unit QA + optional re-splitting (NO OCR)
# =========================
ENABLE_CLAUSE_RESPLIT = True  # True: fix merged clause units; False: keep as-is
MIN_CLAUSE_CHARS = 25
MAX_CLAUSE_CHARS = 2000
MAX_CLAUSE_LINES = 22
MULTI_ITEM_MARKERS_THRESHOLD = 3  # >=3 bullets/number items => likely merged


# =========================
# JSONL helpers
# =========================
def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# =========================
# Text normalization
# =========================
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EASTERN_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
BIDI_CHARS_RE = re.compile(r"[\u200e\u200f\u202a\u202b\u202c\u202d\u202e]")

VIOL_TOKEN_RE = re.compile(r"\[\[VIOLATION_\d+\]\]", re.IGNORECASE)


def normalize_ar(text: str) -> str:
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


def strip_leakage_tokens(text: str) -> str:
    return VIOL_TOKEN_RE.sub(" ", text or "")


def normalize_for_tokens(text: str) -> str:
    t = strip_leakage_tokens(text)
    t = normalize_ar(t).lower()
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_simple(text: str) -> List[str]:
    t = normalize_for_tokens(text)
    if not t:
        return []
    return [w for w in t.split(" ") if len(w) >= 2]


# =========================
# Robust getters
# =========================
def sget(d: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            if s != "":
                return s
    return default


def as_int_label(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def extract_contract_id_any(d: dict) -> str:
    return sget(d, ["contract_id", "contractId", "contract", "case_id", "caseId", "doc_id", "docId", "id"], default="")


def extract_clause_id_any(d: dict) -> str:
    return sget(d, ["clause_id", "clauseId", "clauseid", "id"], default="")


def extract_clause_text_any(d: dict) -> str:
    return sget(d, ["clause_text", "text", "clause", "content", "body"], default="")


def _coerce_violation_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        n = int(v)
        return f"VIOLATION_{n:04d}"
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        s = re.sub(r"[\[\]]", "", s)
        m = re.search(r"(violation[_\s-]*)(\d+)", s, re.IGNORECASE)
        if m:
            return f"VIOLATION_{int(m.group(2)):04d}"
        if re.fullmatch(r"\d+", s):
            return f"VIOLATION_{int(s):04d}"
        return s.upper()
    return None


def extract_violation_ids_any(d: dict) -> List[str]:
    candidate_keys = [
        "violations", "violation_ids", "violationIds", "violation_id", "violationId",
        "labels", "label_ids", "labelIds", "targets", "target_ids", "targetIds"
    ]

    for k in candidate_keys:
        if k not in d:
            continue
        v = d.get(k)

        out: List[str] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    vid = _coerce_violation_id(
                        item.get("id") or item.get("violation_id") or item.get("violationId") or item.get("label")
                    )
                    if vid:
                        out.append(vid)
                else:
                    vid = _coerce_violation_id(item)
                    if vid:
                        out.append(vid)
        elif isinstance(v, dict):
            for subk in ["ids", "violations", "labels", "items"]:
                if subk in v and isinstance(v[subk], list):
                    for item in v[subk]:
                        vid = _coerce_violation_id(item if not isinstance(item, dict) else (item.get("id") or item.get("label")))
                        if vid:
                            out.append(vid)
        else:
            vid = _coerce_violation_id(v)
            if vid:
                out.append(vid)

        if out:
            seen: Set[str] = set()
            uniq = []
            for x in out:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

    return []


def extract_binary_unlawful_label_any(d: dict) -> Optional[int]:
    for k in ["label_unlawful", "unlawful", "is_unlawful", "label", "y"]:
        if k in d:
            return as_int_label(d.get(k))
    return None


# =========================
# Auto-labeling from dataset_violations.jsonl
# =========================
def build_violation_maps(viol_rows: List[dict]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      clause_id -> [VIOLATION_x,...]
      contract_id -> [VIOLATION_x,...]
    Expects each row has: contract_id, violation_id, clause_id (or similar).
    """
    by_clause: Dict[str, List[str]] = {}
    by_contract: Dict[str, List[str]] = {}

    for r in viol_rows:
        cid = extract_contract_id_any(r)
        clid = extract_clause_id_any(r)
        vid = _coerce_violation_id(r.get("violation_id") or r.get("violationId") or r.get("id") or "")
        if not vid:
            continue
        if clid:
            by_clause.setdefault(clid, []).append(vid)
        if cid:
            by_contract.setdefault(cid, []).append(vid)

    def dedup_map(m: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, vs in m.items():
            seen: Set[str] = set()
            uniq = []
            for x in vs:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            out[k] = uniq
        return out

    return dedup_map(by_clause), dedup_map(by_contract)


# =========================
# OPTIONAL: violation -> law article mapping (for retrieval evaluation)
# =========================
def read_violation_to_law_map(path: Path) -> Dict[str, Set[str]]:
    """
    Returns:
      VIOLATION_XXXX -> set({"111","112",...})
    """
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    out: Dict[str, Set[str]] = {}
    for r in rows:
        vid = _coerce_violation_id(r.get("violation_id") or r.get("violationId") or r.get("id"))
        if not vid:
            continue
        arts: Set[str] = set()
        la = r.get("law_article")
        las = r.get("law_articles")
        if isinstance(la, (str, int)):
            arts.add(str(la).strip())
        if isinstance(las, list):
            for x in las:
                if isinstance(x, (str, int)):
                    arts.add(str(x).strip())
        if arts:
            out.setdefault(vid, set()).update(arts)
    return out


# =========================
# Clause unit QA + optional re-splitting (NO OCR)
# =========================
_AR_NUM_ITEM = re.compile(r"(?m)^\s*(?:\d{1,3}[\)\.\-]|[أ-ي]\s*[\)\-]|[-•●])\s+")
_AR_HEADING = re.compile(r"(?m)^\s*(?:البند|مادة|المادة)\s*\(?\s*\d+\s*\)?\s*[:：]?\s*$")
_MULTI_SPACES = re.compile(r"[ \t]+")


@dataclass
class ClauseSplitStats:
    clauses_in: int = 0
    clauses_out: int = 0
    flagged_in: int = 0
    resplit_applied: int = 0
    flags: Dict[str, int] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = {}


def _clause_flags(text: str) -> List[str]:
    t = (text or "").strip()
    flags: List[str] = []
    if len(t) < MIN_CLAUSE_CHARS:
        flags.append("too_short")
    if len(t) > MAX_CLAUSE_CHARS:
        flags.append("too_long")

    lines = [x for x in t.splitlines() if x.strip()]
    if len(lines) > MAX_CLAUSE_LINES:
        flags.append("too_many_lines")

    markers = len(_AR_NUM_ITEM.findall(t))
    if markers >= MULTI_ITEM_MARKERS_THRESHOLD:
        flags.append("multi_item_merged")

    headings = len(_AR_HEADING.findall(t))
    if headings >= 2:
        flags.append("multiple_headings")

    return flags


def _resplit_clause(text: str) -> List[str]:
    """
    Conservative splitting for Arabic/contract formats:
    - split on heading-like lines (المادة/مادة/البند)
    - split on list markers: 1) / 1- / أ) / - / •
    """
    t = (text or "").strip()
    if not t:
        return []
    t = _MULTI_SPACES.sub(" ", t)

    # split by headings if present
    if _AR_HEADING.search(t):
        parts = re.split(r"(?m)(?=^\s*(?:البند|مادة|المادة)\s*\(?\s*\d+\s*\)?)", t)
        parts = [p.strip() for p in parts if p.strip()]
    else:
        parts = [t]

    out: List[str] = []
    for part in parts:
        sub = re.split(r"(?m)(?=^\s*(?:\d{1,3}[\)\.\-]|[أ-ي]\s*[\)\-]|[-•●])\s+)", part)
        sub = [s.strip() for s in sub if s.strip()]
        if len(sub) == 1:
            out.append(sub[0])
        else:
            # merge tiny fragments into neighbors
            buf = ""
            for s in sub:
                if len(s) < MIN_CLAUSE_CHARS and buf:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        out.append(buf)
                    buf = s
            if buf:
                out.append(buf)

    out2 = [x for x in out if len(x.strip()) >= MIN_CLAUSE_CHARS]
    return out2 if out2 else [t]


def apply_clause_unit_refinement(
    clauses_rows: List[dict],
    viol_by_clause_id: Dict[str, List[str]],
) -> Tuple[List[dict], ClauseSplitStats]:
    """
    Runs QA on each clause unit. If ENABLE_CLAUSE_RESPLIT=True and a clause looks merged,
    resplits it into multiple rows.

    Label safety:
    - If a clause_id had labels in viol_by_clause_id, we copy them to each new sub-id
      so we don't lose supervision.
    """
    st = ClauseSplitStats()
    st.clauses_in = len(clauses_rows)

    out_rows: List[dict] = []

    for r in clauses_rows:
        cid = extract_contract_id_any(r)
        clid = extract_clause_id_any(r)
        raw = extract_clause_text_any(r)
        text = strip_leakage_tokens(raw).strip()

        if not cid or not text:
            continue

        flags = _clause_flags(text)
        if flags:
            st.flagged_in += 1
            for f in flags:
                st.flags[f] = st.flags.get(f, 0) + 1

        should_split = ("multi_item_merged" in flags) or ("multiple_headings" in flags) or ("too_long" in flags)

        if ENABLE_CLAUSE_RESPLIT and should_split:
            parts = _resplit_clause(text)
            if len(parts) > 1:
                st.resplit_applied += 1
                inherited = viol_by_clause_id.get(clid, []) if clid else []

                for pi, ptxt in enumerate(parts, start=1):
                    rr = dict(r)
                    new_id = (clid or "clause") + f"__p{pi}"
                    rr["clause_id"] = new_id
                    rr["clause_text"] = ptxt

                    if inherited:
                        viol_by_clause_id[new_id] = inherited

                    out_rows.append(rr)
                continue

        out_rows.append(r)

    st.clauses_out = len(out_rows)
    return out_rows, st


# =========================
# BM25 over law articles
# =========================
@dataclass
class LawDoc:
    article: str
    heading: str
    text: str
    tokens: List[str]


@dataclass
class BM25Index:
    docs: List[LawDoc]
    idf: Dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(law_rows: List[dict]) -> BM25Index:
    docs: List[LawDoc] = []
    df: Dict[str, int] = {}

    for r in law_rows:
        article = sget(r, ["article", "article_no", "articleId", "id"])
        text = sget(r, ["text", "article_text", "content"])
        heading = sget(r, ["heading", "title", "name"], default="")
        if not article or not text:
            continue

        toks = tokenize_simple((heading + " " + text).strip())
        if not toks:
            continue

        docs.append(LawDoc(article=str(article), heading=str(heading), text=str(text), tokens=toks))
        for w in set(toks):
            df[w] = df.get(w, 0) + 1

    N = len(docs)
    if N == 0:
        raise RuntimeError("BM25 index is empty. Check law JSONL fields (article/text).")

    idf: Dict[str, float] = {}
    for w, dfi in df.items():
        idf[w] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    avgdl = sum(len(d.tokens) for d in docs) / max(1, N)
    return BM25Index(docs=docs, idf=idf, avgdl=avgdl)


def bm25_score(index: BM25Index, q: List[str], doc_tokens: List[str]) -> float:
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
        idf = index.idf.get(w, 0.0)
        freq = tf[w]
        denom = freq + index.k1 * (1 - index.b + index.b * (dl / index.avgdl))
        score += idf * (freq * (index.k1 + 1) / (denom + 1e-9))
    return float(score)


def retrieve_bm25_topk(index: BM25Index, clause_text: str, k: int) -> List[Tuple[int, float]]:
    q = tokenize_simple(clause_text)
    if len(q) < MIN_QUERY_TOKENS:
        return []

    scored: List[Tuple[int, float]] = []
    for i, d in enumerate(index.docs):
        s = bm25_score(index, q, d.tokens)
        if s > 0:
            scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# =========================
# TF-IDF reranker (cosine similarity)
# =========================
def build_tfidf_reranker(law_docs: List[LawDoc]) -> Tuple[TfidfVectorizer, Any]:
    corpus = [normalize_for_tokens(d.heading + " " + d.text) for d in law_docs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_df=0.98, sublinear_tf=True)
    X_law = vec.fit_transform(corpus)
    return vec, X_law


def rerank_candidates(
    clause_text: str,
    candidates: List[Tuple[int, float]],
    law_docs: List[LawDoc],
    tfidf_vec: TfidfVectorizer,
    X_law: Any,
) -> List[dict]:
    if not candidates:
        return []

    clause_norm = normalize_for_tokens(clause_text)
    X_clause = tfidf_vec.transform([clause_norm])

    idxs = [i for (i, _) in candidates]
    bm25_scores = np.array([s for (_, s) in candidates], dtype=float)

    X_cand = X_law[idxs]
    cos = cosine_similarity(X_clause, X_cand).ravel().astype(float)

    def minmax(a: np.ndarray) -> np.ndarray:
        if len(a) == 0:
            return a
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-12:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    final = 0.35 * minmax(bm25_scores) + 0.65 * minmax(cos)

    out: List[dict] = []
    for j, law_i in enumerate(idxs):
        d = law_docs[law_i]
        out.append(
            {
                "article": d.article,
                "heading": d.heading,
                "bm25": float(bm25_scores[j]),
                "cosine": float(cos[j]),
                "score": float(final[j]),
                "text": d.text,
            }
        )

    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# =========================
# Evidence formatting safety (DEDUP + TRUNC)
# =========================
def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def build_law_evidence_for_clause_structured(
    clause_text: str,
    bm25: BM25Index,
    tfidf_vec: TfidfVectorizer,
    X_law: Any,
    top_k: int,
) -> Tuple[float, str, List[dict]]:
    """
    Returns:
      bm25_sum, evidence_text, evidence_struct(list)
    """
    bm25_cands = retrieve_bm25_topk(bm25, clause_text, max(BM25_TOP_K, top_k))
    reranked = rerank_candidates(clause_text, bm25_cands, bm25.docs, tfidf_vec, X_law)
    top = reranked[:top_k]

    if EVIDENCE_DEDUP:
        seen = set()
        deduped = []
        for x in top:
            a = str(x.get("article", "")).strip()
            if not a or a in seen:
                continue
            seen.add(a)
            deduped.append(x)
        top = deduped

    bm25_sum = float(sum(x.get("bm25", 0.0) for x in top))

    total_chars = 0
    lines: List[str] = []
    ev_struct: List[dict] = []
    for x in top:
        art = str(x.get("article", "")).strip()
        txt = str(x.get("text", "") or "")
        excerpt = _truncate(txt, EVIDENCE_MAX_CHARS_PER_ARTICLE)

        line = f"المادة {art}: {excerpt}"
        if total_chars + len(line) > EVIDENCE_MAX_TOTAL_CHARS:
            break
        total_chars += len(line)

        lines.append(line)
        ev_struct.append(
            {
                "article": art,
                "heading": str(x.get("heading", "") or ""),
                "bm25": float(x.get("bm25", 0.0)),
                "cosine": float(x.get("cosine", 0.0)),
                "score": float(x.get("score", 0.0)),
                "excerpt": excerpt,
            }
        )

    evidence = "\n\n".join(lines) if lines else ""
    return bm25_sum, evidence, ev_struct


# Backwards compatible wrapper (keeps your original function signature)
def build_law_evidence_for_clause(
    clause_text: str,
    bm25: BM25Index,
    tfidf_vec: TfidfVectorizer,
    X_law: Any,
    top_k: int,
) -> Tuple[float, str]:
    bm25_sum, evidence, _ = build_law_evidence_for_clause_structured(
        clause_text=clause_text,
        bm25=bm25,
        tfidf_vec=tfidf_vec,
        X_law=X_law,
        top_k=top_k,
    )
    return bm25_sum, evidence


# =========================
# Vectorizer + safe calibration
# =========================
def make_vectorizer() -> FeatureUnion:
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 7), min_df=2, max_df=0.97, sublinear_tf=True)
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.97,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )
    return FeatureUnion([("char", char_vec), ("word", word_vec)])


def make_calibrator(base_model, method: str, y_train: np.ndarray):
    """
    Safe calibration: choose cv based on min(#pos, #neg).
    If too few examples, return the base_model (no calibration).
    """
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    m = min(pos, neg)

    if m < 2:
        return base_model  # cannot calibrate safely

    cv = min(3, m)
    try:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=method, cv=cv)


def fbeta_from_pr(p: float, r: float, beta: float) -> float:
    if p <= 0 and r <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (p * r) / (b2 * p + r + 1e-12)


def pick_threshold_binary(y_val: np.ndarray, p_val: np.ndarray, objective: str) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0

    for thr in np.linspace(0.05, 0.95, 37):
        pred = (p_val >= thr).astype(int)
        acc = float((pred == y_val).mean())
        p, r, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", zero_division=0)

        if objective == "acc":
            score = acc
        elif objective == "f1":
            score = float(f1)
        else:
            score = fbeta_from_pr(float(p), float(r), beta=0.5)

        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr, best_score


# =========================
# Metrics (binary + multilabel)
# =========================
def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, Any] = {}
    out["threshold"] = float(threshold)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)

    cm = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()
    out["classification_report"] = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return out


def multilabel_prf(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    return {
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
    }


def jaccard_per_row(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    js: List[float] = []
    for i in range(y_true_bin.shape[0]):
        t = set(np.where(y_true_bin[i] == 1)[0].tolist())
        p = set(np.where(y_pred_bin[i] == 1)[0].tolist())
        if not t and not p:
            js.append(1.0)
            continue
        if not t and p:
            js.append(0.0)
            continue
        inter = len(t & p)
        uni = len(t | p)
        js.append(inter / max(1, uni))
    return float(np.mean(js)) if js else float("nan")


def format_section(title: str, lines: List[str]) -> str:
    return "\n".join([title] + [f"  {x}" for x in lines]) + "\n"


# =========================
# Retrieval evaluation (optional, needs GT mapping)
# =========================
def _recall_at_k(hit: bool) -> float:
    return 1.0 if hit else 0.0


def _precision_at_k(hits: int, k: int) -> float:
    return float(hits) / float(max(1, k))


def _mrr(rank: Optional[int]) -> float:
    return 1.0 / float(rank) if rank and rank > 0 else 0.0


def evaluate_retrieval_for_clause(
    gt_articles: Set[str],
    retrieved_articles: List[str],
    ks: List[int],
) -> Dict[str, float]:
    """
    Compute Recall@K, Precision@K, MRR@K over the ranked list.
    """
    out: Dict[str, float] = {}
    if not gt_articles:
        return out
    ranked = [a for a in retrieved_articles if a]
    for k in ks:
        topk = ranked[:k]
        hits = sum(1 for a in topk if a in gt_articles)
        out[f"recall@{k}"] = _recall_at_k(hits > 0)
        out[f"precision@{k}"] = _precision_at_k(hits, k)
        # MRR@K: rank of first relevant within top-k
        rank = None
        for i, a in enumerate(topk, start=1):
            if a in gt_articles:
                rank = i
                break
        out[f"mrr@{k}"] = _mrr(rank)
    return out


# =========================
# Build law-aware clause examples + labels
# =========================
@dataclass
class ClauseExample:
    clause_id: str
    contract_id: str
    text_aug: str
    y_labels: List[str]
    y_binary: Optional[int]
    clause_text: str
    evidence_struct: List[dict]


def build_clause_examples(
    clauses_rows: List[dict],
    contracts_by_id: Dict[str, dict],
    bm25: BM25Index,
    tfidf_vec: TfidfVectorizer,
    X_law: Any,
    viol_by_clause_id: Dict[str, List[str]],
) -> List[ClauseExample]:
    out: List[ClauseExample] = []

    for i, r in enumerate(clauses_rows):
        contract_id = extract_contract_id_any(r)
        clause_text_raw = extract_clause_text_any(r)
        clause_text = strip_leakage_tokens(clause_text_raw).strip()
        if not contract_id or not clause_text:
            continue

        clause_id = extract_clause_id_any(r) or f"clause_{i}"

        # 1) explicit multi-label IDs if present
        y_ids = extract_violation_ids_any(r)

        # 2) otherwise infer from dataset_violations.jsonl via clause_id match
        if not y_ids and clause_id in viol_by_clause_id:
            y_ids = viol_by_clause_id[clause_id]

        # binary label: explicit if present, else infer from y_ids
        y_bin = extract_binary_unlawful_label_any(r)
        if y_bin is None:
            y_bin = 1 if len(y_ids) > 0 else 0

        # contract meta (optional)
        cmeta = contracts_by_id.get(contract_id, {})
        contract_title = sget(cmeta, ["title", "name"], default="")
        contract_type = sget(cmeta, ["type", "contract_type"], default="")

        bm25_sum, evidence, ev_struct = build_law_evidence_for_clause_structured(
            clause_text=clause_text,
            bm25=bm25,
            tfidf_vec=tfidf_vec,
            X_law=X_law,
            top_k=TOP_K_EVIDENCE_FOR_MODEL,
        )

        text_aug = (
            f"BM25_SUM={bm25_sum:.4f}\n"
            f"CONTRACT_META:\n{contract_title} {contract_type}\n"
            f"CLAUSE:\n{clause_text}\n\n"
            f"EVIDENCE:\n{evidence}\n"
        )

        out.append(
            ClauseExample(
                clause_id=clause_id,
                contract_id=contract_id,
                text_aug=text_aug,
                y_labels=y_ids,
                y_binary=y_bin,
                clause_text=clause_text,
                evidence_struct=ev_struct,
            )
        )

    return out


# =========================
# Train calibrated classifiers per label
# =========================
@dataclass
class MultiLabelModel:
    vectorizer: Any
    label_names: List[str]
    classifiers: Dict[str, Any]
    thresholds: Dict[str, float]
    best_C: Dict[str, float]
    meta: Dict[str, Any]


def predict_multilabel_probs(vec: Any, classifiers: Dict[str, Any], X_texts: List[str], label_names: List[str]) -> np.ndarray:
    Xv = vec.transform(X_texts)
    P = np.zeros((len(X_texts), len(label_names)), dtype=float)
    for j, lab in enumerate(label_names):
        clf = classifiers.get(lab)
        if clf is None:
            continue
        P[:, j] = clf.predict_proba(Xv)[:, 1]
    return P


def probs_to_preds(P: np.ndarray, label_names: List[str], thresholds: Dict[str, float], global_min: float = 0.0) -> np.ndarray:
    Yhat = np.zeros_like(P, dtype=int)
    for j, lab in enumerate(label_names):
        thr = float(thresholds.get(lab, 0.5))
        thr = max(thr, float(global_min))
        Yhat[:, j] = (P[:, j] >= thr).astype(int)
    return Yhat


def train_multilabel(
    X_train: List[str],
    Y_train_bin: np.ndarray,
    X_val: List[str],
    Y_val_bin: np.ndarray,
    label_names: List[str],
) -> Tuple[MultiLabelModel, Dict[str, Any]]:
    vec = make_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    classifiers: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}
    best_C: Dict[str, float] = {}
    val_summary: Dict[str, Any] = {"per_label": {}, "micro": {}}

    for j, lab in enumerate(label_names):
        y_tr = Y_train_bin[:, j].astype(int)
        y_va = Y_val_bin[:, j].astype(int)

        # constant in train → cannot learn
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue

        best_for_label: Optional[Tuple[Any, float, float, float, Dict[str, Any]]] = None

        for C in C_GRID:
            base = LogisticRegression(
                C=C,
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            )

            clf = make_calibrator(base, method=CALIBRATION_METHOD, y_train=y_tr)
            clf.fit(Xtr, y_tr)

            p_val = clf.predict_proba(Xva)[:, 1]
            thr, score = pick_threshold_binary(y_va, p_val, THRESH_OBJECTIVE)

            m = compute_binary_metrics(y_va, p_val, thr)
            m["val_objective"] = float(score)
            m["C"] = float(C)

            if best_for_label is None or score > best_for_label[3]:
                best_for_label = (clf, float(C), float(thr), float(score), m)

        assert best_for_label is not None
        clf_best, C_best, thr_best, _, m_best = best_for_label

        classifiers[lab] = clf_best
        thresholds[lab] = float(thr_best)
        best_C[lab] = float(C_best)
        val_summary["per_label"][lab] = m_best

    # Micro metrics on validation using learned thresholds
    P_val = predict_multilabel_probs(vec, classifiers, X_val, label_names)
    Yhat_val = probs_to_preds(P_val, label_names, thresholds, global_min=GLOBAL_MIN_LABEL_PROB)
    val_summary["micro"] = multilabel_prf(Y_val_bin, Yhat_val)
    val_summary["micro"]["jaccard_row_avg"] = jaccard_per_row(Y_val_bin, Yhat_val)

    model = MultiLabelModel(
        vectorizer=vec,
        label_names=label_names,
        classifiers=classifiers,
        thresholds=thresholds,
        best_C=best_C,
        meta={
            "model_name": "law_aware_multilabel_logreg_calibrated",
            "calibration": CALIBRATION_METHOD,
            "threshold_objective": THRESH_OBJECTIVE,
            "c_grid": C_GRID,
            "top_k_evidence": int(TOP_K_EVIDENCE_FOR_MODEL),
            "bm25_top_k": int(BM25_TOP_K),
            "random_seed": int(RANDOM_SEED),
            "global_min_label_prob": float(GLOBAL_MIN_LABEL_PROB),
            "auto_labels_from_dataset_violations": True,
            "clause_unit_refinement": True,
            "enable_clause_resplit": bool(ENABLE_CLAUSE_RESPLIT),
            "evidence_max_chars_per_article": int(EVIDENCE_MAX_CHARS_PER_ARTICLE),
            "evidence_max_total_chars": int(EVIDENCE_MAX_TOTAL_CHARS),
            "evidence_dedup": bool(EVIDENCE_DEDUP),
            "retrieval_eval_ks": list(RETR_EVAL_KS),
        },
    )
    return model, val_summary


# =========================
# Contract-level aggregation + metrics
# =========================
def aggregate_contract_predictions_max(contract_ids: List[str], P_clause: np.ndarray) -> Dict[str, np.ndarray]:
    by_contract: Dict[str, List[int]] = {}
    for i, cid in enumerate(contract_ids):
        by_contract.setdefault(cid, []).append(i)

    out: Dict[str, np.ndarray] = {}
    for cid, idxs in by_contract.items():
        out[cid] = P_clause[idxs, :].max(axis=0) if idxs else np.zeros((P_clause.shape[1],), dtype=float)
    return out


def contract_truth_by_id(
    contracts_by_id: Dict[str, dict],
    label_names: List[str],
    viol_by_contract_id: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    lab_to_j = {lab: j for j, lab in enumerate(label_names)}
    out: Dict[str, np.ndarray] = {}

    for cid, row in contracts_by_id.items():
        ids = extract_violation_ids_any(row)
        if not ids and cid in viol_by_contract_id:
            ids = viol_by_contract_id[cid]

        y = np.zeros((len(label_names),), dtype=int)
        for vid in ids:
            if vid in lab_to_j:
                y[lab_to_j[vid]] = 1
        out[cid] = y
    return out


def contract_eval_metrics(
    y_true_contract: Dict[str, np.ndarray],
    p_pred_contract: Dict[str, np.ndarray],
    label_names: List[str],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    cids = sorted(set(y_true_contract.keys()) & set(p_pred_contract.keys()))
    if not cids:
        return {"error": "No overlapping contracts for contract-level evaluation."}

    Y_true = np.stack([y_true_contract[c] for c in cids], axis=0)
    P_pred = np.stack([p_pred_contract[c] for c in cids], axis=0)
    Y_hat = probs_to_preds(P_pred, label_names, thresholds, global_min=GLOBAL_MIN_LABEL_PROB)

    m: Dict[str, Any] = {}
    m["num_contracts"] = int(len(cids))
    m.update(multilabel_prf(Y_true, Y_hat))
    m["jaccard_row_avg"] = jaccard_per_row(Y_true, Y_hat)

    exact = []
    for i in range(Y_true.shape[0]):
        exact.append(int(np.array_equal(Y_true[i], Y_hat[i])))
    m["exact_match_ratio"] = float(np.mean(exact))

    any_true = int((Y_true.sum(axis=1) > 0).sum())
    any_pred = int((Y_hat.sum(axis=1) > 0).sum())
    m["contracts_with_any_true_violation"] = any_true
    m["contracts_with_any_pred_violation"] = any_pred

    return m


# =========================
# Stability check: GroupKFold micro PR-AUC on TRAIN
# =========================
def groupkfold_micro_pr_auc_stability(
    X_train: List[str],
    Y_train_bin: np.ndarray,
    groups_train: List[str],
) -> List[str]:
    """
    Stability check ONLY.
    We intentionally DO NOT use calibration here.

    Returns micro PR-AUC over all labels (flattened) for each fold.
    """
    vec = make_vectorizer()
    X_all = vec.fit_transform(X_train)

    unique_groups = len(set(groups_train))
    n_splits = min(5, max(2, unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    lines: List[str] = []
    y_dummy = np.zeros((len(groups_train),), dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_dummy, groups=groups_train), start=1):
        P = np.zeros((len(te_idx), Y_train_bin.shape[1]), dtype=float)

        for j in range(Y_train_bin.shape[1]):
            y_tr = Y_train_bin[tr_idx, j].astype(int)
            pos = int(y_tr.sum())
            neg = int(len(y_tr) - pos)

            if pos == 0 or neg == 0:
                continue

            clf = LogisticRegression(
                C=1.0,
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            )
            clf.fit(X_all[tr_idx], y_tr)
            P[:, j] = clf.predict_proba(X_all[te_idx])[:, 1]

        y_true_flat = Y_train_bin[te_idx, :].ravel()
        y_prob_flat = P.ravel()

        if len(np.unique(y_true_flat)) > 1:
            pr_micro = float(average_precision_score(y_true_flat, y_prob_flat))
            lines.append(f"  Fold {fold}: micro PR-AUC={pr_micro:.4f}")
        else:
            lines.append(f"  Fold {fold}: micro PR-AUC=nan (only one class in fold)")

    return lines


# =========================
# Pretty reporting
# =========================
def format_per_label_summary(val_summary: Dict[str, Any], limit: int = 30) -> str:
    per = val_summary.get("per_label", {})
    if not isinstance(per, dict) or not per:
        return "No per-label validation metrics (labels missing or too sparse)."

    items = []
    for lab, md in per.items():
        obj = md.get("val_objective", -1.0)
        items.append((lab, float(obj), md))
    items.sort(key=lambda x: x[1], reverse=True)

    lines = ["PER-LABEL VALIDATION (top labels by objective):"]
    for (lab, obj, md) in items[:limit]:
        lines.append(
            f"  {lab}: obj={obj:.4f} thr={md.get('threshold', 0.5):.2f} "
            f"PR-AUC={md.get('pr_auc', float('nan')):.4f} "
            f"Prec={md.get('precision', 0.0):.4f} Rec={md.get('recall', 0.0):.4f} F1={md.get('f1', 0.0):.4f}"
        )
    return "\n".join(lines)


def format_multilabel_metrics(title: str, m: Dict[str, Any]) -> str:
    lines = [title]
    keys = [
        "num_contracts",
        "contracts_with_any_true_violation",
        "contracts_with_any_pred_violation",
        "precision_micro", "recall_micro", "f1_micro",
        "precision_macro", "recall_macro", "f1_macro",
        "jaccard_row_avg",
        "exact_match_ratio",
    ]
    for k in keys:
        if k in m:
            v = m[k]
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
    if "error" in m:
        lines.append(f"  error: {m['error']}")
    return "\n".join(lines)


def format_retrieval_eval_section(title: str, metrics_list: List[Dict[str, float]]) -> str:
    if not metrics_list:
        return f"{title}\n  (no retrieval GT available)\n"
    # Average each metric key
    keys = sorted({k for m in metrics_list for k in m.keys()})
    lines = [title]
    for k in keys:
        vals = [m.get(k) for m in metrics_list if k in m]
        if vals:
            lines.append(f"  {k}: {float(np.mean(vals)):.4f}")
    lines.append(f"  n={len(metrics_list)}")
    return "\n".join(lines) + "\n"


# =========================
# Evidence ablation (with vs without evidence)
# =========================
def strip_evidence_from_text_aug(text_aug: str) -> str:
    """
    Removes evidence block so we can test model reliance.
    Keeps BM25_SUM + clause text.
    """
    if not text_aug:
        return ""
    # remove everything after "EVIDENCE:" marker
    parts = text_aug.split("EVIDENCE:")
    if len(parts) <= 1:
        return text_aug
    return parts[0].strip() + "\nEVIDENCE:\n\n"


# =========================
# MAIN
# =========================
def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    law_rows = read_jsonl(LAW_JSONL)
    clauses_rows = read_jsonl(CLAUSES_JSONL)
    contracts_rows = read_jsonl(CONTRACTS_JSONL)

    viol_rows: List[dict] = []
    if VIOLATIONS_JSONL.exists():
        viol_rows = read_jsonl(VIOLATIONS_JSONL)

    violation_to_law = read_violation_to_law_map(VIOLATION_TO_LAW_JSONL)

    contracts_by_id: Dict[str, dict] = {}
    for c in contracts_rows:
        cid = extract_contract_id_any(c)
        if cid:
            contracts_by_id[cid] = c

    # Build violation maps (optional but recommended)
    viol_by_clause_id: Dict[str, List[str]] = {}
    viol_by_contract_id: Dict[str, List[str]] = {}
    if viol_rows:
        viol_by_clause_id, viol_by_contract_id = build_violation_maps(viol_rows)

    # ✅ refine clause units BEFORE training examples are built
    clauses_rows, split_stats = apply_clause_unit_refinement(
        clauses_rows=clauses_rows,
        viol_by_clause_id=viol_by_clause_id,
    )

    # Build retrieval runtime
    bm25 = build_bm25_index(law_rows)
    tfidf_vec, X_law = build_tfidf_reranker(bm25.docs)

    # Build examples
    examples = build_clause_examples(
        clauses_rows=clauses_rows,
        contracts_by_id=contracts_by_id,
        bm25=bm25,
        tfidf_vec=tfidf_vec,
        X_law=X_law,
        viol_by_clause_id=viol_by_clause_id,
    )
    if not examples:
        raise RuntimeError("No usable clause examples found. Check clause_text/contract_id fields in dataset_clauses.jsonl.")

    # Determine mode availability
    any_multilabel = any(len(ex.y_labels) > 0 for ex in examples)
    any_binary = any(ex.y_binary is not None for ex in examples)
    if not any_multilabel and not any_binary:
        raise RuntimeError(
            "No labels found.\n"
            "Provide either:\n"
            "  - clause-level multi-label violation IDs (violations/violation_ids/etc.)\n"
            "  - OR label_unlawful=0/1\n"
            "  - OR provide dataset_violations.jsonl with (contract_id, clause_id, violation_id)\n"
        )

    # Split by contract_id
    groups_all = [ex.contract_id for ex in examples]
    idx = np.arange(len(examples))
    groups_arr = np.array(groups_all)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    trainval, test = next(gss1.split(idx, np.zeros((len(idx),)), groups_arr))

    trainval_groups = groups_arr[trainval]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE / (1.0 - TEST_SIZE), random_state=RANDOM_SEED)
    train_rel, val_rel = next(gss2.split(trainval, np.zeros((len(trainval),)), trainval_groups))

    train_idx = trainval[train_rel]
    val_idx = trainval[val_rel]
    test_idx = test

    def pick_examples(ii: np.ndarray) -> List[ClauseExample]:
        return [examples[int(i)] for i in ii]

    ex_train = pick_examples(train_idx)
    ex_val = pick_examples(val_idx)
    ex_test = pick_examples(test_idx)

    # Report header + integrity stats
    header_lines: List[str] = []
    header_lines.append("TRAINING ONLY: LAW-AWARE (RAG-ish) VIOLATION MODEL")
    header_lines.append(f"Law file:       {LAW_JSONL}")
    header_lines.append(f"Clauses file:   {CLAUSES_JSONL}")
    header_lines.append(f"Contracts file: {CONTRACTS_JSONL}")
    header_lines.append(f"Violations file:{VIOLATIONS_JSONL if VIOLATIONS_JSONL.exists() else '(missing)'}")
    header_lines.append(f"Viol->Law map:  {VIOLATION_TO_LAW_JSONL if VIOLATION_TO_LAW_JSONL.exists() else '(missing)'}")
    header_lines.append("")
    header_lines.append("SPLIT (by contract_id):")
    header_lines.append(f"  Train clauses={len(ex_train)} | Val clauses={len(ex_val)} | Test clauses={len(ex_test)}")
    header_lines.append(
        f"  Unique contracts (train/val/test): "
        f"{len(set(e.contract_id for e in ex_train))} / {len(set(e.contract_id for e in ex_val))} / {len(set(e.contract_id for e in ex_test))}"
    )
    header_lines.append("")
    header_lines.append("CONFIG:")
    header_lines.append(f"  Leakage control: tokens [[VIOLATION_x]] removed from inputs.")
    header_lines.append(f"  Retrieval: BM25_TOP_K={BM25_TOP_K}, EVIDENCE_TOP_K={TOP_K_EVIDENCE_FOR_MODEL}")
    header_lines.append(f"  Evidence trunc: per_article={EVIDENCE_MAX_CHARS_PER_ARTICLE}, total={EVIDENCE_MAX_TOTAL_CHARS}, dedup={EVIDENCE_DEDUP}")
    header_lines.append(f"  Retrieval eval K: {RETR_EVAL_KS}")
    header_lines.append(f"  Threshold objective: {THRESH_OBJECTIVE}")
    header_lines.append(f"  C grid: {C_GRID}")
    header_lines.append(f"  GLOBAL_MIN_LABEL_PROB: {GLOBAL_MIN_LABEL_PROB}")
    header_lines.append("")
    header_lines.append("CLAUSE UNIT QA:")
    header_lines.append(f"  ENABLE_CLAUSE_RESPLIT={ENABLE_CLAUSE_RESPLIT}")
    header_lines.append(f"  clauses_in={split_stats.clauses_in} clauses_out={split_stats.clauses_out}")
    header_lines.append(f"  flagged_in={split_stats.flagged_in} resplit_applied={split_stats.resplit_applied}")
    if split_stats.flags:
        for k, v in sorted(split_stats.flags.items(), key=lambda x: (-x[1], x[0])):
            header_lines.append(f"  flag_{k}={v}")
    header_lines.append("")
    header_txt = "\n".join(header_lines) + "\n"

    report_lines: List[str] = [header_txt]

    # =========================
    # MULTI-LABEL MODE (preferred)
    # =========================
    if any_multilabel:
        # label universe: from clause labels + contract labels + violations file contract grouping
        all_labels: Set[str] = set()
        for ex in examples:
            all_labels.update(ex.y_labels)
        for _, crow in contracts_by_id.items():
            all_labels.update(extract_violation_ids_any(crow))
        for _, vids in viol_by_contract_id.items():
            for v in vids:
                all_labels.add(v)

        label_names = sorted(all_labels)
        if not label_names:
            raise RuntimeError("Multi-label mode selected but no violation IDs found anywhere.")

        mlb = MultiLabelBinarizer(classes=label_names)

        # Strict: clause-level evaluation only on clauses that have GT labels
        def filter_labeled(exs: List[ClauseExample]) -> List[ClauseExample]:
            return [e for e in exs if len(e.y_labels) > 0]

        ex_train_lab = filter_labeled(ex_train)
        ex_val_lab = filter_labeled(ex_val)
        ex_test_lab = filter_labeled(ex_test)

        report_lines.append("LABEL COVERAGE (clause-level GT):")
        report_lines.append(f"  Train labeled clauses: {len(ex_train_lab)}/{len(ex_train)}")
        report_lines.append(f"  Val labeled clauses:   {len(ex_val_lab)}/{len(ex_val)}")
        report_lines.append(f"  Test labeled clauses:  {len(ex_test_lab)}/{len(ex_test)}")
        report_lines.append("")

        if not ex_train_lab or not ex_val_lab:
            raise RuntimeError(
                "Not enough labeled clauses for multi-label training.\n"
                "Fix by ensuring either:\n"
                "  - clauses JSONL has violation IDs per clause, OR\n"
                "  - dataset_violations.jsonl exists and maps clause_id -> violation_id.\n"
            )

        X_train = [e.text_aug for e in ex_train_lab]
        X_val = [e.text_aug for e in ex_val_lab]
        X_test = [e.text_aug for e in ex_test_lab]

        Y_train = mlb.fit_transform([e.y_labels for e in ex_train_lab]).astype(int)
        Y_val = mlb.transform([e.y_labels for e in ex_val_lab]).astype(int)
        Y_test = mlb.transform([e.y_labels for e in ex_test_lab]).astype(int)

        # Remove labels with 0 positives in train
        pos_counts = Y_train.sum(axis=0)
        keep = np.where(pos_counts > 0)[0].tolist()
        if len(keep) < Y_train.shape[1]:
            removed = [label_names[j] for j in range(len(label_names)) if j not in keep]
            report_lines.append(f"NOTE: removed {len(removed)} labels with 0 positives in train (cannot learn).")
            report_lines.append("")

        label_names_kept = [label_names[j] for j in keep]
        Y_train_k = Y_train[:, keep]
        Y_val_k = Y_val[:, keep]
        Y_test_k = Y_test[:, keep]

        # Train (WITH evidence)
        model, val_summary = train_multilabel(X_train, Y_train_k, X_val, Y_val_k, label_names_kept)

        # Clause-level eval on labeled test clauses (WITH evidence)
        P_test = predict_multilabel_probs(model.vectorizer, model.classifiers, X_test, model.label_names)
        Yhat_test = probs_to_preds(P_test, model.label_names, model.thresholds, global_min=GLOBAL_MIN_LABEL_PROB)

        clause_test_metrics = multilabel_prf(Y_test_k, Yhat_test)
        clause_test_metrics["jaccard_row_avg"] = jaccard_per_row(Y_test_k, Yhat_test)

        report_lines.append(format_section("VALIDATION (CLAUSE-LEVEL, LABELED ONLY) [WITH EVIDENCE]", [
            f"precision_micro={val_summary['micro']['precision_micro']:.4f}",
            f"recall_micro={val_summary['micro']['recall_micro']:.4f}",
            f"f1_micro={val_summary['micro']['f1_micro']:.4f}",
            f"precision_macro={val_summary['micro']['precision_macro']:.4f}",
            f"recall_macro={val_summary['micro']['recall_macro']:.4f}",
            f"f1_macro={val_summary['micro']['f1_macro']:.4f}",
            f"jaccard_row_avg={val_summary['micro']['jaccard_row_avg']:.4f}",
        ]))

        report_lines.append(format_section("TEST (CLAUSE-LEVEL, LABELED ONLY) [WITH EVIDENCE]", [
            f"precision_micro={clause_test_metrics['precision_micro']:.4f}",
            f"recall_micro={clause_test_metrics['recall_micro']:.4f}",
            f"f1_micro={clause_test_metrics['f1_micro']:.4f}",
            f"precision_macro={clause_test_metrics['precision_macro']:.4f}",
            f"recall_macro={clause_test_metrics['recall_macro']:.4f}",
            f"f1_macro={clause_test_metrics['f1_macro']:.4f}",
            f"jaccard_row_avg={clause_test_metrics['jaccard_row_avg']:.4f}",
        ]))

        report_lines.append(format_per_label_summary(val_summary, limit=30))
        report_lines.append("")

        # Evidence ablation evaluation (WITHOUT evidence) to verify RAG value
        X_val_noev = [strip_evidence_from_text_aug(x) for x in X_val]
        X_test_noev = [strip_evidence_from_text_aug(x) for x in X_test]
        P_val_noev = predict_multilabel_probs(model.vectorizer, model.classifiers, X_val_noev, model.label_names)
        Yhat_val_noev = probs_to_preds(P_val_noev, model.label_names, model.thresholds, global_min=GLOBAL_MIN_LABEL_PROB)
        ab_val = multilabel_prf(Y_val_k, Yhat_val_noev)
        ab_val["jaccard_row_avg"] = jaccard_per_row(Y_val_k, Yhat_val_noev)

        P_test_noev = predict_multilabel_probs(model.vectorizer, model.classifiers, X_test_noev, model.label_names)
        Yhat_test_noev = probs_to_preds(P_test_noev, model.label_names, model.thresholds, global_min=GLOBAL_MIN_LABEL_PROB)
        ab_test = multilabel_prf(Y_test_k, Yhat_test_noev)
        ab_test["jaccard_row_avg"] = jaccard_per_row(Y_test_k, Yhat_test_noev)

        report_lines.append(format_section("VALIDATION (CLAUSE-LEVEL) [WITHOUT EVIDENCE] (ABLATION)", [
            f"precision_micro={ab_val['precision_micro']:.4f}",
            f"recall_micro={ab_val['recall_micro']:.4f}",
            f"f1_micro={ab_val['f1_micro']:.4f}",
            f"precision_macro={ab_val['precision_macro']:.4f}",
            f"recall_macro={ab_val['recall_macro']:.4f}",
            f"f1_macro={ab_val['f1_macro']:.4f}",
            f"jaccard_row_avg={ab_val['jaccard_row_avg']:.4f}",
        ]))

        report_lines.append(format_section("TEST (CLAUSE-LEVEL) [WITHOUT EVIDENCE] (ABLATION)", [
            f"precision_micro={ab_test['precision_micro']:.4f}",
            f"recall_micro={ab_test['recall_micro']:.4f}",
            f"f1_micro={ab_test['f1_micro']:.4f}",
            f"precision_macro={ab_test['precision_macro']:.4f}",
            f"recall_macro={ab_test['recall_macro']:.4f}",
            f"f1_macro={ab_test['f1_macro']:.4f}",
            f"jaccard_row_avg={ab_test['jaccard_row_avg']:.4f}",
        ]))
        report_lines.append("")

        # Retrieval evaluation if violation->law map exists
        retr_metrics_val: List[Dict[str, float]] = []
        for e in ex_val_lab:
            gt_arts: Set[str] = set()
            for vid in e.y_labels:
                gt_arts |= violation_to_law.get(vid, set())
            if not gt_arts:
                continue
            retrieved = [str(x.get("article", "")).strip() for x in e.evidence_struct]
            retr_metrics_val.append(evaluate_retrieval_for_clause(gt_arts, retrieved, RETR_EVAL_KS))

        retr_metrics_test: List[Dict[str, float]] = []
        for e in ex_test_lab:
            gt_arts: Set[str] = set()
            for vid in e.y_labels:
                gt_arts |= violation_to_law.get(vid, set())
            if not gt_arts:
                continue
            retrieved = [str(x.get("article", "")).strip() for x in e.evidence_struct]
            retr_metrics_test.append(evaluate_retrieval_for_clause(gt_arts, retrieved, RETR_EVAL_KS))

        report_lines.append(format_retrieval_eval_section("RETRIEVAL EVAL (VAL) using violation->law GT", retr_metrics_val))
        report_lines.append(format_retrieval_eval_section("RETRIEVAL EVAL (TEST) using violation->law GT", retr_metrics_test))

        # Contract-level evaluation on test contracts:
        X_test_all = [e.text_aug for e in ex_test]
        test_clause_contract_ids = [e.contract_id for e in ex_test]
        test_contracts = sorted(set(test_clause_contract_ids))

        P_test_all = predict_multilabel_probs(model.vectorizer, model.classifiers, X_test_all, model.label_names)
        P_by_contract = aggregate_contract_predictions_max(test_clause_contract_ids, P_test_all)

        truth_by_contract = contract_truth_by_id(
            contracts_by_id=contracts_by_id,
            label_names=model.label_names,
            viol_by_contract_id=viol_by_contract_id,
        )

        truth_test = {cid: truth_by_contract.get(cid, np.zeros((len(model.label_names),), dtype=int)) for cid in test_contracts}
        pred_test = {cid: P_by_contract.get(cid, np.zeros((len(model.label_names),), dtype=float)) for cid in test_contracts}

        contract_metrics = contract_eval_metrics(truth_test, pred_test, model.label_names, model.thresholds)
        report_lines.append(format_multilabel_metrics("TEST (CONTRACT-LEVEL, BY VIOLATION IDs)", contract_metrics))
        report_lines.append("")

        # Stability check
        report_lines.append("GroupKFold micro PR-AUC on TRAIN (stability check):")
        g_train = [e.contract_id for e in ex_train_lab]
        cv_lines = groupkfold_micro_pr_auc_stability(X_train, Y_train_k, g_train)
        report_lines.extend(cv_lines)
        report_lines.append("")

        # ===== SAVE FULL BUNDLE (INFERENCE READY) =====
        llm_payload_spec = {
            "schema_version": "1.0",
            "payload_fields": [
                "contract_id", "clause_id", "clause_text",
                "predicted_violations[{id,prob,threshold}]",
                "law_evidence[{article,score,bm25,cosine,excerpt}]",
                "model_meta",
            ],
            "safety_rules": [
                "LLM must only cite provided law_evidence excerpts.",
                "If evidence is insufficient, LLM must say: 'Insufficient evidence provided.'",
                "LLM must not invent article numbers.",
            ],
            "suggested_prompt_template": (
                "You are a labor-law compliance assistant.\n"
                "Given a clause and retrieved law evidence excerpts, explain why it is a violation (if it is), "
                "cite the provided articles only, and propose a compliant rewrite.\n"
                "Do NOT invent law articles.\n"
                "If evidence is insufficient, say 'Insufficient evidence provided.'\n\n"
                "CLAUSE:\n{clause_text}\n\n"
                "PREDICTED VIOLATIONS:\n{predicted_violations}\n\n"
                "LAW EVIDENCE (EXCERPTS ONLY):\n{law_evidence}\n"
            ),
        }

        bundle = {
            "meta": model.meta,
            "label_names": model.label_names,
            "thresholds": model.thresholds,
            "best_C": model.best_C,
            "vectorizer": model.vectorizer,
            "classifiers": model.classifiers,

            # ✅ retrieval artifacts saved for inference
            "retrieval": {
                "bm25": {
                    "idf": bm25.idf,
                    "avgdl": bm25.avgdl,
                    "k1": bm25.k1,
                    "b": bm25.b,
                    "docs": [
                        {"article": d.article, "heading": d.heading, "text": d.text, "tokens": d.tokens}
                        for d in bm25.docs
                    ],
                },
                "tfidf_reranker": {
                    "vectorizer": tfidf_vec,
                    "X_law": X_law,
                },
                "settings": {
                    "BM25_TOP_K": BM25_TOP_K,
                    "TOP_K_EVIDENCE_FOR_MODEL": TOP_K_EVIDENCE_FOR_MODEL,
                    "MIN_QUERY_TOKENS": MIN_QUERY_TOKENS,
                    "EVIDENCE_MAX_CHARS_PER_ARTICLE": EVIDENCE_MAX_CHARS_PER_ARTICLE,
                    "EVIDENCE_MAX_TOTAL_CHARS": EVIDENCE_MAX_TOTAL_CHARS,
                    "EVIDENCE_DEDUP": EVIDENCE_DEDUP,
                },
                "law_source": {
                    "law_jsonl": str(LAW_JSONL),
                },
            },

            # ✅ LLM payload spec (for next stage)
            "llm_payload_spec": llm_payload_spec,
        }

        joblib.dump(bundle, MODEL_BUNDLE_PATH)
        report_lines.append(f"Saved FULL inference bundle: {MODEL_BUNDLE_PATH}")

    # =========================
    # BINARY FALLBACK MODE
    # =========================
    else:
        report_lines.append("NOTE: Only binary unlawful labels found. Violation-ID matching is NOT possible.")
        report_lines.append("")

        ex_train_lab = [e for e in ex_train if e.y_binary is not None]
        ex_val_lab = [e for e in ex_val if e.y_binary is not None]
        ex_test_lab = [e for e in ex_test if e.y_binary is not None]
        if not ex_train_lab or not ex_val_lab or not ex_test_lab:
            raise RuntimeError("Binary mode: not enough labeled clauses across splits.")

        X_train = [e.text_aug for e in ex_train_lab]
        y_train = np.array([int(e.y_binary) for e in ex_train_lab], dtype=int)
        X_val = [e.text_aug for e in ex_val_lab]
        y_val = np.array([int(e.y_binary) for e in ex_val_lab], dtype=int)
        X_test = [e.text_aug for e in ex_test_lab]
        y_test = np.array([int(e.y_binary) for e in ex_test_lab], dtype=int)

        vec = make_vectorizer()
        Xtr = vec.fit_transform(X_train)
        Xva = vec.transform(X_val)

        best = None
        for C in C_GRID:
            base = LogisticRegression(
                C=C,
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            )

            clf = make_calibrator(base, method=CALIBRATION_METHOD, y_train=y_train)
            clf.fit(Xtr, y_train)

            p_val = clf.predict_proba(Xva)[:, 1]
            thr, score = pick_threshold_binary(y_val, p_val, THRESH_OBJECTIVE)
            m = compute_binary_metrics(y_val, p_val, thr)
            m["val_objective"] = float(score)
            m["C"] = float(C)

            if best is None or score > best[3]:
                best = (clf, float(C), float(thr), float(score), m)

        assert best is not None
        clf_best, C_best, thr_best, _, m_val = best

        Xte = vec.transform(X_test)
        p_test = clf_best.predict_proba(Xte)[:, 1]
        m_test = compute_binary_metrics(y_test, p_test, thr_best)
        m_test["C"] = float(C_best)

        report_lines.append("BINARY UNLAWFUL MODEL (clause-level)")
        report_lines.append(f"  Best C = {C_best:.3f}")
        report_lines.append(f"  Best threshold = {thr_best:.2f}")
        report_lines.append("")
        report_lines.append("VALIDATION METRICS:\n" + m_val["classification_report"])
        report_lines.append("TEST METRICS:\n" + m_test["classification_report"])
        report_lines.append("")

        llm_payload_spec = {
            "schema_version": "1.0",
            "payload_fields": [
                "contract_id", "clause_id", "clause_text",
                "unlawful_prob", "unlawful_threshold",
                "law_evidence[{article,score,bm25,cosine,excerpt}]",
            ],
        }

        bundle = {
            "meta": {
                "model_name": "law_aware_binary_logreg_calibrated",
                "threshold": float(thr_best),
                "best_C": float(C_best),
                "top_k_evidence": int(TOP_K_EVIDENCE_FOR_MODEL),
                "bm25_top_k": int(BM25_TOP_K),
                "random_seed": int(RANDOM_SEED),
                "calibration": CALIBRATION_METHOD,
                "clause_unit_refinement": True,
                "enable_clause_resplit": bool(ENABLE_CLAUSE_RESPLIT),
                "evidence_max_chars_per_article": int(EVIDENCE_MAX_CHARS_PER_ARTICLE),
                "evidence_max_total_chars": int(EVIDENCE_MAX_TOTAL_CHARS),
                "evidence_dedup": bool(EVIDENCE_DEDUP),
            },
            "vectorizer": vec,
            "classifier": clf_best,

            # save retrieval artifacts even in binary mode (still useful)
            "retrieval": {
                "bm25": {
                    "idf": bm25.idf,
                    "avgdl": bm25.avgdl,
                    "k1": bm25.k1,
                    "b": bm25.b,
                    "docs": [
                        {"article": d.article, "heading": d.heading, "text": d.text, "tokens": d.tokens}
                        for d in bm25.docs
                    ],
                },
                "tfidf_reranker": {
                    "vectorizer": tfidf_vec,
                    "X_law": X_law,
                },
                "settings": {
                    "BM25_TOP_K": BM25_TOP_K,
                    "TOP_K_EVIDENCE_FOR_MODEL": TOP_K_EVIDENCE_FOR_MODEL,
                    "MIN_QUERY_TOKENS": MIN_QUERY_TOKENS,
                    "EVIDENCE_MAX_CHARS_PER_ARTICLE": EVIDENCE_MAX_CHARS_PER_ARTICLE,
                    "EVIDENCE_MAX_TOTAL_CHARS": EVIDENCE_MAX_TOTAL_CHARS,
                    "EVIDENCE_DEDUP": EVIDENCE_DEDUP,
                },
                "law_source": {
                    "law_jsonl": str(LAW_JSONL),
                },
            },

            "llm_payload_spec": llm_payload_spec,
        }
        joblib.dump(bundle, MODEL_BUNDLE_PATH)
        report_lines.append(f"Saved FULL inference bundle: {MODEL_BUNDLE_PATH}")

    # Write report
    write_text(MODEL_REPORT_TXT, "\n".join(report_lines))
    print(f"Saved report: {MODEL_REPORT_TXT}")
    print(f"Saved model:  {MODEL_BUNDLE_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
