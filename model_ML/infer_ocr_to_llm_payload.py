from __future__ import annotations

import json
import math
import re
import unicodedata
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

# ============================================================
# CONFIG
# ============================================================
MODEL_BUNDLE_PATH = Path("models/law_aware_multilabel_model.joblib")

# If your training bundle does NOT include retrieval artifacts, we can rebuild them from LAW_JSONL at inference time.
LAW_JSONL = Path(r"laws\preprocessed\labor14_2025_clean.jsonl")

# Input OCR text file (full contract OCR output)
DEFAULT_OCR_TXT = Path("cases/processed/ocr_output.txt")

# Output: payloads for LLM (JSONL)
OUT_DIR = Path("cases/processed")
DEFAULT_PAYLOAD_JSONL = OUT_DIR / "llm_payload.jsonl"

# Output: ONLY violations (and nothing else)
DEFAULT_VIOLATIONS_TXT = OUT_DIR / "violations_only.txt"

# Optional: store one “prompt per clause”
DEFAULT_PROMPTS_DIR = OUT_DIR / "llm_prompts"

# If you want to actually run a local LLM (Ollama), set RUN_LLM=True
RUN_LLM = False
OLLAMA_MODEL = "qwen2.5:7b-instruct"
LLM_MAX_CHARS = 6000

# Retrieval settings for inference
BM25_TOP_K = 30
TOP_K_EVIDENCE = 6
MIN_QUERY_TOKENS = 3

# Evidence truncation (stable prompt sizes)
EVIDENCE_MAX_CHARS_PER_ARTICLE = 900
EVIDENCE_MAX_TOTAL_CHARS = 3800
EVIDENCE_DEDUP = True

# Probability floors
GLOBAL_MIN_LABEL_PROB = 0.0
HARD_MIN_PROB = 0.15


# ============================================================
# Normalization helpers (Arabic-friendly)
# ============================================================
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


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


# ============================================================
# Clause segmentation (OCR txt -> clauses)
# ============================================================
HEADING_RE = re.compile(
    r"(?m)^\s*(?:البند|بند|المادة|مادة)\s*\(?\s*[\d٠-٩]{1,4}\s*\)?\s*[:：]?\s*$"
)
NUM_ITEM_RE = re.compile(r"(?m)^\s*(?:\d{1,3}[\)\.\-]|[أ-ي]\s*[\)\-]|[-•●])\s+")
EMPTY_LINE_RE = re.compile(r"\n{2,}")


@dataclass
class Clause:
    clause_id: str
    text: str


def segment_ocr_text_to_clauses(ocr_text: str) -> List[Clause]:
    t = normalize_ar(strip_leakage_tokens(ocr_text))
    if not t:
        return []

    if HEADING_RE.search(t):
        chunks = re.split(r"(?m)(?=^\s*(?:البند|بند|المادة|مادة)\s*\(?\s*[\d٠-٩]{1,4}\s*\)?)", t)
        chunks = [c.strip() for c in chunks if c.strip()]
    else:
        chunks = [t]

    clauses: List[Clause] = []
    cidx = 0

    for chunk in chunks:
        markers = len(NUM_ITEM_RE.findall(chunk))
        if markers >= 3:
            parts = re.split(r"(?m)(?=^\s*(?:\d{1,3}[\)\.\-]|[أ-ي]\s*[\)\-]|[-•●])\s+)", chunk)
            parts = [p.strip() for p in parts if p.strip()]
        else:
            parts = [chunk.strip()]

        final_parts: List[str] = []
        for p in parts:
            sub = [x.strip() for x in EMPTY_LINE_RE.split(p) if x.strip()]
            final_parts.extend(sub if sub else [p])

        for p in final_parts:
            p = p.strip()
            if len(p) < 40:
                continue
            cidx += 1
            clauses.append(Clause(clause_id=f"ocr_clause_{cidx:04d}", text=p))

    return clauses


# ============================================================
# Retrieval (BM25 + TFIDF rerank)
# ============================================================
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


def build_retrieval_from_law_jsonl(law_jsonl: Path) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    law_rows = read_jsonl(law_jsonl)

    docs: List[dict] = []
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

    corpus = [normalize_for_tokens((d.get("heading", "") + " " + d.get("text", "")).strip()) for d in docs]
    tfidf_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_df=0.98, sublinear_tf=True)
    X_law = tfidf_vec.fit_transform(corpus)

    return {
        "bm25": {"docs": docs, "idf": idf, "avgdl": float(avgdl), "k1": 1.5, "b": 0.75},
        "tfidf_reranker": {"vectorizer": tfidf_vec, "X_law": X_law},
    }


def bm25_score(idf: Dict[str, float], avgdl: float, k1: float, b: float, q: List[str], doc_tokens: List[str]) -> float:
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


def retrieve_law_evidence(
    clause_text: str,
    bm25_pack: dict,
    tfidf_vec,
    X_law,
    top_k_evidence: int = TOP_K_EVIDENCE,
    bm25_top_k: int = BM25_TOP_K,
) -> Tuple[float, List[dict], str, List[str]]:
    q = tokenize_simple(clause_text)
    if len(q) < MIN_QUERY_TOKENS:
        return 0.0, [], "", []

    docs = bm25_pack["docs"]
    idf = bm25_pack["idf"]
    avgdl = float(bm25_pack["avgdl"])
    k1 = float(bm25_pack.get("k1", 1.5))
    b = float(bm25_pack.get("b", 0.75))

    scored: List[Tuple[int, float]] = []
    for i, d in enumerate(docs):
        s = bm25_score(idf, avgdl, k1, b, q, d.get("tokens", []))
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: max(bm25_top_k, top_k_evidence)]
    if not scored:
        return 0.0, [], "", []

    idxs = [i for (i, _) in scored]
    bm25_scores = np.array([s for (_, s) in scored], dtype=float)

    clause_norm = normalize_for_tokens(clause_text)
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

    reranked = []
    for j, law_i in enumerate(idxs):
        d = docs[law_i]
        reranked.append(
            {
                "article": str(d.get("article", "")).strip(),
                "heading": str(d.get("heading", "") or ""),
                "text": str(d.get("text", "") or ""),
                "bm25": float(bm25_scores[j]),
                "cosine": float(cos[j]),
                "score": float(final[j]),
            }
        )
    reranked.sort(key=lambda r: r["score"], reverse=True)
    top = reranked[:top_k_evidence]

    if EVIDENCE_DEDUP:
        seen = set()
        deduped = []
        for x in top:
            a = x.get("article", "")
            if not a or a in seen:
                continue
            seen.add(a)
            deduped.append(x)
        top = deduped

    bm25_sum = float(sum(x.get("bm25", 0.0) for x in top))

    total_chars = 0
    lines: List[str] = []
    ev_struct: List[dict] = []
    law_citations: List[str] = []

    for x in top:
        art = str(x.get("article", "")).strip()
        txt = _truncate(str(x.get("text", "") or ""), EVIDENCE_MAX_CHARS_PER_ARTICLE)
        line = f"المادة {art}: {txt}"
        if total_chars + len(line) > EVIDENCE_MAX_TOTAL_CHARS:
            break
        total_chars += len(line)
        lines.append(line)
        law_citations.append(art)
        ev_struct.append(
            {
                "article": art,
                "heading": str(x.get("heading", "") or ""),
                "bm25": float(x.get("bm25", 0.0)),
                "cosine": float(x.get("cosine", 0.0)),
                "score": float(x.get("score", 0.0)),
                "excerpt": txt,
            }
        )

    evidence_text = "\n\n".join(lines) if lines else ""
    return bm25_sum, ev_struct, evidence_text, law_citations


# ============================================================
# Model inference
# ============================================================
def build_model_input_text(bm25_sum: float, clause_text: str, evidence_text: str) -> str:
    return (
        f"BM25_SUM={bm25_sum:.4f}\n"
        f"CONTRACT_META:\n\n"
        f"CLAUSE:\n{clause_text}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )


def run_multilabel_inference(bundle: dict, texts: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    vec = bundle["vectorizer"]
    classifiers = bundle["classifiers"]
    label_names = list(bundle["label_names"])
    thresholds = dict(bundle.get("thresholds", {}))

    Xv = vec.transform(texts)
    P = np.zeros((len(texts), len(label_names)), dtype=float)
    for j, lab in enumerate(label_names):
        clf = classifiers.get(lab)
        if clf is None:
            continue
        P[:, j] = clf.predict_proba(Xv)[:, 1]
    return P, label_names, thresholds


def run_binary_inference(bundle: dict, texts: List[str]) -> Tuple[np.ndarray, float]:
    vec = bundle["vectorizer"]
    clf = bundle["classifier"]
    thr = float(bundle.get("meta", {}).get("threshold", 0.5))
    Xv = vec.transform(texts)
    p = clf.predict_proba(Xv)[:, 1]
    return p, thr


# ============================================================
# LLM prompt + optional Ollama runner
# ============================================================
def make_llm_prompt(payload: dict, prompt_template: Optional[str]) -> str:
    clause_text = payload.get("clause_text", "")
    violations = payload.get("violations", []) or []
    law_evidence = payload.get("law_evidence", []) or []

    pred_lines = []
    for x in violations:
        pred_lines.append(f"- {x['id']}: prob={x['prob']:.4f} thr={x['threshold']:.2f}")
    pred_str = "\n".join(pred_lines) if pred_lines else "(none)"

    ev_lines = []
    for e in law_evidence:
        ev_lines.append(f"- المادة {e.get('article','')}: {e.get('excerpt','')}")
    ev_str = "\n".join(ev_lines) if ev_lines else "(no evidence)"

    if prompt_template:
        prompt = prompt_template.format(
            clause_text=clause_text,
            predicted_violations=pred_str,
            law_evidence=ev_str,
        )
    else:
        prompt = (
            "You are a labor-law compliance assistant.\n"
            "Given a clause and retrieved law evidence excerpts, explain why it is a violation (if it is), "
            "cite the provided articles only, and propose a compliant rewrite.\n"
            "Do NOT invent law articles.\n"
            "If evidence is insufficient, say 'Insufficient evidence provided.'\n\n"
            f"CLAUSE:\n{clause_text}\n\n"
            f"PREDICTED VIOLATIONS:\n{pred_str}\n\n"
            f"LAW EVIDENCE (EXCERPTS ONLY):\n{ev_str}\n"
        )

    return _truncate(prompt, LLM_MAX_CHARS)


def try_run_ollama(prompt: str, model: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode != 0:
            return None
        return p.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


# ============================================================
# MAIN PIPELINE
# - violations_only.txt: ONLY violations lines (nothing else)
# - llm_payload.jsonl: clause payloads
# - llm_prompts/: prompts
# ============================================================
def infer_ocr_file_to_outputs(
    ocr_txt_path: Path,
    payload_out_jsonl: Path,
    violations_only_txt: Path,
    run_llm: bool = RUN_LLM,
    ollama_model: str = OLLAMA_MODEL,
) -> None:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}")

    text = ocr_txt_path.read_text(encoding="utf-8", errors="replace")
    clauses = segment_ocr_text_to_clauses(text)
    if not clauses:
        raise RuntimeError("No clauses detected from OCR text. Check OCR output format.")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if not isinstance(bundle, dict):
        raise RuntimeError("Model bundle is not a dict. Re-check how you saved it in training_only.py")

    llm_template = ((bundle.get("llm_payload_spec", {}) or {}).get("suggested_prompt_template"))

    retr = bundle.get("retrieval", {}) or {}
    bm25_pack = (retr.get("bm25", {}) or {})
    tfidf_pack = (retr.get("tfidf_reranker", {}) or {})
    tfidf_vec = tfidf_pack.get("vectorizer")
    X_law = tfidf_pack.get("X_law")

    if not bm25_pack or tfidf_vec is None or X_law is None:
        if not LAW_JSONL.exists():
            raise RuntimeError(
                "Bundle is missing retrieval artifacts AND LAW_JSONL not found.\n"
                "Fix: save bundle['retrieval'] in training OR ensure LAW_JSONL exists."
            )
        rebuilt = build_retrieval_from_law_jsonl(LAW_JSONL)
        bm25_pack = rebuilt["bm25"]
        tfidf_vec = rebuilt["tfidf_reranker"]["vectorizer"]
        X_law = rebuilt["tfidf_reranker"]["X_law"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload_out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    violations_only_txt.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    contract_id = str(ocr_txt_path.stem)

    is_multilabel = ("label_names" in bundle and "classifiers" in bundle and "thresholds" in bundle)

    clause_payloads: List[dict] = []
    best_prob_by_violation: Dict[str, float] = {}

    for c in clauses:
        bm25_sum, ev_struct, evidence_text, law_citations = retrieve_law_evidence(
            clause_text=c.text,
            bm25_pack=bm25_pack,
            tfidf_vec=tfidf_vec,
            X_law=X_law,
            top_k_evidence=TOP_K_EVIDENCE,
            bm25_top_k=BM25_TOP_K,
        )

        model_input = build_model_input_text(bm25_sum, c.text, evidence_text)

        violations: List[dict] = []

        if is_multilabel:
            P, label_names, thresholds = run_multilabel_inference(bundle, [model_input])
            probs = P[0]
            for j, lab in enumerate(label_names):
                thr = float(thresholds.get(lab, 0.5))
                thr = max(thr, float(GLOBAL_MIN_LABEL_PROB))
                pr = float(probs[j])
                if pr >= max(thr, HARD_MIN_PROB):
                    violations.append({"id": lab, "prob": pr, "threshold": thr})
        else:
            p, thr = run_binary_inference(bundle, [model_input])
            prob = float(p[0])
            if prob >= max(thr, HARD_MIN_PROB):
                violations.append({"id": "UNLAWFUL", "prob": prob, "threshold": float(thr)})

        violations.sort(key=lambda x: x["prob"], reverse=True)

        if not violations:
            continue

        for v in violations:
            vid = str(v["id"])
            pr = float(v["prob"])
            if vid not in best_prob_by_violation or pr > best_prob_by_violation[vid]:
                best_prob_by_violation[vid] = pr

        payload = {
            "contract_id": contract_id,
            "clause_id": c.clause_id,
            "clause_text": c.text,
            "violations": violations,
            "law_citations": law_citations,
            "law_evidence": ev_struct,
            "law_evidence_text": evidence_text,
            "bm25_sum": float(bm25_sum),
            "model_meta": bundle.get("meta", {}),
        }
        clause_payloads.append(payload)

    # 1) violations_only.txt -> ONLY violations lines
    # Format: "<VIOLATION_ID>\t<max_prob>"
    contract_viols_sorted = sorted(best_prob_by_violation.items(), key=lambda x: x[1], reverse=True)
    with violations_only_txt.open("w", encoding="utf-8") as ftxt:
        for vid, pr in contract_viols_sorted:
            ftxt.write(f"{vid}\t{pr:.3f}\n")

    # 2) JSONL payloads (one per clause with violations)
    with payload_out_jsonl.open("w", encoding="utf-8") as fjsonl:
        for p in clause_payloads:
            fjsonl.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 3) prompt files (+ optional run Ollama)
    saved_prompts = 0
    saved_llm = 0

    for payload in clause_payloads:
        prompt = make_llm_prompt(payload, llm_template)
        prompt_path = DEFAULT_PROMPTS_DIR / f"{contract_id}__{payload['clause_id']}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        saved_prompts += 1

        if run_llm:
            ans = try_run_ollama(prompt, ollama_model)
            if ans:
                resp_path = DEFAULT_PROMPTS_DIR / f"{contract_id}__{payload['clause_id']}.llm.txt"
                resp_path.write_text(ans, encoding="utf-8")
                saved_llm += 1

    print(f"✅ Saved violations TXT: {violations_only_txt}")
    print(f"✅ Saved payload JSONL:  {payload_out_jsonl} (clauses={len(clause_payloads)})")
    print(f"✅ Saved prompts:        {DEFAULT_PROMPTS_DIR} (files={saved_prompts})")
    if run_llm:
        print(f"✅ LLM run enabled: saved .llm.txt outputs={saved_llm}")
    else:
        print("ℹ️ LLM run disabled.")


# ============================================================
# CLI
# ============================================================
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=str, default=str(DEFAULT_OCR_TXT), help="Path to OCR output .txt file")
    ap.add_argument("--out", type=str, default=str(DEFAULT_PAYLOAD_JSONL), help="Output JSONL payload path")
    ap.add_argument("--viol", type=str, default=str(DEFAULT_VIOLATIONS_TXT), help="Output TXT with ONLY violations")
    ap.add_argument("--run-llm", action="store_true", help="Actually run Ollama if installed")
    ap.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    args = ap.parse_args()

    infer_ocr_file_to_outputs(
        ocr_txt_path=Path(args.ocr),
        payload_out_jsonl=Path(args.out),
        violations_only_txt=Path(args.viol),
        run_llm=bool(args.run_llm),
        ollama_model=str(args.ollama_model),
    )


if __name__ == "__main__":
    main()
