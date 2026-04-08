# -*- coding: utf-8 -*-
"""
BM25-style retriever over law article chunks for law-aware ML training.

Used to retrieve top-K relevant labor-law passages per clause so the model
can see both contract language and governing law (RAG-style input).
No external dependency: uses math + collections only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Tokenize Arabic + English (keep words >= 2 chars)
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    t = (text or "").strip().lower()
    tokens = re.findall(r"[a-z0-9\u0600-\u06ff]+", t)
    return [w for w in tokens if len(w) >= 2]


class BM25LawRetriever:
    """
    BM25 over law chunks. Build index from list of docs {id, text, ...};
    search(query, top_k) returns top_k docs by BM25 score.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0

    def build_index(self, docs: List[Dict[str, Any]]) -> None:
        """docs: list of {id, text, ...} or {page_content, metadata, ...}."""
        self._docs = []
        self._doc_tokens = []
        for d in docs or []:
            text = (d.get("text") or d.get("page_content") or "").strip()
            if not text:
                continue
            self._docs.append(d)
            toks = _tokenize(text)
            self._doc_tokens.append(toks)

        n = len(self._doc_tokens)
        if n == 0:
            self._doc_len = []
            self._avgdl = 0.0
            self._idf = {}
            return

        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / n

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df: Dict[str, int] = {}
        for toks in self._doc_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self._idf = {
            w: math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
            for w in df
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return top_k docs with BM25 score and text."""
        if not self._docs or not query:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            doc_len = self._doc_len[idx]
            score = 0.0
            for w in q_tokens:
                if w not in self._idf:
                    continue
                tf = doc_tokens.count(w)
                if tf == 0:
                    continue
                idf = self._idf[w]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1e-9))
                score += idf * num / den
            if score >= min_score:
                scores.append((score, idx))

        scores.sort(key=lambda x: -x[0])
        out: List[Dict[str, Any]] = []
        for score, idx in scores[: top_k]:
            d = self._docs[idx]
            text = d.get("text") or d.get("page_content") or ""
            meta = d.get("metadata") or {k: d.get(k) for k in ("law", "article", "source", "id") if d.get(k)}
            out.append({
                "score": float(score),
                "text": text,
                "metadata": meta,
            })
        return out


def load_law_docs_for_bm25(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Load law chunks from chunks_dir/*.jsonl into list of {id, text, metadata}."""
    docs: List[Dict[str, Any]] = []
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        return docs
    for fp in sorted(chunks_dir.glob("*.jsonl")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    text = (obj.get("normalized_text") or obj.get("text") or "").strip()
                    if len(text) < 20:
                        continue
                    docs.append({
                        "id": obj.get("id"),
                        "text": text,
                        "metadata": {
                            "law": obj.get("law"),
                            "article": obj.get("article"),
                            "source": obj.get("source") or fp.stem,
                        },
                    })
        except Exception:
            continue
    return docs
