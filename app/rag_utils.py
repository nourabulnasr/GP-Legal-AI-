# app/rag_utils.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

# Try multilingual embedding model (Arabic + English)
_USE_EMBEDDING = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    _USE_EMBEDDING = True
except Exception:
    SentenceTransformer = None  # type: ignore
    faiss = None  # type: ignore


class Retriever:
    """
    Simple local retriever (no external DB).
    - Builds an embedding matrix for docs using a lightweight hashing vector (TF-ish).
    - Searches by cosine similarity.

    ✅ Enhancements:
    - Optional metadata filters: search(..., filters={...})
    - Optional min_score threshold: search(..., min_score=0.0)
    - Backward compatible with old calls: search(query, top_k)
    """

    def __init__(self, dim: int = 2048):
        self.dim = int(dim)
        self._docs: List[Dict[str, Any]] = []
        self._mat: Optional[np.ndarray] = None  # (N, dim)
        self._norms: Optional[np.ndarray] = None  # (N,)

    # ---------------------------
    # Tokenization (Arabic + English friendly)
    # ---------------------------
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        t = text.lower()
        out: List[str] = []
        buf: List[str] = []
        for ch in t:
            if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ("\u0600" <= ch <= "\u06FF"):
                buf.append(ch)
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
        if buf:
            out.append("".join(buf))
        return [x for x in out if len(x) >= 2]

    def _hash_idx(self, token: str) -> int:
        h = 2166136261
        for c in token:
            h ^= ord(c)
            h = (h * 16777619) & 0xFFFFFFFF
        return h % self.dim

    def _text_to_vec(self, text: str) -> np.ndarray:
        vec = np.zeros((self.dim,), dtype=np.float32)
        toks = self._tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            vec[self._hash_idx(tok)] += 1.0
        vec = np.log1p(vec)
        return vec

    # ---------------------------
    # Index building
    # ---------------------------
    def build_index(self, docs: List[Dict[str, Any]]) -> None:
        self._docs = docs or []
        if not self._docs:
            self._mat = None
            self._norms = None
            return

        mat = []
        for d in self._docs:
            text = (d.get("page_content") or d.get("text") or "")
            mat.append(self._text_to_vec(text))
        self._mat = np.vstack(mat)
        self._norms = np.linalg.norm(self._mat, axis=1) + 1e-8

    # ---------------------------
    # Filtering helpers
    # ---------------------------
    def _doc_metadata(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return d.get("metadata") or {
            k: d.get(k)
            for k in ["law", "article", "title", "source", "page", "id"]
            if d.get(k) is not None
        }

    def _match_filters(self, md: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if v is None:
                continue
            if md.get(k) != v:
                return False
        return True

    # ---------------------------
    # Search
    # ---------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if not query or self._mat is None or self._norms is None:
            return []

        qv = self._text_to_vec(query)
        qn = float(np.linalg.norm(qv) + 1e-8)

        sims = (self._mat @ qv) / (self._norms * qn)
        if sims.size == 0:
            return []

        # We'll scan in sorted order and apply filters/min_score until we collect top_k
        order = np.argsort(-sims)
        top_k = max(1, int(top_k))
        min_score = float(min_score or 0.0)

        results: List[Dict[str, Any]] = []
        for idx in order:
            s = float(sims[int(idx)])
            if s < min_score:
                # since sorted descending, we can break early
                break

            d = self._docs[int(idx)]
            md = self._doc_metadata(d)
            if not self._match_filters(md, filters):
                continue

            results.append(
                {
                    "score": s,
                    "text": d.get("page_content") or d.get("text") or "",
                    "metadata": md,
                }
            )
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Embedding-based retriever (upgrade when sentence-transformers + faiss available)
# Same API: build_index(docs), search(query, top_k, filters, min_score)
# ---------------------------------------------------------------------------
if _USE_EMBEDDING and SentenceTransformer is not None and faiss is not None:

    class EmbeddingRetriever(Retriever):
        """
        Multilingual embedding retriever (sentence-transformers + FAISS).
        Uses paraphrase-multilingual-MiniLM-L12-v2 for Arabic/English.
        """

        def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
            super().__init__(dim=384)
            self._model = SentenceTransformer(model_name)
            self._index: Optional[Any] = None

        def build_index(self, docs: List[Dict[str, Any]]) -> None:
            self._docs = docs or []
            if not self._docs:
                self._index = None
                return
            texts = [d.get("page_content") or d.get("text") or "" for d in self._docs]
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)

        def search(
            self,
            query: str,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None,
            min_score: float = 0.0,
        ) -> List[Dict[str, Any]]:
            if not query or self._index is None or not self._docs:
                return []
            qv = self._model.encode([query], convert_to_numpy=True)
            qv = qv.astype(np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-8)
            top_k = max(1, int(top_k))
            scores, indices = self._index.search(qv, min(top_k * 3, len(self._docs)))
            results: List[Dict[str, Any]] = []
            for s, idx in zip(scores[0], indices[0]):
                if idx < 0 or s < min_score:
                    continue
                d = self._docs[int(idx)]
                md = self._doc_metadata(d)
                if not self._match_filters(md, filters):
                    continue
                results.append({
                    "score": float(s),
                    "text": d.get("page_content") or d.get("text") or "",
                    "metadata": md,
                })
                if len(results) >= top_k:
                    break
            return results

    Retriever = EmbeddingRetriever  # type: ignore
