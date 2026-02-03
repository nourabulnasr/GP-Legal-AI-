from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
from docx import Document

# ✅ reuse existing deterministic components (LOCKED) via import
from app.main import _normalize_contract_text  # same normalization you already use
from app.rules import RuleEngine

BASE_DIR = Path(__file__).resolve().parents[2]          # project root
DATA_DIR = BASE_DIR / "data" / "contracts_raw"
OUT_PATH = BASE_DIR / "data" / "dataset" / "silver_dataset.jsonl"

RULES_DIR = BASE_DIR / "rules"
LAWS_DIR  = BASE_DIR / "laws"

PDF_DIR  = DATA_DIR / "pdf"
DOCX_DIR = DATA_DIR / "docx"


# ---------------------------
# 1) Text extraction (simple, deterministic)
# ---------------------------
def extract_pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    return "\n".join(parts)

def extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


# ---------------------------
# 2) Clause chunking (Clause → Sentences evidence later)
# ---------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?]|[؛]|:)\s+|\n+")

def chunk_clauses(text: str, min_len: int = 40, max_len: int = 900, win: int = 3) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # split into sentence-like units (fallback if punctuation is scarce)
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return []

    # sliding windows of sentences
    clauses: List[str] = []
    step = max(1, win // 2)  # overlap to improve recall
    for i in range(0, len(sents), step):
        chunk = " ".join(sents[i:i+win]).strip()
        if min_len <= len(chunk) <= max_len:
            clauses.append(chunk)

    # also keep any long sentence blocks by character slicing (last resort)
    if len(clauses) < 50 and len(text) > 2000:
        for j in range(0, len(text), 600):
            chunk = text[j:j+900].strip()
            if min_len <= len(chunk) <= max_len:
                clauses.append(chunk)

    # dedupe
    seen = set()
    out = []
    for c in clauses:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
# ---------------------------
# 3) Silver labeling via RuleEngine (LOCKED)
# ---------------------------
def label_clause(engine: RuleEngine, clause: str) -> Dict[str, Any]:
    hits = engine.check_text(clause)  # ✅ correct method name in your RuleEngine
    rule_ids = [h.get("rule_id") or h.get("id") for h in hits if isinstance(h, dict)]
    rule_ids = [r for r in rule_ids if r]
    return {"labels": sorted(list(set(rule_ids))), "hits": hits}


def iter_contract_files() -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for p in sorted(PDF_DIR.glob("*.pdf")):
        items.append((p.stem, p))
    for p in sorted(DOCX_DIR.glob("*.docx")):
        items.append((p.stem, p))
    return items


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    engine = RuleEngine(rules_dir=RULES_DIR, laws_dir=LAWS_DIR)

    files = iter_contract_files()
    if not files:
        raise SystemExit("No files found under data/contracts_raw/pdf or docx")

    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for contract_id, path in files:
            if path.suffix.lower() == ".pdf":
                raw_text = extract_pdf_text(path)
            else:
                raw_text = extract_docx_text(path)

            norm = _normalize_contract_text(raw_text)
            clauses = chunk_clauses(norm)

            for i, clause in enumerate(clauses):
                lab = label_clause(engine, clause)
                rec = {
                    "contract_id": contract_id,
                    "chunk_id": f"{contract_id}__{i:04d}",
                    "text": clause,
                    "labels": lab["labels"],       # ✅ multi-label aligned with rule_ids
                    "source": "rule_engine_silver",
                    "file_name": path.name,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"✅ Wrote {n_written} samples to: {OUT_PATH}")


if __name__ == "__main__":
    main()
