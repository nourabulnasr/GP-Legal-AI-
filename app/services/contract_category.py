from __future__ import annotations

import json
import re
from typing import Any, Optional

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "doctor",
        "physician",
        "medical",
        "clinic",
        "hospital",
        "healthcare",
        "patient",
        "surgery",
        "طبيب",
        "طب",
        "مستشفى",
        "عيادة",
        "طبي",
    ],
    "employment": [
        "employment",
        "employee",
        "employer",
        "salary",
        "wages",
        "probation",
        "عقد عمل",
        "موظف",
        "employ",
        "termination",
    ],
    "rental": [
        "lease",
        "rent",
        "landlord",
        "tenant",
        "إيجار",
        "مؤجر",
        "tenant",
    ],
    "nda": [
        "non-disclosure",
        "confidential",
        "nda",
        "سرية",
        "non disclosure",
    ],
    "freelance": [
        "freelance",
        "contractor",
        "independent contractor",
        "consultancy",
        "مستقل",
    ],
}

_CATEGORY_LABELS: dict[str, str] = {
    "medical": "Medical / Doctor contracts",
    "employment": "Employment contracts",
    "rental": "Rental / Lease contracts",
    "nda": "NDA / Confidentiality",
    "freelance": "Freelance / Contractor",
    "general": "General contracts",
}


def detect_contract_category(text: str = "", filename: str = "") -> str:
    """Classify a contract into a discussion category (e.g. medical, employment)."""
    blob = f"{filename or ''} {text or ''}".lower()
    if not blob.strip():
        return "general"
    best = "general"
    best_score = 0
    for cat, words in _CATEGORY_KEYWORDS.items():
        score = sum(1 for w in words if w.lower() in blob or re.search(re.escape(w.lower()), blob))
        if score > best_score:
            best_score = score
            best = cat
    return best if best_score > 0 else "general"


def category_label(category: str) -> str:
    return _CATEGORY_LABELS.get(category or "general", category or "General")


def extract_category_from_analysis(result_json: str, filename: str = "") -> str:
    """Read stored analysis JSON and infer contract category."""
    text_parts: list[str] = [filename or ""]
    try:
        data = json.loads(result_json or "{}")
    except Exception:
        data = {}
    if isinstance(data, dict):
        text_parts.append(str(data.get("contract_type") or ""))
        for key in ("ocr_chunks", "clauses"):
            items = data.get(key)
            if isinstance(items, list):
                for item in items[:40]:
                    if isinstance(item, dict):
                        for field in ("text", "normalized_text", "content", "clause_text"):
                            val = item.get(field)
                            if isinstance(val, str) and val.strip():
                                text_parts.append(val)
                    elif isinstance(item, str):
                        text_parts.append(item)
        hits = data.get("rule_hits")
        if isinstance(hits, list):
            for h in hits[:20]:
                if isinstance(h, dict):
                    text_parts.append(str(h.get("message") or h.get("rule_id") or ""))
    return detect_contract_category(" ".join(text_parts), filename)


def list_known_categories() -> list[dict[str, str]]:
    return [{"id": k, "label": v} for k, v in _CATEGORY_LABELS.items()]
