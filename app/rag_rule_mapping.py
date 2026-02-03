"""
Map RAG-retrieved law chunks to rule_ids for focused rule execution.

Usage: chunks_to_rule_ids(rag_hits, rules_dir=...) -> Set[str]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_rule_mapping.json"
_DEFAULT_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
_CACHE: Dict[str, Any] = {}
_RULES_TEXT_CACHE: Dict[str, Dict[str, str]] = {}


def _pattern_to_keywords(pattern: str) -> str:
    """Strip regex syntax, keep meaningful Arabic/English words."""
    if not pattern or not isinstance(pattern, str):
        return ""
    s = pattern
    s = re.sub(r"\\s\+?", " ", s)
    s = re.sub(r"\\[SdDwWbB.]\+?", " ", s)
    s = re.sub(r"\\[\[\](){}|]", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[?:*+^$\\.,;:]{1,2}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_rules_text(rules_dir: Path) -> Dict[str, str]:
    """Load rules from YAML, return {rule_id: 'description rationale pattern_keywords'}."""
    cache_key = str(rules_dir)
    if cache_key in _RULES_TEXT_CACHE:
        return _RULES_TEXT_CACHE[cache_key]
    result: Dict[str, str] = {}
    try:
        import yaml
        for p in sorted(Path(rules_dir).glob("*.y*ml")):
            with open(p, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, dict):
                data = [data]
            for r in data:
                if not isinstance(r, dict):
                    continue
                rid = r.get("id") or r.get("rule_id")
                if not rid:
                    continue
                desc = (r.get("description") or "").strip()
                rat = (r.get("rationale") or "").strip()
                pat_kw: List[str] = []
                match = r.get("match") or {}
                for item in (match.get("any") or match.get("all") or []):
                    if isinstance(item, dict) and "pattern" in item:
                        kw = _pattern_to_keywords(str(item["pattern"]))
                        if kw and kw not in pat_kw:
                            pat_kw.append(kw)
                    elif isinstance(item, str):
                        kw = _pattern_to_keywords(item)
                        if kw and kw not in pat_kw:
                            pat_kw.append(kw)
                combined = f"{desc} {rat} {' '.join(pat_kw)}".strip()
                if combined:
                    result[str(rid)] = combined
        _RULES_TEXT_CACHE[cache_key] = result
    except Exception:
        pass
    return result


def _load_mapping() -> Dict[str, Any]:
    if not _CACHE:
        path = _DEFAULT_MAPPING_PATH
        if path.exists():
            try:
                _CACHE.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                _CACHE["keywords_to_rules"] = {}
                _CACHE["article_to_rules"] = {}
        else:
            _CACHE["keywords_to_rules"] = {}
            _CACHE["article_to_rules"] = {}
    return _CACHE


def _significant_tokens(s: str, min_len: int = 2) -> Set[str]:
    """Extract significant word tokens (Arabic/English, min length)."""
    s = re.sub(r"[^\w\u0600-\u06FF\s]", " ", s or "")
    return {t for t in s.split() if len(t) >= min_len}


def chunks_to_rule_ids(
    rag_hits: List[Dict[str, Any]],
    rules_dir: Optional[Path] = None,
) -> Set[str]:
    """
    Map RAG-retrieved law chunks to rule_ids to run.
    Uses: (1) static keywords + article from config, (2) rule-text matching from YAML.
    """
    mapping = _load_mapping()
    kw_to_rules: Dict[str, List[str]] = mapping.get("keywords_to_rules") or {}
    art_to_rules: Dict[str, List[str]] = mapping.get("article_to_rules") or {}

    rule_ids: Set[str] = set()
    rules_dir = rules_dir or _DEFAULT_RULES_DIR
    rules_text = _load_rules_text(rules_dir)

    for hit in rag_hits or []:
        text = (hit.get("text") or hit.get("page_content") or "").strip().lower()
        meta = hit.get("metadata") or {}
        article = str(meta.get("article") or "").strip()

        # Article-based
        if article and article in art_to_rules:
            for rid in art_to_rules[article]:
                rule_ids.add(rid)

        # Keyword-based (chunk text)
        for kw, rids in kw_to_rules.items():
            if kw.lower() in text:
                for rid in rids:
                    rule_ids.add(rid)

        # Rule-text matching: chunk contains 2+ significant tokens from rule's description/rationale/patterns
        chunk_tokens = _significant_tokens(text)
        for rid, rule_content in rules_text.items():
            rule_tokens = _significant_tokens(rule_content.lower())
            overlap = chunk_tokens & rule_tokens
            if len(overlap) >= 2:
                rule_ids.add(rid)

    return rule_ids
