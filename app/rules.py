from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json, yaml, re

FLAG_MAP = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "u": re.UNICODE,
}

def _compile_flags(flag_str: Optional[str]) -> int:
    flags = re.UNICODE
    if flag_str:
        for ch in flag_str.lower():
            flags |= FLAG_MAP.get(ch, 0)
    return flags

class RuleEngine:
    def __init__(self, rules_dir: Path | str, laws_dir: Path | str, base_dir: Optional[Path] = None):
        self.rules_dir = Path(rules_dir)
        self.laws_dir = Path(laws_dir)
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.rules: List[Dict[str, Any]] = []
        self.laws: Dict[str, Dict[str, Any]] = {}
        self._load_rules(self.rules_dir)
        self._load_laws(self.laws_dir)

    # -------- loaders --------
    def _load_rules(self, d: Path):
        for p in sorted(Path(d).glob("*.y*ml")):
            with open(p, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f) or []
            if isinstance(data, dict):
                data = [data]
            for r in data:
                if isinstance(r, dict):
                    r["_file"] = str(p)
                    self.rules.append(r)

    def _load_laws(self, d: Path):
        for p in sorted(Path(d).glob("*.json")):
            with open(p, "r", encoding="utf-8-sig") as f:
                self.laws[p.stem] = json.load(f)

    # -------- helpers --------
    def _article_text(self, article: Optional[str | int]) -> Optional[str]:
        if article is None:
            return None
        key = str(article)
        for lawdoc in self.laws.values():
            arts = (lawdoc or {}).get("articles", {})
            if key in arts:
                val = arts[key]
                if isinstance(val, dict):
                    return val.get("text") or val.get("body") or val.get("content")
                if isinstance(val, str):
                    return val
        return None

    def _read_suggestion(self, suggestion_ref: Optional[str]) -> Optional[str]:
        if not suggestion_ref:
            return None
        p = Path(suggestion_ref)
        if not p.is_absolute():
            p = self.base_dir / suggestion_ref
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                return f.read()
        except Exception:
            return None

    # -------- public API --------
    def check_text(self, text: str, law_scope: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
        scopes = set(law_scope or [])
        out: List[Dict[str, Any]] = []

        for r in self.rules:
            # scope filter (lenient if rule has no scope)
            scope_ok = not scopes or (r.get("scope") in scopes) or (r.get("law_scope") in scopes)
            if not scope_ok:
                continue

            m = r.get("match", {}) or {}
            any_rules = m.get("any", []) or []
            flags = _compile_flags(m.get("flags", "iu"))

            hit = False
            for item in any_rules:
                pat = item.get("pattern")
                if not pat:
                    continue
                try:
                    if re.search(pat, text, flags):
                        hit = True
                        break
                except re.error:
                    # ignore broken regex
                    continue
            if not hit:
                continue

            law = r.get("law")
            article = r.get("article")
            suggestion_ref = r.get("suggestion_ref")
            suggestion = self._read_suggestion(suggestion_ref)
            article_text = self._article_text(article)

            out.append({
                "rule_id": r.get("id"),
                "law": law,
                "article": str(article) if article is not None else None,
                "severity": r.get("severity"),
                "description": r.get("description"),
                "rationale": r.get("rationale"),
                "article_text": article_text,
                "suggestion": suggestion,
                "suggestion_ref": suggestion_ref,
            })

        return out
