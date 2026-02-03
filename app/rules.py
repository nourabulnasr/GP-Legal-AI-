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


def _to_western_digits(s: str) -> str:
    ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
    WESTERN = "0123456789"
    trans = str.maketrans(ARABIC_INDIC, WESTERN)
    return (s or "").translate(trans)


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

    def _build_hit(self, rule: Dict[str, Any], matched_text: Optional[str]) -> Dict[str, Any]:
        # IMPORTANT: expose BOTH "id" and "rule_id" to avoid downstream mismatch
        rid = rule.get("id") or rule.get("rule_id")

        law = rule.get("law")
        article = rule.get("article")
        suggestion_ref = rule.get("suggestion_ref")
        article_text = self._article_text(article)
        suggestion = self._read_suggestion(suggestion_ref)

        return {
            "id": rid,                 # ✅ for evidence builders that expect "id"
            "rule_id": rid,            # ✅ keep backward compatibility
            "law": law,
            "article": str(article) if article is not None else None,
            "severity": rule.get("severity"),
            "description": rule.get("description"),
            "rationale": rule.get("rationale"),
            "article_text": article_text,
            "suggestion": suggestion,
            "suggestion_ref": suggestion_ref,
            "matched_text": matched_text,
        }

    # -------- public API --------
    def check_text(
        self,
        text: str,
        law_scope: Optional[List[str]] = None,
        contract_type: Optional[str] = None,
        contract_tags: Optional[List[str]] = None,
        only_rule_ids: Optional[List[str]] = None,   # ✅ NEW (ML shortlist)
    ) -> List[Dict[str, Any]]:
        if not text:
            return []

        scopes = set(law_scope or [])
        out: List[Dict[str, Any]] = []

        # ✅ NEW: if provided, evaluate only these rule ids
        only_set = set(only_rule_ids) if only_rule_ids else None

        for r in self.rules:
            rule_id = r.get("id") or r.get("rule_id")
            if not rule_id:
                continue

            # ✅ NEW: ML pre-filter
            if only_set is not None and rule_id not in only_set:
                continue

            # ✅ scope filter (TRULY lenient if rule has no scope)
            r_scope = r.get("scope") or r.get("law_scope")
            if scopes and r_scope and (r_scope not in scopes):
                continue

            # SAFE contract_type filter
            rule_ct = r.get("contract_type")
            if rule_ct:
                if isinstance(rule_ct, str):
                    rule_ct = [rule_ct]
                if (contract_type is None) or (contract_type not in rule_ct):
                    continue

            # SAFE contract_tags filter
            req_tags = set(contract_tags or [])
            rule_tags = r.get("contract_tags") or r.get("tags")
            if rule_tags:
                if isinstance(rule_tags, str):
                    rule_tags = [rule_tags]
                rule_tags_set = set(rule_tags)
                if not req_tags:
                    continue
                if rule_tags_set.isdisjoint(req_tags):
                    continue

            mconf = r.get("match", {}) or {}
            flags = _compile_flags(mconf.get("flags", "iu"))

            # ==========================
            # SPECIAL: Employer presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_EMPLOYER_INFO":
                pat = re.compile(r"(الطرف\s*الاول|صاحب\s*العمل|الشرك[هة]\s*|شركة\s+.+)", flags)
                m = pat.search(text)
                if m:
                    out.append(self._build_hit(r, m.group(0)))
                continue

            if rule_id == "LABOR25_EMPLOYER_PLACEHOLDER":
                if re.search(r"(الطرف\s*الاول|صاحب\s*العمل).{0,200}(\.{6,}|…{2,})", text, flags):
                    out.append(self._build_hit(r, "placeholders"))
                continue

            if rule_id == "LABOR25_EMPLOYER_NAME_FILLED":
                if re.search(r"(شركة|الشركة)\s*[^\.\n]{3,}", text, flags) and not re.search(r"(شركة|الشركة)\s*(\.{6,}|…{2,})", text, flags):
                    out.append(self._build_hit(r, "company_name_filled"))
                continue

            # ==========================
            # SPECIAL: Employee presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_EMPLOYEE_INFO":
                pat = re.compile(r"(الطرف\s*الثان[ىي]|العامل|الموظف|السيد\s*/)", flags)
                m = pat.search(text)
                if m:
                    out.append(self._build_hit(r, m.group(0)))
                continue

            if rule_id == "LABOR25_EMPLOYEE_PLACEHOLDER":
                if re.search(r"(الطرف\s*الثان[ىي]|العامل|الموظف).{0,200}(\.{6,}|…{2,})", text, flags):
                    out.append(self._build_hit(r, "placeholders"))
                continue

            if rule_id == "LABOR25_EMPLOYEE_NAME_FILLED":
                if re.search(r"(السيد\s*/\s*[^\.\n]{3,})", text, flags) and not re.search(r"(السيد\s*/\s*(\.{6,}|…{2,}))", text, flags):
                    out.append(self._build_hit(r, "employee_name_filled"))
                continue

            # ==========================
            # SPECIAL: Salary presence (VERY tolerant to OCR)
            # ==========================
            if rule_id in ("LABOR25_SALARY_VALUE_PRESENT", "LABOR25_SALARY"):
                pat_salary = re.compile(
                    r"(مرتب|راتب|أجر|اجر).{0,80}(\d{3,7}|[٠-٩]{3,7}).{0,30}(جنيه|جميه|egp|ج\.م|جم)",
                    flags,
                )
                m = pat_salary.search(text)
                if m:
                    out.append(self._build_hit(r, m.group(0)))
                    continue

                pat_currency = re.compile(r"(\d{3,7}|[٠-٩]{3,7}).{0,15}(جنيه|جميه|egp|usd|eur|aed|ج\.م|جم)", flags)
                m2 = pat_currency.search(text)
                if m2:
                    out.append(self._build_hit(r, m2.group(0)))
                continue

            # ==========================
            # SPECIAL: Contract duration presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_CONTRACT_DURATION":
                pat_dur = re.compile(r"(مدة|مد|ة\s*مد).{0,30}(العقد|لعقد|لعا\s*قد|هذا\s*العقد)", flags)
                if pat_dur.search(text):
                    out.append(self._build_hit(r, "duration_clause"))
                    continue

                pat_dates = re.compile(r"(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})", flags)
                dates = pat_dates.findall(text)
                if len(dates) >= 1:
                    out.append(self._build_hit(r, dates[0]))
                continue

            # ==========================
            # SPECIAL: Probation presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_PROBATION_PRESENCE":
                if re.search(r"(فترة\s*الاختبار|مدة\s*الاختبار|تحت\s*ال[اأإآ]?ختبار|اختبار\s*تحت\s*التجرب)", text, flags):
                    out.append(self._build_hit(r, "probation_present"))
                continue

            # ==========================
            # SPECIAL CASE 1: Working hours violation
            # ==========================
            if rule_id == "LABOR25_WORKING_HOURS_VIOLATION":
                pat_with_break = re.compile(r"(\d+)\s*ساعات?.{0,120}?يتخللها.{0,40}?ساعة", flags)
                for m in pat_with_break.finditer(text):
                    try:
                        total_h = int(_to_western_digits(m.group(1)))
                    except Exception:
                        continue
                    effective = total_h - 1
                    if effective > 8:
                        out.append(self._build_hit(r, m.group(0)))

                pat_plain = re.compile(r"(\d+)\s*ساعات?(?:\s+عمل)?(?!.*راحة)", flags)
                for m in pat_plain.finditer(text):
                    try:
                        total_h = int(_to_western_digits(m.group(1)))
                    except Exception:
                        continue
                    if total_h > 8:
                        out.append(self._build_hit(r, m.group(0)))
                continue

            # ==========================
            # SPECIAL: Working hours presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_WORKING_HOURS_PRESENCE":
                if re.search(r"(ساعات\s*العمل|عدد\s*ساعات|الدوام|وقت\s*العمل)", text, flags):
                    out.append(self._build_hit(r, "working_hours_present"))
                continue

            # ==========================
            # SPECIAL CASE 2: Annual leave minimum
            # ==========================
            if rule_id == "LABOR25_ANNUAL_LEAVE_VIOLATION":
                pat_leave = re.compile(r"إجازة\s+سنوية\s+(?:مدتها\s+)?(\d+|[٠-٩]{1,2})\s*يوم", flags)
                for m in pat_leave.finditer(text):
                    try:
                        days = int(_to_western_digits(m.group(1)))
                    except Exception:
                        continue
                    if days < 15:
                        out.append(self._build_hit(r, m.group(0)))
                continue

            # ==========================
            # SPECIAL: Annual leave presence (tolerant)
            # ==========================
            if rule_id == "LABOR25_ANNUAL_LEAVE_PRESENCE":
                if re.search(r"(إجازة\s*سنوية|الاجازة\s*السنوية|اجازه\s*سنويه)", text, flags):
                    out.append(self._build_hit(r, "annual_leave_present"))
                continue

            # ==========================
            # SPECIAL CASE 3: Probation > 3 months
            # ==========================
            if rule_id == "LABOR25_PROBATION_LIMIT":
                word_map = {
                    "اربع": 4, "اربعه": 4, "أربع": 4, "أربعه": 4, "أربعة": 4,
                    "خمس": 5, "خمسه": 5, "خمسة": 5,
                    "ست": 6, "سته": 6, "ستة": 6,
                    "سبع": 7, "سبعه": 7, "سبعة": 7,
                    "ثمان": 8, "ثمانيه": 8, "ثمانية": 8,
                    "تسع": 9, "تسعه": 9, "تسعة": 9,
                }

                pat_prob = re.compile(
                    r"(?:فترة\s+الاختبار|مدة\s+الاختبار|تحت\s+ال[اأإآ]?ختبار)"
                    r".{0,120}?"
                    r"(?:لمدة|مدة)?\s*(?:لا\s*تتجاوز\s*)?"
                    r"(\d{1,2}|[٠-٩]{1,2}|أربع(?:ة)?|خمس(?:ة)?|ست(?:ة)?|سبع(?:ة)?|ثمان(?:ية)?|تسع(?:ة)?)"
                    r"\s*(?:أشهر|اشهر|شهور|شهر)",
                    flags,
                )

                for m in pat_prob.finditer(text):
                    raw = (m.group(1) or "").strip()
                    raw2 = _to_western_digits(raw)

                    months = None
                    if raw2.isdigit():
                        try:
                            months = int(raw2)
                        except Exception:
                            months = None
                    else:
                        key = raw.replace("أ", "").replace("إ", "").replace("آ", "").strip()
                        months = word_map.get(key) or word_map.get(raw)

                    if months is not None and months > 3:
                        out.append(self._build_hit(r, m.group(0)))
                continue

            # ==========================
            # DEFAULT: generic regex matching from YAML
            # ==========================
            any_rules = mconf.get("any", []) or []
            rule_matched = False

            for item in any_rules:
                pat = item.get("pattern")
                if not pat:
                    continue
                try:
                    regex = re.compile(pat, flags)
                except re.error:
                    continue

                for m in regex.finditer(text):
                    out.append(self._build_hit(r, m.group(0)))
                    rule_matched = True
                    break

                if rule_matched:
                    break

        return out