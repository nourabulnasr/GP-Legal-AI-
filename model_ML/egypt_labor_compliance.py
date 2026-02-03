# app/egypt_labor_compliance.py
# ------------------------------------------------------------
# Egypt Labor Compliance Analyzer (Deterministic + Transparent)
#
# Purpose:
# - Identify potential Egypt labor-law compliance issues directly from contract text
# - Complement ML/RAG models with deterministic "signals"
#
# Key features:
# - Robust contract segmentation (Arabic/English headings, numbering, bullets)
# - Clause + contract analysis
# - Findings carry spans in CONTRACT coordinates for UI highlighting
#
# Legal basis (Egypt Labor Law No. 14 of 2025 - translation):
# - Working hours: 8/day, 48/week; max presence 12h/day in cases/exemptions
# - Weekly rest: >= 24 continuous hours after max 6 consecutive days; paid
# - Overtime premiums: >=35% day, >=70% night; rest-day work => extra day wage + substitute day off
# - Probation: <= 3 months; not more than once with same employer
# - Annual leave: 15/21/30/45 + rules (min 15 days yearly incl 6 consecutive; no waiver)
# - Emergency leave: 7 days/year, max 2 at a time; deducted from annual leave
# - Holidays: double wage or substitute day off (on employee written request)
# - Maternity: 4 months paid, >=45 days after delivery; reduced hours from 6th month; no overtime during pregnancy/6 months after
# - Deductions for damage: <= 5 days wages/month; total cap <= 2 months wages (under that mechanism)
# - Dismissal: only labor court can impose dismissal; serious misconduct list exists
#
# NOTE:
# - This is NOT legal advice. It’s a transparent rules layer to boost recall + explainability.
# ------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -------------------------
# Normalization helpers
# -------------------------
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EASTERN_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Remove bidi controls (helps offsets stay stable after OCR copy/paste)
BIDI_CHARS_RE = re.compile(r"[\u200e\u200f\u202a\u202b\u202c\u202d\u202e]")

# Collapse spaces but keep newlines
SPACES_RE = re.compile(r"[ \t]+")

# OCR noise: repeated empty lines
MANY_NEWLINES_RE = re.compile(r"\n{3,}")

# Strip leakage tokens if any got into contracts (optional safety)
VIOL_TOKEN_RE = re.compile(r"\[\[VIOLATION_\d+\]\]", re.IGNORECASE)


def normalize_digits(text: str) -> str:
    return (text or "").translate(ARABIC_INDIC).translate(EASTERN_INDIC)


def normalize_contract_text(text: str) -> str:
    """
    Conservative normalization:
    - keep content mostly unchanged to preserve offsets stability
    - remove bidi controls
    - normalize digits
    - collapse multiple spaces (not newlines)
    - collapse excessive blank lines
    """
    t = text or ""
    t = BIDI_CHARS_RE.sub("", t)
    t = normalize_digits(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = VIOL_TOKEN_RE.sub(" ", t)
    t = SPACES_RE.sub(" ", t)
    t = MANY_NEWLINES_RE.sub("\n\n", t)
    return t


# -------------------------
# Output objects
# -------------------------
@dataclass
class ComplianceFinding:
    finding_id: str
    title_ar: str
    title_en: str
    rationale: str
    confidence: float
    spans: List[Tuple[int, int]]  # spans in CONTRACT coordinates (start,end)


@dataclass
class ContractClause:
    clause_id: str
    start: int
    end: int
    text: str
    heading: str = ""


@dataclass
class ClauseAnalysis:
    clause: ContractClause
    findings: List[ComplianceFinding]
    signal: str


# -------------------------
# Legal thresholds (Egypt Labor Law 14/2025 defaults)
# -------------------------
@dataclass
class LaborComplianceConfig:
    # Art. 117: <= 8 hours/day, 48/week (excluding meal/rest periods)
    max_hours_per_day: int = 8
    max_hours_per_week: int = 48

    # Weekly rest: >= 24 continuous hours after max 6 consecutive days (Art. 120)
    max_consecutive_work_days: int = 6
    min_weekly_rest_hours: int = 24

    # Presence cap (regular + overtime presence): <= 12 hours/day (Art. 120/121)
    max_presence_hours_per_day: int = 12

    # Probation (Art. 90)
    max_probation_months: int = 3

    # Annual leave baseline (Art. 124) – used for "too low" checks
    annual_leave_first_year_days: int = 15
    annual_leave_second_year_days: int = 21
    annual_leave_10y_or_50plus_days: int = 30
    annual_leave_disability_days: int = 45

    # Emergency leave (Art. 128)
    emergency_leave_days_per_year: int = 7
    emergency_leave_max_consecutive: int = 2

    # Overtime minimum premiums (Art. 121)
    overtime_min_premium_day_pct: int = 35
    overtime_min_premium_night_pct: int = 70

    # Deductions for damage (Art. 151)
    damage_deduction_max_days_per_month: int = 5
    damage_deduction_total_cap_months: int = 2


# -------------------------
# Segmentation (robust)
# -------------------------
_SEG_AR_HEADING = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        (?:المادة|مادة)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
      | (?:البند|بند)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
      | (?:الفقرة|فقرة)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
      | (?:الفصل|باب)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
    )
    (?:\s*[:：\-–—]\s*.*)?\s*$
    """
)

_SEG_EN_HEADING = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        (?:article|clause|section|chapter)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
      | (?:appendix|schedule)\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?
    )
    (?:\s*[:：\-–—]\s*.*)?\s*$
    """
)

# numeric headings like: 1) / (1) / 1. / 1- / 1/ ...
_SEG_NUM_HEADING = re.compile(
    r"""(?x)
    ^\s*
    (?:\(?\s*\d{1,4}\s*\)?)
    \s*(?:[.)\-–—/:]|:\s*)
    \s*\S.*
    $
    """
)

# Arabic letter bullets like: أ) ... / ب- ... / ج/ ...
_SEG_AR_LETTER_BULLET = re.compile(
    r"""(?x)
    ^\s*
    (?:[اأإآبجدهوزحطيكلمنسعفصقرشتثخذضظغ])
    \s*(?:[.)\-–—/:]|:\s*)
    \s*\S.*
    $
    """
)

# Roman/alpha bullets in English like: a) / b. / i) ...
_SEG_EN_BULLET = re.compile(
    r"""(?ix)
    ^\s*
    (?:[a-z]|[ivx]{1,6})
    \s*(?:[.)\-–—/:]|:\s*)
    \s*\S.*
    $
    """
)

# extra: "Definitions", "Scope", Arabic equivalents sometimes appear without numbering
_SEG_GENERIC_HEADINGS = re.compile(
    r"""(?ix)
    ^\s*(?:definitions?|scope|purpose|term|duration|wages?|salary|compensation|leave|vacation|overtime|termination|notice|confidentiality|non[-\s]?compete)\s*[:：\-–—]?\s*$
    """
)

_SEG_AR_GENERIC_HEADINGS = re.compile(
    r"""(?x)
    ^\s*(?:تعريفات|التعريفات|نطاق|الغرض|المدة|الأجر|الراتب|المقابل|الإجازات|الاجازات|العمل\s*الإضافي|العمل\s*الاضافي|الإنهاء|الانهاء|الإخطار|الاخطار|السرية)\s*[:：\-–—]?\s*$
    """
)


def _line_offsets(text: str) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start, end, line_text) for each line.
    Offsets refer to indices in `text`. We include newline in end offsets when present.
    """
    lines: List[Tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("\n", i)
        if j == -1:
            j = n
            line = text[i:j]
            lines.append((i, j, line))
            break
        line = text[i:j]
        lines.append((i, j + 1, line))
        i = j + 1
    return lines


def _is_boundary_line(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) < 3:
        return False

    if _SEG_AR_HEADING.match(s) or _SEG_EN_HEADING.match(s):
        return True

    if _SEG_GENERIC_HEADINGS.match(s) or _SEG_AR_GENERIC_HEADINGS.match(s):
        return True

    if _SEG_NUM_HEADING.match(s):
        return True

    # bullets: treat as boundary only if line is somewhat substantial
    if len(s) >= 8 and (_SEG_AR_LETTER_BULLET.match(s) or _SEG_EN_BULLET.match(s)):
        return True

    return False


def segment_contract(
    contract_text: str,
    *,
    min_clause_chars: int = 60,
    merge_short_following: bool = True,
) -> List[ContractClause]:
    """
    Segment a full contract into clauses with stable offsets.

    Strategy:
    - Line-based segmentation using robust heading patterns.
    - Create chunks between boundary lines.
    - Merge extremely short clauses into neighbors (OCR headings alone).
    - If no boundaries: fallback to paragraph segmentation by blank lines.
    """
    t = normalize_contract_text(contract_text)
    if not t.strip():
        return []

    lines = _line_offsets(t)
    if not lines:
        return []

    boundary_lines: List[int] = []
    for li, (_, _, line) in enumerate(lines):
        if _is_boundary_line(line):
            boundary_lines.append(li)

    # fallback: paragraph segmentation by blank lines
    if not boundary_lines:
        clauses: List[ContractClause] = []
        parts: List[Tuple[int, int, str]] = []
        start_idx = 0
        for m in re.finditer(r"\n\s*\n", t):
            end_idx = m.start()
            chunk = t[start_idx:end_idx].strip()
            if chunk:
                parts.append((start_idx, end_idx, chunk))
            start_idx = m.end()
        tail = t[start_idx:].strip()
        if tail:
            parts.append((start_idx, len(t), tail))

        for k, (s, e, chunk) in enumerate(parts, start=1):
            clauses.append(ContractClause(clause_id=f"clause_{k:04d}", start=s, end=e, text=chunk))
        return clauses

    segs: List[Tuple[int, int, str, str]] = []  # (start, end, text, heading)
    for bi, li in enumerate(boundary_lines):
        start = lines[li][0]
        end = lines[boundary_lines[bi + 1]][0] if bi + 1 < len(boundary_lines) else len(t)
        chunk = t[start:end]
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            continue
        heading = lines[li][2].strip()
        segs.append((start, end, chunk_stripped, heading))

    # Prepend intro segment if needed
    first_boundary_start = lines[boundary_lines[0]][0]
    if first_boundary_start > 0:
        intro = t[:first_boundary_start].strip()
        if intro:
            segs.insert(0, (0, first_boundary_start, intro, ""))

    def seg_len(s: str) -> int:
        return len((s or "").strip())

    # merge short forward
    merged: List[Tuple[int, int, str, str]] = []
    i = 0
    while i < len(segs):
        s, e, txt, head = segs[i]
        if merge_short_following and seg_len(txt) < min_clause_chars and i + 1 < len(segs):
            s2, e2, txt2, head2 = segs[i + 1]
            segs[i + 1] = (
                s,
                e2,
                (txt.rstrip() + "\n" + txt2.lstrip()).strip(),
                head or head2,
            )
            i += 1
            continue
        merged.append((s, e, txt, head))
        i += 1

    # merge tiny leftovers backward
    final: List[Tuple[int, int, str, str]] = []
    for (s, e, txt, head) in merged:
        if final and seg_len(txt) < max(30, min_clause_chars // 2):
            ps, pe, ptxt, phead = final[-1]
            final[-1] = (ps, e, (ptxt.rstrip() + "\n" + txt.lstrip()).strip(), phead)
        else:
            final.append((s, e, txt, head))

    clauses: List[ContractClause] = []
    for k, (s, e, txt, head) in enumerate(final, start=1):
        clauses.append(ContractClause(clause_id=f"clause_{k:04d}", start=int(s), end=int(e), text=txt, heading=head))
    return clauses


# -------------------------
# Main analyzer
# -------------------------
class EgyptLaborComplianceAnalyzer:
    """
    Deterministic compliance checks for Egypt Labor Law (14/2025).

    Use:
      - analyze_clause(text) for clause-local checks (spans relative to clause)
      - analyze_contract(contract_text) for segmentation + contract-global spans
    """

    def __init__(self, config: Optional[LaborComplianceConfig] = None):
        self.cfg = config or LaborComplianceConfig()

        # -------------------------
        # Working hours & presence
        # -------------------------
        # e.g. "مدة العمل: 10 ساعات يوم" / "working hours: 10 hours per day"
        self.re_hours_day = re.compile(
            r"(ساعات\s*العمل|مدة\s*العمل|working\s*hours)\s*[:：]?\s*(\d{1,2})\s*(ساعة|hours?)\s*(?:يوم|يومي[اً]?|per\s*day)",
            re.IGNORECASE,
        )
        # broader pattern: "10 ساعات يومياً" without explicit "مدة العمل"
        self.re_hours_day_loose = re.compile(
            r"(?<!\d)(\d{1,2})\s*(?:ساعة|hours?)\s*(?:يوم|يومي[اً]?|per\s*day)\b",
            re.IGNORECASE,
        )

        # e.g. "60 ساعة أسبوع" / "60 hours per week"
        self.re_hours_week = re.compile(
            r"(?<!\d)(\d{1,2})\s*(ساعة|hours?)\s*(?:أسبوع|اسبوع|أسبوعياً|اسبوعياً|per\s*week)\b",
            re.IGNORECASE,
        )

        # presence cap sometimes written as "حتى 12 ساعة" / "not exceed 12 hours"
        self.re_presence_12 = re.compile(
            r"(?:لا\s*تجاوز|لا\s*يتجاوز|بحد\s*أقصى|حد\s*أقصى|max(?:imum)?|not\s*exceed)\s*(\d{1,2})\s*(?:ساعة|hours?)\s*(?:يوم|per\s*day)?",
            re.IGNORECASE,
        )

        # weekly rest / consecutive days
        self.re_weekly_rest_mention = re.compile(r"(راحة\s*أسبوعية|يوم\s*راحة|weekly\s*rest|rest\s*day)", re.IGNORECASE)
        self.re_all_week = re.compile(r"(7\s*أيام|سبعة\s*أيام|طوال\s*أيام\s*الأسبوع|seven\s*days|all\s*week)", re.IGNORECASE)
        self.re_six_consecutive = re.compile(r"(6\s*أيام|ستة\s*أيام|six\s*days)\s*(?:متتالية|consecutive)?", re.IGNORECASE)

        # overtime: unpaid / waiver
        self.re_unpaid_overtime = re.compile(
            r"(دون\s*(أجر|مقابل)|بدون\s*(أجر|مقابل)|لا\s*يستحق\s*أجر\s*إضافي|unpaid\s*overtime|no\s*overtime\s*pay)",
            re.IGNORECASE,
        )
        self.re_overtime_waiver = re.compile(
            r"(يتنازل\s*عن\s*(أجر|بدل)\s*العمل\s*الإضافي|waive\s*overtime|overtime\s*waiver)",
            re.IGNORECASE,
        )

        # overtime premium mentions (try to catch illegal low percentages)
        self.re_overtime_pct = re.compile(
            r"(?:(\d{1,3})\s*%|بنسبة\s*(\d{1,3})\s*%|(\d{1,3})\s*percent)\s*(?:إضافي|زيادة|premium|overtime)?",
            re.IGNORECASE,
        )
        self.re_night = re.compile(r"(ليلي|ليلية|night)", re.IGNORECASE)
        self.re_daytime = re.compile(r"(نهاري|نهارية|day\s*time|daytime|day)", re.IGNORECASE)

        # official holiday work compensation
        self.re_holiday_work = re.compile(
            r"(عطلة\s*رسمية|الأعياد|العطلات|public\s*holiday|official\s*holiday|holiday)",
            re.IGNORECASE,
        )
        self.re_double_pay = re.compile(
            r"(ضعف\s*(?:الأجر|الراتب)|مرتين|double\s*(?:pay|wage)|2x)",
            re.IGNORECASE,
        )
        self.re_substitute_day_off = re.compile(
            r"(يوم\s*بديل|راحة\s*بديلة|substitute\s*day\s*off|day\s*in\s*lieu)",
            re.IGNORECASE,
        )

        # -------------------------
        # Probation
        # -------------------------
        # "فترة الاختبار 6 شهر" / "probation 6 months"
        self.re_probation = re.compile(
            r"(فترة\s*الاختبار|probation)\s*[:：]?\s*(\d{1,2})\s*(شهر|months?)",
            re.IGNORECASE,
        )
        self.re_repeat_probation = re.compile(
            r"(تجديد\s*فترة\s*الاختبار|تمديد\s*فترة\s*الاختبار|probation\s*(?:renewal|extension)|another\s*probation)",
            re.IGNORECASE,
        )

        # -------------------------
        # Annual leave & emergency leave
        # -------------------------
        self.re_annual_leave = re.compile(r"(إجازة\s*سنوية|اجازة\s*سنوية|annual\s*leave|paid\s*leave)", re.IGNORECASE)
        self.re_leave_days = re.compile(r"(?<!\d)(\d{1,3})\s*(?:يوم|days?)\b", re.IGNORECASE)
        self.re_waive_leave = re.compile(
            r"(يتنازل\s*عن\s*الإجازة|يتنازل\s*عن\s*الاجازة|waive\s*(?:annual\s*)?leave|no\s*leave\s*entitlement)",
            re.IGNORECASE,
        )
        self.re_emergency_leave = re.compile(r"(إجازة\s*عارضة|اجازة\s*عارضة|emergency\s*leave|casual\s*leave)", re.IGNORECASE)

        # -------------------------
        # Maternity protections
        # -------------------------
        self.re_maternity = re.compile(r"(إجازة\s*وضع|اجازة\s*وضع|maternity\s*leave)", re.IGNORECASE)
        self.re_months = re.compile(r"(?<!\d)(\d{1,2})\s*(?:شهر|months?)\b", re.IGNORECASE)
        self.re_no_overtime = re.compile(r"(لا\s*يُكلف\s*ب(?:ال)?عمل\s*إضافي|لا\s*يجوز\s*تكليف.*عمل\s*إضافي|no\s*overtime)", re.IGNORECASE)
        self.re_dismiss_pregnant = re.compile(
            r"(إنهاء\s*الخدمة|فصل|dismiss|terminate).*(حمل|pregnan|maternity|وضع)",
            re.IGNORECASE,
        )
        self.re_breastfeeding_breaks = re.compile(
            r"(رضاعة|breastfeed|breastfeeding)\s*(?:استراحة|breaks?)",
            re.IGNORECASE,
        )

        # -------------------------
        # Rights waivers (general)
        # -------------------------
        self.re_general_waiver = re.compile(
            r"(يتنازل\s*ال(عامل|موظف)\s*عن\s*حقوقه|waive\s*all\s*rights|release\s*and\s*discharge\s*all\s*claims)",
            re.IGNORECASE,
        )

        # -------------------------
        # Wage deductions for damage / penalties
        # -------------------------
        self.re_deduct_from_wage = re.compile(r"(خصم|deduct(?:ion)?|اقتطاع|withhold)", re.IGNORECASE)
        self.re_days_wage = re.compile(r"(?<!\d)(\d{1,2})\s*(?:يوم|days?)\s*(?:من\s*)?(?:الأجر|الراتب|wage|salary)", re.IGNORECASE)
        self.re_damage_liability = re.compile(r"(تلف|إتلاف|إضاع[ةه]|damage|loss|lost)", re.IGNORECASE)

        # -------------------------
        # Termination / dismissal structure
        # -------------------------
        self.re_immediate_termination_anytime = re.compile(
            r"(يحق\s*لل(شركة|صاحب\s*العمل|العمل)\s*إنهاء\s*العقد\s*في\s*أي\s*وقت|terminate\s*at\s*any\s*time|at\s*the\s*sole\s*discretion)",
            re.IGNORECASE,
        )
        self.re_no_court = re.compile(r"(دون\s*اللجوء\s*للمحكمة|without\s*court|no\s*need\s*for\s*court)", re.IGNORECASE)

    # -------------------------
    # Public API
    # -------------------------
    def analyze_clause(self, clause_text: str) -> List[ComplianceFinding]:
        """
        Clause-local analysis.
        Spans are relative to the clause text (0..len(clause)).
        """
        t_norm = normalize_digits(clause_text)
        findings: List[ComplianceFinding] = []
        findings += self._check_working_hours(t_norm, base_offset=0)
        findings += self._check_weekly_rest(t_norm, base_offset=0)
        findings += self._check_overtime_rules(t_norm, base_offset=0)
        findings += self._check_probation(t_norm, base_offset=0)
        findings += self._check_leave_rules(t_norm, base_offset=0)
        findings += self._check_maternity_rules(t_norm, base_offset=0)
        findings += self._check_wage_deductions(t_norm, base_offset=0)
        findings += self._check_termination_dismissal_structure(t_norm, base_offset=0)
        findings += self._check_general_waivers(t_norm, base_offset=0)
        return self._deduplicate(findings)

    def analyze_contract(
        self,
        contract_text: str,
        *,
        min_clause_chars: int = 60,
        merge_short_following: bool = True,
    ) -> List[ClauseAnalysis]:
        """
        Correct segmentation + analysis.
        Findings spans are returned in CONTRACT coordinates.
        """
        t = normalize_contract_text(contract_text)
        clauses = segment_contract(t, min_clause_chars=min_clause_chars, merge_short_following=merge_short_following)

        out: List[ClauseAnalysis] = []
        for cl in clauses:
            t_clause = normalize_digits(cl.text)
            findings: List[ComplianceFinding] = []
            findings += self._check_working_hours(t_clause, base_offset=cl.start)
            findings += self._check_weekly_rest(t_clause, base_offset=cl.start)
            findings += self._check_overtime_rules(t_clause, base_offset=cl.start)
            findings += self._check_probation(t_clause, base_offset=cl.start)
            findings += self._check_leave_rules(t_clause, base_offset=cl.start)
            findings += self._check_maternity_rules(t_clause, base_offset=cl.start)
            findings += self._check_wage_deductions(t_clause, base_offset=cl.start)
            findings += self._check_termination_dismissal_structure(t_clause, base_offset=cl.start)
            findings += self._check_general_waivers(t_clause, base_offset=cl.start)

            findings = self._deduplicate(findings)
            signal = self.compliance_signal_text(findings)
            out.append(ClauseAnalysis(clause=cl, findings=findings, signal=signal))

        return out

    def compliance_signal_text(self, findings: List[ComplianceFinding]) -> str:
        if not findings:
            return "COMPLIANCE_SIGNALS: none\n"
        parts = [f"{f.finding_id}({f.confidence:.2f})" for f in findings]
        return "COMPLIANCE_SIGNALS: " + " ".join(parts) + "\n"

    def elevate_findings(
        self,
        findings: List[ComplianceFinding],
        finding_to_violation_id: Dict[str, str],
        min_confidence: float = 0.75,
    ) -> List[str]:
        """
        Converts strong compliance findings into violation IDs (safety-net).
        """
        out: List[str] = []
        seen = set()
        for f in findings:
            if f.confidence < min_confidence:
                continue
            vid = finding_to_violation_id.get(f.finding_id)
            if vid and vid not in seen:
                seen.add(vid)
                out.append(vid)
        return out

    # -------------------------
    # Checks: Working hours / presence
    # -------------------------
    def _check_working_hours(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        # explicit "مدة العمل" pattern
        for m in self.re_hours_day.finditer(t):
            hours = self._safe_int(m.group(2))
            if hours is None:
                continue
            if hours > self.cfg.max_hours_per_day:
                hits.append(
                    ComplianceFinding(
                        finding_id="WORK_HOURS_DAY_EXCESS",
                        title_ar="تجاوز الحد اليومي لساعات العمل",
                        title_en="Daily working hours exceed legal cap",
                        rationale=f"Clause specifies {hours} hours/day (cap is {self.cfg.max_hours_per_day}).",
                        confidence=0.92,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        # loose daily hours mention (only if it seems like an obligation, not e.g. "within 8 hours")
        for m in self.re_hours_day_loose.finditer(t):
            hours = self._safe_int(m.group(1))
            if hours is None:
                continue
            if hours > self.cfg.max_hours_per_day:
                hits.append(
                    ComplianceFinding(
                        finding_id="WORK_HOURS_DAY_EXCESS",
                        title_ar="تجاوز الحد اليومي لساعات العمل",
                        title_en="Daily working hours exceed legal cap",
                        rationale=f"Clause indicates {hours} hours/day (cap is {self.cfg.max_hours_per_day}).",
                        confidence=0.85,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        # weekly hours
        for m in self.re_hours_week.finditer(t):
            hours = self._safe_int(m.group(1))
            if hours is None:
                continue
            if hours > self.cfg.max_hours_per_week:
                hits.append(
                    ComplianceFinding(
                        finding_id="WORK_HOURS_WEEK_EXCESS",
                        title_ar="تجاوز الحد الأسبوعي لساعات العمل",
                        title_en="Weekly working hours exceed legal cap",
                        rationale=f"Clause specifies {hours} hours/week (cap is {self.cfg.max_hours_per_week}).",
                        confidence=0.92,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        # presence cap
        for m in self.re_presence_12.finditer(t):
            h = self._safe_int(m.group(1))
            if h is None:
                continue
            if h > self.cfg.max_presence_hours_per_day:
                hits.append(
                    ComplianceFinding(
                        finding_id="WORK_PRESENCE_EXCESS",
                        title_ar="تجاوز الحد الأقصى للتواجد اليومي بمقر العمل",
                        title_en="Daily workplace presence exceeds legal cap",
                        rationale=f"Clause allows {h} hours/day presence (cap is {self.cfg.max_presence_hours_per_day}).",
                        confidence=0.75,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        return hits

    # -------------------------
    # Checks: Weekly rest
    # -------------------------
    def _check_weekly_rest(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        # If contract says 7 days/all week but doesn't mention weekly rest -> flag
        if self.re_all_week.search(t) and not self.re_weekly_rest_mention.search(t):
            m = self.re_all_week.search(t)
            spans = [(base_offset + m.start(), base_offset + m.end())] if m else []
            return [
                ComplianceFinding(
                    finding_id="NO_WEEKLY_REST_MENTIONED",
                    title_ar="غياب النص على الراحة الأسبوعية",
                    title_en="Weekly rest day not mentioned",
                    rationale="Clause suggests work across all days without clear weekly rest (law requires ≥24h rest).",
                    confidence=0.82,
                    spans=spans,
                )
            ]

        # If clause mentions 6 consecutive days but no rest mention -> still suspicious
        if self.re_six_consecutive.search(t) and not self.re_weekly_rest_mention.search(t):
            m = self.re_six_consecutive.search(t)
            spans = [(base_offset + m.start(), base_offset + m.end())] if m else []
            return [
                ComplianceFinding(
                    finding_id="WEEKLY_REST_UNCLEAR",
                    title_ar="الراحة الأسبوعية غير واضحة",
                    title_en="Weekly rest unclear",
                    rationale="Clause indicates consecutive working days but does not clearly state a paid weekly rest day.",
                    confidence=0.70,
                    spans=spans,
                )
            ]

        return []

    # -------------------------
    # Checks: Overtime rules
    # -------------------------
    def _check_overtime_rules(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        if self.re_unpaid_overtime.search(t) or self.re_overtime_waiver.search(t):
            m = self.re_unpaid_overtime.search(t) or self.re_overtime_waiver.search(t)
            spans = [(base_offset + m.start(), base_offset + m.end())] if m else []
            hits.append(
                ComplianceFinding(
                    finding_id="OVERTIME_UNPAID_OR_WAIVED",
                    title_ar="العمل الإضافي بدون مقابل أو مع التنازل عنه",
                    title_en="Overtime unpaid or waived",
                    rationale="Law requires overtime compensation at not less than specified minimum premiums.",
                    confidence=0.88,
                    spans=spans,
                )
            )

        # If an overtime percentage is explicitly stated and appears below minimums, flag.
        # (We try to infer day/night context; if unknown, use day minimum.)
        for m in self.re_overtime_pct.finditer(t):
            pct = self._first_int_group(m.groups())
            if pct is None:
                continue
            window = t[max(0, m.start() - 30) : min(len(t), m.end() + 30)]
            is_night = bool(self.re_night.search(window))
            min_pct = self.cfg.overtime_min_premium_night_pct if is_night else self.cfg.overtime_min_premium_day_pct
            if 0 < pct < min_pct:
                hits.append(
                    ComplianceFinding(
                        finding_id="OVERTIME_PREMIUM_TOO_LOW",
                        title_ar="بدل العمل الإضافي أقل من الحد الأدنى",
                        title_en="Overtime premium below minimum",
                        rationale=f"Clause suggests overtime premium {pct}% which is below minimum {min_pct}%.",
                        confidence=0.72,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        # Holiday work: if mentions holidays but explicitly denies double pay / substitute day
        if self.re_holiday_work.search(t):
            # if it says "no extra pay" around holidays
            denial = re.search(r"(دون\s*مقابل|بدون\s*أجر|no\s*extra\s*pay|without\s*extra\s*compensation)", t, re.IGNORECASE)
            if denial:
                hits.append(
                    ComplianceFinding(
                        finding_id="HOLIDAY_WORK_COMPENSATION_DENIED",
                        title_ar="إنكار مقابل العمل في العطلات الرسمية",
                        title_en="Holiday work compensation denied",
                        rationale="Law provides double wage or substitute day off for official holidays work.",
                        confidence=0.78,
                        spans=[(base_offset + denial.start(), base_offset + denial.end())],
                    )
                )
            else:
                # If it mentions holiday work but neither double pay nor substitute day off appears, mark as unclear (lower confidence)
                if not self.re_double_pay.search(t) and not self.re_substitute_day_off.search(t):
                    m = self.re_holiday_work.search(t)
                    hits.append(
                        ComplianceFinding(
                            finding_id="HOLIDAY_WORK_COMPENSATION_UNCLEAR",
                            title_ar="مقابل العمل في العطلات الرسمية غير واضح",
                            title_en="Holiday work compensation unclear",
                            rationale="Clause mentions official holiday work but does not clearly state double pay or a substitute day off.",
                            confidence=0.62,
                            spans=[(base_offset + m.start(), base_offset + m.end())] if m else [],
                        )
                    )

        return hits

    # -------------------------
    # Checks: Probation
    # -------------------------
    def _check_probation(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []
        for m in self.re_probation.finditer(t):
            months = self._safe_int(m.group(2))
            if months is None:
                continue
            if months > self.cfg.max_probation_months:
                hits.append(
                    ComplianceFinding(
                        finding_id="PROBATION_EXCEEDS_3_MONTHS",
                        title_ar="فترة الاختبار تتجاوز الحد القانوني",
                        title_en="Probation exceeds legal maximum",
                        rationale=f"Probation set to {months} months (max is {self.cfg.max_probation_months}).",
                        confidence=0.82,
                        spans=[(base_offset + m.start(), base_offset + m.end())],
                    )
                )

        if self.re_repeat_probation.search(t):
            m = self.re_repeat_probation.search(t)
            hits.append(
                ComplianceFinding(
                    finding_id="PROBATION_REPEAT_OR_EXTENSION",
                    title_ar="تمديد/تكرار فترة الاختبار",
                    title_en="Probation repeated or extended",
                    rationale="Law prohibits placing a worker on probation more than once with the same employer.",
                    confidence=0.70,
                    spans=[(base_offset + m.start(), base_offset + m.end())] if m else [],
                )
            )
        return hits

    # -------------------------
    # Checks: Leave (annual + emergency)
    # -------------------------
    def _check_leave_rules(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        if self.re_waive_leave.search(t):
            m = self.re_waive_leave.search(t)
            hits.append(
                ComplianceFinding(
                    finding_id="ANNUAL_LEAVE_WAIVER",
                    title_ar="التنازل عن الإجازة السنوية",
                    title_en="Waiver of annual leave",
                    rationale="Law states employees may not waive annual leave entitlement.",
                    confidence=0.85,
                    spans=[(base_offset + m.start(), base_offset + m.end())] if m else [],
                )
            )

        # If annual leave is mentioned with a specific small number of days, flag "too low" (soft, because entitlement depends on service)
        if self.re_annual_leave.search(t):
            # look for nearby days number
            for m in self.re_leave_days.finditer(t):
                days = self._safe_int(m.group(1))
                if days is None:
                    continue
                # If a contract hard-codes e.g. 7/10/12 annual days -> suspicious vs legal minimum tiers
                if 0 < days < self.cfg.annual_leave_first_year_days:
                    hits.append(
                        ComplianceFinding(
                            finding_id="ANNUAL_LEAVE_TOO_LOW",
                            title_ar="عدد أيام الإجازة السنوية أقل من الحد الأدنى",
                            title_en="Annual leave days below legal minimum",
                            rationale=f"Clause suggests {days} annual leave days, below {self.cfg.annual_leave_first_year_days} (first-year entitlement).",
                            confidence=0.68,
                            spans=[(base_offset + m.start(), base_offset + m.end())],
                        )
                    )
                    break

        # Emergency leave: if contract denies it outright
        if self.re_emergency_leave.search(t):
            denial = re.search(r"(غير\s*مسموح|لا\s*يحق|no\s*emergency\s*leave|not\s*entitled)", t, re.IGNORECASE)
            if denial:
                hits.append(
                    ComplianceFinding(
                        finding_id="EMERGENCY_LEAVE_DENIED",
                        title_ar="إنكار الإجازة العارضة",
                        title_en="Emergency leave denied",
                        rationale="Law grants emergency leave up to 7 days/year (max 2 consecutive), deducted from annual leave.",
                        confidence=0.75,
                        spans=[(base_offset + denial.start(), base_offset + denial.end())],
                    )
                )

        return hits

    # -------------------------
    # Checks: Maternity protections
    # -------------------------
    def _check_maternity_rules(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        if self.re_maternity.search(t):
            # If contract states maternity leave shorter than 4 months => flag
            for m in self.re_months.finditer(t):
                months = self._safe_int(m.group(1))
                if months is None:
                    continue
                if 0 < months < 4:
                    hits.append(
                        ComplianceFinding(
                            finding_id="MATERNITY_LEAVE_TOO_SHORT",
                            title_ar="إجازة الوضع أقل من المقرر قانوناً",
                            title_en="Maternity leave shorter than legal entitlement",
                            rationale=f"Clause suggests {months} months maternity leave; law provides 4 months paid (with ≥45 days after delivery).",
                            confidence=0.78,
                            spans=[(base_offset + m.start(), base_offset + m.end())],
                        )
                    )
                    break

            # If contract requires overtime during pregnancy / after delivery => flag
            if self.re_no_overtime.search(t) is None:
                # We only flag if the contract explicitly imposes overtime obligations near maternity keywords
                around = re.search(r"(maternity|pregnan|حمل|وضع).{0,120}(overtime|عمل\s*إضافي|العمل\s*الإضافي)", t, re.IGNORECASE)
                if around:
                    hits.append(
                        ComplianceFinding(
                            finding_id="PREGNANCY_OVERTIME_RISK",
                            title_ar="خطر تكليف الحامل بعمل إضافي",
                            title_en="Risk of overtime during pregnancy",
                            rationale="Law prohibits requiring overtime during pregnancy and for up to six months after delivery.",
                            confidence=0.70,
                            spans=[(base_offset + around.start(), base_offset + around.end())],
                        )
                    )

        if self.re_dismiss_pregnant.search(t):
            m = self.re_dismiss_pregnant.search(t)
            hits.append(
                ComplianceFinding(
                    finding_id="MATERNITY_DISMISSAL_RISK",
                    title_ar="خطر فصل/إنهاء أثناء أو بعد إجازة الوضع",
                    title_en="Risk of dismissal during/after maternity leave",
                    rationale="Law prohibits dismissal during maternity leave and restricts termination after return unless legitimate reason is proven.",
                    confidence=0.72,
                    spans=[(base_offset + m.start(), base_offset + m.end())],
                )
            )

        if self.re_breastfeeding_breaks.search(t):
            # if breastfeeding mentioned but says unpaid / deducted => flag
            denial = re.search(r"(بدون\s*أجر|تخصم|deduct|unpaid)", t, re.IGNORECASE)
            if denial:
                hits.append(
                    ComplianceFinding(
                        finding_id="BREASTFEEDING_BREAKS_UNPAID",
                        title_ar="استراحات الرضاعة غير مدفوعة/تخصم",
                        title_en="Breastfeeding breaks unpaid/deducted",
                        rationale="Law treats breastfeeding breaks as working time with no pay reduction.",
                        confidence=0.70,
                        spans=[(base_offset + denial.start(), base_offset + denial.end())],
                    )
                )

        return hits

    # -------------------------
    # Checks: Wage deductions (damage / penalties)
    # -------------------------
    def _check_wage_deductions(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        # If clause says employer may deduct large number of days wages (e.g. 10 days) => suspicious vs Art 151 cap (5 days/month)
        if self.re_deduct_from_wage.search(t) and self.re_days_wage.search(t):
            for m in self.re_days_wage.finditer(t):
                days = self._safe_int(m.group(1))
                if days is None:
                    continue
                if days > self.cfg.damage_deduction_max_days_per_month:
                    hits.append(
                        ComplianceFinding(
                            finding_id="WAGE_DEDUCTION_EXCEEDS_MONTHLY_CAP",
                            title_ar="خصم من الأجر يتجاوز الحد الشهري",
                            title_en="Wage deduction exceeds monthly cap",
                            rationale=f"Clause mentions deduction of {days} days' wages; law limits deductions for damages to ≤{self.cfg.damage_deduction_max_days_per_month} days' wages per month under that mechanism.",
                            confidence=0.74,
                            spans=[(base_offset + m.start(), base_offset + m.end())],
                        )
                    )
                    break

        # If deductions for damage are stated but with no investigation / process mention, flag as unclear (low confidence)
        if self.re_damage_liability.search(t) and self.re_deduct_from_wage.search(t):
            process = re.search(r"(تحقيق|إخطار|المحكمة|investigation|notify|court)", t, re.IGNORECASE)
            if not process:
                m = self.re_deduct_from_wage.search(t)
                hits.append(
                    ComplianceFinding(
                        finding_id="WAGE_DEDUCTION_DUE_PROCESS_UNCLEAR",
                        title_ar="إجراءات خصم الأجر غير واضحة",
                        title_en="Due process for wage deduction unclear",
                        rationale="Law ties deductions for damage to investigation/notice and allows challenge before labor court; clause should not allow arbitrary deductions.",
                        confidence=0.58,
                        spans=[(base_offset + m.start(), base_offset + m.end())] if m else [],
                    )
                )

        return hits

    # -------------------------
    # Checks: Termination / dismissal structure
    # -------------------------
    def _check_termination_dismissal_structure(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        hits: List[ComplianceFinding] = []

        # If contract claims employer can terminate at any time at sole discretion (without process)
        if self.re_immediate_termination_anytime.search(t):
            m = self.re_immediate_termination_anytime.search(t)
            conf = 0.60
            rationale = "Clause suggests unilateral termination at any time; dismissal is regulated and, for disciplinary dismissal, only labor court imposes dismissal penalty."
            if self.re_no_court.search(t):
                conf = 0.72
                rationale = "Clause suggests unilateral termination without court/process; disciplinary dismissal penalty is under labor court jurisdiction."
            hits.append(
                ComplianceFinding(
                    finding_id="TERMINATION_UNILATERAL_RISK",
                    title_ar="خطر الإنهاء الانفرادي دون ضوابط",
                    title_en="Risk of unilateral termination without safeguards",
                    rationale=rationale,
                    confidence=conf,
                    spans=[(base_offset + m.start(), base_offset + m.end())],
                )
            )

        return hits

    # -------------------------
    # Checks: General waivers
    # -------------------------
    def _check_general_waivers(self, t: str, *, base_offset: int) -> List[ComplianceFinding]:
        if self.re_general_waiver.search(t):
            m = self.re_general_waiver.search(t)
            return [
                ComplianceFinding(
                    finding_id="GENERAL_RIGHTS_WAIVER",
                    title_ar="تنازل عام عن حقوق العامل",
                    title_en="General waiver of employee rights",
                    rationale="Broad waivers often conflict with non-waivable statutory rights (leave, overtime, etc.).",
                    confidence=0.80,
                    spans=[(base_offset + m.start(), base_offset + m.end())],
                )
            ]
        return []

    # -------------------------
    # Helpers
    # -------------------------
    def _deduplicate(self, findings: List[ComplianceFinding]) -> List[ComplianceFinding]:
        best: Dict[str, ComplianceFinding] = {}
        for f in findings:
            prev = best.get(f.finding_id)
            if prev is None or f.confidence > prev.confidence:
                best[f.finding_id] = f
        return sorted(best.values(), key=lambda x: (-x.confidence, x.finding_id))

    @staticmethod
    def _safe_int(x: Optional[str]) -> Optional[int]:
        try:
            if x is None:
                return None
            return int(str(x).strip())
        except Exception:
            return None

    @staticmethod
    def _first_int_group(groups: Tuple[Optional[str], ...]) -> Optional[int]:
        for g in groups:
            try:
                if g is None:
                    continue
                s = str(g).strip()
                if not s:
                    continue
                return int(s)
            except Exception:
                continue
        return None
