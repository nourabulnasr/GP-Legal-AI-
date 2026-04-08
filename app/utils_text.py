# app/utils_text.py
# helper text normalization + small utilities used across the app

import re
import unicodedata
from typing import List, Optional, Tuple

# Arabic diacritics range
_AR_DIACRITICS = r"[\u064B-\u065F\u0670\u06D6-\u06ED]"

def has_arabic(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return re.search(r"[\u0600-\u06FF]", s) is not None

def detect_language(s: str) -> str:
    """Simple detector: returns 'ar' if the text contains Arabic chars, else 'en'."""
    return "ar" if has_arabic(s) else "en"

def norm_ar(s: str) -> str:
    """
    Basic Arabic normalizer:
      - remove Tashkeel/diacritics
      - remove tatweel
      - normalize whitespace
    Keeps text safe for regex rules and simple matching.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(_AR_DIACRITICS, "", s)
    s = s.replace("\u0640", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_ar(s: str) -> str:
    """Backwards-compatible alias used in other modules."""
    return norm_ar(s)

# ---------- extra normalization and helpers (recommended) ----------
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
WESTERN =    "0123456789"

DIACRITICS_RE = re.compile(
    "["
    "\u0610-\u061A"  # Arabic signs
    "\u064B-\u065F"  # harakat
    "\u0670"
    "\u06D6-\u06ED"
    "]"
)

def convert_arabic_indic_digits(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.translate(str.maketrans(ARABIC_INDIC, WESTERN))

def strip_diacritics(s: str) -> str:
    if not isinstance(s, str):
        return s
    return DIACRITICS_RE.sub("", s)

def normalize_ar_text(s: str) -> str:
    """
    Stronger Arabic normalization used before rule matching:
      - convert Arabic-Indic digits to western
      - remove diacritics, normalize alef/hamza, collapse whitespace
    """
    if not isinstance(s, str) or not s:
        return "" if s is None else s

    s = unicodedata.normalize("NFKC", s)

    # Keep dots placeholders, but remove weird long punctuation sequences
    # (Important: do NOT delete '.' because you detect placeholders with dots)
    s = re.sub(r"[^\w\u0600-\u06FF\s\.\-ـ·]{3,}", " ", s)

    s = convert_arabic_indic_digits(s)
    s = strip_diacritics(s)

    # normalize alef forms
    s = re.sub("[إأآا]", "ا", s)
    s = s.replace("ى", "ي")

    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Arabic number words -> digits ----------
_NUM_WORDS_MAP = {
    "اربعه": "4", "أربعه": "4", "اربعة": "4", "أربعة": "4",
    "خمسه": "5", "خمسة": "5",
    "سته": "6", "ستة": "6",
    "سبعه": "7", "سبعة": "7",
    "ثمانيه": "8", "ثمانية": "8", "ثمان": "8",
    "تسعه": "9", "تسعة": "9",
    "عشر": "10", "عشرة": "10",
}

def normalize_ar_numbers_words(s: str) -> str:
    """
    Converts common Arabic number-words to digits to help rule matching.
    Example: "خمسة أشهر" -> "5 أشهر"
    Safe: only replaces standalone words, keeps original text structure.
    """
    if not isinstance(s, str) or not s:
        return "" if s is None else s

    # Light normalization for matching the words
    t = unicodedata.normalize("NFKC", s)
    t = strip_diacritics(t)
    t = re.sub("[إأآا]", "ا", t)

    def repl(m):
        w = m.group(0)
        return _NUM_WORDS_MAP.get(w, w)

    pattern = r"\b(" + "|".join(re.escape(k) for k in _NUM_WORDS_MAP.keys()) + r")\b"
    t = re.sub(pattern, repl, t)
    return t

# ---------- small word->number fallback ----------
def word_to_num(text):
    """
    Minimal fallback:
    - convert Arabic-Indic digits
    - find the first number-like token and return int/float
    - if nothing convertible, returns the original input
    """
    if text is None:
        return text
    s = str(text).strip()
    s = convert_arabic_indic_digits(s)

    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    if m:
        num_str = m.group(0).replace(",", ".")
        try:
            if "." in num_str:
                return float(num_str)
            return int(num_str)
        except Exception:
            return num_str

    return text

# ============================================================
# NEW: Unified pipeline for rule matching (THIS is what you asked)
# ============================================================
def normalize_for_rules(s: str) -> str:
    """
    One function you call before RuleEngine.check_text().
    This applies:
      1) strong Arabic normalization (digits + diacritics + alef + whitespace)
      2) convert number-words to digits (خمسة -> 5)
    """
    s = normalize_ar_text(s)
    s = normalize_ar_numbers_words(s)
    return s


# ---------- Clause splitting (bilingual: Arabic + English) ----------
def split_into_clauses(text: str, min_clause_len: int = 15) -> List[Tuple[int, int, str]]:
    """
    Split contract text into clauses with (start, end, text) spans.
    Heuristics: numbering (1. 2. أولاً ثانياً), bullets, Arabic punctuation,
    headings (البند الأول / Article 1), newline blocks.
    Returns list of (start, end, clause_text) for linking rule hits to clauses.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    split_positions = [0]

    # Arabic clause headers: البند الأول، البند الثاني، المادة الأولى
    for m in re.finditer(
        r"\n\s*(?:البند\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+)|"
        r"المادة\s+(?:الأولى|الثانية|الثالثة|\d+))",
        text,
    ):
        split_positions.append(m.start())

    # English: Article 1, Section 2, Clause 3
    for m in re.finditer(r"\n\s*(?:Article|Section|Clause)\s*\d+\s*[:\-]?", text, re.I):
        if m.start() not in split_positions:
            split_positions.append(m.start())

    # Numbered list: 1. 2. 3. or 1) 2)
    for m in re.finditer(r"\n\s*\d+[\.\)]\s+", text):
        if m.start() not in split_positions:
            split_positions.append(m.start())

    # Arabic ordinals: أولاً ثانياً
    for m in re.finditer(r"\n\s*(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً)\s*[:\-]?", text):
        if m.start() not in split_positions:
            split_positions.append(m.start())

    # Bullets
    for m in re.finditer(r"\n\s*[\-\•\*]\s+", text):
        if m.start() not in split_positions:
            split_positions.append(m.start())

    # Double newlines (paragraph break)
    for m in re.finditer(r"\n{2,}", text):
        if m.start() not in split_positions:
            split_positions.append(m.start())

    split_positions = sorted(set(split_positions))
    split_positions.append(len(text))

    spans: List[Tuple[int, int, str]] = []
    for i in range(len(split_positions) - 1):
        start, end = split_positions[i], split_positions[i + 1]
        clause = text[start:end].strip()
        if len(clause) >= min_clause_len:
            spans.append((start, end, clause))

    if not spans and text.strip():
        spans.append((0, len(text), text.strip()))
    return spans
