# app/utils_text.py
# helper text normalization + small utilities used across the app

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# Arabic diacritics range
_AR_DIACRITICS = r"[\u064B-\u065F\u0670\u06D6-\u06ED]"

def has_arabic(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return re.search(r"[\u0600-\u06FF]", s) is not None


def _latin_letter_count(s: str) -> int:
    n = 0
    for ch in s:
        if ch.isascii() and ch.isalpha():
            n += 1
    return n


def _arabic_letter_count(s: str) -> int:
    return sum(1 for ch in s if "\u0600" <= ch <= "\u06FF")


def _cyrillic_letter_count(s: str) -> int:
    return sum(1 for ch in s if "\u0400" <= ch <= "\u04FF")


def _cjk_letter_count(s: str) -> int:
    """Han / Hangul / Kana (excludes pure punctuation)."""
    n = 0
    for ch in s:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7AF
        ):
            n += 1
    return n


def _hangul_syllable_count(s: str) -> int:
    return sum(1 for ch in s if "\uAC00" <= ch <= "\uD7AF")


def _kana_count(s: str) -> int:
    return sum(1 for ch in s if "\u3040" <= ch <= "\u30FF")


_ft_lid_model: Any = None  # fasttext model or False if unavailable


def _get_fasttext_lid_model() -> Any:
    """Optional `lid.176.ftz` (or `.bin`) path via FASTTEXT_LID_MODEL; improves noisy Latin LID."""
    global _ft_lid_model
    if _ft_lid_model is False:
        return None
    if _ft_lid_model is not None:
        return _ft_lid_model
    path = (os.getenv("FASTTEXT_LID_MODEL") or "").strip()
    if not path:
        _ft_lid_model = False
        return None
    try:
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.is_file():
            _ft_lid_model = False
            return None
        import fasttext  # type: ignore

        _ft_lid_model = fasttext.load_model(str(p))
    except Exception:
        _ft_lid_model = False
        return None
    return _ft_lid_model


def _euro_latin_lang_hint(s: str) -> Optional[str]:
    """
    When `langdetect` over-predicts English on EU contract boilerplate, nudge toward fr/de
    using high-precision function words (conservative thresholds).
    """
    low = s.lower()
    fr_n = len(
        re.findall(
            r"\b(le|la|les|des|une|un|est|sont|contrat|travail|présent|durée|période|conformément|"
            r"salarié|employeur|préavis|indéterminée)\b",
            low,
        )
    )
    de_n = len(
        re.findall(
            r"\b(der|die|das|und|ist|nicht|ein|eine|arbeitsvertrag|kündigungsfrist|probezeit|für|mit|dieser|"
            r"arbeitnehmer|arbeitgeber|vergütung|monatlich)\b",
            low,
        )
    )
    en_n = len(
        re.findall(
            r"\b(the|and|is|are|this|employment|contract|salary|agreement|employee|employer|notice)\b",
            low,
        )
    )
    if fr_n >= 4 and fr_n >= max(de_n, en_n) + 2:
        return "fr"
    if de_n >= 4 and de_n >= max(fr_n, en_n) + 2:
        return "de"
    return None


def _fasttext_predict_primary(text: str) -> Optional[Tuple[str, float]]:
    m = _get_fasttext_lid_model()
    if m is None:
        return None
    try:
        t = re.sub(r"\s+", " ", (text or "").strip())[:5000]
        if len(t) < 3:
            return None
        labels, probs = m.predict(t.replace("\n", " "), k=1)
        if not labels or not probs:
            return None
        lab = str(labels[0])
        if lab.startswith("__label__"):
            lab = lab[9:]
        code = lab.lower().split("-")[0][:8]
        if len(code) == 3 and code in ("ara", "eng", "deu", "fra", "spa"):
            iso3_to2 = {"ara": "ar", "eng": "en", "deu": "de", "fra": "fr", "spa": "es"}
            code = iso3_to2.get(code, code[:2])
        if len(code) < 2:
            return None
        return code[:2], float(probs[0])
    except Exception:
        return None


def _fuse_fasttext_with_candidate(
    compact: str,
    *,
    ld_code: str,
    ld_prob: float,
    ar_l: int,
    lat_l: int,
    is_mixed: bool,
) -> Tuple[str, float]:
    """Prefer fastText when it disagrees with langdetect on Latin legal boilerplate often mis-tagged as English."""
    ft = _fasttext_predict_primary(compact)
    if ft is None:
        return ld_code, ld_prob
    ft_code, ft_prob = ft
    if ft_code == ld_code:
        return ld_code, min(0.99, max(ld_prob, ft_prob * 0.95))
    # Strong Arabic script wins over fastText on Latin noise inside Arabic pages
    if ar_l > 0 and lat_l > 0 and ar_l >= lat_l * 2:
        return "ar", max(ld_prob, 0.85)
    if lat_l == 0 or ar_l > lat_l * 3:
        return ld_code, ld_prob
    if ft_prob >= 0.55 and ld_code == "en" and ft_code != "en" and ld_prob < 0.82:
        return ft_code, min(0.94, max(ft_prob, ld_prob))
    if ft_prob >= 0.72 and ld_code != ft_code and len(compact) >= 40:
        return ft_code, min(0.93, ft_prob)
    return ld_code, ld_prob


@dataclass(frozen=True)
class LanguageDetectionResult:
    """BCP-47 primary language subtag (ISO 639-1) plus confidence and mixed-script hint."""

    language_code: str
    confidence: float
    is_mixed: bool


def detect_language_detailed(s: str, *, min_chars: int = 20) -> LanguageDetectionResult:
    """
    Offline language identification (`langdetect`) with script fallbacks and optional
    fastText LID (`FASTTEXT_LID_MODEL`) fusion for noisy OCR / legal boilerplate.
    Uses ISO 639-1 primary subtags: ar, en, fr, de, ... or 'und' when unknown.
    """
    if not isinstance(s, str) or not s.strip():
        return LanguageDetectionResult(language_code="und", confidence=0.0, is_mixed=False)

    compact = re.sub(r"\s+", " ", s.strip())
    ar_l = _arabic_letter_count(compact)
    lat_l = _latin_letter_count(compact)
    cy_l = _cyrillic_letter_count(compact)
    cjk_l = _cjk_letter_count(compact)
    hang_l = _hangul_syllable_count(compact)
    kn_l = _kana_count(compact)
    alpha_lat_cy = ar_l + lat_l + cy_l + cjk_l
    is_mixed = (
        alpha_lat_cy > 0
        and ar_l > 0
        and (lat_l + cy_l + cjk_l) > 0
        and min(ar_l, lat_l + cy_l + cjk_l) / float(alpha_lat_cy) >= 0.12
    )

    # Strong Arabic signal (short boilerplate / headers)
    if ar_l > 0 and lat_l == 0 and cy_l == 0 and cjk_l == 0:
        return LanguageDetectionResult(language_code="ar", confidence=0.99, is_mixed=False)
    if ar_l > 0 and lat_l > 0 and ar_l >= lat_l * 2 and cy_l + cjk_l < lat_l:
        return LanguageDetectionResult(language_code="ar", confidence=0.85, is_mixed=is_mixed)

    # Cyrillic-dominant (no Arabic): map to Russian for rules/MT routing
    if cy_l >= 8 and cy_l >= lat_l * 2 and ar_l == 0:
        ft = _fasttext_predict_primary(compact)
        if ft and ft[0] in ("uk", "bg", "sr") and ft[1] >= 0.55:
            return LanguageDetectionResult(language_code=ft[0], confidence=round(ft[1], 4), is_mixed=is_mixed)
        return LanguageDetectionResult(language_code="ru", confidence=0.88, is_mixed=is_mixed)

    # CJK-dominant
    if cjk_l >= 8 and cjk_l >= lat_l * 2 and ar_l == 0:
        if hang_l >= kn_l * 1.5 and hang_l >= cjk_l * 0.35:
            return LanguageDetectionResult(language_code="ko", confidence=0.86, is_mixed=is_mixed)
        if kn_l >= 6 and kn_l >= hang_l:
            return LanguageDetectionResult(language_code="ja", confidence=0.86, is_mixed=is_mixed)
        return LanguageDetectionResult(language_code="zh", confidence=0.84, is_mixed=is_mixed)

    if len(compact) < min_chars:
        if has_arabic(compact):
            return LanguageDetectionResult(language_code="ar", confidence=0.55, is_mixed=is_mixed)
        ft = _fasttext_predict_primary(compact)
        if ft and ft[1] >= 0.45 and ft[0] not in ("en", "und"):
            return LanguageDetectionResult(language_code=ft[0], confidence=round(ft[1], 4), is_mixed=is_mixed)
        return LanguageDetectionResult(language_code="en", confidence=0.5, is_mixed=is_mixed)

    try:
        from langdetect import detect_langs

        langs = detect_langs(compact)
        if langs:
            top = langs[0]
            code = str(top.lang)
            prob = float(top.prob)
            if code == "ar" and lat_l > ar_l * 3:
                code = "en"
                prob = min(prob, 0.75)
            elif code != "ar" and ar_l > lat_l * 3 and has_arabic(compact):
                code = "ar"
                prob = min(0.95, prob + 0.05)
            euro = _euro_latin_lang_hint(compact)
            if euro and code == "en" and len(compact) >= 30:
                code = euro
                prob = max(prob, 0.72)
            if prob < 0.2 and len(compact) < 80:
                if has_arabic(compact):
                    return LanguageDetectionResult(language_code="ar", confidence=round(prob, 4), is_mixed=is_mixed)
                return LanguageDetectionResult(language_code="en", confidence=round(prob, 4), is_mixed=is_mixed)
            code, prob = _fuse_fasttext_with_candidate(
                compact, ld_code=code, ld_prob=prob, ar_l=ar_l, lat_l=lat_l, is_mixed=is_mixed
            )
            return LanguageDetectionResult(language_code=code, confidence=round(prob, 4), is_mixed=is_mixed)
    except Exception:
        pass

    ft2 = _fasttext_predict_primary(compact)
    if ft2 and ft2[1] >= 0.35:
        return LanguageDetectionResult(language_code=ft2[0], confidence=round(ft2[1], 4), is_mixed=is_mixed)

    euro2 = _euro_latin_lang_hint(compact)
    if euro2 and lat_l >= 12:
        return LanguageDetectionResult(language_code=euro2, confidence=0.72, is_mixed=is_mixed)

    if has_arabic(compact):
        return LanguageDetectionResult(language_code="ar", confidence=0.55, is_mixed=is_mixed)
    return LanguageDetectionResult(language_code="en", confidence=0.5, is_mixed=is_mixed)


def detect_language(s: str) -> str:
    """
    Primary document language as ISO 639-1 code (e.g. ar, en, fr).
    Backwards compatible: Arabic script still maps to ar; European Latin maps to detected language when Lingua works.
    """
    return detect_language_detailed(s).language_code


def llm_locale_from_detection(language_code: str) -> str:
    """
    Map detection to LFM prompts (Arabic vs English only).
    English stays English; Arabic stays Arabic; other ISO codes use Arabic prompts so
    clause explain/compare aligns with Arabic-first legal UX (source text may remain FR/DE/…).
    """
    lc = (language_code or "").strip().lower().split("-")[0]
    if lc.startswith("ar"):
        return "ar"
    if lc in ("", "und"):
        return "en"
    if lc.startswith("en"):
        return "en"
    return "ar"


def annotate_ocr_chunks_language(ocr_chunks: Optional[List[Any]]) -> None:
    """
    Sets `detected_lang` and `detected_lang_confidence` on each chunk dict (mutates in place).
    Call after `normalized_text` is populated.
    """
    if not ocr_chunks:
        return
    for c in ocr_chunks:
        if not isinstance(c, dict):
            continue
        t = (c.get("normalized_text") or c.get("text") or "").strip()
        if len(t) < 8:
            c["detected_lang"] = "und"
            c["detected_lang_confidence"] = 0.0
            continue
        sample = t[:3500]
        det = detect_language_detailed(sample)
        c["detected_lang"] = det.language_code
        c["detected_lang_confidence"] = det.confidence


# Subset of BCP 47: language[-subtag]* (subtags alphanumeric, 1–8 chars). Underscores normalized to hyphens.
_BCP47_LOOSE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


def is_valid_bcp47_primary_override(code: str) -> bool:
    """True if `code` looks like a safe BCP-47 language tag for override (e.g. fr, de, en, ar-EG, zh-Hans)."""
    if not isinstance(code, str) or not code.strip():
        return False
    s = code.strip().replace("_", "-")
    if len(s) > 32:
        return False
    return bool(_BCP47_LOOSE_RE.match(s))


def detect_language_for_document(
    full_text: str,
    ocr_chunks: Optional[List[Any]] = None,
) -> LanguageDetectionResult:
    """
    Document-level detection with per-chunk hints for long/multi-page PDFs and mixed languages.
    Refines `is_mixed` when different chunks yield different ISO primary codes (e.g. en + ar).
    Primary `language_code` stays from full-text aggregation (stable for rules/RAG).
    Prefer `annotate_ocr_chunks_language` so chunk `detected_lang` fields are reused.
    """
    base = detect_language_detailed(full_text)
    if not ocr_chunks or len(ocr_chunks) < 2:
        return base

    codes: List[str] = []
    for c in ocr_chunks:
        if not isinstance(c, dict):
            continue
        pre = c.get("detected_lang")
        if isinstance(pre, str) and pre.strip():
            if len(pre.strip()) >= 2:
                codes.append(pre.strip().lower().split("-")[0])
                continue
        t = (c.get("normalized_text") or c.get("text") or "").strip()
        if len(t) < 50:
            continue
        sample = t[:3500]
        codes.append(detect_language_detailed(sample).language_code)

    uniq = {x for x in codes if x and x != "und"}
    is_mixed = base.is_mixed
    if len(uniq) >= 2:
        is_mixed = True

    return LanguageDetectionResult(
        language_code=base.language_code,
        confidence=base.confidence,
        is_mixed=is_mixed or base.is_mixed,
    )


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
