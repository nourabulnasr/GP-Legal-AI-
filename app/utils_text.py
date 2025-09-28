import re

_AR_DIACRITICS = r"[\u064B-\u065F\u0670\u06D6-\u06ED]"

def has_arabic(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return re.search(r"[\u0600-\u06FF]", s) is not None

def detect_language(s: str) -> str:
    return "ar" if has_arabic(s) else "en"

def norm_ar(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(_AR_DIACRITICS, "", s)
    s = s.replace("\u0640", "")  # tatweel
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_ar(s: str) -> str:
    """Back-compat alias for older imports."""
    return norm_ar(s)
