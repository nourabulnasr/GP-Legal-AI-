from __future__ import annotations
import re
from typing import Dict, Any

# Very pragmatic keyword-based detector (good enough for MVP milestone)
CROSS_BORDER_PATTERNS = [
    r"الإمارات|دولة الإمارات|الامارات|UAE|United Arab Emirates",
    r"\bMOHRE\b|وزارة الموارد البشرية والتوطين|التوطين",
    r"قانون العمل الإماراتي|المرسوم بقانون اتحادي|اتحادي رقم\s*\(?33\)?\s*لسنة\s*2021",
    r"داخل أراضي دولة الإمارات|في الإمارات|المكان في الإمارات",
    r"محاكم الإمارات|المحاكم الإماراتية",
    r"درهم|AED",
    r"خارج مصر|outside egypt|work abroad|overseas",
]

def detect_cross_border(text: str) -> Dict[str, Any]:
    """
    Returns:
      { enabled: bool, reason: str, matches: list[str] }
    """
    if not text or not text.strip():
        return {"enabled": False, "reason": "empty_text", "matches": []}

    hits = []
    for pat in CROSS_BORDER_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)

    enabled = len(hits) > 0
    return {
        "enabled": enabled,
        "reason": "keyword_hits" if enabled else "no_hits",
        "matches": hits[:6],  # keep it short
    }
