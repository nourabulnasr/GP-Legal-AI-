from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = Path("laws/preprocessed/labor14_2025_clean.jsonl")
DEBUG_SAMPLE_PATH = Path("laws/preprocessed/labor14_2025_debug_sample.txt")

DEBUG_MARKERS_TSV = Path("laws/preprocessed/labor14_2025_markers_debug.tsv")
DEBUG_MISSING_TXT = Path("laws/preprocessed/labor14_2025_missing_report.txt")

LAW_ID = "labor14_2025"
LAW_NAME = "قانون العمل رقم 14 لسنة 2025"

EXPECTED_MAX_ARTICLE = int(os.environ.get("EXPECTED_MAX_ARTICLE", "298"))
MIN_MARKERS_EXPECTED = int(os.environ.get("MIN_MARKERS_EXPECTED", "200"))

SHOW_PROGRESS = True
STRICT_START = True


# ============================================================
# NORMALIZATION
# ============================================================

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EASTERN_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_ar(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_TATWEEL, "")
    t = _ARABIC_DIACRITICS.sub("", t)
    t = t.translate(_ARABIC_INDIC).translate(_EASTERN_INDIC)
    t = t.replace("\u00a0", " ")
    t = t.replace("ـ", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def normalize_for_match(text: str) -> str:
    t = normalize_ar(text).lower()
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fix_spaced_ocr_keywords(line: str) -> str:
    s = line
    s = re.sub(r"م\s*ا\s*د\s*ة", "مادة", s)
    s = re.sub(r"ا\s*ل\s*م\s*ا\s*د\s*ة", "المادة", s)
    s = re.sub(r"\bماده\b", "مادة", s)

    # reversed parentheses: مادة)1( -> مادة(1)
    s = re.sub(r"(المادة|مادة)\s*\)\s*([0-9٠-٩]{1,4})\s*\(", r"\1(\2)", s)
    s = re.sub(r"(المادة|مادة)\s*\(\s*([0-9٠-٩]{1,4})\s*\)", r"\1(\2)", s)
    return s


def norm_heading_line(s: str) -> str:
    s = normalize_ar(s)
    s = fix_spaced_ocr_keywords(s)
    s = s.translate(_ARABIC_INDIC).translate(_EASTERN_INDIC)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# NOISE REMOVAL (TXT version)
# ============================================================

PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")
GAZETTE_STRONG_RE = re.compile(r"(الجريدة\s*الرسمية)", re.IGNORECASE)
GAZETTE_META_RE = re.compile(r"(العدد|تابع)", re.IGNORECASE)
MONTHS_HINT_RE = re.compile(
    r"(يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر)",
    re.IGNORECASE,
)


def is_noise_line(line: str) -> bool:
    l = normalize_ar(line).strip()
    if not l:
        return False
    if PAGE_NUM_ONLY_RE.match(l):
        return True
    if GAZETTE_STRONG_RE.search(l):
        return True
    if GAZETTE_META_RE.search(l) and (MONTHS_HINT_RE.search(l) or re.search(r"\b20\d{2}\b", l)):
        return True
    return False


def clean_txt_text(txt: str) -> str:
    """
    TXT equivalent of clean_page_text:
    - normalize
    - fix spaced OCR keywords
    - remove safe noise lines
    """
    t = normalize_ar(txt)
    lines = t.splitlines()
    cleaned: List[str] = []
    for ln in lines:
        ln2 = fix_spaced_ocr_keywords(ln).rstrip()
        if not is_noise_line(ln2):
            cleaned.append(ln2)
    # keep structure
    return "\n".join(cleaned).strip()


# ============================================================
# START TRIGGER (same logic)
# ============================================================

def find_start_offset(full_text: str) -> Tuple[bool, int, str]:
    lines = full_text.splitlines(True)
    norm_lines = [normalize_for_match(ln) for ln in lines]

    for i in range(len(lines)):
        if "قانون العمل" in norm_lines[i]:
            for j in range(i, min(i + 50, len(lines))):
                if "الكتاب الاول" in norm_lines[j] or "الكتاب 1" in norm_lines[j]:
                    for k in range(j, min(j + 40, len(lines))):
                        if "الباب الاول" in norm_lines[k] or "الباب 1" in norm_lines[k]:
                            off = sum(len(lines[x]) for x in range(k + 1))
                            return True, off, "قانون العمل → الكتاب الأول → الباب الأول"

    for i in range(len(lines)):
        if "الباب الاول" in norm_lines[i] or "الباب 1" in norm_lines[i]:
            for j in range(i, min(i + 12, len(lines))):
                if "التعاريف" in norm_lines[j]:
                    off = sum(len(lines[x]) for x in range(j + 1))
                    return True, off, "الباب الأول + التعاريف"

    return False, 0, "not found"


# ============================================================
# MARKERS (same logic)
# ============================================================

INLINE_REF_HINT_RE = re.compile(
    r"(من\s+هذا\s+القانون|المشار\s+اليه|المشار\s+إليه|وفق|وفقا|وفقاً|عملا\s+ب|طبقاً|طبقا|مع\s+مراعاة)",
    re.IGNORECASE,
)

ARTICLE_GLOBAL_RE = re.compile(
    r"""(?m)
    ^[^\S\r\n]{0,30}
    [\(\[\{]*\s*[:：\-–—]*\s*
    (?:المادة|مادة)\b
    (?:\s+رقم)?\s*
    (?:[:：\-–—]*\s*)?
    (?:\(\s*)?
    (?P<num>[0-9٠-٩]{1,4})
    (?:\s*\))?
    """,
    re.VERBOSE,
)


@dataclass
class Marker:
    offset: int
    article: int
    heading_line: str
    page: int  # TXT has no pages; we keep 1


def dedup_markers(markers: List[Marker]) -> List[Marker]:
    markers.sort(key=lambda x: x.offset)
    out: List[Marker] = []
    for m in markers:
        if not out:
            out.append(m)
            continue
        last = out[-1]
        if m.article == last.article and (m.offset - last.offset) < 800:
            continue
        if (m.offset - last.offset) < 5:
            continue
        out.append(m)
    return out


def find_article_markers_global(full_text: str, start_offset: int) -> List[Marker]:
    markers: List[Marker] = []
    text = full_text[start_offset:]
    base = start_offset

    for m in ARTICLE_GLOBAL_RE.finditer(text):
        raw_num = m.group("num").translate(_ARABIC_INDIC).translate(_EASTERN_INDIC)
        try:
            num = int(raw_num)
        except Exception:
            continue

        off = base + m.start()
        line_end = full_text.find("\n", off)
        if line_end == -1:
            line_end = min(len(full_text), off + 250)
        heading_line = norm_heading_line(full_text[off:line_end])

        if INLINE_REF_HINT_RE.search(heading_line):
            continue

        markers.append(
            Marker(
                offset=off,
                article=num,
                heading_line=heading_line,
                page=1,  # TXT: no page spans
            )
        )

    return dedup_markers(markers)


def report_missing(markers: List[Marker], expected_max: int) -> str:
    found_set = {m.article for m in markers if 1 <= m.article <= expected_max}
    missing = [i for i in range(1, expected_max + 1) if i not in found_set]
    found = sorted(found_set)

    lines = []
    lines.append(f"Expected max article: {expected_max}")
    lines.append(f"Unique articles found in 1..{expected_max}: {len(found)}")
    lines.append(f"Total markers (after dedup): {len(markers)}")
    lines.append(f"Missing count: {len(missing)}")
    if missing:
        lines.append("First 120 missing: " + ", ".join(map(str, missing[:120])))
    return "\n".join(lines)


def write_markers_debug(markers: List[Marker]) -> None:
    DEBUG_MARKERS_TSV.parent.mkdir(parents=True, exist_ok=True)
    lines = ["idx\tarticle\tpage\toffset\theading"]
    for i, m in enumerate(markers, start=1):
        lines.append(f"{i}\t{m.article}\t{m.page}\t{m.offset}\t{m.heading_line}")
    DEBUG_MARKERS_TSV.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# PARSE CLAUSES (reuse)
# ============================================================

ENUM_ITEM_RE = re.compile(r"^\s*(\d{1,3})\s*[-–—]\s*(.+)\s*$")
SUB_ITEM_RE = re.compile(r"^\s*(?:\(?\s*([اأإآء-ي])\s*\)?\s*[-–—\)]\s*)(.+)\s*$")


def parse_article_clauses(article_text: str) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in normalize_ar(article_text).splitlines() if ln.strip()]
    clauses: List[Dict[str, Any]] = []
    current_enum: Optional[Dict[str, Any]] = None
    current_item: Optional[Dict[str, Any]] = None

    def flush_enum() -> None:
        nonlocal current_enum, current_item
        if current_enum is not None:
            clauses.append(current_enum)
        current_enum = None
        current_item = None

    for ln in lines:
        m = ENUM_ITEM_RE.match(ln)
        if m:
            if current_enum is None:
                current_enum = {"type": "enumeration", "items": []}
            current_item = {"marker": m.group(1), "text": m.group(2).strip()}
            current_enum["items"].append(current_item)
            continue

        m2 = SUB_ITEM_RE.match(ln)
        if m2 and current_enum is not None and current_item is not None:
            subs = current_item.get("subitems")
            if subs is None:
                subs = []
                current_item["subitems"] = subs
            subs.append({"marker": m2.group(1), "text": m2.group(2).strip()})
            continue

        if current_enum is not None and current_item is not None:
            current_item["text"] = (current_item["text"] + "\n" + ln).strip()
            continue

        flush_enum()
        clauses.append({"type": "paragraph", "text": ln})

    flush_enum()

    merged: List[Dict[str, Any]] = []
    for c in clauses:
        if c["type"] == "paragraph" and merged and merged[-1]["type"] == "paragraph":
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + c["text"]).strip()
        else:
            merged.append(c)
    return merged


# ============================================================
# MAIN TXT PREPROCESS
# ============================================================

def write_debug_sample(full_text: str) -> None:
    DEBUG_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_SAMPLE_PATH.write_text(full_text[:250000], encoding="utf-8")


def preprocess_from_txt(txt_path: Path) -> None:
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT not found: {txt_path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 Reading TXT: {txt_path}")
    raw = txt_path.read_text(encoding="utf-8", errors="replace")

    print("🧹 Cleaning + normalization...")
    full = clean_txt_text(raw)

    if len(full) < 500:
        write_debug_sample(full)
        raise RuntimeError(f"TXT too small after cleaning. Debug: {DEBUG_SAMPLE_PATH}")

    # Start offset (optional but recommended)
    found, start_offset, msg = find_start_offset(full)
    if STRICT_START and found:
        print(f"🎯 Start found ({msg}) → start_offset={start_offset}")
    else:
        if STRICT_START:
            print("⚠️ Start trigger not found → start_offset=0")
        start_offset = 0

    print("🔎 Detecting article headings (GLOBAL scan)...")
    markers = find_article_markers_global(full, start_offset=start_offset)

    if len(markers) < MIN_MARKERS_EXPECTED:
        write_debug_sample(full)
        print(f"⚠️ Markers low ({len(markers)} < {MIN_MARKERS_EXPECTED}).")
        print(f"   Debug sample: {DEBUG_SAMPLE_PATH}")

    if not markers:
        write_debug_sample(full)
        raise RuntimeError(
            "❌ No article headings detected.\n"
            f"Debug sample: {DEBUG_SAMPLE_PATH}\n"
            "Check whether headings in TXT look like: المادة (7) / مادة(7) / المادة 7"
        )

    write_markers_debug(markers)

    missing_report = report_missing(markers, expected_max=EXPECTED_MAX_ARTICLE)
    DEBUG_MISSING_TXT.write_text(missing_report, encoding="utf-8")

    print(f"✅ Article markers detected: {len(markers)}")
    print(f"🧾 Markers TSV: {DEBUG_MARKERS_TSV}")
    print(f"🧾 Missing report: {DEBUG_MISSING_TXT}")
    print(missing_report)

    print(f"✍️ Writing JSONL to: {OUTPUT_PATH}")

    written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for idx, mk in enumerate(markers):
            start = mk.offset
            end = markers[idx + 1].offset if idx + 1 < len(markers) else len(full)
            block = full[start:end].strip()

            if len(block) < 80:
                continue

            rec: Dict[str, Any] = {
                "law_id": LAW_ID,
                "law_name": LAW_NAME,
                "article": f"{mk.article:03d}",
                "heading": mk.heading_line,
                "page": mk.page,
                "text": block,
                "clauses": parse_article_clauses(block),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            if SHOW_PROGRESS and (written % 25 == 0):
                print(f"   ...written {written} articles")

    print("✅ Done.")
    print(f"📊 Records written: {written}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python -m app.preprocess_labor_from_txt <TXT_PATH>\n\n"
            "Example:\n"
            "  python -m app.preprocess_labor_from_txt laws\\raw\\labor14_2025.txt\n"
        )

    preprocess_from_txt(Path(sys.argv[1]))
