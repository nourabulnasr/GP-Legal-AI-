# -*- coding: utf-8 -*-
"""Heuristics for whether Egyptian labor-law analysis should apply to contract text."""

from __future__ import annotations


def egyptian_labor_content_related(text: str, source_lang_code: str) -> bool:
    """
    True when the (possibly translated) contract text plausibly concerns Egyptian labor law
    or an Egyptian employment context. Used to skip Egyptian labor analysis when the document
    is clearly unrelated (e.g. non-employment, or foreign-only employment law) after translation.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    src = (source_lang_code or "und").lower().split("-")[0]
    tl = raw.lower()

    explicit = [
        "القانون المصري",
        "قانون العمل المصري",
        "يخضع لأحكام القانون المصري",
        "يخضع للقانون المصري",
        "تطبق أحكام القانون المصري",
        "محاكم مصر",
        "المحاكم المصرية",
        "محكمة مصر",
        "جمهورية مصر العربية",
        "egyptian law",
        "laws of egypt",
        "courts of egypt",
        "ministry of manpower",
        "الهيئة القومية للتأمينات الاجتماعية",
        "14 لسنة 2025",
        "قانون العمل 14",
        "قانون رقم 14",
    ]
    if any(x in tl for x in explicit) or "egypt" in tl:
        return True

    egypt_impl_ar = [
        "مصر ",
        "مصر.",
        "مصر،",
        "مصر\n",
        "القاهرة",
        "الجيزة",
        "الإسكندرية",
        "الاسكندرية",
        "جنيه مصري",
        "الجنيه المصري",
        "الرقم القومي",
        "محافظة القاهرة",
        "محافظة الجيزة",
    ]
    if any(x in raw for x in egypt_impl_ar):
        has_egypt_impl = True
    else:
        has_egypt_impl = any(x in tl for x in (" egp", "egp ", "egp\n", "egp.", "egp,"))
    if "مصر" in raw:
        has_egypt_impl = True

    employment = [
        "عقد عمل",
        "عقد العمل",
        "الموظف",
        "صاحب العمل",
        "الأجر",
        "الراتب",
        "إجازة سنوية",
        "فترة الاختبار",
        "فترة التجربة",
        "إنهاء الخدمة",
        "employment contract",
        "employee",
        "employer",
        "salary",
        "wage",
        "arbeitsvertrag",
        "arbeitnehmer",
        "arbeitgeber",
    ]
    has_emp = any(x in tl for x in employment)

    foreign_only = [
        "قانون العمل الإماراتي",
        "الإمارات العربية المتحدة",
        "united arab emirates",
        "uae",
        "dubai",
        "أبوظبي",
        "ابوظبي",
        "ألمانيا",
        "bundesrepublik deutschland",
        "german law",
        "deutsches recht",
        "bürgerliches gesetzbuch",
        "القانون الألماني",
        "النمسا",
        "austria",
        "فرنسا",
        "code du travail français",
        "المملكة المتحدة",
        "united kingdom",
        "england and wales",
    ]
    has_foreign = any(x in tl for x in foreign_only)

    if has_emp and has_egypt_impl:
        return True
    if has_emp and has_foreign and not has_egypt_impl and "egypt" not in tl:
        return False
    if src == "ar" and has_emp and not has_foreign:
        return True
    if has_emp and not has_foreign and src != "ar":
        return False
    return False
