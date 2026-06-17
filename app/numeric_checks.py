# numeric_checks.py
import re
from utils_text import normalize_ar, word_to_num
# in numeric_checks.py (top)


def extract_number_from_text(text):
    """
    Try to capture an integer from text (arabic digits, english digits, words converted by word_to_num).
    Returns int or None.
    """
    t = normalize_ar(text)
    t = word_to_num(t)  # replace words like "ثلاثة" -> "3" if implemented
    m = re.search(r'(\d{1,3})\s*(?:ساعة|شهر|يوم|أشهر|ساعات|يومًا|يوم)', t)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    # fallback simple number search
    m2 = re.search(r'(\d{1,3})', t)
    if m2:
        return int(m2.group(1))
    return None

def check_numeric_rules_on_chunk(chunk_text, rule_id):
    """
    Example: rule_id = 'labor_probation_limit' -> check <=3
             rule_id = 'labor_annual_leave_minimum' -> check >=15
             rule_id = 'labor_working_hours_limit' -> check <=48
    Returns dict with result boolean, extracted_value
    """
    val = extract_number_from_text(chunk_text)
    if val is None:
        return {"ok": False, "value": None, "reason": "no_number_found"}
    if rule_id == 'labor_probation_limit':
        return {"ok": val <= 3, "value": val}
    if rule_id == 'labor_annual_leave_minimum':
        return {"ok": val >= 15, "value": val}
    if rule_id == 'labor_working_hours_limit':
        return {"ok": val <= 48, "value": val}
    return {"ok": True, "value": val}
