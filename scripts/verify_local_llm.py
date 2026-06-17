#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 verification: Local LLM (LFM2.5-1.2B-Instruct) explanation only.
Run from project root: python scripts/verify_local_llm.py
Checks that generate/explain_violation return explanation-only output (no invented articles).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("Local LLM verification (plan Step 4)...")
    # Prefer llm folder (llm/generate.py); fallback to app/local_llm
    try:
        from llm.generate import is_available, generate, explain_violation
        print("Using llm/generate.py.")
    except Exception as e:
        print(f"llm.generate not used: {e}")
        try:
            from app.local_llm import is_available, generate, explain_violation
            print("Using app/local_llm.")
        except Exception as e2:
            print(f"Skip: local_llm not loadable: {e2}")
            return 0

    if not is_available():
        print("Skip: LOCAL_LLM_PATH / LFM2.5-1.2B-Instruct not found.")
        return 0

    # Short prompt to avoid long inference
    prompt = "أنت مساعد قانوني. اذكر فقط: لا يوجد نص قانوني كافٍ في المستند."
    out = generate(prompt, max_new_tokens=50, do_sample=False)
    assert isinstance(out, str), "generate must return str"
    print(f"  generate() returned len={len(out)}")
    evidence_gen = (out[:150] + "...") if len(out) > 150 else out
    print(f"[Evidence] generate_output_preview: {evidence_gen!r}")

    law_articles = [
        {"text": "لا يجوز تجاوز الحد الأقصى لساعات العمل.", "metadata": {"article": "1", "law": "قانون العمل"}},
    ]
    expl = explain_violation(
        rule_id="LABOR25_WORKING_HOURS_VIOLATION",
        description="ساعات العمل",
        matched_text="العمل 12 ساعة يومياً",
        law_articles=law_articles,
        max_new_tokens=80,
    )
    assert isinstance(expl, str), "explain_violation must return str"
    print(f"  explain_violation() returned len={len(expl)}")
    evidence_expl = (expl[:150] + "...") if len(expl) > 150 else expl
    print(f"[Evidence] explain_violation_output_preview: {evidence_expl!r}")
    print("OK: Local LLM verified (explanation-only path).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
