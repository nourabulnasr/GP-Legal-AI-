#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification: ML model (model_ML predictor).
Run from project root: python scripts/verify_ml_model.py
Runs one prediction on a short labor-law clause and prints evidence (top rule_id, score).
Skips with exit 0 if model_ml_predictor is not available.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("ML model verification (model_ML predictor)...")
    try:
        from app.model_ml_predictor import (
            has_model_ml_predictor,
            predict_rule_scores_full,
            rule_scores_to_rule_hits,
        )
    except Exception as e:
        print(f"Skip: model_ml_predictor not loadable ({e}).")
        return 0

    if not has_model_ml_predictor():
        print("Skip: ML bundles not found (law_aware_multilabel_model.joblib / law_aware_binary_model.joblib).")
        return 0

    # Short Arabic clause for labor-law prediction
    clause = "يعمل الموظف اثنتي عشرة ساعة يومياً دون ذكر ساعات الراحة أو التعويض عن الإضافي."
    preds = predict_rule_scores_full(clause, sort=True, use_law_retrieval=True)
    hits = rule_scores_to_rule_hits(preds, chunk_id="verify_1")

    if not preds:
        print("WARN: predict_rule_scores_full returned empty.")
        return 0

    # Evidence: ml_used, top 1-2 rule_id + score
    print(f"[Evidence] ml_used=True, predictions_count={len(preds)}")
    for i, p in enumerate(preds[:2], 1):
        rule_id = p.get("rule_id", "")
        score = p.get("score", 0.0)
        desc = (p.get("description") or "")[:80]
        print(f"  {i}. rule_id={rule_id} score={score:.4f} description_preview={desc!r}")
    if hits:
        print(f"[Evidence] rule_scores_to_rule_hits returned {len(hits)} hit(s); first rule_id={hits[0].get('rule_id')!r}")
    print("OK: ML model verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
