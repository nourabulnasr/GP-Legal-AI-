#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run unified training in both modes and write an evaluation report.
Useful for testing accuracies and outputs (Task 1).

Run from project root:
  python scripts/run_unified_evaluation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "reports"
ARTIFACTS_DIR = BASE / "app" / "ml" / "artifacts" / "unified"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"modes": [], "summary": {}}

    # 1) Contracts-only
    print("Running unified training (contracts_only)...")
    r1 = subprocess.run(
        [sys.executable, str(BASE / "ml" / "scripts" / "train_unified.py")],
        cwd=str(BASE),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r1.returncode == 0 and (ARTIFACTS_DIR / "metrics.json").exists():
        m1 = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        report["modes"].append({"mode": "contracts_only", "metrics": m1})
    else:
        report["modes"].append({"mode": "contracts_only", "error": r1.stderr or "failed"})

    # 2) Law-aware
    print("Running unified training (law_aware)...")
    r2 = subprocess.run(
        [sys.executable, str(BASE / "ml" / "scripts" / "train_unified.py"), "--law_aware", "--top_k_law", "3"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
        timeout=400,
    )
    if r2.returncode == 0 and (ARTIFACTS_DIR / "metrics.json").exists():
        m2 = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        report["modes"].append({"mode": "law_aware", "metrics": m2})
    else:
        report["modes"].append({"mode": "law_aware", "error": r2.stderr or "failed"})

    # Summary
    f1s = []
    for m in report["modes"]:
        if "metrics" in m and "f1" in m["metrics"]:
            f1s.append((m["mode"], m["metrics"]["f1"], m["metrics"].get("roc_auc"), m["metrics"].get("pr_auc")))
    if f1s:
        report["summary"] = {
            "best_f1_mode": max(f1s, key=lambda x: x[1])[0],
            "f1_scores": {m: f1 for m, f1, _, _ in f1s},
            "roc_auc": {m: roc for m, f1, roc, _ in f1s if roc is not None},
            "pr_auc": {m: pr for m, f1, _, pr in f1s if pr is not None},
        }

    out_path = REPORTS_DIR / "unified_evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Report written to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
