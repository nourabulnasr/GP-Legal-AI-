#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot poll of LAW_OFFICIAL_PDF_URL (conditional GET) and rebuild RAG if the PDF changed.
Intended for cron / CI: run from project root.

  set LAW_OFFICIAL_PDF_URL=https://.../law.pdf
  python scripts/poll_official_law.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    os.chdir(ROOT)
    from app.law_update_service import poll_official_url_once

    try:
        out = poll_official_url_once()
    except Exception as e:
        print(json.dumps({"error": repr(e)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
