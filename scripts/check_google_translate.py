# -*- coding: utf-8 -*-
"""One-off diagnostic: is Google Cloud Translation v2 usable? Run from repo root: python scripts/check_google_translate.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_dotenv_if_present() -> None:
    for p in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            elif v.startswith("'") and v.endswith("'"):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ.setdefault(k, v)


def mask(s: str) -> str:
    if not s:
        return "(empty)"
    if len(s) < 10:
        return f"(set, {len(s)} chars)"
    return f"{s[:4]}...{s[-4:]} ({len(s)} chars)"


def main() -> int:
    load_dotenv_if_present()

    key = (os.getenv("GOOGLE_TRANSLATION_API_KEY") or "").strip()
    creds = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    ext_off = os.getenv("EXTERNAL_MT_DISABLED", "").strip().lower() in ("1", "true", "yes")
    g_off = os.getenv("DISABLE_GOOGLE_MT", "").strip().lower() in ("1", "true", "yes")

    print("=== Google Translation API check ===")
    print("EXTERNAL_MT_DISABLED (blocks Google):", ext_off)
    print("DISABLE_GOOGLE_MT:", g_off)
    print("GOOGLE_TRANSLATION_API_KEY:", "set " + mask(key) if key else "not set")
    print("GOOGLE_APPLICATION_CREDENTIALS:", creds or "not set")
    if creds:
        cp = Path(creds).expanduser()
        print("  file exists:", cp.is_file())

    print()
    print("--- Import google.cloud.translate_v2 ---")
    try:
        from google.cloud import translate_v2 as tv2  # noqa: F401

        print("OK: translate_v2 import succeeded")
    except Exception as e:
        print("FAIL:", type(e).__name__, e)
        print("Fix: pip install google-cloud-translate")
        return 1

    if ext_off or g_off:
        print()
        print("Skipped client/API: EXTERNAL_MT_DISABLED or DISABLE_GOOGLE_MT is on")
        return 2

    print()
    print("--- Build client (app.translation_service._get_translate_v2_client) ---")
    import app.translation_service as ts

    ts._v2_client = None
    client = ts._get_translate_v2_client()
    if client is None and key:
        bad_creds = creds and not Path(creds).expanduser().is_file()
        if bad_creds:
            print("Retry without GOOGLE_APPLICATION_CREDENTIALS (path missing; API key may still work)...")
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            ts._v2_client = None
            client = ts._get_translate_v2_client()
    if client is None:
        print("FAIL: client is None (check API key / ADC / billing / Cloud Translation API enabled)")
        if creds and not Path(creds).expanduser().is_file():
            print("Hint: GOOGLE_APPLICATION_CREDENTIALS points to a missing file; remove it or fix the path.")
            print("      A bad path can prevent the client from initializing even when using an API key.")
        return 3
    print("OK: Translation v2 client created")

    print()
    print("--- Live API call: en -> ar (Hello) ---")
    try:
        out = client.translate("Hello", target_language="ar", source_language="en", format_="text")
        text = out.get("translatedText", out) if isinstance(out, dict) else str(out)
        print("OK: translatedText:", repr(text[:200]))
    except Exception as e:
        print("FAIL:", type(e).__name__, e)
        return 4

    print()
    print("=== Result: Google Translation API is WORKING ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
