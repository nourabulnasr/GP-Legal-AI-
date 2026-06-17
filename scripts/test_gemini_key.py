"""One-off Gemini connectivity check (run inside backend container)."""
import os
import sys

key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
print("key_len", len(key))
print("key_prefix", key[:8] if key else "none")
if not key:
    sys.exit(1)
try:
    from google import genai

    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents="Reply with exactly: ok",
    )
    print("ok", (getattr(resp, "text", None) or str(resp))[:80])
except Exception as e:
    print("fail", type(e).__name__, str(e)[:300])
    sys.exit(2)
