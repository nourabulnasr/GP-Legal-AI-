# Runbook: backend + Legato mobile

## Backend (FastAPI)

```bash
cd GP-Legal-AI--main
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m uvicorn app.main:api --host 0.0.0.0 --port 8002 --reload
```

Use **`--host 0.0.0.0`** (not only `127.0.0.1`) so the **Android emulator** can reach the API via `http://10.0.2.2:8002`. Match the port to **`AppConfig.apiBaseUrl`** in the Flutter app (default **8002**).

### Environment (minimal)

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLite default in `app/core/config.py` if unset |
| `JWT_SECRET` | Required for auth in production |
| `GEMINI_API_KEY` | Chat assistant + negotiation (optional) |
| `LOCAL_LLM_PATH` | Path to LFM folder for explain / document chat |
| `LEGAL_RAG_QUERY_BACKEND` | `chroma_first` (default) or `memory_only` |
| `CHROMA_LEGAL_DIR` | Chroma persistence (see `docs/NEW_LEGAL_RAG.md`) |

## Mobile (Flutter)

1. Set **`lib/config/app_config.dart`** `apiBaseUrl` to your machine (default **8002** must match Uvicorn):
   - Android emulator: `http://10.0.2.2:8002`
   - Physical device: `http://<PC_LAN_IP>:8002`
2. Build / run:
   ```bash
   cd legato_mobile1
   flutter pub get
   flutter run
   ```
3. Release APK:
   ```bash
   flutter build apk --debug
   ```
   Output: `build/app/outputs/flutter-apk/app-debug.apk`

## Smoke test

Follow **`docs/DEMO_SCRIPT.md`**. Confirm **`GET /docs`** lists `/legato` routes when the server is up.
