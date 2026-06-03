# VPS concurrency assessment — Hostinger `76.13.4.148`

**Date:** 2026-06-02  
**Stack:** `/opt/gp-legal-ai`, Docker `legalai-backend`, nginx → `127.0.0.1:8000`  
**Investigation:** SSH inspection + repo code review + safe parallel `/health` on port 8000 (no LFM load test on production).

---

## Executive summary

| Concurrent devices | Realistic today? | Notes |
|-------------------|------------------|--------|
| **2** (mixed: browse + one contract scan) | **Mostly yes** | Light traffic OK; two simultaneous **full** `ocr_check_and_search` + LFM → second user **waits** (serialized inference). |
| **5** (several heavy scans) | **No** | Event-loop blocking, LFM queue, SQLite write contention, CPU/RAM spikes. |
| **10** | **No** | Same bottlenecks; high risk of timeouts and OOM under parallel OCR/RAG/LFM. |

The VPS can **accept connections from many devices**, but **heavy legal-AI work is effectively single-lane** for LFM and largely runs **synchronously inside one uvicorn process**.

---

## 1. Server resources (live)

| Resource | Value |
|----------|--------|
| **RAM** | 7.8 GiB total, ~5.3 GiB available (at check time) |
| **Swap** | 4 GiB file (`/swapfile`), barely used |
| **CPU** | **2 vCPUs** (AMD EPYC host, KVM) |
| **Disk** | 96 GiB, ~69 GiB free on `/` |
| **Load** | ~1.2 (1m) — moderate for 2 cores |
| **Docker `legalai-backend`** | ~**2.4 GiB** RSS, **no** memory/CPU cap; `restart: unless-stopped` |
| **Model on disk** | LFM ~2.2 GiB (`/models/lfm`); Chroma ~6.5 MiB |

Production env (container): `ENABLE_STARTUP_RAG=1`, `WARMUP_LFM_AT_STARTUP=1` — LFM + RAG warmed at startup (good for latency, fixes baseline RAM ~2.4 GiB).

---

## 2. Backend concurrency (code + runtime)

### Uvicorn

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **Single worker process** (no `--workers N`).
- **Do not** add multiple workers on this VPS without a **separate shared inference service** — each worker would load its own LFM (~2+ GiB each) and likely OOM on 8 GiB.

### FastAPI “async” vs blocking work

- Routes like `ocr_check_and_search` are `async def` but run **CPU-heavy OCR, RAG search, rules, and LFM** on the **main event-loop thread** (no `asyncio.to_thread` / `run_in_executor` in `app/`).
- Effect: while one request runs OCR/PDF/RAG, **other requests stall** (even `/health` can lag under extreme load — startup code explicitly moved RAG bootstrap off the loop for that reason).

### LFM — primary bottleneck

`app/local_llm.py`:

- `ThreadPoolExecutor(max_workers=1)` — **all** `generate()` / `explain_violation()` calls are **serialized** (required for Lfm2 thread/device safety).
- `.result(timeout=600)` — callers **block up to 10 minutes** waiting for the single worker.
- `ocr_check_and_search` loops up to `llm_top_k` violations (default 2) → **2 sequential LFM calls per contract** when `use_llm=true`.

**Implication:** 2 users uploading contracts at once → second user’s LFM work queues behind the first; wall-clock can exceed mobile client timeouts unless nginx/app timeouts align (nginx `proxy_read_timeout` **300s**).

---

## 3. Heavy paths under simultaneous users

| Path | Behavior with 2+ users |
|------|-------------------------|
| **`POST /ocr_check_and_search`** | PDF OCR (Tesseract/Document AI), rules, Chroma RAG, then LFM explanations — **long, CPU/RAM heavy**, blocks event loop. |
| **LFM** | Strict **FIFO queue**, 1 at a time globally per container. |
| **RAG (Chroma + embeddings)** | Competes for CPU/RAM with LFM; startup already serializes LFM before RAG embed load. Concurrent searches add latency. |
| **OCR upload** | Up to **20 MB** per file (`main.py`); nginx `client_max_body_size 40m` — OK for size, not for parallel CPU. |
| **Light APIs** | `/health`, auth, reads — fine in parallel (verified: two parallel curls to `:8000/health` → both **200** in &lt;10 ms). |

**OOM risk:** Parallel heavy jobs + 2.4 GiB baseline + OCR peak allocations → possible swap thrash on 8 GiB; **not safe** to run a full LFM load test on production.

---

## 4. Docker / nginx

| Item | Production |
|------|------------|
| **Publish** | `127.0.0.1:8000:8000` (prod compose) |
| **Healthcheck** | `start_period: 300s`, interval 30s — reflects slow LFM/RAG startup |
| **Restart** | `unless-stopped` |
| **nginx** | `worker_connections 768`; `client_max_body_size 40m`; proxy timeouts **300s** |
| **Limits** | No `limit_conn` / rate limiting configured |

Note: `curl http://127.0.0.1/health` on the host returned nginx **404** (default vhost); backend health via **`127.0.0.1:8000/health`** is correct.

---

## 5. Database — SQLite `/data/legalai.db`

| Setting | Value | Impact |
|---------|--------|--------|
| **journal_mode** | `delete` (not WAL) | Weaker concurrent read/write behavior |
| **busy_timeout** | `0` | Writes fail immediately if DB locked |
| **SQLAlchemy** | `check_same_thread=False`, default pool | OK for one process; **not** for multi-worker |

Multiple devices saving analyses / social posts / uploads → **serialized writes**; concurrent writes may surface as `database is locked` under load.

---

## 6. Practical test performed

- **Safe:** 2× parallel `GET http://127.0.0.1:8000/health` → both **200**, ~7–8 ms.
- **Not run:** Parallel `ocr_check_and_search` or LFM stress (would impact live users).

---

## Recommendations (prioritized, minimal scope)

### P0 — Accept current limits; set expectations

- Treat the VPS as **~1 heavy contract analysis at a time**; extra devices can use the app for **light** actions but should expect **queueing** on scan/LLM.
- Document client timeout ≥ **120–300s** for full scans (match nginx).

### P1 — Low-risk config (no architecture change)

1. **SQLite WAL + busy timeout** (single-container, single-worker safe):

   In `app/db/session.py` connect_args (or one-time migration on startup):

   ```python
   connect_args = {
       "check_same_thread": False,
       "timeout": 30,  # seconds waiting on lock
   }
   # After connect: PRAGMA journal_mode=WAL; PRAGMA busy_timeout=30000;
   ```

2. **Optional env** (already in `env.production.template`): keep `WARMUP_LFM_AT_STARTUP=1` for predictable first-request latency.

3. **nginx rate limiting** (protect from accidental stampedes):

   ```nginx
   limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
   location / {
       limit_req zone=api_limit burst=20 nodelay;
       # ... existing proxy_* ...
   }
   ```

   Stricter limit on `POST /ocr_check_and_search` if exposed as separate `location`.

### P2 — Code changes (moderate effort, high value on same VPS)

1. **Offload blocking work** from the event loop: `asyncio.to_thread()` for OCR PDF pipeline, RAG search, rules, and each `explain_violation` call (LFM still serializes inside its executor, but other requests can progress).
2. **Explicit job queue** for LFM (Redis/RQ or in-process `asyncio.Queue` + single worker) with **202 + poll** or WebSocket status — better UX than silent 60–300s waits.
3. **Cap concurrent heavy jobs** per IP/user (middleware) → return **429** with retry-after instead of piling up OCR/LFM.

### P3 — Infrastructure (when you need 5–10 real concurrent heavy users)

| Change | Why |
|--------|-----|
| **Upgrade to 16 GiB RAM + 4 vCPU** | Headroom for OCR peaks + embedding model + one LFM instance |
| **PostgreSQL** (managed or on VPS) | Reliable concurrent writes for auth/social/analyses |
| **Separate inference container** | One LFM process, API workers stay thin; optional GPU later |
| **Do not** scale via `--workers 4` on one box with embedded LFM | 4× model RAM |

---

## Optional compose/env snippets (not applied to production)

See `env.production.template` — add comments only:

```bash
# Concurrency: keep single uvicorn worker; do not set UVICORN_WORKERS>1 without external LFM service.
# SQLITE_BUSY_TIMEOUT_MS=30000  # implement in session.py if added
```

For a future **inference sidecar** (sketch):

```yaml
# docker-compose.prod.yml — illustrative only
services:
  lfm-worker:
    build: .
    command: python -m app.lfm_worker  # dedicated queue consumer
    volumes: [same model mount]
  backend:
    environment:
      - LFM_REMOTE_URL=http://lfm-worker:8081
```

---

## Answers to your questions

1. **Can it handle more than one device?** **Yes** for connectivity and light API usage; **one heavy AI pipeline at a time** in practice.
2. **2 / 5 / 10 concurrent devices?** **2** mixed → yes with queueing on scans; **5–10** heavy → **no** on current hardware/software.
3. **Main limits:** LFM `max_workers=1`, single uvicorn worker, sync CPU work on async routes, 2 vCPU / ~8 GiB RAM, SQLite `delete` journal + `busy_timeout=0`.
4. **Production changes made:** None (report only).
