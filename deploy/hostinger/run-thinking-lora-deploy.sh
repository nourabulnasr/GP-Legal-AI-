#!/usr/bin/env bash
set -euo pipefail
APP_DIR=/opt/gp-legal-ai
MODEL_DIR="$APP_DIR/models/LFM2.5-1.2B-Thinking"
LORA_DIR="$APP_DIR/models/out_adapter"
ENV_FILE="$APP_DIR/.env"

echo "=== verify uploaded files ==="
ls -lh "$APP_DIR/app/local_llm.py" "$APP_DIR/requirements.txt" "$LORA_DIR/adapter_config.json" "$LORA_DIR/adapter_model.safetensors"

echo "=== update .env ==="
touch "$ENV_FILE"
set_kv() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
  fi
}
set_kv LOCAL_LLM_HOST_PATH "$MODEL_DIR"
set_kv LOCAL_LORA_HOST_PATH "$LORA_DIR"
set_kv WARMUP_LFM_AT_STARTUP "1"

echo "=== download Thinking base if missing ==="
mkdir -p "$(dirname "$MODEL_DIR")"
_has_model() {
  [ -f "$MODEL_DIR/config.json" ] || return 1
  [ -f "$MODEL_DIR/model.safetensors" ] && return 0
  ls "$MODEL_DIR"/model-*.safetensors 1>/dev/null 2>&1
}
if ! _has_model; then
  if [ -d "$APP_DIR/models/LFM2.5-1.2B-Instruct" ] && [ ! -d "${APP_DIR}/models/LFM2.5-1.2B-Instruct.backup" ]; then
    cp -a "$APP_DIR/models/LFM2.5-1.2B-Instruct" "${APP_DIR}/models/LFM2.5-1.2B-Instruct.backup"
    echo "Backed up Instruct model"
  fi
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- || true)"
  docker run --rm \
    -v "$(dirname "$MODEL_DIR"):/models" \
    -e HF_TOKEN="${HF_TOKEN}" \
    python:3.11-slim bash -c '
      pip install -q huggingface_hub &&
      python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=\"LiquidAI/LFM2.5-1.2B-Thinking\",
    local_dir=\"/models/LFM2.5-1.2B-Thinking\",
    token=os.environ.get(\"HF_TOKEN\") or None,
)
print(\"Thinking download complete\")
"'
else
  echo "Thinking base already present"
fi

echo "=== rebuild + restart backend ==="
cd "$APP_DIR"
export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml up -d backend

echo "=== wait for health (up to 12 min) ==="
for i in $(seq 1 72); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "HEALTH_OK"
    curl -s http://127.0.0.1:8000/health
    exit 0
  fi
  sleep 10
done
echo "HEALTH_TIMEOUT"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml logs backend --tail 80
exit 1
