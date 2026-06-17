#!/usr/bin/env bash
# Switch an existing Hostinger VPS from LFM Instruct to Thinking + legal LoRA adapter.
# Run on the VPS as root after uploading the adapter zip.
#
# Usage:
#   scp out_adapter.zip root@YOUR_VPS:/root/
#   scp deploy/hostinger/switch-to-thinking-lora.sh root@YOUR_VPS:/root/
#   ssh root@YOUR_VPS 'bash /root/switch-to-thinking-lora.sh /root/out_adapter.zip'
#
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/gp-legal-ai}"
MODEL_DIR="${APP_DIR}/models/LFM2.5-1.2B-Thinking"
LORA_DIR="${APP_DIR}/models/out_adapter"
HF_MODEL_ID="${HF_MODEL_ID:-LiquidAI/LFM2.5-1.2B-Thinking}"
ADAPTER_ZIP="${1:-}"

log() { echo "[switch-thinking-lora] $*"; }

if [ ! -d "$APP_DIR" ]; then
  echo "APP_DIR not found: $APP_DIR" >&2
  exit 1
fi

log "=== 1/6 Backup current Instruct model (if present) ==="
OLD_INSTRUCT="${APP_DIR}/models/LFM2.5-1.2B-Instruct"
if [ -d "$OLD_INSTRUCT" ] && [ ! -d "${OLD_INSTRUCT}.backup" ]; then
  cp -a "$OLD_INSTRUCT" "${OLD_INSTRUCT}.backup"
  log "Backed up to ${OLD_INSTRUCT}.backup"
fi

log "=== 2/6 Download Thinking base (if missing) ==="
mkdir -p "$(dirname "$MODEL_DIR")"
_has_model() {
  [ -f "$MODEL_DIR/config.json" ] || return 1
  [ -f "$MODEL_DIR/model.safetensors" ] && return 0
  ls "$MODEL_DIR"/model-*.safetensors 1>/dev/null 2>&1
}
if ! _has_model; then
  MODEL_BASENAME="$(basename "$MODEL_DIR")"
  log "Downloading ${HF_MODEL_ID} (~2.2GB)..."
  docker run --rm \
    -v "$(dirname "$MODEL_DIR"):/models" \
    -e HF_TOKEN="$(grep -E '^HF_TOKEN=' "$APP_DIR/.env" 2>/dev/null | cut -d= -f2- || true)" \
    -e MODEL_BASENAME="${MODEL_BASENAME}" \
    python:3.11-slim bash -c "
      pip install -q huggingface_hub &&
      python -c \"
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${HF_MODEL_ID}',
    local_dir='/models/' + os.environ['MODEL_BASENAME'],
    token=os.environ.get('HF_TOKEN') or None,
)
print('Thinking base download complete')
\""
else
  log "Thinking base already at ${MODEL_DIR}"
fi

log "=== 3/6 Install LoRA adapter ==="
mkdir -p "$LORA_DIR"
if [ -n "$ADAPTER_ZIP" ] && [ -f "$ADAPTER_ZIP" ]; then
  TMP="/tmp/out_adapter_unpack"
  rm -rf "$TMP"
  mkdir -p "$TMP"
  unzip -qo "$ADAPTER_ZIP" -d "$TMP"
  # Zip may nest under content/out_adapter/ or out_adapter/
  SRC=""
  for candidate in \
    "$TMP/content/out_adapter" \
    "$TMP/out_adapter" \
    "$TMP"; do
    if [ -f "$candidate/adapter_config.json" ]; then
      SRC="$candidate"
      break
    fi
  done
  if [ -z "$SRC" ]; then
    echo "Could not find adapter_config.json inside $ADAPTER_ZIP" >&2
    exit 1
  fi
  rm -rf "$LORA_DIR"
  mkdir -p "$LORA_DIR"
  cp -a "$SRC/." "$LORA_DIR/"
  # Drop training-only checkpoint dirs to save disk
  rm -rf "$LORA_DIR"/checkpoint-*
  log "Adapter installed to ${LORA_DIR}"
elif [ -f "$LORA_DIR/adapter_config.json" ]; then
  log "Adapter already present at ${LORA_DIR}"
else
  log "WARN: No adapter zip passed and ${LORA_DIR}/adapter_config.json missing."
  log "      Pass zip path: bash switch-to-thinking-lora.sh /root/out_adapter.zip"
fi

log "=== 4/6 Update .env paths ==="
ENV_FILE="$APP_DIR/.env"
touch "$ENV_FILE"
for kv in \
  "LOCAL_LLM_HOST_PATH=${MODEL_DIR}" \
  "LOCAL_LORA_HOST_PATH=${LORA_DIR}"; do
  key="${kv%%=*}"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${kv}|" "$ENV_FILE"
  else
    echo "$kv" >> "$ENV_FILE"
  fi
done
grep -q '^WARMUP_LFM_AT_STARTUP=' "$ENV_FILE" || echo 'WARMUP_LFM_AT_STARTUP=1' >> "$ENV_FILE"

log "=== 5/6 Pull latest code + rebuild backend ==="
cd "$APP_DIR"
if [ -d .git ]; then
  git pull --ff-only || log "WARN: git pull failed — continuing with local tree"
fi
export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml up -d backend

log "=== 6/6 Wait for health (up to 10 min) ==="
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "Backend healthy."
    curl -s http://127.0.0.1:8000/health || true
    echo ""
    log "Done. Base: ${MODEL_DIR}"
    log "      LoRA: ${LORA_DIR}"
    exit 0
  fi
  sleep 10
done

log "WARN: health check timed out — inspect logs:"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml logs backend --tail 60
exit 1
