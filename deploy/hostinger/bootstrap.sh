#!/usr/bin/env bash
# Full backend deployment for Hostinger VPS (Ubuntu).
# Run as root: bash bootstrap.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/gp-legal-ai}"
BUNDLE_DIR="${BUNDLE_DIR:-/root/deploy-bundle}"
REPO_URL="${REPO_URL:-https://github.com/nourabulnasr/GP-Legal-AI-.git}"
GIT_BRANCH="${GIT_BRANCH:-final_80%}"
MODEL_DIR="${APP_DIR}/models/LFM2.5-1.2B-Thinking"
HF_MODEL_ID="${HF_MODEL_ID:-LiquidAI/LFM2.5-1.2B-Thinking}"
LORA_DIR="${APP_DIR}/models/out_adapter"
SWAP_GB="${SWAP_GB:-4}"

log() { echo "[deploy] $*"; }

log "=== 1/9 System packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ca-certificates gnupg lsb-release ufw nginx apache2-utils

if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${VERSION_CODENAME:-$VERSION_ID}") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi

log "=== 2/9 Swap (helps 8GB RAM + LFM) ==="
if [ ! -f /swapfile ] && [ "${SWAP_GB}" -gt 0 ] 2>/dev/null; then
  fallocate -l "${SWAP_GB}G" /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB * 1024))
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

log "=== 3/9 Firewall ==="
ufw --force reset || true
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

log "=== 4/9 Clone application ==="
mkdir -p "$(dirname "$APP_DIR")"
if [ -d "$APP_DIR/.git" ]; then
  cd "$APP_DIR"
  git fetch origin
  git checkout "$GIT_BRANCH" 2>/dev/null || git checkout -B "$GIT_BRANCH" "origin/$GIT_BRANCH"
  git pull --ff-only origin "$GIT_BRANCH" || true
else
  git clone --branch "$GIT_BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
  cd "$APP_DIR"
fi

log "=== 5/9 Environment file ==="
mkdir -p "$APP_DIR/deploy/hostinger"
if [ -d "$BUNDLE_DIR" ]; then
  cp -a "$BUNDLE_DIR"/. "$APP_DIR/deploy/hostinger/"
fi
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/deploy/hostinger/env.production.template" "$APP_DIR/.env"
fi

gen_secret() { openssl rand -hex 32; }

if ! grep -q '^SECRET_KEY=.' "$APP_DIR/.env" 2>/dev/null || grep -q 'CHANGE_ME\|your-secret' "$APP_DIR/.env"; then
  SK="$(gen_secret)"
  if grep -q '^SECRET_KEY=' "$APP_DIR/.env"; then
    sed -i "s/^SECRET_KEY=.*/SECRET_KEY=${SK}/" "$APP_DIR/.env"
  else
    echo "SECRET_KEY=${SK}" >> "$APP_DIR/.env"
  fi
fi

if ! grep -q '^JWT_SECRET_KEY=.' "$APP_DIR/.env" 2>/dev/null || grep -q 'CHANGE_ME' "$APP_DIR/.env"; then
  JK="$(gen_secret)"
  if grep -q '^JWT_SECRET_KEY=' "$APP_DIR/.env"; then
    sed -i "s/^JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${JK}/" "$APP_DIR/.env"
  else
    echo "JWT_SECRET_KEY=${JK}" >> "$APP_DIR/.env"
  fi
fi

if [ -n "${DEPLOY_GEMINI_API_KEY:-}" ]; then
  if grep -q '^GEMINI_API_KEY=' "$APP_DIR/.env"; then
    sed -i "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=${DEPLOY_GEMINI_API_KEY}|" "$APP_DIR/.env"
  else
    echo "GEMINI_API_KEY=${DEPLOY_GEMINI_API_KEY}" >> "$APP_DIR/.env"
  fi
fi

if [ -n "${DEPLOY_HF_TOKEN:-}" ]; then
  if grep -q '^HF_TOKEN=' "$APP_DIR/.env"; then
    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${DEPLOY_HF_TOKEN}|" "$APP_DIR/.env"
  else
    echo "HF_TOKEN=${DEPLOY_HF_TOKEN}" >> "$APP_DIR/.env"
  fi
fi

export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
grep -q '^LOCAL_LLM_HOST_PATH=' "$APP_DIR/.env" || echo "LOCAL_LLM_HOST_PATH=${MODEL_DIR}" >> "$APP_DIR/.env"
grep -q '^LOCAL_LORA_HOST_PATH=' "$APP_DIR/.env" || echo "LOCAL_LORA_HOST_PATH=${LORA_DIR}" >> "$APP_DIR/.env"

log "=== 6/9 Download LFM model (if missing) ==="
mkdir -p "$(dirname "$MODEL_DIR")"
_has_model() {
  [ -f "$MODEL_DIR/config.json" ] || return 1
  [ -f "$MODEL_DIR/model.safetensors" ] && return 0
  ls "$MODEL_DIR"/model-*.safetensors 1>/dev/null 2>&1
}
if ! _has_model; then
  log "Downloading ${HF_MODEL_ID} (~2.2GB) — may take 10–30 min..."
  MODEL_BASENAME="$(basename "$MODEL_DIR")"
  docker run --rm \
    -v "$(dirname "$MODEL_DIR"):/models" \
    -e HF_TOKEN="${DEPLOY_HF_TOKEN:-}" \
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
print('Model download complete')
\""
else
  log "LFM model already present at ${MODEL_DIR}"
fi

if [ ! -f "$LORA_DIR/adapter_config.json" ]; then
  log "WARN: LoRA adapter not found at ${LORA_DIR}/adapter_config.json"
  log "      Upload out_adapter (adapter_config.json + adapter_model.safetensors + tokenizer files)"
  log "      then restart backend. Until then, base Thinking model runs without fine-tune."
fi

log "=== 7/9 Build and start backend (Docker) ==="
cd "$APP_DIR"
export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml up -d backend

log "Waiting for backend health (up to 8 min for RAG + LFM warmup)..."
for i in $(seq 1 48); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "Backend healthy."
    break
  fi
  sleep 10
  if [ "$i" -eq 48 ]; then
    log "WARN: health check timed out — inspect: docker compose logs backend --tail 80"
  fi
done

log "=== 8/9 Nginx ==="
cp "$APP_DIR/deploy/hostinger/nginx-legalai.conf" /etc/nginx/sites-available/legalai
ln -sf /etc/nginx/sites-available/legalai /etc/nginx/sites-enabled/legalai
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

log "=== 9/9 Done ==="
echo ""
echo "API:      http://76.13.4.148/health"
echo "Docs:     http://76.13.4.148/docs"
echo "App dir:  ${APP_DIR}"
echo ""
echo "Update Flutter: API_BASE_URL=http://76.13.4.148"
echo "Add GEMINI_API_KEY to ${APP_DIR}/.env if chat is disabled, then: docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml restart backend"
