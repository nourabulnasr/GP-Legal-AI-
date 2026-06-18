#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/gp-legal-ai
MODEL_DIR="$APP_DIR/models/LFM2.5-1.2B-Thinking"
LORA_DIR="$APP_DIR/models/out_adapter"

log() { echo "[deploy] $*"; }

log "Fetch and switch to Final"
cd "$APP_DIR"
git fetch origin Final
git checkout -f -B Final FETCH_HEAD
git log --oneline -1

log "Build and start backend"
export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml up -d backend

log "Reload nginx"
cp "$APP_DIR/deploy/hostinger/nginx-legalai.conf" /etc/nginx/sites-available/legalai
ln -sf /etc/nginx/sites-available/legalai /etc/nginx/sites-enabled/legalai
nginx -t
systemctl reload nginx

log "Wait for health (up to 8 min)"
for i in $(seq 1 48); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "DEPLOY_OK"
    curl -s http://127.0.0.1:8000/health
    exit 0
  fi
  sleep 10
done

log "DEPLOY_PENDING — backend not healthy yet"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml logs backend --tail 40
exit 1
