#!/usr/bin/env bash
set -euo pipefail
exec > /root/deploy2.log 2>&1

APP_DIR=/opt/gp-legal-ai
MODEL_DIR="$APP_DIR/models/LFM2.5-1.2B-Thinking"
LORA_DIR="$APP_DIR/models/out_adapter"

echo "[continue] model download if needed"
mkdir -p "$(dirname "$MODEL_DIR")"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  docker run --rm \
    -v "$APP_DIR/models:/models" \
    python:3.11-slim bash -c "
      pip install -q huggingface_hub &&
      python -c \"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LiquidAI/LFM2.5-1.2B-Thinking',
    local_dir='/models/LFM2.5-1.2B-Thinking',
)
print('model ok')
\""
fi

echo "[continue] docker build + start"
cd "$APP_DIR"
export LOCAL_LLM_HOST_PATH="$MODEL_DIR"
export LOCAL_LORA_HOST_PATH="$LORA_DIR"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml up -d backend

echo "[continue] nginx"
cp "$APP_DIR/deploy/hostinger/nginx-legalai.conf" /etc/nginx/sites-available/legalai
ln -sf /etc/nginx/sites-available/legalai /etc/nginx/sites-enabled/legalai
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "[continue] wait for health"
for i in $(seq 1 48); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "DEPLOY_OK"
    curl -s http://127.0.0.1:8000/health
    exit 0
  fi
  sleep 10
done
echo "DEPLOY_PENDING - check docker logs"
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml logs backend --tail 40
