# Hostinger VPS — full backend deployment

**VPS:** `root@76.13.4.148`  
**Mode:** Full (LFM + RAG + Docker)  
**Mobile frontend:** https://legatoappgp2026.web.app/

## One-time: enable SSH from your PC

1. Hostinger hPanel → **VPS** → your server → **Settings** → **SSH Keys** → **Add key**
2. Paste the public key (generated on this machine):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILNImvdNfcpP5ZaER5ntxBFClEaTJxrnHz0bKcG+gZts legato-vps-deploy
```

3. From project root:

```powershell
powershell -ExecutionPolicy Bypass -File deploy/hostinger/remote-deploy.ps1
```

Optional env vars before running:

```powershell
$env:DEPLOY_GEMINI_API_KEY = "your-gemini-key"
$env:DEPLOY_HF_TOKEN = "your-hf-token"
```

## After deploy

- Health: http://76.13.4.148/health  
- Docs: http://76.13.4.148/docs  
- Flutter: `--dart-define=API_BASE_URL=http://76.13.4.148`

## Logs / restart

```bash
ssh -i $env:USERPROFILE\.ssh\id_ed25519_legato_vps root@76.13.4.148
cd /opt/gp-legal-ai
docker compose -f docker-compose.yml -f deploy/hostinger/docker-compose.prod.yml logs -f backend
```
