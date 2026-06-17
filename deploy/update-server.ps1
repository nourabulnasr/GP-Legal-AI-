# Upload updated backend files to the VPS and restart.
# Run from project root:  powershell -ExecutionPolicy Bypass -File deploy/update-server.ps1

param(
    [string]$VpsHost = "76.13.4.148",
    [string]$SshUser = "root",
    [string]$SshKey  = "$env:USERPROFILE\.ssh\id_ed25519_legato_vps",
    [string]$AppDir  = "/opt/gp-legal-ai"
)

$sshArgs = @("-i", $SshKey, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes")
$target  = "${SshUser}@${VpsHost}"
$root    = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "Testing SSH to $target ..." -ForegroundColor Cyan
& ssh @sshArgs $target "echo SSH OK"
if ($LASTEXITCODE -ne 0) {
    Write-Host @"

SSH failed. Do this first:
  1. Go to hpanel.hostinger.com -> VPS -> your server -> SSH Keys
  2. Click 'Add SSH Key' and paste this public key:

$(Get-Content "$SshKey.pub" -Raw)

  3. Re-run this script.
"@ -ForegroundColor Yellow
    exit 1
}

Write-Host "Uploading updated backend files..." -ForegroundColor Cyan

$files = @(
    "app/routers/auth.py",
    "app/routers/lawyer.py",
    "app/routers/admin_lawyers.py",
    "app/schemas/auth.py",
    "app/db/models.py",
    "app/db/init_db.py",
    "app/main.py",
    "app/services/user_deletion.py"
)

foreach ($f in $files) {
    $local  = Join-Path $root $f
    $remote = "$AppDir/$f"
    Write-Host "  $f" -ForegroundColor Gray
    # Ensure remote directory exists
    $dir = ($remote -split '/' | Select-Object -SkipLast 1) -join '/'
    & ssh @sshArgs $target "mkdir -p $dir"
    & scp @sshArgs $local "${target}:${remote}"
    if ($LASTEXITCODE -ne 0) { Write-Host "Failed: $f" -ForegroundColor Red; exit 1 }
}

Write-Host "Restarting backend container..." -ForegroundColor Cyan
& ssh @sshArgs $target @"
cd $AppDir
if docker compose ps 2>/dev/null | grep -q backend; then
    docker compose restart backend
elif docker-compose ps 2>/dev/null | grep -q backend; then
    docker-compose restart backend
else
    echo 'Backend container not found, trying systemd...'
    systemctl restart gunicorn 2>/dev/null || systemctl restart uvicorn 2>/dev/null || pkill -HUP uvicorn || true
fi
echo 'Restart done'
"@

Write-Host ""
Write-Host "Done! Verify at https://srv1723974.hstgr.cloud/docs" -ForegroundColor Green
Write-Host "The /lawyer/* endpoints should now appear in the docs." -ForegroundColor Green
