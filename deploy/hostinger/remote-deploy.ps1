# Run from project root on Windows (after adding SSH public key to Hostinger VPS):
#   powershell -ExecutionPolicy Bypass -File deploy/hostinger/remote-deploy.ps1
param(
    [string]$VpsHost = "76.13.4.148",
    [string]$SshUser = "root",
    [string]$SshKey = "$env:USERPROFILE\.ssh\id_ed25519_legato_vps",
    [string]$GitBranch = "final_80%"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if (-not (Test-Path $SshKey)) {
    Write-Error "SSH key not found: $SshKey. Run: ssh-keygen -t ed25519 -f `"$SshKey`" -N '""'"
}

$PubKey = Get-Content "$SshKey.pub" -Raw
Write-Host "`n=== Add this SSH public key in Hostinger VPS dashboard (SSH Keys) ===" -ForegroundColor Yellow
Write-Host $PubKey.Trim()
Write-Host "================================================================`n" -ForegroundColor Yellow

$sshArgs = @("-i", $SshKey, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes")
$target = "${SshUser}@${VpsHost}"

Write-Host "Testing SSH to $target ..."
& ssh @sshArgs $target "echo ok"
if ($LASTEXITCODE -ne 0) {
    Write-Host "SSH failed. Add the public key above to Hostinger, then re-run this script." -ForegroundColor Red
    exit 1
}

Write-Host "Uploading deploy bundle..."
& ssh @sshArgs $target "mkdir -p /root/deploy-bundle"
& scp @sshArgs (Join-Path $PSScriptRoot "bootstrap.sh") "${target}:/root/bootstrap.sh"
& scp @sshArgs `
  (Join-Path $PSScriptRoot "docker-compose.prod.yml") `
  (Join-Path $PSScriptRoot "nginx-legalai.conf") `
  (Join-Path $PSScriptRoot "env.production.template") `
  "${target}:/root/deploy-bundle/"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$envExports = ""
if ($env:DEPLOY_GEMINI_API_KEY) { $envExports += "export DEPLOY_GEMINI_API_KEY='$($env:DEPLOY_GEMINI_API_KEY)'; " }
if ($env:DEPLOY_HF_TOKEN) { $envExports += "export DEPLOY_HF_TOKEN='$($env:DEPLOY_HF_TOKEN)'; " }

Write-Host "Running bootstrap on VPS (20-60 min for model + Docker build)..."
& ssh @sshArgs $target "${envExports}export GIT_BRANCH='$GitBranch'; export BUNDLE_DIR=/root/deploy-bundle; chmod +x /root/bootstrap.sh; bash /root/bootstrap.sh"
exit $LASTEXITCODE
