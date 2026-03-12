# scripts\02_start_db.ps1
. (Join-Path $PSScriptRoot "00_go_project.ps1")

if (-not (Test-Path ".\docker\docker-compose.yml")) {
  Write-Host "ERROR: docker-compose.yml not found in .\docker\"
  exit 1
}

Set-Location ".\docker"
docker compose up -d

Write-Host "Postgres (pgvector) started."
