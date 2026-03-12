# scripts/03_load_env.ps1
# Purpose:
#   Load environment variables from .\docker\.env into the CURRENT PowerShell session ("Process").
#   This is required before running Python modules directly (ingest, REPL, etc.)
#
# Usage:
#   From project root:
#     .\scripts\03_load_env.ps1
#
# Verify:
#   echo $env:DATABASE_URL

$ErrorActionPreference = "Stop"

# --- 1) Ensure we are running from the project root ---
# If you run this from inside /scripts by mistake, paths won't match.
if (-not (Test-Path ".\docker\.env")) {
    Write-Host "ERROR: Could not find .\docker\.env" -ForegroundColor Red
    Write-Host "Run this script from the PROJECT ROOT (same folder that contains docker\)." -ForegroundColor Yellow
    Write-Host "Example:" -ForegroundColor Yellow
    Write-Host "  cd C:\Users\crb33\Desktop\AI-Projects\agentic-sql-rag" -ForegroundColor Yellow
    Write-Host "  .\scripts\03_load_env.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "----------------------------------------" -ForegroundColor DarkGray
Write-Host "Loading environment variables from: .\docker\.env" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor DarkGray
Write-Host ""

# --- 2) Read docker/.env line-by-line ---
Get-Content ".\docker\.env" | ForEach-Object {

    # Clean whitespace
    $line = $_.Trim()

    # Skip blank lines and comments
    if (-not $line -or $line.StartsWith("#")) { return }

    # Split on FIRST '=' only (so values can contain '=')
    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) { return }

    $key = $parts[0].Trim()
    $value = $parts[1].Trim()

    # Remove optional surrounding quotes
    $value = $value.Trim('"').Trim("'")

    # Set env var for current PowerShell process
    [Environment]::SetEnvironmentVariable($key, $value, "Process")

    Write-Host ("Set: {0}" -f $key) -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Quick check:" -ForegroundColor Cyan
Write-Host ("DATABASE_URL = {0}" -f $env:DATABASE_URL) -ForegroundColor White
Write-Host ""

# --- 3) Hard fail if DATABASE_URL is missing ---
if (-not $env:DATABASE_URL) {
    Write-Host "ERROR: DATABASE_URL was not loaded." -ForegroundColor Red
    Write-Host "Open .\docker\.env and make sure it contains a line like:" -ForegroundColor Yellow
    Write-Host "DATABASE_URL=postgresql+psycopg://user:pass@localhost:55432/ragdb" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Environment loaded successfully into this session." -ForegroundColor Green