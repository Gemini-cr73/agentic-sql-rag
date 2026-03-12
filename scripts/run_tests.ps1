# scripts/run_tests.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure we're at repo root even if launched elsewhere
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Activate venv
& "$repoRoot\.venv\Scripts\Activate.ps1"

# Load docker/.env into this PowerShell process (DATABASE_URL, etc.)
Get-Content "$repoRoot\docker\.env" | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k, $v = $_.Split('=', 2)
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}

# Install test deps into the venv (safe to re-run)
python -m pip install -U pip
python -m pip install -U pytest httpx

# Run tests
python -m pytest -v
