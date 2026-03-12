# scripts\01_bootstrap_venv.ps1
. (Join-Path $PSScriptRoot "00_go_project.ps1")

if (-not (Test-Path ".\requirements.txt")) {
  Write-Host "ERROR: requirements.txt not found in project root: $(Get-Location)"
  Write-Host "Fix: Create requirements.txt at the root (same level as app/, docker/, scripts/)."
  exit 1
}

if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"

Write-Host "Venv ready: .\.venv"
