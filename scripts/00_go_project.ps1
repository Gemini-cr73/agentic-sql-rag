$ProjectRoot = "C:\Users\crb33\Desktop\AI-Projects\agentic-sql-rag"

if (-not (Test-Path $ProjectRoot)) {
  Write-Host "ERROR: Project root not found: $ProjectRoot"
  exit 1
}

Set-Location $ProjectRoot
Write-Host "Now in project root: $(Get-Location)"
