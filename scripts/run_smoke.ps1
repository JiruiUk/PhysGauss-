param([string]$RunName = 'smoke')
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProjectRoot
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run bootstrap.ps1 first' }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain letters, numbers, hyphens or underscores' }
if (Test-Path -LiteralPath "runs\$RunName\checkpoint.pt") { throw 'Run already exists; choose a new RunName or use the CLI --resume option' }
if (-not (Test-Path -LiteralPath 'data\synthetic\manifest.json')) {
    & $taskPython -m physgauss make-synthetic --output data/synthetic
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic generation failed' }
}
& $taskPython -m physgauss validate-data --manifest data/synthetic/manifest.json
if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed' }
& $taskPython -m physgauss train --manifest data/synthetic/manifest.json --config configs/smoke.json --output "runs/$RunName"
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $taskPython -m physgauss evaluate --manifest data/synthetic/manifest.json --checkpoint "runs/$RunName/checkpoint.pt" --output "runs/$RunName/evaluation"
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
& $taskPython -m physgauss export-state --manifest data/synthetic/manifest.json --checkpoint "runs/$RunName/checkpoint.pt" --time 0.5 --output "runs/$RunName/state_t050.npz"
if ($LASTEXITCODE -ne 0) { throw 'State export failed' }
Write-Host "Finished: runs/$RunName"
