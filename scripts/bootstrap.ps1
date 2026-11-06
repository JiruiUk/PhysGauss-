param(
    [string]$PythonExecutable = 'python',
    [switch]$Cpu
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProjectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $PythonExecutable -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python environment' }
}
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
if ($Cpu) {
    & $taskPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install CPU PyTorch' }
}
& $taskPython -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Failed to install PhysGauss' }
Write-Host 'Ready. Run .\scripts\run_smoke.ps1'
