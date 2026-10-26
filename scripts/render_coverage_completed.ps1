$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
Push-Location $taskRoot
try {
    & (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -u (Join-Path $taskRoot 'scripts\render_coverage_completed.py')
    if($LASTEXITCODE -ne 0){throw "Priority render failed: $LASTEXITCODE"}
} finally { Pop-Location }
