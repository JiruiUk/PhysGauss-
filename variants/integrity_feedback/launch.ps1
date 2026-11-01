$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction/scripts/enable_cuda.ps1') -ProjectRoot $taskRoot
Push-Location $taskRoot
try { & (Join-Path $taskRoot 'envs/pidg-cu128/python.exe') -u -m variants.integrity_feedback.run
    if($LASTEXITCODE -ne 0){throw "Integrity feedback failed: $LASTEXITCODE"}
} finally { Pop-Location }
