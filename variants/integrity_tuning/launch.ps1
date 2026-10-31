param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction/scripts/enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u','-m','variants.integrity_tuning.run')
if($PreflightOnly){$taskArgs+='--preflight-only'}
Push-Location $taskRoot
try { & (Join-Path $taskRoot 'envs/pidg-cu128/python.exe') @taskArgs
    if($LASTEXITCODE -ne 0){throw "Integrity queue failed: $LASTEXITCODE"}
} finally { Pop-Location }
